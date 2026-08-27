#include "com_host.h"
#include "com_proto.h"
#include "usart.h"

#include <stdio.h>

#define RX_RING_SIZE 256U
/* 0：单帧 speed 后保持角度，不超时急停。改回 1 即恢复 300ms 掉线保护。 */
#ifndef COM_CMD_TIMEOUT_EN
#define COM_CMD_TIMEOUT_EN 0
#endif

/* ISR 写入、主循环读取。
 * 缓冲区本身也要 volatile，避免主循环把 rx_ring[] 当不变数据优化掉。
 * Cortex-M4 对齐的 16 位索引访问是原子的，ISR 只改 rx_w，主循环只改 rx_r。 */
static volatile uint8_t rx_ring[RX_RING_SIZE];
static volatile uint16_t rx_w;
static volatile uint16_t rx_r;
static volatile uint8_t rx_err;

static ComJsonFramer framer;
static ComWatchdog watchdog;
static ComDedup dedup;
static int16_t cmd_T;
static int16_t cmd_Y;

/* RS485 总线空闲→驱动的第一个起始沿最易失真（表现为 D03 侧首字节
 * 畸变成 D5/EA/FA）。帧前垫 2 字节 0x00 让收发器先把总线驱动稳；
 * D03 的滑窗解析器会把 0x00 当垃圾前缀滑过，两端无需协商。 */
#define RS485_PREAMBLE_LEN 2U
static const uint8_t rs485_preamble[RS485_PREAMBLE_LEN] = {0x00U, 0x00U};

static void uart2_write(const uint8_t *p, uint16_t n)
{
  while (n > 0U) {
    uint32_t t0 = HAL_GetTick();
    while (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_TXE) == RESET) {
      if ((HAL_GetTick() - t0) > 5U) {
        return;
      }
    }
    huart2.Instance->DR = *p++;
    n--;
  }
  {
    uint32_t t0 = HAL_GetTick();
    while (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_TC) == RESET) {
      if ((HAL_GetTick() - t0) > 5U) {
        break;
      }
    }
  }
}

static void uart4_puts(const char *s)
{
  uint32_t t0 = HAL_GetTick();

  while (*s != '\0') {
    if (__HAL_UART_GET_FLAG(&huart4, UART_FLAG_TXE) != RESET) {
      huart4.Instance->DR = (uint8_t)*s++;
    }
    if ((HAL_GetTick() - t0) > 30U) {
      break;
    }
  }
}

static void send_estop(void)
{
  uint8_t f[RS485_PREAMBLE_LEN + D03_ESTOP_FRAME_LEN];
  uint8_t i;

  for (i = 0U; i < RS485_PREAMBLE_LEN; i++) {
    f[i] = rs485_preamble[i];
  }
  D03_BuildEstop(&f[RS485_PREAMBLE_LEN]);
  uart2_write(f, (uint16_t)sizeof(f));
}

static void send_speed(int16_t left, int16_t right)
{
  uint8_t f[RS485_PREAMBLE_LEN + D03_SPEED_FRAME_LEN];
  uint8_t i;

  for (i = 0U; i < RS485_PREAMBLE_LEN; i++) {
    f[i] = rs485_preamble[i];
  }
  D03_BuildSpeed(&f[RS485_PREAMBLE_LEN], left, right);
  uart2_write(f, (uint16_t)sizeof(f));
}

static void uart4_putln(const char *s)
{
  char line[96];
  size_t n = 0U;

  while (s[n] != '\0' && n < (sizeof(line) - 3U)) {
    line[n] = s[n];
    n++;
  }
  line[n] = '\r';
  line[n + 1U] = '\n';
  line[n + 2U] = '\0';
  uart4_puts(line);
}

/* 解析失败回 NACK，让 MQTTX/串口侧能看到指令被拒，而不是静默丢弃 */
static void send_err(int code)
{
  const char *why;

  switch (code) {
    case COM_ERR_MODE:
      why = "ERR MODE (stop|speed)";
      break;
    case COM_ERR_MISSING:
      why = "ERR MISSING (mode/T)";
      break;
    case COM_ERR_RANGE:
      why = "ERR RANGE (T,Y in -100..100)";
      break;
    default:
      why = "ERR SYNTAX";
      break;
  }
  uart4_putln(why);
}

static void apply_cmd(const ComCmd *cmd, uint32_t now)
{
  int16_t left = 0;
  int16_t right = 0;
  char ack[80];
  int send;
  const char *tag;

  cmd_T = cmd->t;
  cmd_Y = cmd->y;
  ComWatchdog_OnValidCmd(&watchdog, now);

  send = ComDedup_ShouldSend(&dedup, cmd, now);

  if (cmd->is_stop != 0U) {
    /* 只有显式 stop 走急停帧；speed 模式 T=0 改发速度帧 (0,0)，
     * 避免 D03 侧急停锁窗把紧跟速度帧的停车指令吞掉。 */
    send_estop();
    tag = "STOP";
  } else {
    left = Com_MixLeft(cmd_T, cmd_Y);
    right = Com_MixRight(cmd_T, cmd_Y);
    if (send != 0) {
      send_speed(left, right);
      tag = "OK";
    } else {
      tag = "OK DUP";
    }
  }

  (void)snprintf(ack, sizeof(ack),
#if COM_CMD_TIMEOUT_EN
                 "%s T=%d Y=%d L=%d R=%d t=%lu",
#else
                 "%s HOLD T=%d Y=%d L=%d R=%d t=%lu",
#endif
                 tag, (int)cmd_T, (int)cmd_Y, (int)left, (int)right,
                 (unsigned long)now);
  uart4_putln(ack);
}

void ComHost_Uart4Irq(void)
{
  USART_TypeDef *u = huart4.Instance;
  uint32_t sr = u->SR;
  uint8_t b;

  if ((sr & (USART_SR_RXNE | USART_SR_ORE | USART_SR_FE | USART_SR_NE | USART_SR_PE)) == 0U) {
    return;
  }

  /* 先读 DR：清除 RXNE，并按手册用 SR+DR 序列清除 ORE/FE/NE */
  b = (uint8_t)(u->DR & 0xFFU);

  if ((sr & (USART_SR_ORE | USART_SR_FE | USART_SR_NE | USART_SR_PE)) != 0U) {
    rx_err = 1U;
    return;
  }

  if ((sr & USART_SR_RXNE) != 0U) {
    uint16_t w = rx_w;
    uint16_t next = (uint16_t)((w + 1U) % RX_RING_SIZE);
    if (next == rx_r) {
      rx_err = 1U;
    } else {
      rx_ring[w] = b;
      rx_w = next;
    }
  }
}

void ComHost_Init(void)
{
  cmd_T = 0;
  cmd_Y = 0;
  rx_w = 0U;
  rx_r = 0U;
  rx_err = 0U;
  ComJsonFramer_Reset(&framer);
  ComWatchdog_Init(&watchdog, HAL_GetTick());
  ComDedup_Init(&dedup);

  (void)huart4.Instance->SR;
  (void)huart4.Instance->DR;

  ATOMIC_CLEAR_BIT(huart4.Instance->CR3, USART_CR3_DMAR);
  ATOMIC_CLEAR_BIT(huart4.Instance->CR1, USART_CR1_PEIE);
  ATOMIC_SET_BIT(huart4.Instance->CR3, USART_CR3_EIE);
  ATOMIC_SET_BIT(huart4.Instance->CR1, USART_CR1_RXNEIE | USART_CR1_UE);
  huart4.RxState = HAL_UART_STATE_READY;
  huart4.ErrorCode = HAL_UART_ERROR_NONE;
#if COM_CMD_TIMEOUT_EN
  uart4_puts("READY\r\n");
#else
  uart4_puts("READY HOLD\r\n");
#endif
}

void ComHost_Poll(void)
{
  if (rx_err != 0U) {
    uint16_t w;
    rx_err = 0U;
    w = rx_w;
    rx_r = w;
    ComJsonFramer_Reset(&framer);
  }

  while (rx_r != rx_w) {
    uint8_t b;
    int fr;

    b = rx_ring[rx_r];
    rx_r = (uint16_t)((rx_r + 1U) % RX_RING_SIZE);
    fr = ComJsonFramer_Feed(&framer, b);
    if (fr == 1) {
      ComCmd cmd;
      int rc = ComJson_Parse((const char *)framer.buf, &cmd);
      if (rc == 0) {
        apply_cmd(&cmd, HAL_GetTick());
      } else {
        send_err(rc);
      }
      ComJsonFramer_Reset(&framer);
    }
  }

#if COM_CMD_TIMEOUT_EN
  {
    uint32_t now = HAL_GetTick();
    if (ComWatchdog_Poll(&watchdog, now) != 0) {
      cmd_T = 0;
      cmd_Y = 0;
      send_estop();
    }
  }
#endif
}
