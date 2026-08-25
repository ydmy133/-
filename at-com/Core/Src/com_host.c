#include "com_host.h"
#include "com_proto.h"
#include "usart.h"

#include <stdio.h>

#define RX_RING_SIZE 256U

/* ISR 写入、主循环读取。
 * 缓冲区本身也要 volatile，避免主循环把 rx_ring[] 当不变数据优化掉。
 * Cortex-M4 对齐的 16 位索引访问是原子的，ISR 只改 rx_w，主循环只改 rx_r。 */
static volatile uint8_t rx_ring[RX_RING_SIZE];
static volatile uint16_t rx_w;
static volatile uint16_t rx_r;
static volatile uint8_t rx_err;

static ComJsonFramer framer;
static ComWatchdog watchdog;
static int16_t cmd_T;
static int16_t cmd_Y;

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
  uint8_t f[D03_ESTOP_FRAME_LEN];
  D03_BuildEstop(f);
  uart2_write(f, (uint16_t)D03_ESTOP_FRAME_LEN);
}

static void send_speed(int16_t left, int16_t right)
{
  uint8_t f[D03_SPEED_FRAME_LEN];
  D03_BuildSpeed(f, left, right);
  uart2_write(f, (uint16_t)D03_SPEED_FRAME_LEN);
}

static void apply_cmd(const ComCmd *cmd, uint32_t now)
{
  int16_t left;
  int16_t right;
  char ack[64];

  cmd_T = cmd->t;
  cmd_Y = cmd->y;
  ComWatchdog_OnValidCmd(&watchdog, now);

  left = Com_MixLeft(cmd_T, cmd_Y);
  right = Com_MixRight(cmd_T, cmd_Y);

  if (cmd->is_stop != 0U || (cmd_T == 0 && cmd_Y == 0)) {
    send_estop();
  } else {
    send_speed(left, right);
  }

  (void)snprintf(ack, sizeof(ack), "OK T=%d Y=%d L=%d R=%d t=%lu\r\n",
                 (int)cmd_T, (int)cmd_Y, (int)left, (int)right,
                 (unsigned long)now);
  uart4_puts(ack);
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

  (void)huart4.Instance->SR;
  (void)huart4.Instance->DR;

  ATOMIC_CLEAR_BIT(huart4.Instance->CR3, USART_CR3_DMAR);
  ATOMIC_CLEAR_BIT(huart4.Instance->CR1, USART_CR1_PEIE);
  ATOMIC_SET_BIT(huart4.Instance->CR3, USART_CR3_EIE);
  ATOMIC_SET_BIT(huart4.Instance->CR1, USART_CR1_RXNEIE | USART_CR1_UE);
  huart4.RxState = HAL_UART_STATE_READY;
  huart4.ErrorCode = HAL_UART_ERROR_NONE;
  uart4_puts("READY\r\n");
}

void ComHost_Poll(void)
{
  uint32_t now = HAL_GetTick();

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
      if (ComJson_Parse((const char *)framer.buf, &cmd) == 0) {
        now = HAL_GetTick();
        apply_cmd(&cmd, now);
      }
      ComJsonFramer_Reset(&framer);
    }
  }

  now = HAL_GetTick();
  if (ComWatchdog_Poll(&watchdog, now) != 0) {
    cmd_T = 0;
    cmd_Y = 0;
    send_estop();
  }
}
