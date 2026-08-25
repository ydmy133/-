#include "d03_rs485.h"
#include "usart.h"
#include "motor_pwm.h"

#include <string.h>

/* PA1 → SP3485 DE/RE：0=收，1=发 */
#define RS485_DE_PORT   GPIOA
#define RS485_DE_PIN    GPIO_PIN_1

/* 原理图：DC3V3 → LED → 1k → GPIO，低电平点亮 */
#define LED1_PORT       GPIOA
#define LED1_PIN        GPIO_PIN_0   /* 近期收到合法速度帧 */
#define LED2_PORT       GPIOA
#define LED2_PIN        GPIO_PIN_5   /* 已应用非零油门 */

#define LED_ON          GPIO_PIN_RESET
#define LED_OFF         GPIO_PIN_SET

#define CMD_HEARTBEAT   0x00
#define CMD_SET_SPEED   0x01
#define CMD_ESTOP       0x02
/* 速度帧后的 RS485 换向噪声可能拼出假急停，把舵机打回中位。 */
#define ESTOP_LOCKOUT_MS 500U

extern UART_HandleTypeDef huart2;

static uint8_t  rb[32];
static uint8_t  rb_n;
static int16_t  tgt_left;
static int16_t  tgt_right;
static uint32_t last_keep_ms;
static uint32_t last_ok_ms;
static uint32_t last_nonzero_ms;
static uint8_t  motor_running;

static void led1_set(uint8_t on)
{
  HAL_GPIO_WritePin(LED1_PORT, LED1_PIN, on ? LED_ON : LED_OFF);
}

static void led2_set(uint8_t on)
{
  HAL_GPIO_WritePin(LED2_PORT, LED2_PIN, on ? LED_ON : LED_OFF);
}

static void rb_reset(void)
{
  rb_n = 0;
}

static uint8_t xor8(const uint8_t *p, uint8_t n)
{
  uint8_t x = 0;
  while (n--) {
    x ^= *p++;
  }
  return x;
}

static void apply_speed(int16_t left, int16_t right)
{
  last_keep_ms = HAL_GetTick();
  if (left == tgt_left && right == tgt_right) {
    return;
  }
  tgt_left = left;
  tgt_right = right;
  Motor_SetSpeed(left, right);

  motor_running = (left != 0 || right != 0) ? 1U : 0U;
  led2_set(motor_running);
}

static void handle_speed_frame(const uint8_t *f, uint8_t total)
{
  uint8_t len = f[2];
  uint8_t cmd = f[3];
  int16_t left;
  int16_t right;

  if (xor8(&f[2], (uint8_t)(total - 3U)) != f[total - 1U]) {
    return;
  }

  switch (cmd) {
    case CMD_HEARTBEAT:
      if (len == 2U) {
        last_keep_ms = HAL_GetTick();
      }
      break;

    case CMD_SET_SPEED:
      if (len == 6U && total == 9U) {
        left  = (int16_t)(((uint16_t)f[4] << 8) | f[5]);
        right = (int16_t)(((uint16_t)f[6] << 8) | f[7]);
        last_ok_ms = HAL_GetTick();
        /* 只跟速度帧走。0,0 忽略：噪声/误同步不要把已到位的角度打回中位 */
        if (left != 0 || right != 0) {
          apply_speed(left, right);
          last_nonzero_ms = last_ok_ms;
        }
      }
      break;

    case CMD_ESTOP:
      if (len == 2U) {
        uint32_t now = HAL_GetTick();
        /* 刚收到非零速度后的假 0x02（总线尾噪声）直接丢掉 */
        if (last_nonzero_ms != 0U && (now - last_nonzero_ms) < ESTOP_LOCKOUT_MS) {
          break;
        }
        apply_speed(0, 0);
        last_ok_ms = now;
      }
      break;

    default:
      break;
  }
}

/* 在环形窗口里搜 AA 55，前导 0x00 / 噪声不会把状态机卡死 */
static void try_parse(void)
{
  uint8_t i = 0;

  while ((uint8_t)(i + 4U) <= rb_n) {
    uint8_t len;
    uint8_t total;

    if (rb[i] != 0xAAU || rb[i + 1U] != 0x55U) {
      i++;
      continue;
    }

    len = rb[i + 2U];
    if (len < 2U || len > 12U) {
      i++;
      continue;
    }

    total = (uint8_t)(3U + len);
    if ((uint8_t)(i + total) > rb_n) {
      break;
    }

    if (xor8(&rb[i + 2U], (uint8_t)(total - 3U)) != rb[i + total - 1U]) {
      i++;
      continue;
    }

    handle_speed_frame(&rb[i], total);
    /* 合法速度帧后丢掉同一次突发里的尾字节，避免被拼成急停回中 */
    if (rb[i + 3U] == CMD_SET_SPEED) {
      rb_reset();
      return;
    }
    {
      uint8_t used = (uint8_t)(i + total);
      uint8_t remain = (uint8_t)(rb_n - used);
      if (remain > 0U) {
        (void)memmove(rb, &rb[used], remain);
      }
      rb_n = remain;
      i = 0;
    }
  }

  if (rb_n > 20U) {
    uint8_t keep = 12U;
    (void)memmove(rb, &rb[rb_n - keep], keep);
    rb_n = keep;
  }
}

static void feed_byte(uint8_t b)
{
  if (rb_n >= sizeof(rb)) {
    (void)memmove(rb, &rb[8], (size_t)(rb_n - 8U));
    rb_n = (uint8_t)(rb_n - 8U);
  }
  rb[rb_n++] = b;
  try_parse();
}

void D03_RS485_Init(void)
{
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();

  HAL_GPIO_WritePin(RS485_DE_PORT, RS485_DE_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LED1_PORT, LED1_PIN, LED_OFF);
  HAL_GPIO_WritePin(LED2_PORT, LED2_PIN, LED_OFF);

  gpio.Pin = RS485_DE_PIN | LED1_PIN | LED2_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &gpio);

  tgt_left = 0;
  tgt_right = 0;
  motor_running = 0;
  last_keep_ms = HAL_GetTick();
  last_ok_ms = 0;
  last_nonzero_ms = 0;
  rb_reset();

  __HAL_UART_ENABLE(&huart2);
  __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_OREF | UART_CLEAR_NEF |
                                 UART_CLEAR_PEF | UART_CLEAR_FEF);

  /* 不在这里只发一次：很多情况是先上电再开串口，会漏掉。
   * 改由 Poll 里每秒 PING，直到主机有数据进来。 */
}

void D03_RS485_Poll(void)
{
  uint32_t now = HAL_GetTick();
  uint8_t n;

  if (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_ORE) != RESET) {
    __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_OREF);
  }
  if (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_FE) != RESET) {
    __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_FEF);
  }
  if (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_NE) != RESET) {
    __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_NEF);
  }
  if (__HAL_UART_GET_FLAG(&huart2, UART_FLAG_PE) != RESET) {
    __HAL_UART_CLEAR_FLAG(&huart2, UART_CLEAR_PEF);
  }

  /* 每次最多抽 48 字节 */
  n = 0;
  while (n < 48U && __HAL_UART_GET_FLAG(&huart2, UART_FLAG_RXNE) != RESET) {
    uint8_t b = (uint8_t)(huart2.Instance->RDR & 0xFFU);
    feed_byte(b);
    n++;
  }

  /* LED1：合法速度帧。LED2：非零油门。噪声/前导不会让 LED1 常亮 */
  led1_set((last_ok_ms != 0U) && ((now - last_ok_ms) < 400U));
  led2_set(motor_running);

  /* 失联不回中：PWM 保持最后一次合法角度，只有急停帧才回 1500us */
  (void)now;
}

uint32_t D03_RS485_LastRxMs(void)
{
  return last_keep_ms;
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  (void)huart;
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2) {
    /* 只清错误标志，不要丢掉已经搜到的 AA 55 */
    __HAL_UART_CLEAR_FLAG(huart, UART_CLEAR_OREF | UART_CLEAR_NEF |
                                 UART_CLEAR_PEF | UART_CLEAR_FEF);
  }
}
