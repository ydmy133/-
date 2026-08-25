#include "com_host.h"
#include "usart.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define JSON_LINE_SIZE   192U
#define JSON_Q_SIZE      8U
#define SPEED_MAX        1000
#define CMD_SET_SPEED    0x01U
#define CMD_ESTOP        0x02U

static uint8_t line_buf[JSON_LINE_SIZE];
static volatile uint16_t line_len;
static char json_q[JSON_Q_SIZE][JSON_LINE_SIZE];
static volatile uint8_t json_q_w;
static volatile uint8_t json_q_r;

static int16_t cmd_T;
static int16_t cmd_Y;

static int16_t clamp_speed(int32_t v)
{
  if (v > SPEED_MAX) {
    return SPEED_MAX;
  }
  if (v < -SPEED_MAX) {
    return (int16_t)(-SPEED_MAX);
  }
  return (int16_t)v;
}

static uint8_t xor8(const uint8_t *p, uint8_t n)
{
  uint8_t x = 0;
  while (n--) {
    x ^= *p++;
  }
  return x;
}

static const char *skip_ws(const char *s)
{
  while (*s != '\0' && isspace((unsigned char)*s)) {
    s++;
  }
  return s;
}

static const char *json_after_key(const char *js, const char *key)
{
  char pat[24];
  const char *p;
  size_t n = strlen(key);

  if (n + 4U >= sizeof(pat)) {
    return NULL;
  }
  pat[0] = '"';
  memcpy(&pat[1], key, n);
  pat[n + 1U] = '"';
  pat[n + 2U] = ':';
  pat[n + 3U] = '\0';

  p = strstr(js, pat);
  if (p == NULL) {
    return NULL;
  }
  return skip_ws(p + n + 3U);
}

static int json_get_str(const char *js, const char *key, char *out, uint32_t out_sz)
{
  const char *p = json_after_key(js, key);
  uint32_t i = 0;

  if (p == NULL || *p != '"' || out_sz == 0U) {
    return -1;
  }
  p++;
  while (p[i] != '\0' && p[i] != '"' && (i + 1U) < out_sz) {
    out[i] = p[i];
    i++;
  }
  out[i] = '\0';
  return (p[i] == '"') ? 0 : -1;
}

static int json_get_int(const char *js, const char *key, int *out)
{
  const char *p = json_after_key(js, key);
  int sign = 1;
  int v = 0;
  uint8_t got = 0;

  if (p == NULL || out == NULL) {
    return -1;
  }
  if (*p == '-') {
    sign = -1;
    p++;
  } else if (*p == '+') {
    p++;
  }
  while (*p >= '0' && *p <= '9') {
    v = v * 10 + (*p - '0');
    p++;
    got = 1;
  }
  if (got == 0U) {
    return -1;
  }
  *out = sign * v;
  return 0;
}

static int parse_json_line(const char *js)
{
  char mode[16];
  int t_pct = 0;
  int y_pct = 0;

  if (js == NULL) {
    return -1;
  }
  js = skip_ws(js);
  if (js[0] != '{') {
    return -1;
  }
  if (json_get_str(js, "mode", mode, sizeof(mode)) != 0) {
    return -1;
  }

  if (strcmp(mode, "stop") == 0) {
    cmd_T = 0;
    cmd_Y = 0;
    return 0;
  }
  if (strcmp(mode, "speed") != 0) {
    return -1;
  }
  if (json_get_int(js, "T", &t_pct) != 0) {
    return -1;
  }
  if (json_get_int(js, "Y", &y_pct) != 0) {
    y_pct = 0;
  }
  if (t_pct > 100) {
    t_pct = 100;
  }
  if (t_pct < -100) {
    t_pct = -100;
  }
  if (y_pct > 100) {
    y_pct = 100;
  }
  if (y_pct < -100) {
    y_pct = -100;
  }

  cmd_T = (int16_t)(t_pct * 10);
  cmd_Y = (int16_t)(y_pct * 10);
  return 0;
}

/* 只发 USART2，不在这里抽 UART4，避免和 UART4 中断抢同一接收缓冲 */
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

static void d03_send_speed(int16_t left, int16_t right)
{
  uint8_t f[9];

  left = clamp_speed(left);
  right = clamp_speed(right);

  f[0] = 0xAA;
  f[1] = 0x55;
  f[2] = 0x06;
  f[3] = CMD_SET_SPEED;
  f[4] = (uint8_t)((uint16_t)left >> 8);
  f[5] = (uint8_t)((uint16_t)left & 0xFFU);
  f[6] = (uint8_t)((uint16_t)right >> 8);
  f[7] = (uint8_t)((uint16_t)right & 0xFFU);
  f[8] = xor8(&f[2], 6);

  /* 连发两遍：第一遍可能被自动换向吃掉帧头，第二遍完整。之后不再续发。 */
  uart2_write(f, sizeof(f));
  uart2_write(f, sizeof(f));
}

static void d03_send_estop(void)
{
  uint8_t f[5];

  f[0] = 0xAA;
  f[1] = 0x55;
  f[2] = 0x02;
  f[3] = CMD_ESTOP;
  f[4] = xor8(&f[2], 2);

  uart2_write(f, sizeof(f));
  uart2_write(f, sizeof(f));
}

static void uart4_feed_byte(uint8_t b)
{
  if (line_len == 0U && b != '{') {
    return;
  }
  if (line_len >= (JSON_LINE_SIZE - 1U)) {
    line_len = 0;
    if (b != '{') {
      return;
    }
  }
  line_buf[line_len++] = b;
  line_buf[line_len] = '\0';
  if (b == '}') {
    uint8_t next = (uint8_t)((json_q_w + 1U) % JSON_Q_SIZE);
    if (next != json_q_r) {
      (void)memcpy(json_q[json_q_w], line_buf, (size_t)line_len + 1U);
      json_q_w = next;
    }
    line_len = 0;
    line_buf[0] = '\0';
  }
}

static void uart4_poll_rx(void)
{
  if (__HAL_UART_GET_FLAG(&huart4, UART_FLAG_ORE) != RESET) {
    __HAL_UART_CLEAR_OREFLAG(&huart4);
  }
  while (__HAL_UART_GET_FLAG(&huart4, UART_FLAG_RXNE) != RESET) {
    uart4_feed_byte((uint8_t)(huart4.Instance->DR & 0xFFU));
  }
}

void ComHost_Uart4Irq(void)
{
  uart4_poll_rx();
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

void ComHost_Init(void)
{
  cmd_T = 0;
  cmd_Y = 0;
  line_len = 0;
  json_q_w = 0;
  json_q_r = 0;

  if (huart4.hdmarx != NULL) {
    (void)HAL_DMA_Abort(huart4.hdmarx);
  }
  ATOMIC_CLEAR_BIT(huart4.Instance->CR3, USART_CR3_DMAR);
  ATOMIC_CLEAR_BIT(huart4.Instance->CR1, USART_CR1_PEIE);
  ATOMIC_CLEAR_BIT(huart4.Instance->CR3, USART_CR3_EIE);
  ATOMIC_SET_BIT(huart4.Instance->CR1, USART_CR1_RXNEIE | USART_CR1_UE);
  huart4.RxState = HAL_UART_STATE_READY;
  huart4.ErrorCode = HAL_UART_ERROR_NONE;
  uart4_puts("READY\r\n");
}

void ComHost_Poll(void)
{
  uint32_t now = HAL_GetTick();
  char ack[64];

  /* JSON 只在 UART4 中断入队，主循环只出队，不再关总中断、也不在发 485 时抽串口 */
  while (json_q_r != json_q_w) {
    if (parse_json_line(json_q[json_q_r]) == 0) {
      int16_t left;
      int16_t right;

      left = clamp_speed((int32_t)cmd_T - (int32_t)cmd_Y);
      right = clamp_speed((int32_t)cmd_T + (int32_t)cmd_Y);

      if (cmd_T == 0 && cmd_Y == 0) {
        d03_send_estop();
      } else {
        d03_send_speed(left, right);
      }

      (void)snprintf(ack, sizeof(ack), "OK T=%d Y=%d L=%d R=%d t=%lu\r\n",
                     (int)cmd_T, (int)cmd_Y, (int)left, (int)right,
                     (unsigned long)now);
      uart4_puts(ack);
    }
    json_q_r = (uint8_t)((json_q_r + 1U) % JSON_Q_SIZE);
  }
}
