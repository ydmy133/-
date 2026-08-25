#include "com_proto.h"

#include <string.h>

static uint8_t xor8(const uint8_t *p, uint8_t n)
{
  uint8_t x = 0U;
  while (n > 0U) {
    x ^= *p++;
    n--;
  }
  return x;
}

void ComJsonFramer_Reset(ComJsonFramer *f)
{
  if (f == NULL) {
    return;
  }
  f->len = 0U;
  f->brace_depth = 0;
  f->in_string = 0U;
  f->escaped = 0U;
  f->buf[0] = 0U;
}

int ComJsonFramer_Feed(ComJsonFramer *f, uint8_t b)
{
  if (f == NULL) {
    return -1;
  }

  /* 完整帧未复位时，先丢掉旧帧再重新同步 */
  if (f->len > 0U && f->brace_depth == 0) {
    ComJsonFramer_Reset(f);
  }

  if (f->len == 0U) {
    if (b != (uint8_t)'{') {
      return 0;
    }
  }

  if (f->len >= (JSON_LINE_SIZE - 1U)) {
    ComJsonFramer_Reset(f);
    if (b != (uint8_t)'{') {
      return -1;
    }
  }

  f->buf[f->len] = b;
  f->len++;
  f->buf[f->len] = 0U;

  if (f->in_string != 0U) {
    if (f->escaped != 0U) {
      f->escaped = 0U;
    } else if (b == (uint8_t)'\\') {
      f->escaped = 1U;
    } else if (b == (uint8_t)'"') {
      f->in_string = 0U;
    }
    return 0;
  }

  if (b == (uint8_t)'"') {
    f->in_string = 1U;
    return 0;
  }

  if (b == (uint8_t)'{') {
    f->brace_depth++;
    if (f->brace_depth > 1) {
      ComJsonFramer_Reset(f);
      return -1;
    }
    return 0;
  }

  if (b == (uint8_t)'}') {
    f->brace_depth--;
    if (f->brace_depth < 0) {
      ComJsonFramer_Reset(f);
      return -1;
    }
    if (f->brace_depth == 0) {
      return 1;
    }
    return 0;
  }

  return 0;
}

static void skip_ws(const char **pp)
{
  const char *p = *pp;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
    p++;
  }
  *pp = p;
}

static int skip_json_string(const char **pp)
{
  const char *p = *pp;
  if (*p != '"') {
    return -1;
  }
  p++;
  while (*p != '\0') {
    if (*p == '\\') {
      p++;
      if (*p == '\0') {
        return -1;
      }
      p++;
      continue;
    }
    if (*p == '"') {
      *pp = p + 1;
      return 0;
    }
    p++;
  }
  return -1;
}

/* 协议字段字符串：禁止转义和控制字符，避免畸形数据。 */
static int parse_quoted_strict(const char **pp, char *out, uint32_t out_sz)
{
  const char *p = *pp;
  uint32_t n = 0U;

  if (*p != '"') {
    return -1;
  }
  p++;
  while (*p != '\0' && *p != '"') {
    if (*p == '\\' || (unsigned char)*p < 0x20U) {
      return -1;
    }
    if ((n + 1U) >= out_sz) {
      return -1;
    }
    out[n] = *p;
    n++;
    p++;
  }
  if (*p != '"') {
    return -1;
  }
  out[n] = '\0';
  *pp = p + 1;
  return 0;
}

static int skip_json_number(const char **pp)
{
  const char *p = *pp;
  uint32_t digits = 0U;

  if (*p == '-') {
    p++;
  } else if (*p == '+') {
    return -1;
  }
  if (*p < '0' || *p > '9') {
    return -1;
  }
  if (*p == '0') {
    p++;
    if (*p >= '0' && *p <= '9') {
      return -1;
    }
  } else {
    while (*p >= '0' && *p <= '9') {
      digits++;
      if (digits > 10U) {
        return -1;
      }
      p++;
    }
  }
  if (*p == '.' || *p == 'e' || *p == 'E') {
    return -1;
  }
  *pp = p;
  return 0;
}

static int skip_json_value(const char **pp)
{
  const char *p;

  skip_ws(pp);
  p = *pp;
  if (*p == '"') {
    return skip_json_string(pp);
  }
  if (*p == '{' || *p == '[') {
    return -1;
  }
  if (*p == '-' || (*p >= '0' && *p <= '9')) {
    return skip_json_number(pp);
  }
  if (strncmp(p, "true", 4) == 0) {
    *pp = p + 4;
    return 0;
  }
  if (strncmp(p, "false", 5) == 0) {
    *pp = p + 5;
    return 0;
  }
  if (strncmp(p, "null", 4) == 0) {
    *pp = p + 4;
    return 0;
  }
  return -1;
}

/*
 * 解析 JSON 整数。超长/溢出直接失败，不做 wrap。
 * 调用方再检查业务范围，禁止溢出后再 clamp。
 */
static int parse_json_i32(const char **pp, int32_t *out)
{
  const char *p = *pp;
  uint8_t neg = 0U;
  uint32_t mag = 0U;
  uint32_t nd = 0U;

  if (*p == '-') {
    neg = 1U;
    p++;
  } else if (*p == '+') {
    return -1;
  }
  if (*p < '0' || *p > '9') {
    return -1;
  }
  if (*p == '0') {
    p++;
    if (*p >= '0' && *p <= '9') {
      return -1;
    }
    *out = 0;
    *pp = p;
    return 0;
  }

  while (*p >= '0' && *p <= '9') {
    uint32_t d = (uint32_t)(*p - '0');
    if (nd >= 10U) {
      return -1;
    }
    if (mag > 214748364U) {
      return -1;
    }
    if (mag == 214748364U) {
      if (neg != 0U) {
        if (d > 8U) {
          return -1;
        }
      } else if (d > 7U) {
        return -1;
      }
    }
    mag = mag * 10U + d;
    nd++;
    p++;
  }

  if (nd == 0U) {
    return -1;
  }
  if (*p == '.' || *p == 'e' || *p == 'E') {
    return -1;
  }
  if (neg != 0U) {
    if (mag > 2147483647U) {
      return -1;
    }
    *out = -(int32_t)mag;
  } else {
    *out = (int32_t)mag;
  }
  *pp = p;
  return 0;
}

static int parse_pct_field(const char **pp, int32_t *out)
{
  if (parse_json_i32(pp, out) != 0) {
    return -1;
  }
  if (*out < COM_PCT_MIN || *out > COM_PCT_MAX) {
    return -1;
  }
  return 0;
}

int ComJson_Parse(const char *js, ComCmd *out)
{
  const char *p;
  char key[16];
  char mode[16];
  int32_t t_pct = 0;
  int32_t y_pct = 0;
  uint8_t seen_mode = 0U;
  uint8_t seen_t = 0U;
  uint8_t seen_y = 0U;
  uint8_t first = 1U;

  if (js == NULL || out == NULL) {
    return -1;
  }

  p = js;
  skip_ws(&p);
  if (*p != '{') {
    return -1;
  }
  p++;

  for (;;) {
    skip_ws(&p);
    if (*p == '}') {
      p++;
      break;
    }
    if (first == 0U) {
      if (*p != ',') {
        return -1;
      }
      p++;
      skip_ws(&p);
      if (*p == '}') {
        return -1; /* 尾逗号 */
      }
    }
    first = 0U;

    if (parse_quoted_strict(&p, key, (uint32_t)sizeof(key)) != 0) {
      return -1;
    }
    skip_ws(&p);
    if (*p != ':') {
      return -1;
    }
    p++;
    skip_ws(&p);

    if (strcmp(key, "mode") == 0) {
      if (seen_mode != 0U) {
        return -1;
      }
      if (parse_quoted_strict(&p, mode, (uint32_t)sizeof(mode)) != 0) {
        return -1;
      }
      seen_mode = 1U;
    } else if (strcmp(key, "T") == 0) {
      if (seen_t != 0U) {
        return -1;
      }
      if (parse_pct_field(&p, &t_pct) != 0) {
        return -1;
      }
      seen_t = 1U;
    } else if (strcmp(key, "Y") == 0) {
      if (seen_y != 0U) {
        return -1;
      }
      if (parse_pct_field(&p, &y_pct) != 0) {
        return -1;
      }
      seen_y = 1U;
    } else {
      if (skip_json_value(&p) != 0) {
        return -1;
      }
    }
  }

  skip_ws(&p);
  if (*p != '\0') {
    return -1;
  }
  if (seen_mode == 0U) {
    return -1;
  }

  memset(out, 0, sizeof(*out));

  if (strcmp(mode, "stop") == 0) {
    out->t = 0;
    out->y = 0;
    out->is_stop = 1U;
    out->y_omitted = 0U;
    return 0;
  }
  if (strcmp(mode, "speed") != 0) {
    return -1;
  }
  if (seen_t == 0U) {
    return -1;
  }
  out->is_stop = 0U;
  out->y_omitted = (seen_y == 0U) ? 1U : 0U;
  if (seen_y == 0U) {
    y_pct = 0;
  }
  out->t = (int16_t)(t_pct * 10);
  out->y = (int16_t)(y_pct * 10);
  return 0;
}

void ComWatchdog_Init(ComWatchdog *w, uint32_t now_ms)
{
  if (w == NULL) {
    return;
  }
  w->last_valid_ms = now_ms;
  w->last_estop_ms = 0U;
  w->timed_out = 0U;
}

void ComWatchdog_OnValidCmd(ComWatchdog *w, uint32_t now_ms)
{
  if (w == NULL) {
    return;
  }
  w->last_valid_ms = now_ms;
  w->last_estop_ms = 0U;
  w->timed_out = 0U;
}

int ComWatchdog_Poll(ComWatchdog *w, uint32_t now_ms)
{
  if (w == NULL) {
    return 0;
  }
  if ((now_ms - w->last_valid_ms) < CMD_TIMEOUT_MS) {
    return 0;
  }
  w->timed_out = 1U;
  if (w->last_estop_ms == 0U || (now_ms - w->last_estop_ms) >= CMD_TIMEOUT_MS) {
    w->last_estop_ms = now_ms;
    return 1;
  }
  return 0;
}

int16_t Com_ClampSpeed(int32_t v)
{
  if (v > COM_SPEED_MAX) {
    return (int16_t)COM_SPEED_MAX;
  }
  if (v < -COM_SPEED_MAX) {
    return (int16_t)(-COM_SPEED_MAX);
  }
  return (int16_t)v;
}

int16_t Com_MixLeft(int16_t t, int16_t y)
{
  return Com_ClampSpeed((int32_t)t - (int32_t)y);
}

int16_t Com_MixRight(int16_t t, int16_t y)
{
  return Com_ClampSpeed((int32_t)t + (int32_t)y);
}

void D03_BuildSpeed(uint8_t out[D03_SPEED_FRAME_LEN], int16_t left, int16_t right)
{
  left = Com_ClampSpeed(left);
  right = Com_ClampSpeed(right);
  out[0] = 0xAAU;
  out[1] = 0x55U;
  out[2] = 0x06U;
  out[3] = D03_CMD_SET_SPEED;
  out[4] = (uint8_t)((uint16_t)left >> 8);
  out[5] = (uint8_t)((uint16_t)left & 0xFFU);
  out[6] = (uint8_t)((uint16_t)right >> 8);
  out[7] = (uint8_t)((uint16_t)right & 0xFFU);
  out[8] = xor8(&out[2], 6U);
}

void D03_BuildEstop(uint8_t out[D03_ESTOP_FRAME_LEN])
{
  out[0] = 0xAAU;
  out[1] = 0x55U;
  out[2] = 0x02U;
  out[3] = D03_CMD_ESTOP;
  out[4] = xor8(&out[2], 2U);
}
