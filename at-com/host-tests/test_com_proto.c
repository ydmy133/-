/*
 * 宿主机单元测试（不依赖 STM32 HAL / Keil）。
 *
 *   gcc -std=c99 -Wall -Wextra -Werror -I../Core/Inc \
 *       -o test_com_proto.exe test_com_proto.c ../Core/Src/com_proto.c
 *   ./test_com_proto.exe
 *
 * 覆盖：边界 T/Y、超范围、超长数字、缺 T、非法 mode、字符串内 }、
 * 不完整 JSON、连续多包、超长 JSON、watchdog、stop、D03 单帧。
 */

#include "com_proto.h"

#include <stdio.h>
#include <string.h>

static int g_fail;

static void expect(int cond, const char *msg)
{
  if (!cond) {
    printf("FAIL %s\n", msg);
    g_fail++;
  }
}

static int feed_str(ComJsonFramer *f, const char *s)
{
  int last = 0;
  while (*s != '\0') {
    last = ComJsonFramer_Feed(f, (uint8_t)*s);
    s++;
    if (last == 1) {
      return 1;
    }
  }
  return last;
}

static void test_parse_valid(void)
{
  ComCmd cmd;

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":100,\"Y\":100}", &cmd) == 0, "T=100 Y=100 parse");
  expect(cmd.is_stop == 0U && cmd.t == 1000 && cmd.y == 1000, "T=100 Y=100 scale");

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":-100,\"Y\":-100}", &cmd) == 0, "T=-100 Y=-100 parse");
  expect(cmd.t == -1000 && cmd.y == -1000, "T=-100 Y=-100 scale");

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50,\"Y\":20}", &cmd) == 0, "frontend speed");
  expect(cmd.t == 500 && cmd.y == 200, "frontend scale");

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50}", &cmd) == 0, "Y omitted");
  expect(cmd.y == 0 && cmd.y_omitted == 1U && cmd.t == 500, "Y default 0");

  expect(ComJson_Parse("{\"mode\":\"stop\"}", &cmd) == 0, "stop");
  expect(cmd.is_stop == 1U && cmd.t == 0 && cmd.y == 0, "stop zeros");

  expect(ComJson_Parse(" { \"mode\" : \"stop\" } ", &cmd) == 0, "stop ws");
  expect(ComJson_Parse("{\"T\":1,\"mode\":\"speed\",\"Y\":2}", &cmd) == 0, "key order");
  expect(cmd.t == 10 && cmd.y == 20, "key order values");
}

static void test_parse_invalid(void)
{
  ComCmd cmd;

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":150,\"Y\":0}", &cmd) != 0, "T=150");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":-150,\"Y\":0}", &cmd) != 0, "T=-150");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":100,\"Y\":101}", &cmd) != 0, "Y=101");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"Y\":20}", &cmd) != 0, "missing T");
  expect(ComJson_Parse("{\"mode\":\"sped\",\"T\":1,\"Y\":0}", &cmd) != 0, "illegal mode");
  expect(ComJson_Parse("{\"mode\":\"stopplus\"}", &cmd) != 0, "stop prefix");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":2147483647}", &cmd) != 0, "huge T");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":999999999999}", &cmd) != 0, "overlong digits");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50.5,\"Y\":0}", &cmd) != 0, "float T");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":+1,\"Y\":0}", &cmd) != 0, "plus sign");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":01,\"Y\":0}", &cmd) != 0, "leading zero");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"XT\":1}", &cmd) != 0, "XT not T");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50abc,\"Y\":0}", &cmd) != 0, "junk after number");
  expect(ComJson_Parse("{mode:\"stop\"}", &cmd) != 0, "unquoted key");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50,}", &cmd) != 0, "trailing comma");
  expect(ComJson_Parse("not-json", &cmd) != 0, "not json");
  expect(ComJson_Parse("{}", &cmd) != 0, "empty object");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":50,\"T\":1}", &cmd) != 0, "dup T");
}

static void test_framer(void)
{
  ComJsonFramer f;
  ComCmd cmd;
  int done;
  char longbuf[JSON_LINE_SIZE + 32];
  size_t i;

  ComJsonFramer_Reset(&f);
  expect(feed_str(&f, "{\"mode\":\"stop\"}") == 1, "framer stop");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.is_stop, "framer stop parse");
  ComJsonFramer_Reset(&f);

  expect(feed_str(&f, "{\"mode\":\"speed\",\"note\":\"has}brace\",\"T\":1,\"Y\":0}") == 1,
         "brace in string");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.t == 10, "brace in string parse");
  ComJsonFramer_Reset(&f);

  expect(feed_str(&f, "{\"mode\":\"speed\",\"note\":\"say \\\"}x\",\"T\":2,\"Y\":0}") == 1,
         "escaped quote then brace");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.t == 20,
         "escaped quote then brace parse");
  ComJsonFramer_Reset(&f);

  expect(feed_str(&f, "{\"mode\":\"speed\",\"T\":1") != 1, "incomplete");
  ComJsonFramer_Reset(&f);

  done = 0;
  expect(ComJsonFramer_Feed(&f, '{') == 0, "first of concat");
  {
    const char *p = "\"mode\":\"stop\"}";
    while (*p != '\0') {
      if (ComJsonFramer_Feed(&f, (uint8_t)*p) == 1) {
        done = 1;
        break;
      }
      p++;
    }
    expect(done == 1, "first json complete");
    expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.is_stop, "first concat stop");
  }
  ComJsonFramer_Reset(&f);
  expect(feed_str(&f, "{\"mode\":\"speed\",\"T\":1,\"Y\":2}") == 1, "second concat");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.t == 10 && cmd.y == 20,
         "second concat speed");

  ComJsonFramer_Reset(&f);
  longbuf[0] = '{';
  for (i = 1; i < (sizeof(longbuf) - 2U); i++) {
    longbuf[i] = 'x';
  }
  longbuf[sizeof(longbuf) - 2U] = '}';
  longbuf[sizeof(longbuf) - 1U] = '\0';
  expect(feed_str(&f, longbuf) != 1, "overlong discarded");

  ComJsonFramer_Reset(&f);
  expect(feed_str(&f, "{{") != 1, "illegal nest");
}

static void test_watchdog(void)
{
  ComWatchdog w;
  ComCmd cmd;

  ComWatchdog_Init(&w, 0);
  expect(ComWatchdog_Poll(&w, 299) == 0, "wd 299");
  expect(ComWatchdog_Poll(&w, 300) == 1, "wd 300 fire");
  expect(w.timed_out == 1U, "wd timed out");
  expect(ComWatchdog_Poll(&w, 301) == 0, "wd no spam");
  expect(ComWatchdog_Poll(&w, 600) == 1, "wd repeat estop");

  expect(ComJson_Parse("not-json", &cmd) != 0, "invalid does not arm");
  expect(w.timed_out == 1U, "invalid json does not clear timeout");

  ComWatchdog_OnValidCmd(&w, 800);
  expect(w.timed_out == 0U, "valid cmd recovers");
  expect(ComWatchdog_Poll(&w, 1099) == 0, "wd after cmd 299ms");
  expect(ComWatchdog_Poll(&w, 1100) == 1, "wd after cmd 300ms");

  ComWatchdog_OnValidCmd(&w, 2000);
  expect(ComWatchdog_Poll(&w, 2000) == 0, "same tick no timeout");
}

static void test_d03(void)
{
  uint8_t f[D03_SPEED_FRAME_LEN];
  uint8_t e[D03_ESTOP_FRAME_LEN];
  uint8_t x;
  int16_t left;
  int16_t right;

  D03_BuildSpeed(f, 0, 0);
  expect(f[0] == 0xAA && f[1] == 0x55 && f[2] == 0x06 && f[3] == 0x01, "speed header");
  expect(f[4] == 0 && f[5] == 0 && f[6] == 0 && f[7] == 0, "zero speed");
  x = (uint8_t)(f[2] ^ f[3] ^ f[4] ^ f[5] ^ f[6] ^ f[7]);
  expect(f[8] == x, "speed xor");

  D03_BuildSpeed(f, 1000, -1000);
  expect(f[4] == 0x03 && f[5] == 0xE8, "1000 be");
  expect(f[6] == 0xFC && f[7] == 0x18, "-1000 be");

  D03_BuildSpeed(f, -1, -1);
  expect(f[4] == 0xFF && f[5] == 0xFF && f[6] == 0xFF && f[7] == 0xFF, "neg one");

  left = Com_MixLeft(1000, 1000);
  right = Com_MixRight(1000, 1000);
  expect(left == 0 && right == 1000, "mix clamp right");

  D03_BuildEstop(e);
  expect(e[0] == 0xAA && e[1] == 0x55 && e[2] == 0x02 && e[3] == 0x02, "estop header");
  expect(e[4] == (uint8_t)(e[2] ^ e[3]), "estop xor");
}

int main(void)
{
  test_parse_valid();
  test_parse_invalid();
  test_framer();
  test_watchdog();
  test_d03();
  if (g_fail != 0) {
    printf("%d failed\n", g_fail);
    return 1;
  }
  printf("com_proto host tests ok\n");
  return 0;
}
