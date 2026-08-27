/*
 * 宿主机单元测试（不依赖 STM32 HAL / Keil）。
 *
 *   gcc -std=c99 -Wall -Wextra -Werror -I../Core/Inc \
 *       -o test_com_proto.exe test_com_proto.c ../Core/Src/com_proto.c
 *   ./test_com_proto.exe
 *
 * 覆盖：边界 T/Y、超范围、超长数字、缺 T、非法 mode、字符串内 }、
 * 不完整 JSON、连续多包、超长 JSON、watchdog 3s、stop、D03 单帧、
 * 无人船协议 control_mode / 断连保护 / 导航拒绝。
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
  expect(cmd.dc_prot == 1U && cmd.nav_unsupported == 0U, "legacy dc_prot default");

  expect(ComJson_Parse("{\"mode\":\"stop\"}", &cmd) == 0, "stop");
  expect(cmd.is_stop == 1U && cmd.t == 0 && cmd.y == 0 && cmd.dc_prot == 1U,
         "stop zeros dc_prot default");

  expect(ComJson_Parse("{\"mode\":\"speed\",\"T\":0,\"Y\":0}", &cmd) == 0, "speed zeros");
  expect(cmd.is_stop == 1U && cmd.t == 0 && cmd.y == 0, "speed zeros is stop");

  expect(ComJson_Parse(" { \"mode\" : \"stop\" } ", &cmd) == 0, "stop ws");
  expect(ComJson_Parse("{\"T\":1,\"mode\":\"speed\",\"Y\":2}", &cmd) == 0, "key order");
  expect(cmd.t == 10 && cmd.y == 20, "key order values");
}

static void test_parse_invalid(void)
{
  ComCmd cmd;
  int rc;

  rc = ComJson_Parse("{\"mode\":\"speed\",\"T\":150,\"Y\":0}", &cmd);
  expect(rc == COM_ERR_RANGE, "T=150 range code");
  rc = ComJson_Parse("{\"mode\":\"speed\",\"T\":-150,\"Y\":0}", &cmd);
  expect(rc == COM_ERR_RANGE, "T=-150 range code");
  rc = ComJson_Parse("{\"mode\":\"speed\",\"T\":100,\"Y\":101}", &cmd);
  expect(rc == COM_ERR_RANGE, "Y=101 range code");
  rc = ComJson_Parse("{\"mode\":\"speed\",\"Y\":20}", &cmd);
  expect(rc == COM_ERR_MISSING, "missing T code");
  rc = ComJson_Parse("{\"mode\":\"sped\",\"T\":1,\"Y\":0}", &cmd);
  expect(rc == COM_ERR_MODE, "illegal mode code");
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
  rc = ComJson_Parse("{\"mode\":1}", &cmd);
  expect(rc == COM_ERR_SYNTAX, "mode not string code");
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
  expect(ComWatchdog_Poll(&w, 2999) == 0, "wd 2999");
  expect(ComWatchdog_Poll(&w, 3000) == 1, "wd 3000 fire");
  expect(w.timed_out == 1U, "wd timed out");
  expect(ComWatchdog_Poll(&w, 3001) == 0, "wd no spam");
  expect(ComWatchdog_Poll(&w, 6000) == 1, "wd repeat estop");

  expect(ComJson_Parse("not-json", &cmd) != 0, "invalid does not arm");
  expect(w.timed_out == 1U, "invalid json does not clear timeout");

  ComWatchdog_OnValidCmd(&w, 800);
  expect(w.timed_out == 0U, "valid cmd recovers");
  expect(ComWatchdog_Poll(&w, 3799) == 0, "wd after cmd 2999ms");
  expect(ComWatchdog_Poll(&w, 3800) == 1, "wd after cmd 3000ms");

  ComWatchdog_OnValidCmd(&w, 2000);
  expect(ComWatchdog_Poll(&w, 2000) == 0, "same tick no timeout");
}

static void test_dedup(void)
{
  ComDedup d;
  ComCmd cmd;

  ComDedup_Init(&d);

  /* 首条必发 */
  memset(&cmd, 0, sizeof(cmd));
  cmd.t = 100;
  cmd.y = 0;
  expect(ComDedup_ShouldSend(&d, &cmd, 1000) == 1, "first speed sent");

  /* 窗口内同值副本拦截（QoS 重传场景） */
  expect(ComDedup_ShouldSend(&d, &cmd, 1050) == 0, "dup within window");
  expect(ComDedup_ShouldSend(&d, &cmd, 1100) == 0, "dup at window edge");
  /* 窗口外同值放行（100ms 周期重发的合法场景） */
  expect(ComDedup_ShouldSend(&d, &cmd, 1201) == 1, "same value after window");

  /* 值变了立即放行 */
  cmd.t = -100;
  expect(ComDedup_ShouldSend(&d, &cmd, 1210) == 1, "changed value sent");
  cmd.t = 100;
  expect(ComDedup_ShouldSend(&d, &cmd, 1215) == 1, "value back sent");

  /* 急停永不去重 */
  memset(&cmd, 0, sizeof(cmd));
  cmd.is_stop = 1U;
  expect(ComDedup_ShouldSend(&d, &cmd, 1220) == 1, "estop 1 sent");
  expect(ComDedup_ShouldSend(&d, &cmd, 1221) == 1, "estop 2 sent");
  expect(ComDedup_ShouldSend(&d, &cmd, 1222) == 1, "estop 3 sent");

  /* 急停后的 speed (0,0)：值不同（last_stop=1）必须放行 */
  memset(&cmd, 0, sizeof(cmd));
  expect(ComDedup_ShouldSend(&d, &cmd, 1230) == 1, "speed 0 after stop sent");
  expect(ComDedup_ShouldSend(&d, &cmd, 1231) == 0, "speed 0 dup blocked");
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

static const char kProtoManual[] =
    "{\"control_mode\":\"manual\",\"move_speed\":50,\"steer_speed\":0,"
    "\"target_lat\":0.0,\"target_lon\":0.0,"
    "\"return_lat\":31.230400,\"return_lon\":121.473700,"
    "\"dc_prot\":1,\"lb_prot\":1}";

static const char kProtoLeft[] =
    "{\"control_mode\":\"manual\",\"move_speed\":30,\"steer_speed\":50,"
    "\"target_lat\":0.0,\"target_lon\":0.0,"
    "\"return_lat\":31.230400,\"return_lon\":121.473700,"
    "\"dc_prot\":1,\"lb_prot\":1}";

static const char kProtoNav[] =
    "{\"control_mode\":\"navigate\",\"move_speed\":0,\"steer_speed\":0,"
    "\"target_lat\":31.230416,\"target_lon\":121.473701,"
    "\"return_lat\":31.230400,\"return_lon\":121.473700,"
    "\"dc_prot\":1,\"lb_prot\":1}";

static void test_protocol_parse(void)
{
  ComCmd cmd;
  int rc;

  rc = ComJson_Parse(kProtoManual, &cmd);
  expect(rc == 0, "proto manual parse");
  expect(cmd.nav_unsupported == 0U && cmd.is_stop == 0U, "proto manual flags");
  expect(cmd.t == 500 && cmd.y == 0 && cmd.dc_prot == 1U, "proto manual scale");

  rc = ComJson_Parse(kProtoLeft, &cmd);
  expect(rc == 0 && cmd.t == 300 && cmd.y == 500, "proto left turn");
  expect(Com_MixLeft(cmd.t, cmd.y) == -200 && Com_MixRight(cmd.t, cmd.y) == 800,
         "proto mix left");

  expect(ComJson_Parse("{\"control_mode\":\"manual\",\"move_speed\":-40,"
                       "\"steer_speed\":0}",
                       &cmd) == 0,
         "proto reverse");
  expect(cmd.t == -400 && cmd.y == 0 && cmd.is_stop == 0U, "proto reverse scale");

  expect(ComJson_Parse("{\"control_mode\":\"manual\",\"move_speed\":0,"
                       "\"steer_speed\":0}",
                       &cmd) == 0,
         "proto stop 0/0");
  expect(cmd.is_stop == 1U && cmd.t == 0 && cmd.y == 0, "proto 0/0 is stop");

  expect(ComJson_Parse("{\"control_mode\":\"cruise_speed\",\"move_speed\":60,"
                       "\"steer_speed\":50}",
                       &cmd) == 0,
         "cruise_speed");
  expect(cmd.t == 600 && cmd.y == 0 && cmd.y_omitted == 1U, "cruise ignores steer");

  expect(ComJson_Parse("{\"control_mode\":\"cruise_dir\",\"move_speed\":60}",
                       &cmd) == 0,
         "cruise_dir");
  expect(cmd.t == 600 && cmd.y == 0, "cruise_dir straight");

  rc = ComJson_Parse(kProtoNav, &cmd);
  expect(rc == 0 && cmd.nav_unsupported == 1U, "navigate parse ok not apply");
  expect(ComJson_Parse("{\"control_mode\":\"stable_anchor\"}", &cmd) == 0 &&
             cmd.nav_unsupported == 1U,
         "stable_anchor unsupported");
  expect(ComJson_Parse("{\"control_mode\":\"fixed_point\",\"target_lat\":1.0,"
                       "\"target_lon\":2.0}",
                       &cmd) == 0 &&
             cmd.nav_unsupported == 1U,
         "fixed_point unsupported");

  expect(ComJson_Parse("{\"control_mode\":\"manual\",\"move_speed\":10,"
                       "\"dc_prot\":0}",
                       &cmd) == 0 &&
             cmd.dc_prot == 0U && cmd.t == 100,
         "dc_prot 0");
  rc = ComJson_Parse("{\"control_mode\":\"manual\",\"move_speed\":10,"
                     "\"dc_prot\":2}",
                     &cmd);
  expect(rc == COM_ERR_RANGE, "dc_prot 2 range");

  rc = ComJson_Parse("{\"control_mode\":\"auto\"}", &cmd);
  expect(rc == COM_ERR_MODE, "unknown control_mode");
  rc = ComJson_Parse("{\"move_speed\":10}", &cmd);
  expect(rc == COM_ERR_MISSING, "missing control_mode");
  expect(ComJson_Parse("{\"mode\":\"speed\",\"control_mode\":\"manual\","
                       "\"T\":1}",
                       &cmd) != 0,
         "mode and control_mode");
}

static void test_protocol_framer(void)
{
  ComJsonFramer f;
  ComCmd cmd;
  char long_ok[300];
  size_t n;
  size_t i;

  ComJsonFramer_Reset(&f);
  expect(feed_str(&f, kProtoManual) == 1, "framer proto manual");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.t == 500,
         "framer proto parse");

  /* 超过旧 192 上限、仍小于 384：必须拼出完整对象 */
  {
    static const char kPrefix[] =
        "{\"control_mode\":\"manual\",\"move_speed\":10,\"pad\":\"";
    n = sizeof(kPrefix) - 1U;
    memcpy(long_ok, kPrefix, n);
  }
  for (i = 0; i < 160U; i++) {
    long_ok[n++] = 'a';
  }
  long_ok[n++] = '"';
  long_ok[n++] = '}';
  long_ok[n] = '\0';
  expect(n > 192U && n < JSON_LINE_SIZE, "mid-size length");
  ComJsonFramer_Reset(&f);
  expect(feed_str(&f, long_ok) == 1, "framer mid-size >192");
  expect(ComJson_Parse((const char *)f.buf, &cmd) == 0 && cmd.t == 100,
         "framer mid-size parse");
}

int main(void)
{
  test_parse_valid();
  test_parse_invalid();
  test_framer();
  test_watchdog();
  test_dedup();
  test_d03();
  test_protocol_parse();
  test_protocol_framer();
  if (g_fail != 0) {
    printf("%d failed\n", g_fail);
    return 1;
  }
  printf("com_proto host tests ok\n");
  return 0;
}
