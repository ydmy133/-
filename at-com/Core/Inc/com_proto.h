#ifndef COM_PROTO_H
#define COM_PROTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CMD_TIMEOUT_MS
#define CMD_TIMEOUT_MS 300U
#endif

#define JSON_LINE_SIZE 192U
#define COM_PCT_MIN (-100)
#define COM_PCT_MAX 100
#define COM_SPEED_MAX 1000
#define D03_SPEED_FRAME_LEN 9U
#define D03_ESTOP_FRAME_LEN 5U
#define D03_CMD_SET_SPEED 0x01U
#define D03_CMD_ESTOP 0x02U

/*
 * 纯逻辑（JSON 拼帧 / 解析 / watchdog / D03 组帧），不依赖 HAL。
 * 宿主机测试：
 *   gcc -std=c99 -Wall -Wextra -I../Core/Inc -o test_com_proto \
 *       test_com_proto.c ../Core/Src/com_proto.c
 */

typedef struct {
  int16_t t;      /* 电机单位 = 网页百分比 * 10，范围 [-1000, 1000] */
  int16_t y;
  uint8_t is_stop;
  uint8_t y_omitted; /* speed 且未给 Y 时为 1，Y 按 0 处理 */
} ComCmd;

typedef struct {
  uint8_t buf[JSON_LINE_SIZE];
  uint16_t len;
  int16_t brace_depth;
  uint8_t in_string;
  uint8_t escaped;
} ComJsonFramer;

typedef struct {
  uint32_t last_valid_ms;
  uint32_t last_estop_ms;
  uint8_t timed_out;
} ComWatchdog;

void ComJsonFramer_Reset(ComJsonFramer *f);
/* 1=得到完整顶层对象（在 f->buf），0=继续，-1=丢弃并已复位 */
int ComJsonFramer_Feed(ComJsonFramer *f, uint8_t b);

int ComJson_Parse(const char *js, ComCmd *out);

void ComWatchdog_Init(ComWatchdog *w, uint32_t now_ms);
void ComWatchdog_OnValidCmd(ComWatchdog *w, uint32_t now_ms);
/* 1=调用方应立即停车（边沿或周期性补发急停），0=无需动作 */
int ComWatchdog_Poll(ComWatchdog *w, uint32_t now_ms);

int16_t Com_ClampSpeed(int32_t v);
int16_t Com_MixLeft(int16_t t, int16_t y);
int16_t Com_MixRight(int16_t t, int16_t y);

void D03_BuildSpeed(uint8_t out[D03_SPEED_FRAME_LEN], int16_t left, int16_t right);
void D03_BuildEstop(uint8_t out[D03_ESTOP_FRAME_LEN]);

#ifdef __cplusplus
}
#endif

#endif
