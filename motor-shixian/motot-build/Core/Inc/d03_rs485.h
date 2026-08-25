#ifndef D03_RS485_H
#define D03_RS485_H

#include "main.h"

/* AT-COM → AT-D03：AA 55 ...，115200 8N1，CN2 A/B */
void D03_RS485_Init(void);
void D03_RS485_Poll(void);           /* 主循环调用：收帧打角；失联保持最后角度，不回中 */
uint32_t D03_RS485_LastRxMs(void);   /* 上次合法 0x00/0x01 时刻 */

#endif
