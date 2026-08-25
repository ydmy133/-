#ifndef MOTOR_PWM_H
#define MOTOR_PWM_H

#include "main.h"

/* 速度：-1000 ~ +1000，0=中位 1500us。
 * PWM 周期 20ms；脉宽 = 1500 + speed → 500~2500us（FT6335M 360° 定位）。
 */
void Motor_Init(void);
void Motor_SetSpeed(int16_t left, int16_t right);
void Motor_Stop(void);

#endif
