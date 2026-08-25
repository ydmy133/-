#ifndef MOTOR_PWM_H
#define MOTOR_PWM_H

#include <stdint.h>

/* 速度：-1000 ~ +1000，0=停。脉宽 1500±500us，周期 20ms。 */
void Motor_Init(void);
void Motor_SetSpeed(int16_t left, int16_t right);
void Motor_Stop(void);

#endif
