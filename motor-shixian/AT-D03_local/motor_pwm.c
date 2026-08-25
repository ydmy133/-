#include "motor_pwm.h"

extern TIM_HandleTypeDef htim14; /* PA4 SIG0 左  TIM14_CH1 */
extern TIM_HandleTypeDef htim16; /* PA6 SIG1 右  TIM16_CH1 */

#define SPEED_MAX     1000
#define PULSE_MID_US  1500
#define PULSE_SPAN_US 500

static int16_t clamp_speed(int16_t s)
{
    if (s > SPEED_MAX)  return SPEED_MAX;
    if (s < -SPEED_MAX) return -SPEED_MAX;
    return s;
}

/* 1 tick = 1us 时，CCR 直接等于脉宽微秒数 */
static uint16_t speed_to_ccr(int16_t speed)
{
    speed = clamp_speed(speed);
    return (uint16_t)(PULSE_MID_US + (speed * PULSE_SPAN_US) / SPEED_MAX);
}

void Motor_Init(void)
{
    HAL_TIM_PWM_Start(&htim14, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim16, TIM_CHANNEL_1);
    Motor_Stop();
}

void Motor_SetSpeed(int16_t left, int16_t right)
{
    left = clamp_speed(left);
    right = clamp_speed(right);
    __HAL_TIM_SET_COMPARE(&htim14, TIM_CHANNEL_1, speed_to_ccr(left));
    __HAL_TIM_SET_COMPARE(&htim16, TIM_CHANNEL_1, speed_to_ccr(right));
}

void Motor_Stop(void)
{
    Motor_SetSpeed(0, 0);
}
