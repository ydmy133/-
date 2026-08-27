#include "motor_pwm.h"

extern TIM_HandleTypeDef htim14; /* PA4 SIG0 */
extern TIM_HandleTypeDef htim16; /* PA6 SIG1 */

#define SPEED_MAX     1000
#define PULSE_MID_US  1500
#define PULSE_SPAN_US 1000  /* Feetech FT6335M：500~2500us = 360° 定位 */

/* CN3 使能：部分板子用 S0/S1 给马达回路供电，停车保持低 */
#define MOTOR_L_EN_PORT  GPIOB
#define MOTOR_L_EN_PIN   GPIO_PIN_6
#define MOTOR_R_EN_PORT  GPIOB
#define MOTOR_R_EN_PIN   GPIO_PIN_7

static int16_t last_left = 0;
static int16_t last_right = 0;

static int16_t clamp_speed(int16_t s)
{
    if (s > SPEED_MAX)  return SPEED_MAX;
    if (s < -SPEED_MAX) return -SPEED_MAX;
    return s;
}

static uint16_t speed_to_ccr(int16_t speed)
{
    int32_t s = clamp_speed(speed);
    return (uint16_t)(PULSE_MID_US + (s * (int32_t)PULSE_SPAN_US) / SPEED_MAX);
}

/* 按实际内核时钟把定时器配成 1us/tick、20ms 周期、PWM1 */
static void tim_force_servo(TIM_TypeDef *tim, uint16_t pulse_us, uint8_t has_bdtr)
{
    uint32_t psc;

    SystemCoreClockUpdate();
    psc = SystemCoreClock / 1000000UL;
    if (psc == 0U) {
        psc = 1U;
    }

    tim->CR1 = 0U;
    tim->PSC = psc - 1U;
    tim->ARR = 19999U;
    tim->CCR1 = pulse_us;
    /* OC1M = 110 PWM mode 1，OC1PE 预装载 */
    tim->CCMR1 = (6U << TIM_CCMR1_OC1M_Pos) | TIM_CCMR1_OC1PE;
    tim->CCER = TIM_CCER_CC1E;
    if (has_bdtr != 0U) {
        tim->BDTR = TIM_BDTR_MOE;
    }
    tim->EGR = TIM_EGR_UG;
    tim->CR1 = TIM_CR1_CEN | TIM_CR1_ARPE;
}

static void motor_en(uint8_t on)
{
    GPIO_PinState s = on ? GPIO_PIN_SET : GPIO_PIN_RESET;
    HAL_GPIO_WritePin(MOTOR_L_EN_PORT, MOTOR_L_EN_PIN, s);
    HAL_GPIO_WritePin(MOTOR_R_EN_PORT, MOTOR_R_EN_PIN, s);
}

void Motor_Init(void)
{
    last_left = 0;
    last_right = 0;
    motor_en(1);
    /* 上电不输出 1500us 中位，避免舵机通电自行回中；CCR=0 无脉宽，等首条速度指令 */
    tim_force_servo(TIM14, 0U, 0);
    tim_force_servo(TIM16, 0U, 1);
}

void Motor_SetSpeed(int16_t left, int16_t right)
{
    uint16_t pl;
    uint16_t pr;

    left = clamp_speed(left);
    right = clamp_speed(right);
    if (left == last_left && right == last_right) {
        return;
    }
    last_left = left;
    last_right = right;
    pl = speed_to_ccr(left);
    pr = speed_to_ccr(right);

    /* 供电保持；到位后只改 CCR，不要反复开关定时器，否则脉宽会抖 */
    motor_en(1);
    TIM14->CCR1 = pl;
    TIM16->CCR1 = pr;
}

void Motor_Stop(void)
{
    Motor_SetSpeed(0, 0);
}
