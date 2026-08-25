/* 贴到 CubeMX 生成的 main.c 里：Motor_Init() 放在 while 前，循环体用下面其中一种。 */

#include "motor_pwm.h"

/* 方案 A：左右同速正转 20%，看马达会不会转 */
void Motor_Demo_Constant(void)
{
    Motor_SetSpeed(200, 200);
    while (1) {
        HAL_Delay(100);
    }
}

/* 方案 B：停 → 正转 → 停 → 反转，循环 */
void Motor_Demo_Sweep(void)
{
    for (;;) {
        Motor_Stop();
        HAL_Delay(1500);

        Motor_SetSpeed(200, 200);   /* 约 1600us，正转 */
        HAL_Delay(2000);

        Motor_Stop();
        HAL_Delay(1500);

        Motor_SetSpeed(-200, -200); /* 约 1400us，反转 */
        HAL_Delay(2000);
    }
}
