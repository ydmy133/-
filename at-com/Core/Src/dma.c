/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    dma.c
  * @brief   DMA 已不再用于 UART4 RX（改为 RXNE + ring buffer）。
  *          保留空实现，避免 CubeMX 工程组里缺文件。
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "dma.h"

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

void MX_DMA_Init(void)
{
  /* UART4 不再使用 DMA，不使能 DMA1 时钟与 DMA1_Stream2 IRQ */
}

/* USER CODE BEGIN 2 */

/* USER CODE END 2 */
