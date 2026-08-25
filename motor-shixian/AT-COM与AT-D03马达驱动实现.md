# AT-COM + AT-D03 马达驱动实现说明

依据：

- `E:\逐电\原理图\AT-COM_20260609.pdf`（上位机，STM32F405RGT6）
- `E:\逐电\原理图\AT-D03.pdf`（收发与下发，STM32G030F6P6TR）
- `修改8.17.pdf`（控制律：T / Y、差速混控、定速巡航）

本文只解决一件事：**AT-COM 算出左右速度，经 RS485 发给 AT-D03，D03 用 PWM 驱动两路马达/舵机。**

---

## 1. 角色与数据流

```
地面站 / 串口助手（JSON，可选）
        │  AT-COM CN6  UART4  PA0/PA1
        ▼
AT-COM  STM32F405RGT6     ← 上位机
  解析指令 → 选模式 → 算 T、Y
  左 = T - Y
  右 = T + Y
  限幅后打包 RS485 帧
        │  AT-COM CN2 A0/B0  →  AT-D03 CN2 A/B
        ▼
AT-D03  STM32G030F6       ← 收发与下发
  USART2 + SP3485 收帧
  TIM14_CH1 / TIM16_CH1 出 PWM
        │
        ▼
CN4：SIG0 + 7.4V1、SIG1 + 7.4V2 → 左右马达/舵机
```

| 板 | 芯片 | 职责 |
|---|---|---|
| AT-COM | STM32F405RGT6 | 导航/控制律、混控、对下位机发速度 |
| AT-D03 | STM32G030F6P6TR | 收 RS485、把速度变成 PWM、给舵机供电 |

G030 Flash 仅 32 KB，**不要在 AT-D03 上跑 cJSON**。JSON 只在 AT-COM 的 UART4 解析。

---

## 2. 硬件接线

### 2.1 供电（必须分开）

| 板 | 座子 | 网络 | 说明 |
|---|---|---|---|
| AT-COM | CN1（2 针） | DC24V / GND | 通信板供电 |
| AT-D03 | CN1（2 针） | VCC / GND | 主电源，板上再降出 7.4V1、7.4V2、5V、3.3V |

两板 **GND 必须共地**。  
**不要把 AT-COM 的 24V 接到 AT-D03 的 CN2。** D03 的 CN2 是信号座，没有 24V 输入脚；马达电源走 D03 自己的 CN1 → 两路 TPS54531 → 7.4V。

### 2.2 通信：RS485 A/B（主链路）

AT-COM 四路 RS485 与 AT-D03 的 SP3485 都是半双工 485，只接 **A、B、GND**。

一块 D03 同时管左右两路马达时，只占用 AT-COM 的 **CN2（USART2）**：

| AT-COM CN2（4 针 3.81） | AT-D03 CN2（MX1.25 6 针） | 说明 |
|---|---|---|
| 丝印 **A0** | 针 5 **A** | 485 A |
| 丝印 **B0** | 针 6 **B** | 485 B |
| GND | 电源地（CN1 地，或 CN5 针 5） | 共地 |

AT-D03 CN2 针脚：

| 针 | 网络 | 是否接 AT-COM |
|---|---|---|
| 1 | TX+ | 否（本期不用） |
| 2 | TX− | 否 |
| 3 | RX+ | 否 |
| 4 | RX− | 否 |
| 5 | A | **是** |
| 6 | B | **是** |

CN5 是同一组 TX/RX 的引出/级联（针 1 RX− … 针 6 VCC），本期可不接。

若左右各用一块 D03：左接 AT-COM **CN2 / USART2**，右接 **CN3 / USART3**。

AT-COM 其余座子用途（本期控马达不用）：

| 座子 | 外设 | 引脚 |
|---|---|---|
| CN6 | UART4，对电脑 JSON/调试 | PA0 TX、PA1 RX |
| CN7 | CAN1 | PA11/PA12 |
| CN4 / CN5 | USART6 / USART1，备用 485 | PC6/PC7、PA9/PA10 |
| H2 | SWD | PA13/PA14 |

### 2.3 马达：只接 AT-D03 的 CN4

转速/转向由 **PWM 信号线 SIG0 / SIG1** 决定，电源是板上两路 7.4 V。

| CN4 针 | 网络 | G030 引脚 | 定时器 | 接到 |
|---|---|---|---|---|
| 1 | SIG0 | PA4 | TIM14_CH1 | 左舵机/电调信号 |
| 2 | 7.4V1 | 电源 | — | 左电源 |
| 3 | GND | 地 | — | 左地 |
| 4 | SIG1 | PA6 | TIM16_CH1 | 右信号 |
| 5 | 7.4V2 | 电源 | — | 右电源 |
| 6 | GND | 地 | — | 右地 |

### 2.4 不要接错的口

AT-D03 **CN3**（Y0 / Y1 / Y2 / S0 / S1）是 MOSFET 开关量（PA7、PA8、PA11、PB6、PB7），**不是调速口**。当前实现马达转速不要用 CN3。

---

## 3. 控制量（与《修改8.17》对齐）

控制周期 **100 ms**。

| 符号 | 含义 |
|---|---|
| T | 公共推进速度 |
| Y | 差速转向量 |
| 左 | T − Y |
| 右 | T + Y |

定速巡航（先把马达转起来用这个）：

- 上位机给 T
- Y = 0
- 左右同速，直线转

限幅：T、Y 先限幅，再算左右，左右再限一次。量程与下面协议一致：`[-1000, +1000]`，单位 0.1%（1000 = 100%）。

地面站 JSON 走 AT-COM UART4（空闲中断收包，主循环 cJSON 解析）。建议字段：

```json
{"mode":"stop"}
{"mode":"speed","T":20}
{"mode":"speed","T":20,"Y":8}
```

`T`/`Y` 为百分数。AT-COM 内部乘 10 变成 int16 再下发。  
JSON 非法（不是 `{...}` 或长度 < 2）则丢弃。  
**超过 5200 ms 收不到合法地面指令：停车（左右 = 0）。** 用串口助手调试时打开定时发送（建议 500 ms）。

---

## 4. 板间协议（AT-COM → AT-D03）

波特率：**115200 8N1**（原理图未写死，两板必须一致）。

D03 半双工：平时 **PA1 = 0 只收**；仅回状态时拉高 DE。  
COM 发一帧前把该路 485 的 **EN 拉高**，等发送完成（TC）再拉低。

### 4.1 下发速度帧

| 偏移 | 长度 | 字段 | 说明 |
|---|---|---|---|
| 0 | 1 | 帧头 | `0xAA` |
| 1 | 1 | 帧头 | `0x55` |
| 2 | 1 | LEN | 后面字节数，速度帧为 `0x06` |
| 3 | 1 | CMD | `0x01` = 设置速度 |
| 4 | 2 | LEFT | int16 大端，-1000~+1000 |
| 6 | 2 | RIGHT | int16 大端，-1000~+1000 |
| 8 | 1 | XOR | 从 LEN 到 RIGHT 低字节的异或 |

定速 20% 正转（左=右=200）：

```
AA 55 06 01 00 C8 00 C8 0F
```

XOR 计算：`0x06 ^ 0x01 ^ 0x00 ^ 0xC8 ^ 0x00 ^ 0xC8 = 0x0F`。

停车：

```
AA 55 06 01 00 00 00 00 07
```

### 4.2 其它命令（可选）

| CMD | 含义 | 数据 |
|---|---|---|
| 0x00 | 心跳 / 保活 | 无（LEN=2，仅 CMD+XOR） |
| 0x01 | 设置速度 | LEFT、RIGHT |
| 0x02 | 急停 | 无，D03 立即 PWM 回中位 |

D03 连续 **300 ms** 收不到合法 0x01/0x00，自动停车。

---

## 5. PWM 映射（AT-D03）

标准舵机/多数电调：周期 20 ms（50 Hz），用高电平脉宽表示油门。

| 速度值 | 脉宽 | 含义 |
|---|---|---|
| −1000 | 1000 µs | 最大反转 |
| 0 | 1500 µs | 停 / 中位 |
| +1000 | 2000 µs | 最大正转 |

公式：

```
pulse_us = 1500 + speed * 0.5
```

`speed` 为 int16，先限幅到 ±1000。  
若实机转向相反：对该通道 `speed = -speed`。  
若电调中位或行程不是 1000~2000 µs，只改本表，协议不变。

CubeMX / 寄存器建议（G030，TIM14 / TIM16）：

- 内部时钟 64 MHz 时：PSC 使计数频率 = 1 MHz（1 tick = 1 µs）
- ARR = 19999 → 周期 20 ms
- CCR = pulse_us（1000~2000）
- PWM 模式 1，极性高

---

## 6. AT-COM 软件要点（F405）

### 6.1 外设

| 外设 | 引脚 | 用途 |
|---|---|---|
| UART4 | PA0 TX、PA1 RX | 地面站 JSON，IDLE + DMA |
| USART2 | PA2 TX、PA3 RX | 对 D03 的 RS485（CN2） |
| 该路 EN | 原理图 EN0 | 发前 1、发完 0 |

中断里禁止 `cJSON_Parse` / `malloc`。只拷贝一帧、置标志、重新开接收。

### 6.2 100 ms 循环

```
收 JSON（若有）→ 更新 mode / T / Y
若失联 > 5200 ms → T=0, Y=0
左 = clamp(T-Y)
右 = clamp(T+Y)
组帧 AA 55 … 发 USART2
```

定速模式强制可用 `Y = 0`（除非 JSON 显式带了 Y，用于台架差速测试）。

### 6.3 发送伪代码

```c
#define T_MAX  1000

static uint8_t xor8(const uint8_t *p, int n)
{
    uint8_t x = 0;
    for (int i = 0; i < n; i++) x ^= p[i];
    return x;
}

void d03_send_speed(int16_t left, int16_t right)
{
    uint8_t f[9];
    if (left  >  T_MAX) left  =  T_MAX;
    if (left  < -T_MAX) left  = -T_MAX;
    if (right >  T_MAX) right =  T_MAX;
    if (right < -T_MAX) right = -T_MAX;

    f[0] = 0xAA; f[1] = 0x55; f[2] = 0x06; f[3] = 0x01;
    f[4] = (uint8_t)(left  >> 8); f[5] = (uint8_t)(left  & 0xFF);
    f[6] = (uint8_t)(right >> 8); f[7] = (uint8_t)(right & 0xFF);
    f[8] = xor8(&f[2], 6);

    rs485_tx_enable(1);                 /* EN0 = 1 */
    HAL_UART_Transmit(&huart2, f, 9, 10);
    /* 等待 USART_SR.TC 置位 */
    rs485_tx_enable(0);
}
```

地面站 `{"mode":"speed","T":20}` → `left = right = 200` → 调用 `d03_send_speed(200, 200)`。

---

## 7. AT-D03 软件要点（G030）

### 7.1 外设

| 外设 | 引脚 | 用途 |
|---|---|---|
| USART2 TX | PA2 | → SP3485 DI |
| USART2 RX | PA3 | ← SP3485 RO |
| DE/RE | PA1 | 0=收，1=发 |
| TIM14_CH1 | PA4 | SIG0 左 PWM |
| TIM16_CH1 | PA6 | SIG1 右 PWM |
| LED1 / LED2 | PA0 / PA5 | 心跳、收包指示 |
| SWD | PA13 / PA14 | H1 下载 |

上电：PA1=0，PWM 输出 1500 µs（中位停，不做大油门自检），再开 USART 接收。

### 7.2 收包

建议 DMA + IDLE，或逐字节状态机找 `AA 55`。校验：

1. 帧头 `AA 55`
2. LEN 与实际长度一致
3. XOR 正确
4. CMD 认识

CMD=0x01 时更新左右目标速度，并刷新“上次收包时刻”。主循环 20 ms 内把 CCR 写到对应脉宽（可做斜率限制，避免电调丢波）。

### 7.3 脉宽换算

```c
static uint16_t speed_to_ccr(int16_t s)
{
    if (s >  1000) s =  1000;
    if (s < -1000) s = -1000;
    return (uint16_t)(1500 + s / 2);   /* 1 µs / tick */
}

void motor_apply(int16_t left, int16_t right)
{
    __HAL_TIM_SET_COMPARE(&htim14, TIM_CHANNEL_1, speed_to_ccr(left));
    __HAL_TIM_SET_COMPARE(&htim16, TIM_CHANNEL_1, speed_to_ccr(right));
}
```

---

## 8. 联调步骤

1. **只上电、不接桨。** AT-COM 24V，AT-D03 CN1 上电，两板共地。ST-Link 能连 F405 与 G030。
2. **D03 空载 PWM。** 示波器看 CN4 针 1、针 4：上电应为 50 Hz、高电平约 1.5 ms。
3. **PC 直连验证 D03（可选）。** USB 转 485 接到 D03 的 A/B，发 `AA 55 06 01 00 C8 00 C8 0F`，脉宽应变到约 1.6 ms，马达轻转。
4. **AT-COM → D03。** COM 上电循环发停车帧，再用定速 10%（left=right=100）。
5. **接地面站 JSON。** USB 转 TTL 接 AT-COM CN6（TX↔RX 交叉），115200，ASCII，定时 500 ms 发送：
   - `{"mode":"stop"}`
   - `{"mode":"speed","T":10}`
   - `{"mode":"speed","T":-10}`
6. **确认左右方向。** 若某侧反了，只在 D03 该通道取反，或在 COM 混控后对该侧取反。
7. **再上《修改8.17》其它模式**（定向 / 点到点 / 稳泊）。那些模式仍只向 D03 发左右速度，导航闭环全部留在 AT-COM。

---

## 9. 引脚速查

### AT-COM（F405）

| 功能 | 引脚 | 座子 |
|---|---|---|
| UART4 TX/RX | PA0 / PA1 | CN6 |
| USART2 TX/RX（D03） | PA2 / PA3 | CN2 |
| USART3 TX/RX | PB10 / PB11 | CN3 |
| USART6 TX/RX | PC6 / PC7 | CN4 |
| USART1 TX/RX | PA9 / PA10 | CN5 |
| CAN1 | PA11 / PA12 | CN7 |
| SWD | PA13 / PA14 | H2 |
| 485 方向 | EN0~EN3（经反相器驱动 3088 的 DE/RE） | 发前拉高对应 EN |

### AT-D03（G030）

| 功能 | 引脚 | 座子 |
|---|---|---|
| USART2 TX/RX | PA2 / PA3 | 经 SP3485 到 CN2 A/B |
| 485 DE/RE | PA1 | — |
| 左 PWM SIG0 | PA4 TIM14_CH1 | CN4 针 1 |
| 右 PWM SIG1 | PA6 TIM16_CH1 | CN4 针 4 |
| Y0/Y1/Y2 开关 | PA7 / PA8 / PA11 | CN3，本期不用 |
| S0/S1 开关 | PB6 / PB7 | CN3，本期不用 |
| LED | PA0 / PA5 | 板载 |
| SWD | PA13 / PA14 | H1 |

---

## 10. 未写入原理图、实现时必须统一的项

1. 板间波特率：本文定为 **115200 8N1**。
2. 速度单位：int16，**0.1%**，±1000。
3. PWM 中位 1500 µs、行程 ±500 µs；与电调不符时只改 D03 映射。
4. AT-COM CN2 的 4 针物理顺序以板子丝印为准，按 **A / B / GND** 连接，不接 24V 到 D03 CN2。
5. 电调若需上电油门校准（油门行程标定），先按厂家手册做一次，再接入闭环。

按第 8 节做到第 5 步，即完成「AT-COM 当上位机、AT-D03 收发并下发、马达按速度转动」。
