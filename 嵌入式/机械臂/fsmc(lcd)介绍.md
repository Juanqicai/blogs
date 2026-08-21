# FSMC / LCD 介绍

> 入门向。讲清楚：**FSMC 是什么、干嘛用、原理、怎么配置、以及我们踩过的坑**。
> 我们用它接一块 **LCD 屏**（启明欣欣板，`ILI9341` / `SSD1963` 驱动）。
> 前置：上篇 [STM32 网络通信](./stm32网络通信.md) 提到"FSMC 时序太慢会饿死主循环"，这里展开讲。
> 参考：`motor` 项目的 `Core/Src/fsmc.c`、`App/UI/lcd_driver/lcd.c/h`、`text/RS05_LATENCY_DEBUG.md`。

## 0. FSMC 是什么

**FSMC = Flexible Static Memory Controller**（灵活静态存储控制器）。

翻译成大白话：**STM32 把"外部存储器的地址线、数据线、控制线"全部引出来，让你能像访问内部 RAM 一样，直接读写外部器件。**

本来外部器件（SRAM、NOR Flash、PSRAM、LCD）要通过"时序打点"一个一个字节地读，很麻烦。FSMC 把这些复杂时序**封装成一次内存访问**，你用一句：

```c
*(uint16_t *)0x6C000000 = data;    // 就像写内部 RAM 一样
```

FSMC 就自动在总线上帮你拉起地址线、数据线、写使能，按正确时序把数据送出去。

**核心价值**：把"读写外部并口器件"变成"读写一个内存地址"。**速度比 GPIO 模拟时序快得多**，代码也简洁。

## 1. 作用：为什么用 FSMC 接 LCD

LCD（尤其是带并口的大屏）刷新要写**海量像素数据**。比如一次刷一屏 `240×320` 个点，每点一个 16 位颜色值，就是**几十万次** 16 位写。

- 如果用 **GPIO 模拟时序**（手动拉 CS、RS、WR、数据线）：一次写要几十条指令，慢得离谱。
- 用 **FSMC**：一次 `*(u16*)addr = color` 就是一次并口写，**硬件自动完成时序**，速度是 GPIO 的几十倍。

所以我们用 FSMC 把 LCD 挂到外部总线（Bank4，`NE4` 片选），让它能快速刷屏。

### 1.1 我们用 FSMC 时还注意到一个"副作用"

- FSMC 很快，但**每刷一屏仍会占用主循环大量时间**（一次性写几十万次）。
- 我们曾把 FSMC 时序配成默认的极保守值，导致刷一屏 **383 ms**，把主循环拖死，**网络处理被饿死**，`ping` 高达 180 ms。
- 所以 FSMC 时序**不能乱配**，既要快，又别把主循环占满（详见"坑"一节）。

## 2. 原理（够用就行）

### 2.1 内存映射：Bank 4

STM32F407 的 FSMC 把外部空间分成几块（Bank），每块有固定的基址。我们用 **Bank 4**，基址 `0x6C000000`，片选是 `NE4`（`PG12`）。

- `0x60000000` 区：Bank1（NE1~NE4，接 NOR/PSRAM/SRAM）
- `0x6C000000`：**Bank4（NE4）** ← 我们的 LCD

### 2.2 数据线和地址线

- **数据线**：16 位（`FSMC_D0~D15`），分布在不同 GPIO 口（`PD0/1/8/9/10/14/15`、`PE7~PE15`）。LCD 是 16 位并口。
- **地址线**：`FSMC_A0~A5`（`PF0~PF5`），还有 `FSMC_A12`（`PG2`）。
- **控制线**：`FSMC_NOE`（读使能，`PD4`）、`FSMC_NWE`（写使能，`PD5`）、`FSMC_NE4`（片选，`PG12`）。

### 2.3 用地址位区分"命令"和"数据"（关键技巧）

LCD 驱动一般有根 **RS（命令/数据）线**：低电平是写"命令/寄存器"，高电平是写"数据/GRAM"。

我们没用 GPIO 单独拉 RS，而是**用一个地址位来区分**——把 RS 接到 FSMC 的地址线 `A12`（`PG2`）：

- 写**命令**：让 `A12 = 0` → 访问 `0x6C000000 | 0x1FFE`
- 写**数据**：让 `A12 = 1` → 访问 `0x6C000000 | 0x2000`

`lcd.h` 里就是：

```c
#define CMD_BASE    ((u32)(0x6C000000 | 0x00001FFE))
#define DATA_BASE   ((u32)(0x6C000000 | 0x00002000))
#define LCD_CMD     (*(u16 *)CMD_BASE)
#define LCD_DATA    (*(u16 *)DATA_BASE)
```

于是 `LCD_CMD = 0x2C`（写命令），`LCD_DATA = color`（写像素）。**读写同一个器件，只是地址的第 12 位不同，FSMC 就自动把 RS 线拉对了。** 这比额外用一根 GPIO 控制 RS 更省事。

> 注意：STM32 内部对 16 位外部总线**右移一位对齐**，所以实际 `A12` 对应 CPU 地址的某一位。代码里用 `0x1FFE` 和 `0x2000` 这个差值正好只差第 13 位（`0x2000 - 0x1FFE = 2`），配合右移对齐正好落到 `A12`。**这部分由硬件保证，不用手算，记住"用地址位区分命令/数据"这个思路即可。**

### 2.4 一次读写怎么发生

当你 `LCD_CMD = reg` 时，FSMC 自动：
1. 拉低片选 `NE4`；
2. 把地址放到地址线（含 `A12` 决定是命令还是数据）；
3. 拉低写使能 `NWE`；
4. 把 `reg` 放到 16 位数据线；
5. 等待时序（地址建立、数据保持）；
6. 恢复信号。

整个过程**一次总线访问完成**，不需要你管时序细节。速度取决于你配的时序参数（见下）。

---

## 3. 配置方法（CubeMX）

配置在 `Connectivity > FSMC`，生成后落在 `Core/Src/fsmc.c`。

### 3.1 基本项

| 项 | 值 | 说明 |
|---|---|---|
| 片选 Bank | **Bank4**（NE4） | `FSMC_NORSRAM_BANK4` |
| 存储器类型 | **SRAM** | `FSMC_MEMORY_TYPE_SRAM` |
| 数据总线宽度 | **16 bits** | 屏是 16 位并口，必须 16 |
| 写操作 | Enable | 允许写 |
| 扩展模式 | Enable | 读写分开时序（`FSMC_EXTENDED_MODE_ENABLE`） |

### 3.2 时序（读/写分开）

因为开了扩展模式，**读时序（Timing）和写时序（ExtTiming）分开**。我们项目最终用的是：

| 项 | 读（Timing） | 写（ExtTiming） | 说明 |
|---|---:|---:|---|
| Address setup | 15 | 9 | 地址建立时间 |
| Data setup | 60 | 8 | 数据建立时间（**最影响速度**） |
| Bus turn around | 0 | 0 | 总线转向时间 |
| Access mode | A | A | 访问模式 |

- **读**：`ADDSET15 / DATAST60`。
- **写**：`ADDSET9 / DATAST8`（写访问约 `9+8+15(ADDHLD)+1 ≈ 33` 周期 ≈ 196 ns/次，是默认值 1.5 µs 的 **1/8**）。
- **注意**：F407 的 CubeMX GUI **没有 `AddressHoldTime` 这一项**，它固定 15，不影响正确性，不用管。

### 3.3 引脚（CubeMX 自动分配）

- `PF0~PF5` → `FSMC_A0~A5`
- `PE7~PE15` → `FSMC_D4~D12`
- `PD0, PD1, PD8, PD9, PD10, PD14, PD15` → `FSMC_D0~D3, D13~D15`
- `PD4` → `FSMC_NOE`，`PD5` → `FSMC_NWE`
- `PG12` → `FSMC_NE4`
- `PG2` → `FSMC_A12`（命令/数据 RS 线，`lcd.c` 里单独配了 `GPIO_AF12_FSMC`）

> 我们在 `lcd.c` 的 `LCD_Init()` 里补配了 `PG2`（`FSMC_A12`）和背光 `PF10`，因为 CubeMX 生成的 FSMC 只配了总线引脚，没配 A12 和背光。

### 3.4 配置完做什么

1. CubeMX 生成后，`main.c` 里会调用 `MX_FSMC_Init()`。
2. 在 `app_main.c` / `LCD_Init()` 里调用 `LCD_Init()` 初始化 LCD 驱动（识别 ID、写初始化序列、开显示）。
3. 之后就能 `LCD_CMD = ...` / `LCD_DATA = ...` 刷屏了。

---

## 4. 坑（重要，看 `RS05_LATENCY_DEBUG.md`）

### 4.1 默认时序太慢 → LCD 饿死主循环 → 网络变慢

CubeMX **默认生成的 FSMC 时序极保守**：

```c
Timing.DataSetupTime = 255;   // 每次访问 ≈ 256 个 HCLK ≈ 1.5 µs
```

而 LCD 底层是**逐点写 GRAM**（`LCD_Fill_onecolor` / `LCD_DisplayString` 每像素 `LCD_DATA = color`）。一次仪表盘刷新约 25 万次 16 位访问 × 1.5 µs ≈ **383 ms**，刷新周期才 100 ms → **LCD 几乎一直占 CPU**。

后果：主循环被占满，网络处理（含 ICMP 回包）被饿死，`ping 192.168.1.240` 平均 **~180 ms**。

**修复**：把时序改成上面 `ADDSET15/DATAST60`（读）、`ADDSET9/DATAST8`（写）。LCD 刷屏从 383 ms 降到 ~50 ms，`ping` 回到 **<1 ms**。

### 4.2 FSMC 寄存器坑（别手改）

`RS05_LATENCY_DEBUG.md` 特别强调：

- `FSMC_NORSRAM_BANK4 = 6`，所以 **Bank4 写时序寄存器是 `FSMC_Bank1E->BWTR[6]`**（不是别的）。
- **写时序 = `BWTR`**（扩展模式使能时），**读时序 = `BTR`**（`BTCR[Bank+1]` = `BTCR[7]`）。
- **`BTCR[6]` 是 `BCR4` 控制寄存器**（含 `MBKEN` 使能位），**不是**时序寄存器。往它写时序值会清掉 Bank4 使能位 → **屏幕黑屏**。
- **`FSMC_Bank1E->BWTR[3]`** 是保留/空闲地址，**不是** Bank4 的写时序。

> **结论：不要手改时序寄存器。** 时序只在 CubeMX GUI 里配，重新生成即生效。手改寄存器很容易改错导致黑屏。

### 4.3 LCD 刷屏和实时控制的取舍

我们项目在**主从（bilateral）实时模式**下，会**跳过 LCD 刷新**：

```c
if ((now - last_lcd_tick) >= LCD_UPDATE_PERIOD_MS &&
    (NetRealtime_IsBilateralEnabled() == 0U))
{
    LCD_Dashboard_Update();
}
```

因为 LCD 刷屏是**阻塞式**的，会在短时间内占用主循环，可能把 LwIP 饿到触发 **100 ms 对端安全超时**。所以实时模式下宁可牺牲显示，保证控制/网络不死。

---

## 5. 一句话总结

- **FSMC** = 把外部并口器件当"内存"一样直接读写，硬件自动时序，快、省代码。
- **接 LCD**：Bank4 + 16 位数据线，用**地址位 A12 区分命令/数据**（`CMD_BASE`/`DATA_BASE`）。
- **配置**：CubeMX 选 Bank4/SRAM/16 位，**读 `ADDSET15/DATAST60`，写 `ADDSET9/DATAST8`**。
- **最大坑**：默认时序太慢会饿死主循环连累网络；别手改时序寄存器（会黑屏）；实时模式下跳过 LCD 刷新保证控制安全。

> 至此机械臂文档基本齐了：看 [README](./README.md) 按学习路线来。上篇 [STM32 网络通信](./stm32网络通信.md)。
