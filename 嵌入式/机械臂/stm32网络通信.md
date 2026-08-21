# STM32 网络通信（LwIP）

> 入门向。分三块：
> 1. **LwIP 原理**（先讲一点 TCP/UDP 和通信分层，不用太深）；
> 2. **CubeMX 配置方法**（ETH + LwIP，配引脚、PHY、静态 IP）；
> 3. **项目开发里怎么用**（初始化、轮询、发/收、我们的 TCP 维护 + UDP 实时两套）。
> 以我们 [`motor`](/home/juanqicai/AAA_user_file/motor) 项目（STM32F407 + LAN8720）为准。
> 上篇：[FSMC/LCD](./fsmc(lcd)介绍.md)。

## 0. 为什么 STM32 要联网

我们这台机械臂需要**和上位机/另一块板通信**（下发目标关节角、回传状态、主从联动）。STM32F407 自带 **ETH 以太网外设**，配一颗 **PHY 芯片**（我们用 LAN8720）就能上网。真正让"能联网"变成"能用网络协议"的，是 **LwIP 协议栈**。

## 1. LwIP 原理（够用就行）

### 1.1 通信分层：一个"包"是怎么走的

网络通信从底到顶大致分四层。以 STM32 为例：

| 层 | 干了啥 | 我们的对应 |
|---|---|---|
| **应用层** | 你的业务逻辑，比如"发关节角" | `App/Net`（TCP/UDP 应用） |
| **传输层** | 端口到端口：TCP/UDP | LwIP 的 `tcp.c` / `udp.c` |
| **网络层** | IP 寻址、路由 | LwIP 的 `ip4.c` |
| **链路层** | 网卡帧、MAC、PHY | `ethernetif.c` + HAL ETH + LAN8720 |

数据从上往下**逐层打包**（每层加个头），到最底层变成一串以太网帧发出去；收到时反向逐层拆包。这就是"封装/解封装"。

### 1.2 TCP vs UDP（最关键的区别）

| | **TCP** | **UDP** |
|---|---|---|
| 连接 | 面向连接（要握手） | 无连接 |
| 可靠性 | 可靠：确认、重传、排序、流控 | 不可靠：发出去就不管，可能丢/乱序/重复 |
| 开销 | 大（握手、头部、状态机） | 小（几乎无状态） |
| 速度/延迟 | 较慢、较稳 | 快、低延迟 |
| 适合 | 文件、命令、维护（要可靠） | 实时数据、视频、控制（要快） |

**我们项目怎么选**：**UDP 走实时控制，TCP 走低频维护**。

- **UDP**：500 Hz 下发目标、500 Hz 回传状态。丢几帧无所谓，只要"最新的值"到就行——所以用 UDP，快、省资源。
- **TCP**：`enable/disable/hold/zero/param` 这些**破坏性维护命令**，要可靠、要确认，所以走 TCP，且**要 `CONFIRM` 二次确认**。

> 这是非常典型的分工：**实时数据用 UDP，控制/维护用 TCP。** 别反过来。

### 1.3 LwIP 是什么

**LwIP = Lightweight IP**，是专为**嵌入式**做的轻量级 TCP/IP 协议栈。

- 特点：内存占用小、CPU 要求低，**可以不跑操作系统**（裸机也能用）。
- **两种运行模式**：
  1. **`NO_SYS`（无 RTOS）**：协议栈没有自己的线程，你在主循环里**轮询**调用 `MX_LWIP_Process()`。简单、无锁、资源少。我们就是这种。
  2. **带 RTOS（`NO_SYS=0`）**：协议栈有独立线程，通过信号量/邮箱和你的线程通信。更复杂，一般配 FreeRTOS。
- **三种 API**：
  1. **Raw/Callback API**（底层）：你注册回调函数，协议栈收到包就调你。高效但难写。我们 TCP/UDP 都用这个。
  2. **Netconn API**（中层）：类似 socket，稍微高层。
  3. **Socket API**（高层）：和 PC 端几乎一样。我们**没用**（`LWIP_SOCKET=0`，为了省内存）。

> 我们项目的 `lwipopts.h` 里：`NO_SYS=1`、`LWIP_NETCONN=0`、`LWIP_SOCKET=0` → **纯裸机 + Raw/Callback API**。这是最省内存、也最"嵌入式"的配置。

---

## 2. CubeMX 配置方法

配置分两步：**先配 ETH（硬件/PHY），再配 LwIP（软件栈）**。

### 2.1 配 ETH（Connectivity → ETH）

1. **模式**：选 `RMII`（我们板子用 RMII 接口，比 MII 省引脚）。
   - 对应 CubeMX 里的 `Media Interface = RMII`。
2. **引脚**：RMII 需要这些引脚，CubeMX 选对模式后会自动分配（不用手动一个个点）：
   - `PA1` → `ETH_REF_CLK`（参考时钟 50 MHz）
   - `PA2` → `ETH_MDIO`，`PC1` → `ETH_MDC`（管理接口）
   - `PA7` → `ETH_CRS_DV`，`PC4` → `ETH_RXD0`，`PC5` → `ETH_RXD1`
   - `PG11` → `ETH_TX_EN`，`PG13` → `ETH_TXD0`，`PG14` → `ETH_TXD1`
3. **PHY 地址**：LAN8720 的地址是 **0**（改板子上某电阻可改）。CubeMX 里选 PHY 库为 `LAN8742`。
4. **复位引脚**：LAN8720 硬件复位（`PE2`，低有效脉冲），给它起个标签 `ETH_RESET`。

### 2.2 配 LwIP（Middleware → LWIP）

1. 打开 `LwIP` 中间件。
2. **版本**：默认 `v2.1.2_Cube`。
3. **IP 地址**：默认静态 `192.168.1.240` / 掩码 `255.255.255.0` / 网关 `192.168.1.1`，**DHCP 关**（`LWIP_DHCP=0`）。
   - ⚠️ 这些只是 CubeMX 的**默认值**，我们项目里真正生效的是 `App/conf/net_conf.h` 里的值（见下）。
4. **PHY**：选 `LAN8742`。
5. **生成代码**：CubeMX 会生成 `LwIP/App/lwip.c`（`MX_LWIP_Init/Process`）、`LwIP/Target/lwipopts.h`、`LwIP/Target/ethernetif.c`。

> ⚠️ **重新生成时注意**：`ethernetif.c` 的 **TX 释放补丁**（`HAL_ETH_ReleaseTxPacket`）和 **`HAL_SRAM_MODULE_ENABLED`** 可能被 CubeMX 覆盖，需要按 `Middlewares/Third_Party/README.md` 重做一次。这是 CubeMX 全托管工程的常见坑。

### 2.3 一个容易踩的坑：ETH 和 FSMC 互相抢时间

- FSMC 是用来接 **LCD** 的（见 [FSMC 篇](./fsmc(lcd)介绍.md)）。
- 如果 FSMC **时序太慢**，LCD 刷一屏会占用主循环很久（我们曾到 **383 ms**），把网络处理**饿死**，导致 `ping` 高达 **~180 ms**、回包极慢。
- 修复：把 FSMC 时序调快（`Data setup 60 → 写 8`），LCD 刷屏降到 ~50 ms 量级，`ping` 回到 **<1 ms**。
- **结论**：FSMC 和 ETH 共用 CPU/主循环，**FSMC 时序一定要合理**，否则网络一起遭殃。这是"表面无关、实际耦合"的典型。

---

## 3. 项目开发里怎么用

### 3.1 初始化（`Net_Init`）

`app_main.c` 里 `App_Init()` 调用：

```c
NetHandler_Init();                      // 先注册接收回调
Net_Init(NET_COMM_MODE_STRING, NULL);   // 根据配置启动网络
```

`Net_Init` 读 `NET_COMM_MODE_STRING`，决定用哪种模式（`tcp_server` / `tcp_client` / `udp`），并绑定端口、注册回调。**注意顺序：先注册回调再 Init**，否则收到的数据会被静默丢弃。

### 3.2 主循环轮询（`Net_Process`）

`app_main.c` 的 `App_Process()` 里调用 `Net_Process()`，它内部：

1. `MX_LWIP_Process()` → 收包、跑 TCP/UDP 定时器、链路检测；
2. 根据模式调用 `TcpServer_Process()` / `TcpClient_Process()` / `UdpSync_Process()`。

> 因为 `NO_SYS`，**所有网络处理都在主循环里**。主循环不能卡太久（比如 LCD 刷屏），否则网络就卡。这就是为什么 FSMC 时序要调快。

### 3.3 静态 IP 在 `net_conf.h` 配置（覆盖 CubeMX）

`App/conf/net_conf.h` 里通过 `USER_NET_IP0..3`、`USER_NET_NETMASK*`、`USER_NET_GATEWAY*`、`USER_NET_MAC*` 配静态地址。`lwip.c` 的 `MX_LWIP_Init()` 在 `USER CODE` 区用它**覆盖 CubeMX 的默认值**。

- **主板**（`net_master.h`）：`192.168.1.240`，节点 `1`，对端 `192.168.1.241`。
- **从板**（`net_slave.h`）：`192.168.1.241`，节点 `2`，对端 `192.168.1.240`。
- 烧录前**手动改 `NET_CONFIG_PROFILE`**（`MASTER` / `SLAVE`），一个固件实例用一套。

### 3.4 TCP 维护通道（文本协议）

TCP 走**文本行协议**，每行以 `\r\n` 结束。`net_protocol.c` 里的 `NetLineReader` 负责按行分割（自动剥 `\r`）。

收到一行，`net_handler.c` 解析命令：

| 命令 | 说明 | 要确认吗 |
|---|---|---|
| `help` | 列出命令 | 否 |
| `status` | 立即返回一行状态 | 否 |
| `info` | 返回电机数、节点、IP | 否 |
| `enable,CONFIRM` | 使能电机 | **是** |
| `disable` | 失能电机 | 否 |
| `hold` | 保持 | 否 |
| `zero,CONFIRM` | 当前角度清零 | **是** |
| `param,index,value,CONFIRM` | 写参数 | **是** |

- **破坏性操作必须 `CONFIRM`**，且只在"失能 + 静止 + 反馈新鲜"时才接受（`RobStride_App_Request` 里校验）。
- 状态回传格式：`MSTATUS,n=3,tx=...,txe=...,rx=...,rxe=...;id=1,a=..,v=..,t=..,temp=..,m=..,o=..,s=..,safe=..,age=..,miss=..`。

### 3.5 UDP 实时通道（二进制协议）

UDP 走**自定义二进制协议**（`net_realtime.c`），**不重传**、**只认最新的值**：

- 报文 = `20 字节头 | payload | 2 字节 CRC16-CCITT`，**小端序**，不用 C 结构体强转。
- 头部：magic `0x5352`(RS)、版本、消息类型、节点 ID、会话 ID、序号、时间戳、payload 长度。
- **消息类型**：`HELLO`（建立会话）、`CMD_BATCH`（下发一批轴目标）、`STATE`（回传状态）、`PEER_STATE`（主从板互发状态）、`MNT/ACK`（TCP 维护）。
- **安全校验**：magic、版本、长度、CRC、节点 ID、会话、序号都合法才接受，否则直接拒收。**UDP 拒绝维护命令**（只走 TCP）。
- 每轴状态记录 24 字节：`can_id, online, app_state, mode_state, feedback_age_ms, angle, speed, torque, temperature`。

### 3.6 收包模型：轮询（够用）

- 当前是**纯轮询**：`Net_Process() → MX_LWIP_Process() → ethernetif_input()` 在主循环里收包。
- ICMP 回包延迟 ≈ 主循环轮一圈的时间。FSMC 修复后 `ping` 基本 **<1 ms**，够用。
- **可选优化（未做）**：ETH 接收走中断（`ETH_IRQHandler → HAL_ETH_IRQHandler → ethernetif_input`），让回包不依赖主循环。但要在中断上下文里收包，需保护可重入，且叠加 CAN/TIM6 中断要考虑抢占抖动。**当前收益不大，先不动。**

---

## 4. 一句话总结

- **LwIP** = 嵌入式轻量协议栈，裸机用 `NO_SYS` + Raw API 轮询最省。
- **TCP/UDP 分工**：实时数据 UDP（快、不重传），维护命令 TCP（可靠、要确认）。
- **CubeMX**：先配 ETH（RMII + LAN8720 + 引脚 + PHY 地址 0），再配 LwIP（静态 IP、DHCP 关、NO_SYS）。
- **开发使用**：静态 IP 在 `net_conf.h` 覆盖；主循环轮询 `Net_Process`；TCP 文本行协议 + UDP 二进制协议。
- **最大坑**：FSMC 时序太慢会把主循环拖死，网络跟着饿死 → 一定要把 FSMC 时序调合理。

> 下一篇：[FSMC/LCD](./fsmc(lcd)介绍.md) —— 为什么 LCD 这么占时间、FSMC 怎么把刷屏变快、以及配置的坑。
