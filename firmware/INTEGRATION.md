# 固件集成说明（生产模式流水线 → tinyml_f407 工程）

## 文件清单与放置

| 文件 | 放置位置（CubeIDE 工程内） |
|------|--------------------------|
| `features_fw.c/.h` | `X-CUBE-AI/App/`（或自建 `App/` 目录并加入包含路径） |
| `pipeline_fw.c/.h` | 同上 |
| `mpu6050_fw.c/.h` | 同上 |
| `scalers_gonzalez.h` | 同上（任意被包含路径内） |

在 CubeIDE 里：把文件复制进工程目录 → 右键工程 → Refresh（F5）即可参与编译
（工程用 make 自动收集 .c）。

## CubeMX 需补充启用的外设（生成代码前配好）

1. **I²C（Connectivity → I2C → I2C）**——**两块板均 V3，已定稿走分支 A**：
   MPU6050 为**外接 GY-521 模块**（仅 A 板装），插 ATK-MODULE 接口，
   I²C 引脚 **PB10=SCL / PB11=SDA** → CubeMX 配 **I2C2**，`pipeline_init(&hi2c2)`
   - ⚠️ **必须拔掉 ATK-MODULE 接口的 USART3 跳线帽**（PB10/PB11 与 USART3 TXD/RX 复用，不拔会总线冲突）
   - GY-521 自带 3.3 V 稳压与上拉，接模块口的 5V/GND/PB10/PB11 四根即可；AD0 接地 → 地址 0x68
     （`mpu6050_fw.c` 已做 0x68/0x69 地址自适应，无需改驱动）
   - 分支 B（V2 板载 MPU6050，I2C1 @ PB8/PB9）仅作备用记录，当前不采用
   - 速率 100 kHz（Standard Mode，MPU6050 支持到 400 kHz 可后调）
2. **USART1**（已启用，验证/调试共用，PA9/PA10 → 板载 CH340；
   ⚠️ 板上 USART1 跳线帽须接到 CH340 一侧，否则串口验证全部失败）
3. 可选：**TIM 定时器**（推理耗时精确测量用，功能验证阶段不需要）

## main.c 挂载（全部写在 USER CODE 区，CubeMX 重新生成不会丢）

```c
/* USER CODE BEGIN Includes */
#include "pipeline_fw.h"
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
static int s_prod_mode = 1;   /* 1=生产模式(自主推理报警) 0=验证模式(PC 比对) */
/* USER CODE END PV */

/* USER CODE BEGIN 2 */
if (s_prod_mode) {
    if (pipeline_init(&hi2c2) != 0) Error_Handler();  /* V3 板：I2C2+外接 GY-521；V2 板改 &hi2c1 */
} else {
    MX_X_CUBE_AI_Init();      /* ST 验证模式（等待 PC 通过 stm_ai_runner 连接）*/
}
/* USER CODE END 2 */

while (1) {
    /* USER CODE BEGIN 3 */
    if (s_prod_mode) pipeline_process();
    else             MX_X_CUBE_AI_Process();
    /* USER CODE END 3 */
}
```

## 报警输出钩子与无线帧格式

`pipeline_fw.c` 里 `alarm_output()` 是弱符号，在业务模块里重写即可接三路输出：
**NRF24L01 上报**、**W25Q128 存储**、**LCD 显示**。无线帧分两种：

| 帧 | 周期 | 内容 | 大小 | 用途 |
|---|---|---|---|---|
| 心跳帧 | 每次推理后（步进 0.5 s，窗长仍 2.56 s）；协议 v2 起为 16 B（含活动电平） | 帧头+窗序号+z 轴 RMS+当前判定+置信度 | ~8 B | 链路活性监测（keepalive），B 板驱动实时活动电平条 |
| 事件帧 | 事件发生时（状态机去重后） | 帧头+类别+置信度+窗口号+**8 个 FFT 频段能量（8×uint16）**+三轴 RMS | ~30 B | B 板更新频谱条形图（分类依据可视化）+ 事件列表 + 存储 |

数据率口径：最坏工况合计 ~38 B/s（心跳 32 + 事件 6，协议 v2 心跳 16B 帧长），vs 特征流方案 320 B/s **低 93%**，
vs 原始波形回传 ~600 B/s 低 98.5%。8 频段能量取自 24 维特征子集，推理路径零额外计算。

⚠️ **5 s 去重的语义（勿混淆）**：去重只作用于**事件帧的报警计数**——同一事件（如一个坑洞，
振动持续 2~3 窗）不去重会被记成 2~3 条，事件列表/Flash/统计全部失真；语义 = "一个坑洞 = 一条记录"。
它**不是发包周期**：心跳帧每次推理（0.5 s 步进）必发，B 板"当前判定+置信度+活动电平"以 0.5 s 刷新；每次推理看的仍是最近 2.56 s（128 点@50Hz，与训练口径一致）。
5 s 为初始值，写在 `pipeline_fw.c` 顶部，实路测试时标定（车速低事件拉长则加大）。

## W25Q128 存储（核心要求：传感节点至少具有数据存储能力）

- 硬件：板载 W25Q128（16 MB），**与 nRF24L01 共用 SPI1**（PB3/PB4/PB5）——
  W25Q128 片选 **PB14**，nRF 片选 **PG7**；访问 Flash 时必须保证 nRF CSN 为高，分时复用（对方组同款板验证过的配置）
- 记录格式（定长 16 B，页缓冲 256 B = 16 条/页，写满 4 KB 扇区提前擦除下一扇区）：

  | 字段 | 字节 | 说明 |
  |---|---|---|
  | 窗口时间戳 | 4 | 窗口号 × 窗长 2.56 s |
  | RMS_x/y/z | 6 | float16 精度足够展示 |
  | 类别 argmax | 1 | 0 正常 / 1 坑洞 / 2 减速带 / 3 磨损 |
  | 置信度 | 2 | uint16 = 概率×1000 |
  | 报警位+序号 | 3 | 报警 1 bit + 帧序号 |

  容量核算：约 0.4 窗/s × 16 B ≈ 6.4 B/s，16 MB 可连续记录 **约 29 天**——容量无忧
- 导出：串口命令触发，按记录序号导出 CSV（时间戳、RMS 三轴、类别、置信度、报警位）
- 新增文件：`w25q128_fw.c/.h`（读写/擦除/页缓冲，正点原子 SPI Flash 例程移植）

## 两板固件（同一工程，编译开关区分）

| 板 | 角色 | 编译宏 | 启用的链路 |
|---|---|---|---|
| A 板 | 传感节点 | `-DROLE_SENSOR` | I2C2 采样 → pipeline 推理 → NRF TX + W25Q128 存储 + LCD 状态 |
| B 板 | 网关/基站节点 | `-DROLE_GATEWAY` | NRF RX + CRC 校验 → LCD **频谱条形图主显**（事件帧 8 频段能量）+ 活动电平条（心跳帧）+ 事件列表 + 报警 + W25Q128 存储 + **轻量分析（频谱/类别统计/已知工况命中统计）** |
| 共用 | 验证模式 | `s_prod_mode=0` | PC 逐条比对（与角色无关，两块板都可跑一致性验证） |

## ⚠️ 重新生成代码的已知代价

每次在 CubeMX 点 GENERATE CODE，`app_x-cube-ai.h/.c` 的模板 bug 会重现
（双输入模型令牌留空），需按下列清单重修（共 4 处，位置固定）：

1. `app_x-cube-ai.h`: `#include ".h"` → `#include "network.h"`
2. `app_x-cube-ai.h`: `#include "_data.h"` → `#include "network_data.h"`
3. `app_x-cube-ai.h`: 三个 `SIZE_BYTES 1` → 引用 `AI_NETWORK_IN_1/2_SIZE_BYTES`、`AI_NETWORK_OUT_1_SIZE_BYTES`
4. `app_x-cube-ai.c`: 入口表 `AI__*`→`AI_NETWORK_*`、`ai__*`→`ai_network_*`（10 个符号）；
   `pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE]` → `pool0[AI_NETWORK_DATA_ACTIVATIONS_SIZE]`

修完编译验收：**text ≥ 30 KB**。

## 两个固件模式

| 模式 | 用途 | 入口 |
|------|------|------|
| 验证模式（已编译通过） | PC 逐条下发输入比对输出，量化板上一致性 | `MX_X_CUBE_AI_Init/Process` |
| 生产模式（本骨架） | 自主采样→推理→报警 | `pipeline_init/process` |
