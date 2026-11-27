# 基于无线传感网络的车载路面状态监测系统

**简体中文** | [English](README.en.md)

双 STM32F407 节点 + 端侧 TinyML 推理 + nRF24L01+ 无线链路的路面状态实时监测系统。传感节点在端侧完成振动信号特征提取与轻量级卷积神经网络推理，将振动实时分类为**平整 / 坑洞 / 减速带 / 磨损**四类路面状态；接收节点完成链路监测、频谱显示、类别计数与记录存储。

> 本项目模型基于公开的 [González 2017 路面事件数据集](#引用)训练。

## 系统架构

```
┌─────────────── A 板（传感 / 推理 / 上报）───────────────┐      ┌─────────────── B 板（接收 / 显示 / 汇聚）──────────────┐
│  GY-521 ──I2C2──► STM32F407ZGT6 ◄──SPI1── W25Q128      │      │  nRF24L01+ ──SPI1──► STM32F407ZGT6 ◄──SPI1── W25Q128  │
│  (±8g, 50Hz)     24维特征·CNN推理    (16B定长记录)      │ ───► │  (心跳+事件)       链路三态·类别计数   (事件落库)       │
│                  │                                     │ 2.4G │                  │                                    │
│                  ▼                                     │ 无线 │                  ▼                                    │
│              3.5寸 TFTLCD（FSMC）                       │ 链路 │              3.5寸 TFTLCD（事件卡·频谱·计数）           │
└────────────────────────────────────────────────────────┘      └────────────────────────────────────────────────────────┘
```

- **A 板**：以 50 Hz 采集三轴加速度 → 128 点滑窗（2.56 s）→ 静止预筛 → 24 维特征 + 双输入 CNN 推理 → 报警状态机（3 连窗一致 + 置信度 ≥ 0.90 + 5 s 去重）→ 事件写入 W25Q128 并经 nRF24L01+ 上报
- **B 板**：接收"心跳（7 B @ 2 Hz）+ 事件（15 B）"双语义帧 → 模 256 丢帧检测与链路三态判定（连接/暂停/失联）→ 事件卡、8 频段频谱条形图、类别计数显示，事件同步落库
- **无线协议**：自动应答 + 自动重传 + 动态载荷三级兜底，持续数据率约 **17 B/s**

## 关键指标

| 项目 | 结果 |
|---|---|
| 模型精度（十折交叉验证 / 留出测试） | 82.8%±4.1% / 84.0% |
| 模型规模 | 3,652 参数 / 19.7 KB（TFLite） |
| 板端推理耗时（168 MHz，单窗约 11.2 万次乘加） | 约 7.3 ms |
| 板端 vs PC 端一致性（75 条验证事件） | 判类 75/75，概率最大偏差 2.98×10⁻⁷ |
| 无线链路对账 | 发送 584 帧 = 接收 572 + 事件 12，**零丢帧** |
| 存储容量（W25Q128，16 B 定长记录） | 约 105 万条 / 约 30 天 |
| 模拟路面激励应用测试 | 8 次报警全部正确 |

## 硬件清单

| 部件 | 型号 | 说明 |
|---|---|---|
| 主控 ×2 | 正点原子探索者 STM32F407ZGT6 | Cortex-M4 168 MHz，板载 W25Q128 |
| 振动传感 | GY-521（MPU6050 兼容） | ±8 g / 50 Hz / DLPF 44 Hz，I2C |
| 无线收发 | nRF24L01+（Si24R1 兼容） | 2.4 GHz GFSK，信道 76，1 Mbps，DPL |
| 存储 | W25Q128（板载） | SPI NOR Flash 16 MB |
| 显示 | 3.5 寸 TFTLCD 320×480 | FSMC 16 位并口 |
| 工具链 | STM32CubeIDE + X-CUBE-AI；TensorFlow / Keras | |

## 目录结构

```
tinyml_project/
├── TinyML路面识别_模型训练与部署全程记录.md   # 训练篇：数据集→基线→调参→定版全程
├── TinyML路面识别_双板部署与联调全程记录.md   # 部署篇：双板联调、协议演进、排障案例
├── 坑洞误判分析与重训方案.md               # 坑洞误判分析与重训方案
├── prepare_data_gonzalez.py                  # 数据集切窗与预处理
├── gonzalez_cropaug.py                       # 裁剪增强训练（定版模型出处）
├── gonzalez_experiments.py / train_model_gonzalez.py / tune_gonzalez.py
├── stage_a_convert.py                        # Keras → TFLite 转换与一致性验证
├── cross_eval_gonzalez_on_roadsens.py        # 跨数据集泛化评估
├── *_log_gonzalez.txt                        # 全部训练日志（报告数据出处）
├── model_gonzalez_crop.h5 / .tflite          # ★ 定版模型
├── model_gonzalez_crop_scalers.json          # 定版归一化常数
├── norm_constants_gonzalez.json              # 固件端特征归一化头文件来源
├── data/gonzalez.npz                         # 预处理后的数据集（128 点窗口）
├── validation_data/                          # 75 条板前验证包与生成脚本
├── features_c/                               # C 语言特征参考实现与 Python 对照工具
├── pc_tools/compare_board.py                 # 板端 vs PC 端一致性比对工具
├── firmware/                                 # A/B 板驱动源码权威副本
├── tinyml_f407/                              # ★ STM32CubeIDE 完整工程（A/B 角色宏切换）
└── backup/
    ├── 双板定版/                    # ★ 定版固件（A/B 板 bin/elf + MD5）
    └── 双板固件备份.zip               # 双板定版固件合并包
```

## 复现步骤

### 1. 模型训练（PC 端，Python 3.9+ / TensorFlow 2.x）

```bash
pip install numpy tensorflow scikit-learn

python prepare_data_gonzalez.py    # 数据集切窗 → data/gonzalez.npz
python gonzalez_cropaug.py         # 裁剪增强训练 → model_gonzalez_crop.h5
python stage_a_convert.py          # TFLite 转换 + 一致性验证（75/75）
```

各阶段指标见 `cropaug_log_gonzalez.txt` 等日志文件。

### 2. 固件烧录（STM32CubeIDE 或 DFU）

1. 从 `backup/双板固件备份.zip` 解压，或用 STM32CubeIDE 打开 `tinyml_f407/` 自行编译；
2. A/B 板由工程内**角色宏**一行切换，分别编译烧录；
3. 或直接使用 `backup/双板定版/` 下的 `tinyml_f407_A.bin` / `tinyml_f407_B.bin` 通过 DFU 烧录。

### 3. 端侧验证

```bash
python pc_tools/compare_board.py   # 板端经串口输出推理概率，与 PC 参考实现逐条比对
python validation_data/gen_validation_pack.py   # 重新生成 75 条验证包
```

## 引用

本项目模型基于 González 等人 2017 年公开发布的路面事件数据集训练：500 条车载垂直加速度时间序列（50 Hz，MEMS 加速度计，实车采集），覆盖减速带、金属凸起、坑洞、正常路面、磨损路面五类事件，每类 100 条。数据集随原论文公开发布，使用本仓库代码或数据时请引用：

> Luis C. González, Ricardo Moreno, Hugo Jair Escalante, Fernando Martínez, Manuel Ricardo Carlos. Learning Roadway Surface Disruption Patterns Using the Bag of Words Representation. **IEEE Transactions on Intelligent Transportation Systems**, 2017, 18(11): 2916–2928. DOI: [10.1109/TITS.2017.2662483](https://doi.org/10.1109/TITS.2017.2662483)

<details>
<summary>BibTeX</summary>

```bibtex
@article{gonzalez2017roadway,
  author  = {Gonz{\'a}lez, Luis C. and Moreno, Ricardo and Escalante, Hugo Jair and Mart{\'i}nez, Fernando and Carlos, Manuel Ricardo},
  title   = {Learning Roadway Surface Disruption Patterns Using the Bag of Words Representation},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {18},
  number  = {11},
  pages   = {2916--2928},
  year    = {2017},
  doi     = {10.1109/TITS.2017.2662483}
}
```

</details>

## License

代码部分采用 MIT License 发布；正点原子板级驱动（`firmware/`、`tinyml_f407/Drivers` 等）版权归原厂商，仅作学习交流用途。
