# Vehicle-Mounted Road Surface Condition Monitoring System Based on a Wireless Sensor Network

[简体中文](README.md) | **English**

A real-time road surface condition monitoring system built on dual STM32F407 nodes with on-device TinyML inference and an nRF24L01+ wireless link. The sensor node performs vibration feature extraction and lightweight CNN inference on the edge, classifying vibration signals in real time into four road surface states: **flat / pothole / speed bump / worn**; the receiver node provides link monitoring, spectrum display, class counting, and record storage.

> The model in this project was trained on the public [González 2017 road surface event dataset](#citation).

## System Architecture

```
┌──────────── Node A (sensing / inference / report) ───────┐      ┌──────────── Node B (receive / display / sink) ──────────┐
│  GY-521 ──I2C2──► STM32F407ZGT6 ◄──SPI1── W25Q128        │      │  nRF24L01+ ──SPI1──► STM32F407ZGT6 ◄──SPI1── W25Q128    │
│  (±8g, 50Hz)    24-dim features · CNN   (16B records)    │ ───► │  (heartbeat+event)  link state · counts  (event store)  │
│                  │                                       │ 2.4G │                  │                                      │
│                  ▼                                       │ link │                  ▼                                      │
│           3.5" TFTLCD (FSMC)                             │      │           3.5" TFTLCD (event card · spectrum · counts)  │
└──────────────────────────────────────────────────────────┘      └─────────────────────────────────────────────────────────┘
```

- **Node A**: samples 3-axis acceleration at 50 Hz → 128-point sliding window (2.56 s) → static pre-screening → 24-dim features + dual-input CNN inference → alarm state machine (3 consecutive consistent windows + confidence ≥ 0.90 + 5 s deduplication) → event written to W25Q128 and reported via nRF24L01+
- **Node B**: receives dual-semantic frames ("heartbeat: 7 B @ 2 Hz + event: 15 B") → modulo-256 frame-loss detection and tri-state link determination (connected / paused / lost) → event card, 8-band spectrum bar chart, and class counting display, with events stored to flash
- **Wireless protocol**: hardware auto-ACK + auto-retransmission + three-level dynamic-payload fallback, sustained data rate ≈ **17 B/s**

## Key Metrics

| Item | Result |
|---|---|
| Model accuracy (10-fold CV / held-out test) | 82.8%±4.1% / 84.0% |
| Model size | 3,652 parameters / 19.7 KB (TFLite) |
| On-board inference latency (168 MHz, ~112k MACs per window) | ≈ 7.3 ms |
| Board vs PC consistency (75 validation events) | 75/75 class match, max probability deviation 2.98×10⁻⁷ |
| Wireless link audit | 584 sent = 572 received + 12 events, **zero frame loss** |
| Storage capacity (W25Q128, 16 B fixed-length records) | ≈ 1.05 M records / ≈ 30 days |
| Simulated road-excitation application test | 8/8 alarms correct |

## Hardware

| Part | Model | Notes |
|---|---|---|
| MCU ×2 | ALIENTEK Explorer STM32F407ZGT6 | Cortex-M4 168 MHz, on-board W25Q128 |
| Vibration sensor | GY-521 (MPU6050 compatible) | ±8 g / 50 Hz / DLPF 44 Hz, I2C |
| Wireless transceiver | nRF24L01+ (Si24R1 compatible) | 2.4 GHz GFSK, channel 76, 1 Mbps, DPL |
| Storage | W25Q128 (on-board) | SPI NOR Flash 16 MB |
| Display | 3.5" TFTLCD 320×480 | FSMC 16-bit parallel |
| Toolchain | STM32CubeIDE + X-CUBE-AI; TensorFlow / Keras | |

## Repository Layout

Documentation files are written in Chinese; code and logs use standard identifiers.

```
tinyml_project/
├── TinyML路面识别_模型训练与部署全程记录.md   # (Chinese) full record: model training & deployment
├── TinyML路面识别_双板部署与联调全程记录.md   # (Chinese) full record: dual-board integration
├── 坑洞误判分析与重训方案.md               # (Chinese) pothole misclassification analysis & retraining plan
├── prepare_data_gonzalez.py                  # dataset windowing & preprocessing
├── gonzalez_cropaug.py                       # crop-augmentation training (source of the final model)
├── gonzalez_experiments.py / train_model_gonzalez.py / tune_gonzalez.py
├── stage_a_convert.py                        # Keras → TFLite conversion & consistency verification
├── cross_eval_gonzalez_on_roadsens.py        # cross-dataset generalization evaluation
├── *_log_gonzalez.txt                        # all training logs (source of reported numbers)
├── model_gonzalez_crop.h5 / .tflite          # ★ final model
├── model_gonzalez_crop_scalers.json          # normalization constants of the final model
├── norm_constants_gonzalez.json              # source of the firmware-side feature normalization header
├── data/gonzalez.npz                         # preprocessed dataset (128-point windows)
├── validation_data/                          # 75-event pre-deployment validation pack + generator
├── features_c/                               # C reference implementation of features + Python comparison tools
├── pc_tools/compare_board.py                 # board-vs-PC consistency comparison tool
├── firmware/                                 # authoritative copy of A/B board driver sources
├── tinyml_f407/                              # ★ complete STM32CubeIDE project (A/B role macro switch)
└── backup/
    ├── 双板定版/                    # ★ final firmware (A/B bin/elf + MD5)
    └── 双板固件备份.zip               # combined final-firmware package
```

## Reproduction

### 1. Model training (PC, Python 3.9+ / TensorFlow 2.x)

```bash
pip install numpy tensorflow scikit-learn

python prepare_data_gonzalez.py    # dataset windowing → data/gonzalez.npz
python gonzalez_cropaug.py         # crop-augmentation training → model_gonzalez_crop.h5
python stage_a_convert.py          # TFLite conversion + consistency verification (75/75)
```

Per-stage metrics are recorded in `cropaug_log_gonzalez.txt` and other log files.

### 2. Firmware flashing (STM32CubeIDE or DFU)

1. Extract `backup/双板固件备份.zip`, or open `tinyml_f407/` in STM32CubeIDE and build it yourself;
2. Node A / Node B firmware is selected by a single **role macro** in the project; build and flash each separately;
3. Alternatively, flash `tinyml_f407_A.bin` / `tinyml_f407_B.bin` from `backup/双板定版/` directly via DFU.

### 3. On-device verification

```bash
python pc_tools/compare_board.py   # board outputs inference probabilities over UART; compare with the PC reference
python validation_data/gen_validation_pack.py   # regenerate the 75-event validation pack
```

## Citation

The model in this project was trained on the road surface event dataset publicly released by González et al. (2017): 500 in-vehicle vertical acceleration time series (50 Hz, MEMS accelerometer, real-vehicle collection) covering five event classes — speed bumps, metal bumps, potholes, regular road, and worn road — 100 series each. The dataset was released publicly alongside the original paper. If you use code or data from this repository, please cite:

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

The code in this repository is released under the MIT License. ALIENTEK board-level driver code (`firmware/`, `tinyml_f407/Drivers`, etc.) remains the property of its original vendor and is included for educational purposes only.
