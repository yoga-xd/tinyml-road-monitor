双板定版固件(定版验收实测通过)
=====================================================
## 烧录文件
- A 板(传感/推理/上报): tinyml_f407_A.elf  (md5 92ab1f80)
  = 界面 v4 + 心跳协议 v3(暂停同步/cls-conf 修复) + 频谱去直流数据源
    + 开机 gk 标定修正量程 + DPL 自检(7B/15B 变长帧)
- B 板(接收/显示/汇聚): tinyml_f407_B.elf  (md5 7b714023)
  = LINK 三态(CONNECT/PAUSED/LOST) + 频谱入卡(开方刻度 60px) + 链路质量计数
    (RECEIVED/LOST/EVENTS/FLASH 四行) + 历史页/清空 + 灰字黑 + v4 类名
- 配对约束: 心跳协议 v3 + 事件帧 15B(去直流频谱), 两板必须同为双板定版
## 验收结论(实测)
- 三类路况(减速带/坑洞/磨损)识别全通, 频谱形态分类可辨
- 零丢帧(HEARTBEATS 375 = SEQUENCE 119 + 一次 uint8 回绕)
- LINK 三态/暂停同步/历史页/清空/频谱入卡全部实测通过
## 烧录命令(DFU, BOOT0=3.3V 侧)
A: STM32_Programmer_CLI -c port=usb1 -w tinyml_f407_A.elf -v
B: STM32_Programmer_CLI -c port=usb1 -w tinyml_f407_B.elf -v
