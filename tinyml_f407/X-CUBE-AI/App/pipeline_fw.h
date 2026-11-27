/*
 * pipeline_fw.h —— 生产模式流水线（采集→标准化→特征→推理→报警）
 */
#ifndef PIPELINE_FW_H
#define PIPELINE_FW_H

#include "main.h"

typedef struct {
    uint8_t  cls;        /* 报警类别: 1坑洞 2减速带 3磨损 */
    float    conf;       /* 置信度 */
    uint32_t tick;       /* 报警时刻 (HAL_GetTick) */
    uint32_t window_id;  /* 窗口号 */
} alarm_event_t;

/* 初始化: MPU6050 + 网络。传 CubeMX 生成的 I2C 句柄。返回 0=成功 */
int pipeline_init(I2C_HandleTypeDef *hi2c);

/* 主循环调用（20ms 粒度轮询即可；内部自定时 50Hz 采样 / 500ms 推理）*/
void pipeline_process(void);

/* 最近一次报警事件（无报警时 cls=0）*/
alarm_event_t pipeline_last_alarm(void);

/* 最近窗口活动电平: 去直流 RMS (m/s^2), 供心跳帧上报 */
float pipeline_activity(void);

/* KEY2 清空: 复位锁存事件/报警计数/候选窗口 */
void pipeline_alarm_clear(void);

/* 暂停恢复保护: 清空报警候选窗口计数(KEY0 恢复运行时调用) */
void pipeline_resume_reset(void);

/* 调试统计 */
typedef struct {
    uint32_t samples, inferences, alarms;
    float    last_probs[4];   /* 最近一次推理的 4 类概率 */
} pipeline_stats_t;
pipeline_stats_t pipeline_stats(void);

/* 诊断读出: 最近窗口 std(特征1, m/s²) / 报警候选类别(-1=无) / 候选连续计数 */
float pipeline_window_std(void);
int   pipeline_cand_cls(void);
int   pipeline_cand_cnt(void);

/* 最近窗 8 频段归一化能量(特征17-24) → uint8(0~255, 和≈255): 事件帧频谱字段 */
void pipeline_band_e8(uint8_t out[8]);

#endif
