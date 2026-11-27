/*
 * features_fw.h —— 固件版 24 维特征（float32）
 * 与 features_ref.c（double 参考实现）逻辑一致，已通过 PC 数值对照
 * （24/24 PASS，最大相对误差 5.9e-8 = float32 存储极限）
 */
#ifndef FEATURES_FW_H
#define FEATURES_FW_H

#include <stdint.h>

#define N_SAMP   128
#define N_FFT    128
#define N_FEAT   24
#define FEAT_EPS 1e-6f

/* 频段边界 = np.array_split(65,8)：65 个 rFFT bin 分 8 组 */
static const int BAND_EDGE[9] = {0, 9, 17, 25, 33, 41, 49, 57, 65};

/* 计算 24 维特征（顺序与训练端一致）
 * x  : 原始加速度窗口 (m/s², 含重力, z 轴朝上)
 * f  : 输出 24 维特征（未标准化） */
void features_compute(const float *x, float *f);

/* 显示专用: 最近窗去直流 8 频段归一化能量 → uint8(0~255, 和≈255)
 * 与特征 17-24(含直流口径)解耦, 供事件帧频谱可视化 */
void features_bands_disp8(uint8_t out[8]);

#endif
