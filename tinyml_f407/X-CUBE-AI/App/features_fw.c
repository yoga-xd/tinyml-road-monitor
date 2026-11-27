/*
 * features_fw.c —— 固件版 24 维特征（float32）
 * 数值验证: 见 features_c/compare_features.py（PC 上 double 版对照 24/24 PASS）
 * 本版改为 float32 内部运算，板上预期再叠加 ~1e-6 相对偏差（验证模式实测）
 */
#include "features_fw.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* 显示专用频谱(去直流 8 频段, 归一化和=1): features_compute 每窗更新, 供事件帧频谱可视化
 * 与特征 17-24(含直流口径)完全解耦 —— 模型输入零影响 */
static float s_bands_disp[8];

/* 迭代式基-2 复数 FFT（float32，原地）。板上可换 CMSIS-DSP arm_rfft_fast_f32，
 * 精度差异在验证模式中量化；先用自己的实现保证零外部依赖 */
static void fft_128f(float *re, float *im)
{
    const int n = N_FFT;
    int i, j, len;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265f / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float vr = re[b] * cr - im[b] * ci;
                float vi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - vr;  im[b] = im[a] - vi;
                re[a] += vr;         im[a] += vi;
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

/* float32 版线性插值分位数（需已排序数组） */
static float percentile_f(const float *srt, float q)
{
    float h = (float)(N_SAMP - 1) * q / 100.0f;
    int lo = (int)h;
    int hi = lo + 1; if (hi > N_SAMP - 1) hi = N_SAMP - 1;
    return srt[lo] + (h - (float)lo) * (srt[hi] - srt[lo]);
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

void features_compute(const float *x, float *f)
{
    float srt[N_SAMP], dm[N_SAMP], re[N_FFT], im[N_FFT];
    int i, k;

    /* 1-2: mean / std(总体) */
    float mean = 0, sum2 = 0, m2 = 0, minv = x[0], maxv = x[0], absv = 0;
    for (i = 0; i < N_SAMP; i++) {
        float v = x[i];
        mean += v;  sum2 += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        if (fabsf(v) > absv) absv = fabsf(v);
    }
    mean /= N_SAMP;
    for (i = 0; i < N_SAMP; i++) { float d = x[i] - mean; m2 += d * d; }
    float std = sqrtf(m2 / N_SAMP);
    float rms = sqrtf(sum2 / N_SAMP);

    /* 3-7: peak / ptp / crest / energy */
    f[0] = mean;             /* mean */
    f[1] = std;              /* std */
    f[2] = rms;              /* rms */
    f[3] = absv;             /* peak */
    f[4] = maxv - minv;      /* peak_to_peak */
    f[5] = absv / (rms + FEAT_EPS);          /* crest */
    f[6] = sum2;             /* energy */

    /* 8: zcr —— np.sign 语义 {-1,0,1}，零样本产生两次跳变 */
    int sgn = (x[0] > 0) - (x[0] < 0), flips = 0;
    for (i = 1; i < N_SAMP; i++) {
        int s = (x[i] > 0) - (x[i] < 0);
        if (s != sgn) flips++;
        sgn = s;
    }
    f[7] = (float)flips / (N_SAMP - 1);

    /* 9: iqr */
    memcpy(srt, x, sizeof(float) * N_SAMP);
    qsort(srt, N_SAMP, sizeof(float), cmp_float);
    f[8] = percentile_f(srt, 75.0f) - percentile_f(srt, 25.0f);

    /* 10-11: skew / kurt（总体矩）*/
    float m3 = 0, m4 = 0;
    for (i = 0; i < N_SAMP; i++) {
        dm[i] = x[i] - mean;
        m3 += dm[i] * dm[i] * dm[i];
        m4 += dm[i] * dm[i] * dm[i] * dm[i];
    }
    m3 /= N_SAMP;  m4 /= N_SAMP;
    float std3 = std * std * std;
    f[9]  = m3 / (std3 + FEAT_EPS);
    f[10] = m4 / (std3 * std + FEAT_EPS);

    /* 12-16: 自相关滞后 1-5 */
    float std2 = std * std;
    for (k = 1; k <= 5; k++) {
        float s = 0;
        int cnt = N_SAMP - k;
        for (i = 0; i < cnt; i++) s += dm[i] * dm[i + k];
        f[10 + k] = (s / cnt) / (std2 + FEAT_EPS);
    }

    /* 17-24: 128 点 FFT 幅值² → 8 频段归一化能量 */
    for (i = 0; i < N_FFT; i++) { re[i] = x[i]; im[i] = 0.0f; }
    fft_128f(re, im);
    float be[8] = {0}, tot = 0;
    for (i = 0; i < 65; i++) {
        float p = re[i] * re[i] + im[i] * im[i];   /* |F|² 免开方 */
        int b;
        for (b = 0; b < 8; b++)
            if (i >= BAND_EDGE[b] && i < BAND_EDGE[b + 1]) { be[b] += p; break; }
    }
    for (i = 0; i < 8; i++) tot += be[i];
    for (i = 0; i < 8; i++) f[16 + i] = be[i] / (tot + FEAT_EPS);

    /* 显示专用(20261125): 去直流(dm 去均值) FFT → 8 频段归一化能量(和=1), 存 s_bands_disp
     * 供 B 板频谱可视化 —— 与特征 17-24 完全解耦(特征仍用含直流口径, 保训练/部署一致)。
     * 复用 re/im(17-24 已用完), 零栈增长; +1 次 128 点 FFT/窗(<1ms) */
    for (i = 0; i < N_FFT; i++) { re[i] = dm[i]; im[i] = 0.0f; }
    fft_128f(re, im);
    float dbe[8] = {0}, dtot = 0;
    for (i = 0; i < 65; i++) {
        float p = re[i] * re[i] + im[i] * im[i];
        int b;
        for (b = 0; b < 8; b++)
            if (i >= BAND_EDGE[b] && i < BAND_EDGE[b + 1]) { dbe[b] += p; break; }
    }
    for (i = 0; i < 8; i++) dtot += dbe[i];
    for (i = 0; i < 8; i++) s_bands_disp[i] = dbe[i] / (dtot + FEAT_EPS);
}

/* 显示专用去直流 8 频段(0~1, 和=1) → uint8(0~255, 和≈255): 事件帧频谱字段 */
void features_bands_disp8(uint8_t out[8])
{
    int i;
    for (i = 0; i < 8; i++) {
        float v = s_bands_disp[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        out[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
}
