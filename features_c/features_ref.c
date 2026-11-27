/*
 * features_ref.c —— Gonzalez 24 维特征参考实现（C 版）
 *
 * 与 Python 训练端 features_batch (gonzalez_experiments.py) 逐项对应：
 *   0  mean        窗口均值
 *   1  std         总体标准差（ddof=0）
 *   2  rms         均方根
 *   3  peak        |x| 最大值
 *   4  ptp         峰峰值 (max-min)
 *   5  crest       波峰因子 peak/(rms+EPS)
 *   6  energy      Σx²
 *   7  zcr         过零率（np.sign 语义：{-1,0,1}，零样本产生两次跳变）/127
 *   8  iqr         p75-p25（np.percentile linear 插值法）
 *   9  skew        三阶矩/std³
 *   10 kurt        四阶矩/std⁴
 *   11-15 autocorr 滞后 1-5：mean(dm[i]*dm[i+k])/(std²+EPS)
 *   16-23 band_e   128 点 rFFT 幅值²按 8 频段求和后归一化
 *                  频段边界 = np.array_split(65,8) = [9,8,8,8,8,8,8,8]
 *
 * 输入:  windows.bin  [int32 N][float32 N*128]   原始 m/s²
 * 输出:  feats_c.bin  [int32 N][float32 N*24]    特征矩阵
 *
 * 内部全部用 double 计算（数值参考实现）；STM32 固件版用 float32 +
 * CMSIS-DSP arm_rfft_fast_f32 替换 fft_128，其余逻辑一字不改。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SAMP 128
#define N_FFT  128
#define N_FEAT 24
#define EPS    1e-6
#define PI     3.14159265358979323846

/* np.array_split(65, 8) 的频段边界（65 个 rFFT bin） */
static const int BAND_EDGE[9] = {0, 9, 17, 25, 33, 41, 49, 57, 65};

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* numpy.percentile(method="linear")：虚拟下标 h=(N-1)*q/100，线性插值 */
static double percentile_linear(const double *srt, double q)
{
    double h = (N_SAMP - 1) * q / 100.0;
    int lo = (int)h;
    int hi = lo + 1; if (hi > N_SAMP - 1) hi = N_SAMP - 1;
    double frac = h - lo;
    return srt[lo] + frac * (srt[hi] - srt[lo]);
}

/* 迭代式基-2 复数 FFT，n=128，原地计算（实输入→取 bin 0..64 即 rfft） */
static void fft_128(double *re, double *im)
{
    const int n = N_FFT;
    int i, j, len;
    for (i = 1, j = 0; i < n; i++) {          /* 位反转 */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / (double)len;
        double wr = cos(ang), wi = sin(ang);
        for (i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                double vr = re[b] * cr - im[b] * ci;
                double vi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - vr;  im[b] = im[a] - vi;
                re[a] += vr;         im[a] += vi;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

/* 单窗口 24 特征计算：x = 原始 m/s² (float32)，f = 输出 (double) */
static void compute_features(const float *x, double *f)
{
    double srt[N_SAMP], dm[N_SAMP], re[N_FFT], im[N_FFT];
    int i, k;

    double mean = 0, sum2 = 0, m2 = 0, minv = x[0], maxv = x[0], absv = 0;
    for (i = 0; i < N_SAMP; i++) {
        double v = (double)x[i];
        mean += v;
        sum2 += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        if (fabs(v) > absv) absv = fabs(v);
    }
    mean /= N_SAMP;
    for (i = 0; i < N_SAMP; i++) {
        double d = (double)x[i] - mean;
        m2 += d * d;
    }
    double std = sqrt(m2 / N_SAMP);
    double rms = sqrt(sum2 / N_SAMP);
    double ptp = maxv - minv;
    double crest = absv / (rms + EPS);
    double energy = sum2;

    int sgn = (x[0] > 0) - (x[0] < 0), flips = 0;
    for (i = 1; i < N_SAMP; i++) {
        int s = (x[i] > 0) - (x[i] < 0);
        if (s != sgn) flips++;
        sgn = s;
    }
    double zcr = (double)flips / (N_SAMP - 1);

    for (i = 0; i < N_SAMP; i++) srt[i] = (double)x[i];
    qsort(srt, N_SAMP, sizeof(double), cmp_double);
    double iqr = percentile_linear(srt, 75.0) - percentile_linear(srt, 25.0);

    double m3 = 0, m4 = 0;
    for (i = 0; i < N_SAMP; i++) {
        dm[i] = (double)x[i] - mean;
        m3 += dm[i] * dm[i] * dm[i];
        m4 += dm[i] * dm[i] * dm[i] * dm[i];
    }
    m3 /= N_SAMP;  m4 /= N_SAMP;
    double std3 = std * std * std;
    double std4 = std3 * std;
    double std2 = std * std;

    f[0] = mean;   f[1] = std;   f[2] = rms;   f[3] = absv;
    f[4] = ptp;    f[5] = crest; f[6] = energy; f[7] = zcr;
    f[8] = iqr;    f[9] = m3 / (std3 + EPS);
    f[10] = m4 / (std4 + EPS);

    for (k = 1; k <= 5; k++) {                 /* 自相关滞后 1-5 */
        double s = 0;
        int cnt = N_SAMP - k;
        for (i = 0; i < cnt; i++) s += dm[i] * dm[i + k];
        f[10 + k] = (s / cnt) / (std2 + EPS);
    }

    for (i = 0; i < N_FFT; i++) { re[i] = (double)x[i]; im[i] = 0.0; }
    fft_128(re, im);
    double be[8] = {0}, tot = 0;
    for (i = 0; i < 65; i++) {
        double p = re[i] * re[i] + im[i] * im[i];   /* |F|² 直接由复数算，免开方 */
        int b;
        for (b = 0; b < 8; b++)
            if (i >= BAND_EDGE[b] && i < BAND_EDGE[b + 1]) { be[b] += p; break; }
    }
    for (i = 0; i < 8; i++) tot += be[i];
    for (i = 0; i < 8; i++) f[16 + i] = be[i] / (tot + EPS);
}

int main(void)
{
    FILE *fi = fopen("windows.bin", "rb");
    FILE *fo = fopen("feats_c.bin", "wb");
    int n, i, w;
    float *win;
    double f[N_FEAT];
    if (!fi || !fo) { fprintf(stderr, "文件打开失败\n"); return 1; }
    fread(&n, 4, 1, fi);
    win = (float *)malloc((size_t)n * N_SAMP * sizeof(float));
    fread(win, sizeof(float), (size_t)n * N_SAMP, fi);
    fwrite(&n, 4, 1, fo);
    for (w = 0; w < n; w++) {
        compute_features(win + (size_t)w * N_SAMP, f);
        for (i = 0; i < N_FEAT; i++) {
            float v = (float)f[i];
            fwrite(&v, sizeof(float), 1, fo);
        }
    }
    fclose(fi); fclose(fo);
    free(win);
    printf("C 特征计算完成: %d 窗口 x %d 特征 -> feats_c.bin\n", n, N_FEAT);
    return 0;
}
