/*
 * pipeline_fw.c —— 生产模式流水线
 * 依赖: features_fw / scalers_gonzalez / mpu6050_fw / app_x-cube-ai（工程内已有）
 *
 * 时序: 50 Hz 采样入环形缓冲；每 500 ms 取最近 128 点（2.56 s 窗）推理一次
 * 报警: 连续 3 窗同一异常类且置信度≥0.90 才触发；5 s 去重
 * RMS 预筛: 窗口活动度（特征1=std）< 0.15 m/s² 视为静止，跳过推理直接判正常
 */
#include "pipeline_fw.h"
#include "features_fw.h"
#include <math.h>
#include "scalers_gonzalez.h"
#include "mpu6050_fw.h"
#include "app_x-cube-ai.h"
#include <string.h>

/* 可调参数（上车实测后调优）*/
#define SAMPLE_PERIOD_MS   20      /* 50 Hz */
#define INFER_PERIOD_MS    500     /* 2 Hz 推理 */
#define CONF_TH            0.90f   /* 异常候选置信度门槛 */
#define CONSEC_N           3       /* 连续一致次数 */
#define DEDUP_MS           5000    /* 同事件去重窗口 */
#define STD_IDLE_TH        0.15f   /* 静止判据: 窗口 std (m/s²) */

static float    s_ring[N_WINDOW];
static int      s_ring_idx = 0, s_filled = 0;
static uint32_t s_last_sample = 0, s_last_infer = 0;
static uint32_t s_win_id = 0;

static float    s_net_sig[N_WINDOW], s_net_feat[N_FEAT], s_probs[4];
static ai_network_report s_report;   /* X-CUBE-AI 10.x IO 缓冲容器 */
static float *s_p_in0, *s_p_in1, *s_p_out;   /* 实际 IO 数据指针(可能指向 activations) */
static float    s_feat_raw[N_FEAT];
static ai_handle s_net = AI_HANDLE_NULL;

static alarm_event_t s_last_alarm;
static int  s_cand_cls = -1, s_cand_cnt = 0;
static uint32_t s_fail_cnt = 0;   /* 连续采样失败计数(>=25 即 0.5s: 传感器失联) */
static pipeline_stats_t s_stats;

/* 报警输出钩子——弱符号，NRF24L01 上报/屏幕显示在各自模块里重写 */
__weak void alarm_output(const alarm_event_t *ev) { (void)ev; }

int pipeline_init(I2C_HandleTypeDef *hi2c)
{
    memset(&s_last_alarm, 0, sizeof(s_last_alarm));
    memset(&s_stats, 0, sizeof(s_stats));

    if (mpu6050_init(hi2c) != 0) return -1;
    /* 静态自检: z 轴朝上时应约 +9.8（调试串口打印确认符号约定）*/
    /* float z0 = mpu6050_static_check_z();  ← 上板先开这句确认 */

    ai_error err = ai_mnetwork_create(AI_NETWORK_MODEL_NAME, &s_net,
                                      AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) return -2;
    if (!ai_mnetwork_init(s_net)) return -3;

    /* 取网络 IO 描述符，数据指针指向自己的数组（float32 双输入）*/
    /* X-CUBE-AI 10.x: IO 缓冲经 mnetwork report 获取(ai_network_inputs_get 已废弃, 对
     * mnetwork 句柄返回 NULL —— 与 aiValidation.c 参考实现一致) */
    if (!ai_mnetwork_get_report(s_net, &s_report)) return -41;   /* report 获取失败 */
    if (s_report.n_inputs != AI_NETWORK_IN_NUM ||
        s_report.n_outputs != AI_NETWORK_OUT_NUM) return -42;    /* IO 数量不符 */
    if (!s_report.inputs[0].data)
        s_report.inputs[0].data = AI_HANDLE_PTR(s_net_sig);
    if (!s_report.inputs[1].data)
        s_report.inputs[1].data = AI_HANDLE_PTR(s_net_feat);
    if (!s_report.outputs[0].data)
        s_report.outputs[0].data = AI_HANDLE_PTR(s_probs);
    /* 10.x 的 size 字段按字节计; 兼容按元素计的语义: 两种都接受 */
    if (!((s_report.inputs[0].size == N_WINDOW || s_report.inputs[0].size == N_WINDOW * 4) &&
          (s_report.inputs[1].size == N_FEAT  || s_report.inputs[1].size == N_FEAT  * 4)))
        return -43;
    s_p_in0 = (float *)s_report.inputs[0].data;
    s_p_in1 = (float *)s_report.inputs[1].data;
    s_p_out = (float *)s_report.outputs[0].data;
    return 0;
}

static void alarm_update(uint32_t now)
{
    int am = 0;
    for (int i = 1; i < 4; i++) if (s_probs[i] > s_probs[am]) am = i;
    float conf = s_probs[am];

    if (am == 0) { s_cand_cnt = 0; s_cand_cls = -1; return; }   /* 正常窗复位 */
    if (am == s_cand_cls && conf >= CONF_TH) s_cand_cnt++;
    else if (conf >= CONF_TH) { s_cand_cls = am; s_cand_cnt = 1; }
    else s_cand_cnt = 0;

    if (s_cand_cnt >= CONSEC_N) {
        if (now - s_last_alarm.tick >= DEDUP_MS || am != s_last_alarm.cls) {
            s_last_alarm.cls = (uint8_t)am;
            s_last_alarm.conf = conf;
            s_last_alarm.tick = now;
            s_last_alarm.window_id = s_win_id;
            alarm_output(&s_last_alarm);
            s_stats.alarms++;
        }
        s_cand_cnt = 0;   /* 触发后重新计数 */
    }
}

void pipeline_process(void)
{
    uint32_t now = HAL_GetTick();

    /* 50 Hz 采样入环形缓冲 */
    if (now - s_last_sample >= SAMPLE_PERIOD_MS) {
        s_last_sample = now;
        float z;
        if (mpu6050_read_acc_z(&z) == 0) {
            if (s_fail_cnt >= 25) { s_cand_cnt = 0; s_cand_cls = -1; }   /* 失联恢复: 清候选防幻影报警 */
            s_fail_cnt = 0;
            s_ring[s_ring_idx] = z;
            s_ring_idx = (s_ring_idx + 1) % N_WINDOW;
            if (s_filled < N_WINDOW) s_filled++;
            s_stats.samples++;
        } else if (++s_fail_cnt >= 25) {
            /* 修复#3: 传感器失联 0.5s -> 强制正常语义, 冻结报警(防冻结窗幻影报警) */
            s_probs[0] = 1.0f; s_probs[1] = s_probs[2] = s_probs[3] = 0.0f;
            memcpy(s_stats.last_probs, s_probs, sizeof(s_probs));
            s_cand_cnt = 0; s_cand_cls = -1;
        }
    }

    /* 500 ms 推理 */
    if (s_filled < N_WINDOW) return;
    if (now - s_last_infer < INFER_PERIOD_MS) return;
    s_last_infer = now;
    s_win_id++;
    if (s_fail_cnt >= 25) {          /* 修复#3: 失联中不推理, 保持正常语义 */
        s_probs[0] = 1.0f; s_probs[1] = s_probs[2] = s_probs[3] = 0.0f;
        memcpy(s_stats.last_probs, s_probs, sizeof(s_probs));
        s_stats.inferences++;
        return;
    }

    /* 组装时序窗（旧→新，与训练窗口方向一致）*/
    float win[N_WINDOW];
    for (int i = 0; i < N_WINDOW; i++)
        win[i] = s_ring[(s_ring_idx + i) % N_WINDOW];

    /* RMS 预筛: 活动度低于阈值判静止，不推理 */
    features_compute(win, s_feat_raw);
    if (s_feat_raw[1] < STD_IDLE_TH) {
        s_probs[0] = 1.0f; s_probs[1] = s_probs[2] = s_probs[3] = 0.0f;
        memcpy(s_stats.last_probs, s_probs, sizeof(s_probs));   /* 静止窗口也刷新概率快照 */
        s_cand_cnt = 0; s_cand_cls = -1;
        s_stats.inferences++;
        return;
    }

    /* 网络双输入 */
    for (int i = 0; i < N_WINDOW; i++)
        s_net_sig[i] = (win[i] - POS_MEAN[i]) / POS_STD[i];
    for (int j = 0; j < N_FEAT; j++)
        s_net_feat[j] = (s_feat_raw[j] - FEAT_MEAN[j]) / FEAT_STD[j];

    memcpy(s_p_in0, s_net_sig,  N_WINDOW * 4);
    memcpy(s_p_in1, s_net_feat, N_FEAT  * 4);
    if (ai_mnetwork_run(s_net, s_report.inputs, s_report.outputs) <= 0) {
        /* 推理失败: 保持上一状态，计数排查 */
        return;
    }
    memcpy(s_probs, s_p_out, 4 * 4);
    s_stats.inferences++;
    memcpy(s_stats.last_probs, s_probs, sizeof(s_probs));
    alarm_update(now);
}

/* 暂停恢复保护: 清空报警候选窗口计数, 防止暂停期间的静止旧数据凑齐连续一致而假报警 */
void pipeline_resume_reset(void)
{
    s_cand_cnt = 0;
    s_cand_cls = -1;
}

/* 最近窗口活动电平: 去直流 RMS (m/s^2) —— 心跳帧上报, B 板活动条显示 */
float pipeline_activity(void)
{
    float win[N_WINDOW];
    float m = 0.0f, acc = 0.0f;
    for (int i = 0; i < N_WINDOW; i++) win[i] = s_ring[(s_ring_idx + i) % N_WINDOW];
    for (int i = 0; i < N_WINDOW; i++) m += win[i];
    m /= N_WINDOW;
    for (int i = 0; i < N_WINDOW; i++) { float d = win[i] - m; acc += d * d; }
    return sqrtf(acc / N_WINDOW);
}

/* KEY2 清空: 复位锁存事件/报警计数/候选窗口(与 w25_erase_records 配套) */
void pipeline_alarm_clear(void)
{
    s_last_alarm.cls = 0;
    s_last_alarm.conf = 0.0f;
    s_stats.alarms = 0;
    s_cand_cnt = 0;
    s_cand_cls = -1;
}

alarm_event_t pipeline_last_alarm(void) { return s_last_alarm; }
pipeline_stats_t pipeline_stats(void)   { return s_stats; }

/* 诊断读出: 最近窗 std(特征1) / 报警候选类别(-1=无) / 候选连续计数(遥测自检用) */
float pipeline_window_std(void) { return s_feat_raw[1]; }
int  pipeline_cand_cls(void)    { return s_cand_cls; }
int  pipeline_cand_cnt(void)    { return s_cand_cnt; }

/* 最近窗 8 频段能量(事件帧频谱字段): 委托显示专用去直流频谱(features_fw) */
void pipeline_band_e8(uint8_t out[8])
{
    features_bands_disp8(out);
}
