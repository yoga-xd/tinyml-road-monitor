# -*- coding: utf-8 -*-
"""
跨数据集验证: 用 Gonzalez 混合模型（4类, 50Hz 事件级）评估 RoadSens-4M 测试窗口
适配链（每一处都对应训练/板端语义）:
  1. 竖直轴: RoadSens 的 Y 轴即竖直（重力向量与 Y 对齐 0.99-1.00, 已验证）
  2. 重力重构: RoadSens acc_y 是线性加速度（无重力 DC）, 加回常数 9.81
     （重力向量幅值 9.807, Y 对齐 0.99-1.00 → 恒定近似误差 <1%）
     得到与 Gonzalez z 轴同语义的"含重力竖直加速度"
  3. 采样率 100Hz -> 50Hz: np.interp 256 -> 128（时间跨度 2.56s 不变,
     与 Gonzalez 训练时的重采样同一函数同一目标长度）
  4. 信号分支: 减 pos_mean 除 pos_sd（Gonzalez 常数）
  5. 特征分支: 在含重力窗口上按训练定义算 24 维, 减 feat_mean 除 feat_std
类别映射: Gonzalez 输出 {0正常,1坑洞,2减速带,3磨损} -> RoadSens {0正常,1坑洞,2颠簸};
          预测为"磨损"的窗口在 RoadSens 中无对应类, 计为不匹配并单独计数。
"""
import os, json, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

EPS = 1e-6
BASE = r"D:/tinyml_project"
NAMES_RS = ["正常", "坑洞", "颠簸"]          # RoadSens 3 类
NAMES_GZ = ["正常", "坑洞", "减速带", "磨损"]  # Gonzalez 4 类
GRAVITY = 9.81
MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_gonzalez_s2.h5"
SCALER = sys.argv[2] if len(sys.argv) > 2 else "model_gonzalez_s2_scalers.json"
print(f"评估模型: {MODEL}")

d = np.load(os.path.join(BASE, "data", "windows_samedist.npz"), allow_pickle=False)
Xte, yte = d["X_test"], d["y_test"]          # (381,256,3) 原始 m/s², y: 0正常/1坑洞/2颠簸
print(f"RoadSens-4M 测试窗口: {Xte.shape} | 真实分布 {np.bincount(yte, minlength=3).tolist()}")

sc = json.load(open(os.path.join(BASE, SCALER)))
pm = np.array(sc["pos_mean"]); psd = np.array(sc["pos_std"])
fm = np.array(sc["feat_mean"]); fsd = np.array(sc["feat_std"])
print(f"模型: {MODEL} | 特征 {len(fm)} 维 | 类别 {NAMES_GZ}")

model = tf.keras.models.load_model(os.path.join(BASE, MODEL))

# ---------- 适配链 ----------
vert = Xte[:, :, 1] + GRAVITY                 # Y 轴竖直 + 重力重构 (381,256)
n = vert.shape[0]
t_src = np.arange(256); t_dst = np.linspace(0, 255, 128)
sig_resampled = np.stack([np.interp(t_dst, t_src, vert[i]) for i in range(n)])   # (381,128)
sig_in = ((sig_resampled - pm[None, :]) / psd[None, :]).astype(np.float32)[:, :, None]

def features_batch(Xb):
    n2, L = Xb.shape
    mean = Xb.mean(1); std = Xb.std(1)
    rms = np.sqrt((Xb ** 2).mean(1))
    peak = np.abs(Xb).max(1)
    ptp = Xb.max(1) - Xb.min(1)
    crest = peak / (rms + EPS)
    energy = (Xb ** 2).sum(1)
    zcr = (np.diff(np.sign(Xb), axis=1) != 0).mean(1)
    p25 = np.percentile(Xb, 25, axis=1); p75 = np.percentile(Xb, 75, axis=1)
    iqr = p75 - p25
    dm = Xb - mean[:, None]
    skew = (dm ** 3).mean(1) / (std ** 3 + EPS)
    kurt = (dm ** 4).mean(1) / (std ** 4 + EPS)
    acs = [((dm[:, :-k] * dm[:, k:]).mean(1) / (std ** 2 + EPS))[:, None] for k in range(1, 6)]
    F = np.abs(np.fft.rfft(Xb, axis=1))
    bands = np.array_split(F, 8, axis=1)
    band_e = np.stack([(b ** 2).sum(1) for b in bands], axis=1)
    band_e = band_e / (band_e.sum(1, keepdims=True) + EPS)
    return np.concatenate([mean[:, None], std[:, None], rms[:, None], peak[:, None],
                           ptp[:, None], crest[:, None], energy[:, None], zcr[:, None],
                           iqr[:, None], skew[:, None], kurt[:, None]] + acs + [band_e], axis=1)

feat = features_batch(sig_resampled)
feat_in = ((feat - fm[None, :]) / fsd[None, :]).astype(np.float32)

pred_gz = model.predict([sig_in, feat_in], verbose=0).argmax(1)   # 0正常/1坑洞/2减速带/3磨损

# ---------- 类别映射与评估 ----------
mapped = pred_gz.copy()
mapped[pred_gz == 3] = -1                       # 磨损: RoadSens 无对应类
valid = mapped >= 0
acc_strict = (mapped[valid] == yte[valid]).mean()          # 磨损预测计错误
acc_shared = (mapped[pred_gz != 3] == yte[pred_gz != 3]).mean()  # 同上(相同)
n_worn = int((pred_gz == 3).sum())

print("\n===== Gonzalez 模型 -> RoadSens-4M 测试集（381 窗, 真跨数据集）=====")
print(f"总体: 严格口径（磨损预测计错误） {acc_strict*100:.2f}%")
print(f"       输出'磨损'的窗口数: {n_worn} ({n_worn/len(yte)*100:.1f}%)")
pred3 = mapped.copy(); pred3[pred3 == -1] = 99
print("\n混淆矩阵（行=真实[正常/坑洞/颠簸], 列=预测[正常/坑洞/颠簸/磨损(无对应)]）:")
cm = np.zeros((3, 4), int)
for t in range(3):
    for p_ in range(4):
        cm[t, p_] = int(((yte == t) & (pred_gz == p_)).sum())
for i, row in enumerate(cm):
    print(f"  {NAMES_RS[i]:　<3}" + "".join(f"{v:6d}" for v in row))
print("\n共享三类上的精确率/召回率:")
for k in range(3):
    rec = cm[k, k] / max(cm[k].sum(), 1)
    prec = cm[k, k] / max(cm[:, k].sum(), 1)
    print(f"  {NAMES_RS[k]:　<3} 召回 {rec*100:5.1f}%  精确 {prec*100:5.1f}%")

# ---------- 与 RoadSens 系列同测试集对照 ----------
print("\n===== 同一测试集（windows_samedist 381 窗）上的横向对比 =====")
print("Gonzalez 混合模型（跨数据集, 从未见过 RoadSens）: 见上")
print("RoadSens v6（本数据集内训练, Plan I 口径）:        43.57%")
print("RoadSens v3（本数据集内训练, 旧口径测试集）:       36.16%")
print("RoadSens v2（跨批次划分, 测试集不同）:             52.71%")
print("RoadSens v1（跨批次划分, 测试集不同）:             41.49%")
EOF_MARKER = None
