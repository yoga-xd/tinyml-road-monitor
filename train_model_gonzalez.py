# -*- coding: utf-8 -*-
"""
Gonzalez 2017 四分类训练 —— 事件级协议, 双口径评估
数据: data/gonzalez.npz（prepare_data_gonzalez.py 产出, 500 事件, 128 点, 逐窗标准化）
模型: Conv1D(16,5) -> MaxPool(2) -> Conv1D(32,3) -> GAP -> Dropout(0.3) -> Dense(4)
      约 1800 参数, float32 权重约 7 KB（板上部署无压力）
口径:
  A. 十折分层交叉验证（文献/仓库作者主口径）: 每折独立训练+测试, 报告平均±标准差
  B. 70/15/15 固定留出: 训练最终部署模型, 报告测试集明细
基准对照（仓库 outputs 自带, 同一份数据）:
  XGBoost 测试 94% / 十折 CV 87.5%; SVM 94% / 89.25%
增强（仅训练集）: 幅值缩放 0.7-1.4 | 高斯噪声 σ=0.1 | 随机时间平移 ±16 点
运行: C:/miniconda3/envs/tinyml/python.exe D:/tinyml_project/train_model_gonzalez.py
"""
import os, json, time, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

BASE = r"D:/tinyml_project"
EPS = 1e-6
AUG_SCALE = (0.7, 1.4)
AUG_SHIFT = 16
AUG_NOISE = 0.10

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

def build_model(n_classes=4):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 1)),
        tf.keras.layers.Conv1D(16, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="road_event_cnn_gonzalez")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

class AugSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, class_weight, batch_size, seed):
        self.X, self.y = X, y
        self.cw = class_weight
        self.bs = batch_size
        self.idx = np.arange(len(y))
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))
    def on_epoch_end(self):
        self.rng.shuffle(self.idx)
    def __getitem__(self, i):
        sl = self.idx[i * self.bs:(i + 1) * self.bs]
        Xb = self.X[sl].copy()
        yb = self.y[sl]
        n = len(sl)
        scale = self.rng.uniform(AUG_SCALE[0], AUG_SCALE[1], size=(n, 1))
        Xb = (Xb * scale.astype(np.float32)).reshape(n, 128, 1)
        shifts = self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, size=n)
        for j in range(n):
            Xb[j] = np.roll(Xb[j], shifts[j], axis=0)
        Xb += self.rng.normal(0, AUG_NOISE, size=Xb.shape).astype(np.float32)
        w = np.array([self.cw[c] for c in yb], np.float32)
        return Xb, yb, w

def train_one(Xtr, ytr, Xva, yva, tag, n_classes=4):
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    cw = {i: float(len(ytr) / (n_classes * counts[i])) for i in range(n_classes)}
    seq = AugSequence(Xtr.reshape(-1, 128), ytr, cw, batch_size=16, seed=SEED)
    m = build_model(n_classes)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                           restore_best_weights=True, verbose=0)]
    m.fit(seq, validation_data=(Xva.reshape(-1, 128, 1), yva),
          epochs=100, callbacks=cb, verbose=0)
    acc = accuracy_score(yva, m.predict(Xva.reshape(-1, 128, 1), verbose=0).argmax(1))
    return m, acc, cw

# ---------- 数据 ----------
d = np.load(os.path.join(BASE, "data", "gonzalez.npz"), allow_pickle=False)
meta = json.loads(str(d["meta"]))
NAMES = meta["classes"]
fold_ids = d["fold_ids"]
X_all, y_all = d["X_all"], d["y_all"]
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
Xte, yte = d["X_test"], d["y_test"]

log("=" * 62)
log("Gonzalez 四分类训练（事件级协议, 128 点, 逐窗标准化, seed=%d）" % SEED)
log(f"数据: 共 {len(y_all)} 事件 | 留出 训{len(ytr)}/验{len(yva)}/测{len(yte)} | 十折 CV")
n_params = build_model().count_params()
log(f"参数量: {n_params} | float32 权重约 {n_params*4/1024:.1f} KB")

# ========== 口径 A: 十折分层交叉验证 ==========
log("-" * 62)
log("[口径 A] 十折分层交叉验证")
fold_accs, fold_cms = [], []
t0 = time.time()
for k in range(meta["n_folds"]):
    te = fold_ids == k
    tr = ~te
    X_tr_f, y_tr_f = X_all[tr], y_all[tr]
    itr_i, iva_i = train_test_split(
        np.arange(len(y_tr_f)), test_size=0.15, random_state=SEED + k, stratify=y_tr_f)
    X_tr_i, y_tr_i = X_tr_f[itr_i], y_tr_f[itr_i]
    X_va_i, y_va_i = X_tr_f[iva_i], y_tr_f[iva_i]
    m, va_acc, _ = train_one(X_tr_i, y_tr_i, X_va_i, y_va_i, tag=f"fold{k+1}")
    pred = m.predict(X_all[te].reshape(-1, 128, 1), verbose=0).argmax(1)
    acc = accuracy_score(y_all[te], pred)
    cm = confusion_matrix(y_all[te], pred, labels=[0, 1, 2, 3])
    fold_accs.append(acc); fold_cms.append(cm)
    log(f"  fold {k+1:2d}: 测试准确率 {acc*100:5.1f}%  (折内验证 {va_acc*100:.1f}%)")
fold_accs = np.array(fold_accs)
cm_total = np.sum(fold_cms, axis=0)
log(f"十折 CV 结果: 平均 {fold_accs.mean()*100:.2f}% ± {fold_accs.std()*100:.2f}%  (耗时 {time.time()-t0:.0f}s)")
log("十折累计混淆矩阵（行=真实, 列=预测, 顺序 " + "/".join(NAMES) + "）:")
for i, row in enumerate(cm_total):
    log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))
log("基准对照（仓库 outputs, 同一数据）: XGBoost CV 87.50%±4.47 | SVM CV 89.25%±5.37")

# ========== 口径 B: 70/15/15 固定留出（最终部署模型） ==========
log("-" * 62)
log("[口径 B] 固定留出 70/15/15")
m_final, va_acc, cw = train_one(Xtr, ytr, Xva, yva, tag="final")
log(f"类别权重: " + ", ".join(f"{NAMES[i]}: {cw[i]:.2f}" for i in range(4)))
log(f"留出验证准确率 {va_acc*100:.2f}%")
pred = m_final.predict(Xte.reshape(-1, 128, 1), verbose=0).argmax(1)
acc_te = accuracy_score(yte, pred)
log(classification_report(yte, pred, target_names=NAMES, digits=4))
cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
log("测试混淆矩阵（行=真实, 列=预测）:")
for i, row in enumerate(cm):
    log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))
log(f"基准对照（仓库 outputs, 同一数据 20% 留出）: XGBoost 94% | SVM 94% | RandomForest 90% | MLP 90%")

# ---------- 导出 ----------
m_final.save(os.path.join(BASE, "model_gonzalez.h5"))
with open(os.path.join(BASE, "norm_constants_gonzalez.json"), "w", encoding="utf-8") as f:
    json.dump({
        "note": "逐事件 z-score 标准化: 板端对每个 128 点窗口(2.56s@50Hz)求均值/方差后标准化, 无需固化常数",
        "normalization": "per-window per-channel z-score",
        "unit": "acc_z m/s^2 (含重力直流, 标准化后去除)",
        "sample_rate_hz": 50, "window_points": 128, "window_seconds": 2.56,
        "classes": NAMES, "n_params": int(n_params), "seed": SEED,
        "cv_accuracy_mean": float(fold_accs.mean()), "cv_accuracy_std": float(fold_accs.std()),
        "holdout_test_accuracy": float(acc_te),
        "baseline_repo": {"XGBoost_holdout": 0.94, "XGBoost_cv": 0.875, "SVM_holdout": 0.94},
    }, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_total, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(NAMES)
ax.set_yticks(range(4)); ax.set_yticklabels(NAMES)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm_total[i, j], ha="center", va="center",
                color="white" if cm_total[i, j] > cm_total.max()/2 else "black")
ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"十折累计混淆矩阵 (CV {fold_accs.mean()*100:.1f}%±{fold_accs.std()*100:.1f}%)")
fig.colorbar(im); fig.tight_layout()
fig.savefig(os.path.join(BASE, "gonzalez_cv_confusion.png"), dpi=150)

log("=" * 62)
log(f"总结: 十折 CV {fold_accs.mean()*100:.2f}%±{fold_accs.std()*100:.2f}% | 留出测试 {acc_te*100:.2f}%")
log("验收参考线: 90%（文献/仓库基准口径）")

with open(os.path.join(BASE, "train_log_gonzalez.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")

log("交付物:")
for f in ["model_gonzalez.h5", "norm_constants_gonzalez.json",
          "gonzalez_cv_confusion.png", "train_log_gonzalez.txt"]:
    p = os.path.join(BASE, f)
    log(f"  {p}  ({os.path.getsize(p)/1024:.1f} KB)" if os.path.exists(p) else f"  {p}  (未生成)")
