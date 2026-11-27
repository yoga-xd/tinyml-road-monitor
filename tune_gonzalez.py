# -*- coding: utf-8 -*-
"""Gonzalez 混合模型调参: 8 组超参 × 3 折快评 -> 最优配置 -> 十折 CV + 留出"""
import os, sys, json, time, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE = r"D:/tinyml_project"
EPS = 1e-6
AUG_SCALE, AUG_SHIFT, AUG_NOISE = (0.7, 1.4), 16, 0.10
NAMES = ["正常", "坑洞", "减速带", "磨损"]

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

d = np.load(os.path.join(BASE, "data", "gonzalez.npz"), allow_pickle=False)
X_raw, y_all, fold_ids = d["X_all"], d["y_all"], d["fold_ids"]
Xtr, ytr, Xva, yva, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]

def features_batch(Xb):
    n, L = Xb.shape
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

N_FEAT = features_batch(X_raw[:2]).shape[1]

class AugSeq(tf.keras.utils.Sequence):
    def __init__(self, Xr, y, cw, pm, psd, fm, fsd, batch_size, seed):
        self.X, self.y, self.cw = Xr, y, cw
        self.pm, self.psd, self.fm, self.fsd = pm, psd, fm, fsd
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
        Xb = Xb * scale.astype(np.float32)
        shifts = self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, size=n)
        for j in range(n):
            Xb[j] = np.roll(Xb[j], shifts[j], axis=0)
        sig = ((Xb - self.pm[None, :]) / self.psd[None, :]).astype(np.float32)[:, :, None]
        ft = features_batch(Xb)
        ft = ((ft - self.fm[None, :]) / self.fsd[None, :]).astype(np.float32)
        Xb_n = Xb + self.rng.normal(0, AUG_NOISE * self.X.std(), size=Xb.shape).astype(np.float32)
        sig_aug = ((Xb_n - self.pm[None, :]) / self.psd[None, :]).astype(np.float32)[:, :, None]
        ft_aug = features_batch(Xb_n)
        ft_aug = ((ft_aug - self.fm[None, :]) / self.fsd[None, :]).astype(np.float32)
        w = np.array([self.cw[c] for c in yb], np.float32)
        return [sig_aug, ft_aug], yb, w

def build_model(lr, dropout, feat_dense, n_classes=4):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    sig_in = tf.keras.layers.Input(shape=(128, 1))
    h = tf.keras.layers.Conv1D(16, 5, activation="relu")(sig_in)
    h = tf.keras.layers.MaxPooling1D(2)(h)
    h = tf.keras.layers.Conv1D(32, 3, activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    h = tf.keras.layers.Dropout(dropout)(h)
    feat_in = tf.keras.layers.Input(shape=(N_FEAT,))
    f = tf.keras.layers.Dense(feat_dense, activation="relu")(feat_in)
    f = tf.keras.layers.Dropout(dropout)(f)
    h = tf.keras.layers.Concatenate()([h, f])
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(h)
    m = tf.keras.Model([sig_in, feat_in], out, name="gonzalez_hybrid")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def as_mi(Xr, pm, psd, fm, fsd):
    sig = ((Xr - pm[None, :]) / psd[None, :]).astype(np.float32)[:, :, None]
    ft = ((features_batch(Xr) - fm[None, :]) / fsd[None, :]).astype(np.float32)
    return [sig, ft]

def train_one(Xr_tr, y_tr, Xr_va, y_va, cfg):
    lr, dropout, feat_dense = cfg
    pm, psd = Xr_tr.mean(0), Xr_tr.std(0) + EPS
    Ftr = features_batch(Xr_tr)
    fm, fsd = Ftr.mean(0), Ftr.std(0) + EPS
    counts = np.bincount(y_tr, minlength=4).astype(np.float64)
    cw = {i: float(len(y_tr) / (4 * counts[i])) for i in range(4)}
    seq = AugSeq(Xr_tr, y_tr, cw, pm, psd, fm, fsd, batch_size=16, seed=SEED)
    m = build_model(lr, dropout, feat_dense)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                           restore_best_weights=True, verbose=0)]
    m.fit(seq, validation_data=(as_mi(Xr_va, pm, psd, fm, fsd), y_va),
          epochs=100, callbacks=cb, verbose=0)
    va_acc = accuracy_score(y_va, m.predict(as_mi(Xr_va, pm, psd, fm, fsd), verbose=0).argmax(1))
    return m, va_acc, cw, (pm, psd, fm, fsd)

# ---------- 1. 网格快评（前三折） ----------
grid = [
    (5e-4, 0.3, 64),                # 当前配置
    (3e-4, 0.3, 64), (1e-3, 0.3, 64),
    (3e-4, 0.2, 64), (5e-4, 0.2, 64), (1e-3, 0.4, 64),
    (5e-4, 0.3, 128), (3e-4, 0.4, 32),
]
log(f"网格 {len(grid)} 组, 每组 3 折快评（折 0/1/2）| 特征 {N_FEAT} 维")
grid_res = []
t0 = time.time()
for gi, cfg in enumerate(grid):
    accs = []
    for k in range(3):
        te = fold_ids == k
        tr = ~te
        tr_idx = np.where(tr)[0]
        itr_i, iva_i = train_test_split(np.arange(tr.sum()), test_size=0.15,
                                        random_state=SEED + k, stratify=y_all[tr])
        m, va_acc, _, sc = train_one(X_raw[tr_idx[itr_i]], y_all[tr_idx[itr_i]],
                                     X_raw[tr_idx[iva_i]], y_all[tr_idx[iva_i]], cfg)
        pred = m.predict(as_mi(X_raw[te], *sc), verbose=0).argmax(1)
        accs.append(accuracy_score(y_all[te], pred))
    mean = float(np.mean(accs))
    grid_res.append((mean, cfg, accs))
    log(f"  组{gi+1} lr={cfg[0]:.0e} dropout={cfg[1]} feat_dense={cfg[2]}: "
        f"3折均值 {mean*100:.2f}%  各折 {[f'{a*100:.0f}' for a in accs]}")
log(f"网格快评耗时 {time.time()-t0:.0f}s")
best_mean, best_cfg, _ = max(grid_res, key=lambda r: r[0])
log(f"最优配置: lr={best_cfg[0]:.0e}, dropout={best_cfg[1]}, feat_dense={best_cfg[2]} (3折 {best_mean*100:.2f}%)")

# ---------- 2. 最优配置完整评估: 十折 CV + 留出 ----------
log("-" * 62)
log(f"[最优配置正式评估] lr={best_cfg[0]:.0e} dropout={best_cfg[1]} feat_dense={best_cfg[2]}")
accs, cms = [], []
for k in range(10):
    te = fold_ids == k
    tr = ~te
    tr_idx = np.where(tr)[0]
    itr_i, iva_i = train_test_split(np.arange(tr.sum()), test_size=0.15,
                                    random_state=SEED + k, stratify=y_all[tr])
    m, va_acc, _, sc = train_one(X_raw[tr_idx[itr_i]], y_all[tr_idx[itr_i]],
                                 X_raw[tr_idx[iva_i]], y_all[tr_idx[iva_i]], best_cfg)
    pred = m.predict(as_mi(X_raw[te], *sc), verbose=0).argmax(1)
    accs.append(accuracy_score(y_all[te], pred))
    cms.append(confusion_matrix(y_all[te], pred, labels=[0, 1, 2, 3]))
    log(f"  fold {k+1:2d}: 测试准确率 {accs[-1]*100:5.1f}%")
accs = np.array(accs); cm_total = np.sum(cms, axis=0)
log(f"十折 CV: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")
for i, row in enumerate(cm_total):
    log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))

log("-" * 62)
log("[最终部署模型] 70/15/15 留出")
m_final, va_acc, cw, (pm, psd, fm, fsd) = train_one(Xtr, ytr, Xva, yva, best_cfg)
pred = m_final.predict(as_mi(Xte, pm, psd, fm, fsd), verbose=0).argmax(1)
acc_te = accuracy_score(yte, pred)
log(f"留出验证 {va_acc*100:.2f}% | 测试 {acc_te*100:.2f}%")
log(classification_report(yte, pred, target_names=NAMES, digits=4, zero_division=0))
cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
for i, row in enumerate(cm):
    log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))

m_final.save(os.path.join(BASE, "model_gonzalez_tuned.h5"))
json.dump({"lr": best_cfg[0], "dropout": best_cfg[1], "feat_dense": best_cfg[2],
           "pos_mean": pm.tolist(), "pos_std": psd.tolist(),
           "feat_mean": fm.tolist(), "feat_std": fsd.tolist(),
           "classes": NAMES, "n_features": int(N_FEAT),
           "cv_mean": float(accs.mean()), "cv_std": float(accs.std()),
           "holdout_test": float(acc_te), "seed": SEED,
           "sample_rate_hz": 50, "window_points": 128},
          open(os.path.join(BASE, "model_gonzalez_tuned_scalers.json"), "w"), indent=2)
log("基准对照: 仓库 XGBoost 留出 94%/CV 87.5% | SVM 94%/89.25%")
with open(os.path.join(BASE, "tune_log_gonzalez.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
print("DONE")
