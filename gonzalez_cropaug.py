# -*- coding: utf-8 -*-
"""
Gonzalez 子窗裁剪增强实验（对齐板上滑窗语义）
增强: 训练时 50% 概率把事件合成为"部分可见"形态——随机裁剪 48-128 点
      嵌入类平整路填充基底（均值/方差取自训练集平整路的稳健统计）, 标签不变。
对照: 基线(无裁剪) vs 裁剪增强, 两种测试集:
      (i) 完整事件测试集(75条, 与此前口径一致)
      (ii) 板上形态部分可见窗测试集(每条事件合成 2 窗, 种子固定, 共 150 窗)
模型: 24 维混合 CNN（当前定版架构, lr 5e-4 / dropout 0.3 / feat_dense 64）
用法: python gonzalez_cropaug.py smoke | full
"""
import os, sys, json, time, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE = r"D:/tinyml_project"
MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
N_FOLDS_RUN = 1 if MODE == "smoke" else 10
EPOCHS = 2 if MODE == "smoke" else 100
AUG_SCALE, AUG_SHIFT, AUG_NOISE = (0.7, 1.4), 16, 0.10
EPS = 1e-6
NAMES = ["正常", "坑洞", "减速带", "磨损"]
CROP_PROB = 0.5
CROP_MIN = 48      # 部分可见窗内事件最少 48 点(~0.96s)

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
# 填充基底: 训练集平整路(类0)的稳健统计——类"平滑路"的典型形态
fill_mu = float(np.median(Xtr[ytr == 0].mean(1)))
fill_sd = float(np.median(Xtr[ytr == 0].std(1)))
log(f"特征 {N_FEAT} 维 | 填充基底: 均值 {fill_mu:.2f}, 标准差 {fill_sd:.2f} (训练集平整路统计)")

class AugSeq(tf.keras.utils.Sequence):
    def __init__(self, Xr, y, cw, pm, psd, fm, fsd, use_crop, batch_size, seed):
        self.X, self.y, self.cw = Xr, y, cw
        self.pm, self.psd, self.fm, self.fsd = pm, psd, fm, fsd
        self.use_crop = use_crop
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
        if self.use_crop:
            for j in range(n):
                if self.rng.random() < CROP_PROB:
                    Lvis = int(self.rng.integers(CROP_MIN, 129))       # 可见事件长度
                    s = int(self.rng.integers(0, 128 - Lvis + 1))      # 事件在窗内起点
                    c = int(self.rng.integers(0, 128 - Lvis + 1))      # 事件内部裁剪起点
                    w = self.fill_mu + self.rng.normal(0, self.fill_sd, 128)
                    w[s:s + Lvis] = Xb[j, c:c + Lvis]
                    Xb[j] = w
        scale = self.rng.uniform(AUG_SCALE[0], AUG_SCALE[1], size=(n, 1))
        Xb = Xb * scale.astype(np.float32)
        shifts = self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, size=n)
        for j in range(n):
            Xb[j] = np.roll(Xb[j], shifts[j], axis=0)
        Xb = Xb + self.rng.normal(0, AUG_NOISE * self.X.std(), size=Xb.shape).astype(np.float32)
        sig = ((Xb - self.pm[None, :]) / self.psd[None, :]).astype(np.float32)[:, :, None]
        ft = features_batch(Xb)
        ft = ((ft - self.fm[None, :]) / self.fsd[None, :]).astype(np.float32)
        w = np.array([self.cw[c] for c in yb], np.float32)
        return [sig, ft], yb, w

def build_model(n_classes=4):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    sig_in = tf.keras.layers.Input(shape=(128, 1))
    h = tf.keras.layers.Conv1D(16, 5, activation="relu")(sig_in)
    h = tf.keras.layers.MaxPooling1D(2)(h)
    h = tf.keras.layers.Conv1D(32, 3, activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    feat_in = tf.keras.layers.Input(shape=(N_FEAT,))
    f = tf.keras.layers.Dense(64, activation="relu")(feat_in)
    f = tf.keras.layers.Dropout(0.3)(f)
    h = tf.keras.layers.Concatenate()([h, f])
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(h)
    m = tf.keras.Model([sig_in, feat_in], out, name="gonzalez_hybrid")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def as_mi(Xr, pm, psd, fm, fsd):
    sig = ((Xr - pm[None, :]) / psd[None, :]).astype(np.float32)[:, :, None]
    ft = ((features_batch(Xr) - fm[None, :]) / fsd[None, :]).astype(np.float32)
    return [sig, ft]

def train_one(Xr_tr, y_tr, Xr_va, y_va, use_crop):
    pm, psd = Xr_tr.mean(0), Xr_tr.std(0) + EPS
    Ftr = features_batch(Xr_tr)
    fm, fsd = Ftr.mean(0), Ftr.std(0) + EPS
    counts = np.bincount(y_tr, minlength=4).astype(np.float64)
    cw = {i: float(len(y_tr) / (4 * counts[i])) for i in range(4)}
    seq = AugSeq(Xr_tr, y_tr, cw, pm, psd, fm, fsd, use_crop, batch_size=16, seed=SEED)
    seq.fill_mu, seq.fill_sd = fill_mu, fill_sd
    m = build_model()
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                           restore_best_weights=True, verbose=0)]
    m.fit(seq, validation_data=(as_mi(Xr_va, pm, psd, fm, fsd), y_va),
          epochs=EPOCHS, callbacks=cb, verbose=0)
    return m, cw, (pm, psd, fm, fsd)

def eval_model(m, Xr, y, sc, tag):
    pred = m.predict(as_mi(Xr, *sc), verbose=0).argmax(1)
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    return acc, cm, pred

# ---------- 板上形态部分可见测试集（固定种子合成） ----------
rng_t = np.random.default_rng(SEED + 999)
Xp, yp = [], []
for j in range(len(yte)):
    for _ in range(2):   # 每条测试事件合成 2 个部分可见窗
        Lvis = int(rng_t.integers(CROP_MIN, 129))
        s = int(rng_t.integers(0, 128 - Lvis + 1))
        c = int(rng_t.integers(0, 128 - Lvis + 1))
        w = fill_mu + rng_t.normal(0, fill_sd, 128)
        w[s:s + Lvis] = Xte[j, c:c + Lvis]
        Xp.append(w); yp.append(yte[j])
Xp = np.array(Xp); yp = np.array(yp)
log(f"板上形态测试集: {Xp.shape[0]} 窗（75 事件 × 2, 部分可见 48-128 点嵌入平整路基底）")

log("=" * 62)
log(f"子窗裁剪增强实验 ({MODE}) | 裁剪概率 {CROP_PROB}, 可见长度 {CROP_MIN}-128 点")

# ---------- 基线（无裁剪）与裁剪增强, 各训一个留出模型 ----------
outs = {}
for name, use_crop in [("基线(无裁剪)", False), ("裁剪增强", True)]:
    t0 = time.time()
    m, cw, sc = train_one(Xtr, ytr, Xva, yva, use_crop)
    acc_full, cm_full, _ = eval_model(m, Xte, yte, sc, name)
    acc_part, cm_part, _ = eval_model(m, Xp, yp, sc, name)
    outs[name] = (m, sc, acc_full, acc_part)
    log("-" * 62)
    log(f"[{name}] 训练 {time.time()-t0:.0f}s")
    log(f"  完整事件测试集: {acc_full*100:.2f}%")
    for i, row in enumerate(cm_full):
        log(f"    {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))
    log(f"  板上形态部分可见窗: {acc_part*100:.2f}%")
    for i, row in enumerate(cm_part):
        log(f"    {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))

# ---------- 裁剪增强配置的十折 CV ----------
log("-" * 62)
if MODE == "full":
    log("[裁剪增强] 十折分层交叉验证")
    accs, cms = [], []
    for k in range(10):
        te = fold_ids == k
        tr = ~te
        tr_idx = np.where(tr)[0]
        itr_i, iva_i = train_test_split(np.arange(tr.sum()), test_size=0.15,
                                        random_state=SEED + k, stratify=y_all[tr])
        m, _, _ = train_one(X_raw[tr_idx[itr_i]], y_all[tr_idx[itr_i]],
                            X_raw[tr_idx[iva_i]], y_all[tr_idx[iva_i]], True)
        pred = m.predict(as_mi(X_raw[te],
                               X_raw[tr_idx[itr_i]].mean(0), X_raw[tr_idx[itr_i]].std(0) + EPS,
                               features_batch(X_raw[tr_idx[itr_i]]).mean(0),
                               features_batch(X_raw[tr_idx[itr_i]]).std(0) + EPS), verbose=0).argmax(1)
        accs.append(accuracy_score(y_all[te], pred))
        cms.append(confusion_matrix(y_all[te], pred, labels=[0, 1, 2, 3]))
    accs = np.array(accs)
    log(f"十折 CV: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%  (基线 83.40% ± 6.33%)")
    cm_total = np.sum(cms, axis=0)
    for i, row in enumerate(cm_total):
        log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))
    # 定版: 保存裁剪增强模型
    m_crop, _, sc_crop = train_one(Xtr, ytr, Xva, yva, True)
    m_crop.save(os.path.join(BASE, "model_gonzalez_crop.h5"))
    json.dump({"pos_mean": sc_crop[0].tolist(), "pos_std": sc_crop[1].tolist(),
               "feat_mean": sc_crop[2].tolist(), "feat_std": sc_crop[3].tolist(),
               "crop_prob": CROP_PROB, "crop_min": CROP_MIN,
               "filler_mean": fill_mu, "filler_std": fill_sd,
               "classes": NAMES, "n_features": int(N_FEAT), "sample_rate_hz": 50,
               "window_points": 128, "seed": SEED,
               "cv_mean": float(accs.mean()), "cv_std": float(accs.std())},
              open(os.path.join(BASE, "model_gonzalez_crop_scalers.json"), "w"), indent=2)
    log("已保存 model_gonzalez_crop.h5 + model_gonzalez_crop_scalers.json")

log("=" * 62)
log(f"总结: 完整事件测试 基线 {outs['基线(无裁剪)'][2]*100:.2f}% / 裁剪 {outs['裁剪增强'][2]*100:.2f}% | "
    f"板上形态测试 基线 {outs['基线(无裁剪)'][3]*100:.2f}% / 裁剪 {outs['裁剪增强'][3]*100:.2f}%")

with open(os.path.join(BASE, "cropaug_log_gonzalez.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
print("DONE")
