# -*- coding: utf-8 -*-
"""
Gonzalez 实验脚本: 阶段1=全局逐位置标准化CNN; 阶段2=混合特征分支（阶段1不达标时自动启用）
用法:
  python gonzalez_experiments.py smoke   # 冒烟: 1折x2轮
  python gonzalez_experiments.py full    # 全量: 十折CV + 留出
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

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

d = np.load(os.path.join(BASE, "data", "gonzalez.npz"), allow_pickle=False)
X_raw, y_all, fold_ids = d["X_all"], d["y_all"], d["fold_ids"]
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
Xte, yte = d["X_test"], d["y_test"]

# ---------- 特征工程（板上可廉价计算的时域+频段特征, 在原始幅值上计算） ----------
def features_batch(Xb):
    """Xb: (n,128) 原始 m/s² -> (n,F) 特征矩阵"""
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
    acs = [ ((dm[:, :-k] * dm[:, k:]).mean(1) / (std ** 2 + EPS))[:, None] for k in range(1, 6) ]  # 自相关滞后 1-5
    F = np.abs(np.fft.rfft(Xb, axis=1))
    bands = np.array_split(F, 8, axis=1)
    band_e = np.stack([(b ** 2).sum(1) for b in bands], axis=1)
    band_e = band_e / (band_e.sum(1, keepdims=True) + EPS)
    return np.concatenate([mean[:, None], std[:, None], rms[:, None], peak[:, None],
                           ptp[:, None], crest[:, None], energy[:, None], zcr[:, None],
                           iqr[:, None], skew[:, None], kurt[:, None]] + acs + [band_e], axis=1)

N_FEAT = features_batch(X_raw[:2]).shape[1]
log(f"特征维数: {N_FEAT}")

class AugSeq(tf.keras.utils.Sequence):
    """两阶段通用: 输出 (归一化信号, 缩放后特征) 双输入; stage1 时 FEATS=False"""
    def __init__(self, Xr, y, cw, pos_mu, pos_sd, feat_mu, feat_sd, use_feats, batch_size, seed):
        self.X, self.y = Xr, y
        self.cw = cw
        self.pm, self.psd = pos_mu, pos_sd
        self.fm, self.fsd = feat_mu, feat_sd
        self.use_feats = use_feats
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
        Xb = self.X[sl].copy()                       # 原始幅值
        yb = self.y[sl]
        n = len(sl)
        scale = self.rng.uniform(AUG_SCALE[0], AUG_SCALE[1], size=(n, 1))
        Xb = Xb * scale.astype(np.float32)           # 幅值缩放（增强安装差异鲁棒性）
        shifts = self.rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, size=n)
        for j in range(n):
            Xb[j] = np.roll(Xb[j], shifts[j], axis=0)
        sig = ((Xb - self.pm[None, :]) / self.psd[None, :]).astype(np.float32)[:, :, None]
        if self.use_feats:
            ft = features_batch(Xb)
            ft = ((ft - self.fm[None, :]) / self.fsd[None, :]).astype(np.float32)
            x = [sig, ft]
        else:
            x = sig
        Xb_n = Xb + self.rng.normal(0, AUG_NOISE * self.X.std(), size=Xb.shape).astype(np.float32)
        sig_aug = ((Xb_n - self.pm[None, :]) / self.psd[None, :]).astype(np.float32)[:, :, None]
        if self.use_feats:
            ft_aug = features_batch(Xb_n)
            ft_aug = ((ft_aug - self.fm[None, :]) / self.fsd[None, :]).astype(np.float32)
            x = [sig_aug, ft_aug]
        else:
            x = sig_aug
        w = np.array([self.cw[c] for c in yb], np.float32)
        return x, yb, w

def fit_scalers(Xr, stage):
    pm, psd = Xr.mean(0), Xr.std(0) + EPS
    fm, fsd = None, None
    if stage == 2:
        F = features_batch(Xr)
        fm, fsd = F.mean(0), F.std(0) + EPS
    return pm, psd, fm, fsd

def build_model(stage, n_classes=4):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    sig_in = tf.keras.layers.Input(shape=(128, 1))
    h = tf.keras.layers.Conv1D(16, 5, activation="relu")(sig_in)
    h = tf.keras.layers.MaxPooling1D(2)(h)
    h = tf.keras.layers.Conv1D(32, 3, activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    if stage == 2:
        feat_in = tf.keras.layers.Input(shape=(N_FEAT,))
        f = tf.keras.layers.Dense(64, activation="relu")(feat_in)
        f = tf.keras.layers.Dropout(0.2)(f)
        h = tf.keras.layers.Concatenate()([h, f])
        inputs = [sig_in, feat_in]
    else:
        inputs = sig_in
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(h)
    m = tf.keras.Model(inputs, out, name=f"gonzalez_stage{stage}")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def run_fold(Xr_tr, y_tr, Xr_va, y_va, Xr_te, y_te, stage, tag, save_path=None):
    pm, psd, fm, fsd = fit_scalers(Xr_tr, stage)
    counts = np.bincount(y_tr, minlength=4).astype(np.float64)
    cw = {i: float(len(y_tr) / (4 * counts[i])) for i in range(4)}
    seq = AugSeq(Xr_tr, y_tr, cw, pm, psd, fm, fsd, stage == 2, batch_size=16, seed=SEED)
    m = build_model(stage)
    def as_model_input(Xr):
        sig = ((Xr - pm[None, :]) / psd[None, :]).astype(np.float32)[:, :, None]
        if stage == 2:
            ft = features_batch(Xr)
            ft = ((ft - fm[None, :]) / fsd[None, :]).astype(np.float32)
            return [sig, ft]
        return sig
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                           restore_best_weights=True, verbose=0)]
    m.fit(seq, validation_data=(as_model_input(Xr_va), y_va),
          epochs=EPOCHS, callbacks=cb, verbose=0)
    pred = m.predict(as_model_input(Xr_te), verbose=0).argmax(1)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2, 3])
    if save_path:
        m.save(save_path)
        json.dump({"pos_mean": pm.tolist(), "pos_std": psd.tolist(),
                   "feat_mean": (fm.tolist() if fm is not None else None),
                   "feat_std": (fsd.tolist() if fsd is not None else None),
                   "stage": stage, "classes": NAMES, "n_params": int(m.count_params()),
                   "sample_rate_hz": 50, "window_points": 128, "seed": SEED},
                  open(save_path.replace(".h5", "_scalers.json"), "w"), indent=2)
    return m, acc, cm, cw

log("=" * 62)
log(f"Gonzalez 实验 ({MODE}): 阶段1=全局标准化CNN -> 不达标自动升级阶段2混合特征")
log(f"特征维数 {N_FEAT} | 折数 {N_FOLDS_RUN} | 轮数上限 {EPOCHS}")

results = {}
for stage in (1, 2):
    if stage == 2 and results.get(1, 0) >= 0.88:
        log(f"阶段1 CV 已达 {results[1]*100:.2f}% >= 88%, 跳过阶段2")
        break
    log("-" * 62)
    log(f"[阶段 {stage}] 十折分层交叉验证")
    accs, cms = [], []
    t0 = time.time()
    for k in range(N_FOLDS_RUN):
        te = fold_ids == k
        tr = ~te
        itr_i, iva_i = train_test_split(np.arange(tr.sum()), test_size=0.15,
                                        random_state=SEED + k, stratify=y_all[tr])
        tr_idx = np.where(tr)[0]
        Xr_tr_i, y_tr_i = X_raw[tr_idx[itr_i]], y_all[tr_idx[itr_i]]
        Xr_va_i, y_va_i = X_raw[tr_idx[iva_i]], y_all[tr_idx[iva_i]]
        m, acc, cm, _ = run_fold(Xr_tr_i, y_tr_i, Xr_va_i, y_va_i,
                                 X_raw[te], y_all[te], stage, f"fold{k+1}")
        accs.append(acc); cms.append(cm)
        log(f"  fold {k+1:2d}: 测试准确率 {acc*100:5.1f}%")
    accs = np.array(accs); cm_total = np.sum(cms, axis=0)
    cv_mean = float(accs.mean())
    results[stage] = cv_mean
    log(f"[阶段 {stage}] 十折 CV: {cv_mean*100:.2f}% ± {accs.std()*100:.2f}%  (耗时 {time.time()-t0:.0f}s)"
        + ("" if N_FOLDS_RUN > 1 else "  [冒烟, 折数=1]"))
    log("累计混淆矩阵（行=真实, 列=预测, 顺序 " + "/".join(NAMES) + "）:")
    for i, row in enumerate(cm_total):
        log(f"  {NAMES[i]:　<3}" + "".join(f"{v:5d}" for v in row))
    if MODE == "full":
        log("-" * 62)
        log(f"[阶段 {stage}] 固定留出 70/15/15（最终部署模型）")
        m_fin, acc_te, cm_h, cw = run_fold(Xtr, ytr, Xva, yva, Xte, yte, stage, "final",
                                           save_path=os.path.join(BASE, f"model_gonzalez_s{stage}.h5"))
        log(f"留出测试准确率 {acc_te*100:.2f}% | 类别权重 " +
            ", ".join(f"{NAMES[i]}:{cw[i]:.2f}" for i in range(4)))
        log(classification_report(yte, m_fin.predict(
            ([((Xte - Xtr.mean(0)[None, :]) / (Xtr.std(0)[None, :] + EPS)).astype(np.float32)[:, :, None],
              ((features_batch(Xte) - features_batch(Xtr).mean(0)[None, :]) / (features_batch(Xtr).std(0)[None, :] + EPS)).astype(np.float32)]
             if stage == 2 else
             ((Xte - Xtr.mean(0)[None, :]) / (Xtr.std(0)[None, :] + EPS)).astype(np.float32)[:, :, None]),
            verbose=0).argmax(1), target_names=NAMES, digits=4, zero_division=0))

log("基准对照: 仓库 XGBoost 留出 94%/CV 87.5% | SVM 94%/89.25%")
with open(os.path.join(BASE, "experiments_log_gonzalez.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
print("DONE")
