# -*- coding: utf-8 -*-
"""
Gonzalez 2017 数据集事件级预处理（车辆振动 TinyML, 四分类）
输入: D:/pothole-detection-and-severity-management-main/data/ 下 5 类 × 100 个 CSV
      （单列 acc_z, 50 Hz, 变长 59-378 点, 单位 m/s² 含重力直流）
输出: data/gonzalez.npz + data/prepare_gonzalez_log.txt

处理规则（已确认方案）:
  - 四分类映射: 0=正常(regular_road), 1=坑洞(potholes),
               2=减速带(asphalt_bumps + metal_bumps 合并), 3=磨损(worn_out_road)
  - 长度对齐: 线性插值重采样到固定 128 点（= 2.56 s @50 Hz, 板端窗口一致）
  - 标准化: 逐窗 z-score（每条事件减均值除方差, 顺带去掉重力直流; 板端同规则）
  - 划分: 双口径
      (1) 十折分层交叉验证: fold_ids 数组（种子 42, 文献/仓库作者主口径）
      (2) 70/15/15 固定留出（种子 42）: 训练最终部署模型
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = r"D:/pothole-detection-and-severity-management-main/data"
OUT_DIR = r"D:/tinyml_project/data"
N_POINTS = 128      # 对齐长度（2.56 s @50 Hz）
SEED = 42
N_FOLDS = 10

CLASS_MAP = {       # 四分类映射
    "regular_road": 0,     # 正常
    "potholes": 1,         # 坑洞
    "asphalt_bumps": 2,    # 减速带（沥青）
    "metal_bumps": 2,      # 减速带（金属）
    "worn_out_road": 3,    # 磨损
}
NAMES = ["正常", "坑洞", "减速带", "磨损"]

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

def resample(a, n=N_POINTS):
    """线性插值重采样到固定长度（保留原始幅值信息, 标准化在训练脚本内按划分拟合）"""
    L = len(a)
    if L == n:
        return a.astype(np.float64)
    return np.interp(np.linspace(0, L - 1, n), np.arange(L), a)

# ---------- 读取全部事件 ----------
X, y, src = [], [], []
lens_raw = {}
for cls in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(p) or cls not in CLASS_MAP:
        continue
    for f in sorted(glob.glob(os.path.join(p, "*.csv"))):
        a = pd.read_csv(f).iloc[:, 0].to_numpy(np.float64)
        lens_raw.setdefault(cls, []).append(len(a))
        w = resample(a)
        X.append(w.astype(np.float32))
        y.append(CLASS_MAP[cls])
        src.append(cls)
X = np.stack(X); y = np.array(y, np.int8)
log("=" * 62)
log("Gonzalez 2017 事件级预处理（四分类, 128 点对齐, 逐窗标准化, 种子 %d）" % SEED)
for cls in sorted(lens_raw):
    L = np.array(lens_raw[cls])
    log(f"  {cls:15} -> 类别 {CLASS_MAP[cls]}({NAMES[CLASS_MAP[cls]]}) | {len(L)} 条 | "
        f"原始长度 中位 {np.median(L):.0f} 范围 [{L.min()},{L.max()}]")
c_all = np.bincount(y, minlength=4)
log(f"四分类样本分布: " + " | ".join(f"{NAMES[i]} {c_all[i]}" for i in range(4)))
log(f"对齐后窗口: {X.shape} | 标准化后范围 [{X.min():.2f}, {X.max():.2f}] | NaN {int(np.isnan(X).sum())}")

# ---------- 口径一: 十折分层交叉验证索引 ----------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_ids = np.zeros(len(y), np.int8)
for k, (_, te) in enumerate(skf.split(X, y)):
    fold_ids[te] = k
log(f"十折分层 CV: 每折测试样本 {np.bincount(fold_ids, minlength=N_FOLDS).tolist()}")

# ---------- 口径二: 70/15/15 固定留出 ----------
idx = np.arange(len(y))
itr, irest = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
iva, ite = train_test_split(irest, test_size=0.5, random_state=SEED, stratify=y[irest])
log(f"留出划分: 训练 {len(itr)} / 验证 {len(iva)} / 测试 {len(ite)}")
for nm, ii in [("训练", itr), ("验证", iva), ("测试", ite)]:
    c = np.bincount(y[ii], minlength=4)
    log(f"  [{nm}] " + " | ".join(f"{NAMES[i]} {c[i]}" for i in range(4)))

npz_path = os.path.join(OUT_DIR, "gonzalez.npz")
np.savez_compressed(
    npz_path,
    X_all=X, y_all=y, fold_ids=fold_ids,
    X_train=X[itr], y_train=y[itr],
    X_val=X[iva], y_val=y[iva],
    X_test=X[ite], y_test=y[ite],
    meta=json.dumps({
        "n_points": N_POINTS, "sample_rate_hz": 50, "seed": SEED, "n_folds": N_FOLDS,
        "classes": NAMES,
        "class_map": {"regular_road": 0, "potholes": 1,
                      "asphalt_bumps": 2, "metal_bumps": 2, "worn_out_road": 3},
        "normalization": "raw (未标准化; 全局逐位置标准化在训练脚本内按训练集拟合)",
        "source": "Gonzalez 2017 via aadidesign/pothole-detection-and-severity-management",
        "holdout_idx": {"train": itr.tolist(), "val": iva.tolist(), "test": ite.tolist()},
    }, ensure_ascii=False),
)
log(f"已保存 {npz_path} ({os.path.getsize(npz_path)/1e3:.0f} KB)")

with open(os.path.join(OUT_DIR, "prepare_gonzalez_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
log("日志已保存 " + os.path.join(OUT_DIR, "prepare_gonzalez_log.txt"))
