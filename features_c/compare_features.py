# -*- coding: utf-8 -*-
"""
compare_features.py —— Python 训练端 vs C 参考实现的 24 维特征数值对照
1) 从 gonzalez.npz 取全部 500 条事件（float32 原始 m/s²）
2) Python 端用与训练完全一致的 features_batch 计算（float64 参考精度）
3) 生成 windows.bin 供 C 读取，编译并运行 C 参考实现
4) 逐特征对比最大绝对误差 / 最大相对误差
"""
import numpy as np
import subprocess
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-6
NAMES = ["mean", "std", "rms", "peak", "ptp", "crest", "energy", "zcr", "iqr",
         "skew", "kurt", "ac_lag1", "ac_lag2", "ac_lag3", "ac_lag4", "ac_lag5",
         "band1", "band2", "band3", "band4", "band5", "band6", "band7", "band8"]


def features_batch(Xb):
    """与 gonzalez_experiments.py 训练端一字不差的特征函数"""
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


def main():
    d = np.load(r"D:/tinyml_project/data/gonzalez.npz", allow_pickle=False)
    X = d["X_all"].astype(np.float64)          # (500,128) 双精度参考
    assert X.shape[1] == 128, "窗口长度应为 128"
    F_py = features_batch(X)                    # (500,24) float64
    print(f"Python 参考特征: {F_py.shape}")

    # 生成 C 侧输入（float32 落盘，模拟板上 float32 采集值）
    with open("windows.bin", "wb") as fp:
        np.array([len(X)], np.int32).tofile(fp)
        X.astype(np.float32).tofile(fp)
    with open("feats_py.bin", "wb") as fp:
        np.array([len(X)], np.int32).tofile(fp)
        F_py.astype(np.float32).tofile(fp)

    # 编译并运行 C 参考实现
    gcc = r"C:\Program Files\mingw64\bin\gcc.exe"
    subprocess.run([gcc, "-O2", "-Wall", "-o", "features_ref.exe",
                    "features_ref.c", "-lm"], check=True)
    subprocess.run([os.path.join(".", "features_ref.exe")], check=True)

    # 读回 C 结果并对比（计数为 int32，数据为 float32）
    n = int(np.fromfile("feats_c.bin", np.int32, count=1)[0])
    F_c = np.fromfile("feats_c.bin", np.float32, offset=4).reshape(n, 24).astype(np.float64)
    assert n == len(X) and F_c.shape == F_py.shape, "形状不一致"

    print(f"\n{'特征':<12}{'最大绝对误差':>14}{'最大相对误差':>14}  判定(<1e-6)")
    print("-" * 56)
    worst_abs = worst_rel = 0.0
    all_ok = True
    for j, name in enumerate(NAMES):
        diff = np.abs(F_c[:, j] - F_py[:, j])
        denom = np.maximum(np.abs(F_py[:, j]), 1e-12)
        rel = (diff / denom).max()
        ab = diff.max()
        ok = (ab < 1e-6) or (rel < 1e-6)
        all_ok &= ok
        worst_abs = max(worst_abs, ab)
        worst_rel = max(worst_rel, rel)
        print(f"{name:<12}{ab:>14.3e}{rel:>14.3e}  {'PASS' if ok else 'FAIL'}")
    print("-" * 56)
    print(f"总体: 最大绝对误差 {worst_abs:.3e} | 最大相对误差 {worst_rel:.3e}")
    print("结论:", "C 实现与训练端数值一致 ✅" if all_ok else "存在偏差，需排查 ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
