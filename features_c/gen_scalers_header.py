# -*- coding: utf-8 -*-
"""gen_scalers_header.py —— 由 model_gonzalez_crop_scalers.json 生成固件 C 头文件"""
import json
import numpy as np

sc = json.load(open(r"D:/tinyml_project/model_gonzalez_crop_scalers.json"))
pos_mean = np.array(sc["pos_mean"], np.float32)
pos_std = np.maximum(np.array(sc["pos_std"], np.float32), 1e-6)
feat_mean = np.array(sc["feat_mean"], np.float32)
feat_std = np.maximum(np.array(sc["feat_std"], np.float32), 1e-6)


def arr(name, v, per_line=4):
    lines = []
    for i in range(0, len(v), per_line):
        chunk = ", ".join(f"{x:.7f}f" for x in v[i:i + per_line])
        lines.append("    " + chunk + ",")
    body = "\n".join(lines).rstrip(",")
    return f"static const float {name}[{len(v)}] = {{\n{body}\n}};"


hdr = f"""/*
 * scalers_gonzalez.h —— 部署固化常数（由 model_gonzalez_crop_scalers.json 自动生成）
 * 模型: model_gonzalez_crop (四类: {"/".join(sc["classes"])})
 * 采样率 {sc["sample_rate_hz"]} Hz | 窗口 {sc["window_points"]} 点
 *
 * 流水线（固件内顺序，与训练端一致）:
 *   raw acc_z (m/s², 含重力, z 轴朝上)
 *   → net_sig = (raw - POS_MEAN[i]) / POS_STD[i]          ← 网络输入 1 (128)
 *   → features_fw(raw) 得 24 维原始特征
 *   → net_feat = (feat[j] - FEAT_MEAN[j]) / FEAT_STD[j]   ← 网络输入 2 (24)
 *   → ai_network_run → 4 类概率 (正常/坑洞/减速带/磨损)
 */
#ifndef SCALERS_GONZALEZ_H
#define SCALERS_GONZALEZ_H

#define N_WINDOW   128
#define N_FEATURES  24

{arr("POS_MEAN", pos_mean)}

{arr("POS_STD", pos_std)}

{arr("FEAT_MEAN", feat_mean)}

{arr("FEAT_STD", feat_std)}

#endif /* SCALERS_GONZALEZ_H */
"""
open(r"D:/tinyml_project/firmware/scalers_gonzalez.h", "w", encoding="utf-8", newline="\n").write(hdr)
print("scalers_gonzalez.h 已生成 (128+128+24+24 个常数)")
