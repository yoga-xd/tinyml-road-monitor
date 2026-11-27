# -*- coding: utf-8 -*-
"""阶段 A: model_gonzalez_crop.h5 -> TFLite 转换 + PC 端一致性验证"""
import os, json, time
import numpy as np
import tensorflow as tf

BASE = r"D:/tinyml_project"
EPS = 1e-6
log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

# ---------- 数据（与训练/评估同源） ----------
d = np.load(os.path.join(BASE, "data", "gonzalez.npz"), allow_pickle=False)
Xte, yte = d["X_test"], d["y_test"]
sc = json.load(open(os.path.join(BASE, "model_gonzalez_crop_scalers.json")))
pm, psd = np.array(sc["pos_mean"]), np.array(sc["pos_std"])
fm, fsd = np.array(sc["feat_mean"]), np.array(sc["feat_std"])

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

def as_mi(Xr):
    sig = ((Xr - pm[None, :]) / psd[None, :]).astype(np.float32)[:, :, None]
    ft = ((features_batch(Xr) - fm[None, :]) / fsd[None, :]).astype(np.float32)
    return [sig, ft]

sig_in, feat_in = as_mi(Xte)          # (75,128,1) + (75,24)
NAMES = ["正常", "坑洞", "减速带", "磨损"]

# ---------- 1. 加载 Keras 模型 ----------
model = tf.keras.models.load_model(os.path.join(BASE, "model_gonzalez_crop.h5"))
log("=" * 62)
log("阶段 A: TFLite 转换与一致性验证")
log(f"Keras 模型输入: {[i.shape for i in model.inputs]}")

# ---------- 2. float32 TFLite 转换 ----------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
p_float = os.path.join(BASE, "model_gonzalez_crop.tflite")
open(p_float, "wb").write(tflite_float)
log(f"float32 TFLite: {len(tflite_float)/1024:.1f} KB -> {p_float}")

# ---------- 3. int8 全整数量化（仅作参考） ----------
def rep_dataset():
    for i in range(0, len(Xte), 16):
        yield [sig_in[i:i+16], feat_in[i:i+16]]
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = rep_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.float32
converter_int8.inference_output_type = tf.float32
try:
    tflite_int8 = converter_int8.convert()
    p_int8 = os.path.join(BASE, "model_gonzalez_crop_int8.tflite")
    open(p_int8, "wb").write(tflite_int8)
    log(f"int8 量化 TFLite（参考）: {len(tflite_int8)/1024:.1f} KB -> {p_int8}")
    int8_ok = True
except Exception as e:
    log(f"int8 量化失败（不影响主线）: {str(e)[:120]}")
    int8_ok = False

# ---------- 4. 解释器逐条一致性比对 ----------
interp = tf.lite.Interpreter(model_path=p_float)
interp.allocate_tensors()
idetails = interp.get_input_details(); odetails = interp.get_output_details()
log(f"解释器输入: {[(x['shape'].tolist(), x['dtype']) for x in idetails]}")
log(f"解释器输出: {odetails[0]['shape'].tolist()}")

# 输入顺序对齐: 按形状匹配 (128,1) 信号 / (24,) 特征
sig_idx = next(i for i, x in enumerate(idetails) if x["shape"][-1] == 1 and x["shape"][1] == 128)
feat_idx = next(i for i, x in enumerate(idetails) if x["shape"][-1] == feat_in.shape[1])
log(f"输入对齐: 信号->槽位{sig_idx}, 特征->槽位{feat_idx}")

keras_pred = model.predict([sig_in, feat_in], verbose=0)
diffs, agree = [], 0
for i in range(len(Xte)):
    interp.set_tensor(idetails[sig_idx]["index"], sig_in[i:i+1])
    interp.set_tensor(idetails[feat_idx]["index"], feat_in[i:i+1])
    interp.invoke()
    out = interp.get_tensor(odetails[0]["index"])[0]
    diffs.append(np.abs(out - keras_pred[i]).max())
    if out.argmax() == keras_pred[i].argmax():
        agree += 1
diffs = np.array(diffs)
log(f"75 条测试集逐条比对: 最大绝对差 {diffs.max():.2e} | 平均绝对差 {diffs.mean():.2e}")
log(f"argmax 判类一致率: {agree}/{len(Xte)} = {agree/len(Xte)*100:.1f}%")
verdict = "通过" if diffs.max() < 1e-4 and agree == len(Xte) else "需排查"
log(f"一致性判定（阈值 <1e-4 且判类 100%）: {verdict}")

# ---------- 5. int8 精度参考 ----------
if int8_ok:
    interp8 = tf.lite.Interpreter(model_path=p_int8)
    interp8.allocate_tensors()
    id8 = interp8.get_input_details(); od8 = interp8.get_output_details()
    s8 = next(i for i, x in enumerate(id8) if x["shape"][-1] == 1 and x["shape"][1] == 128)
    f8 = next(i for i, x in enumerate(id8) if x["shape"][-1] == feat_in.shape[1])
    agree8, diffs8 = 0, []
    for i in range(len(Xte)):
        interp8.set_tensor(id8[s8]["index"], sig_in[i:i+1])
        interp8.set_tensor(id8[f8]["index"], feat_in[i:i+1])
        interp8.invoke()
        out8 = interp8.get_tensor(od8[0]["index"])[0]
        diffs8.append(np.abs(out8 - keras_pred[i]).max())
        if out8.argmax() == keras_pred[i].argmax():
            agree8 += 1
    log(f"int8 参考: 最大绝对差 {np.max(diffs8):.2e} | 判类一致率 {agree8}/{len(Xte)}")

# ---------- 6. 转换验证报告 ----------
report = {
    "model": "model_gonzalez_crop.h5",
    "arch": "Conv1D(16,5)->MP2->Conv1D(32,3)->GAP->Drop(0.3) + Dense(64) 混合双输入",
    "keras_inputs": [str(i.shape) for i in model.inputs],
    "tflite_float32": {"file": p_float, "size_kb": len(tflite_float)/1024,
                       "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
                       "argmax_agree": f"{agree}/{len(Xte)}", "verdict": verdict},
    "tflite_int8_ref": ({"file": p_int8, "size_kb": len(tflite_int8)/1024,
                         "max_abs_diff": float(np.max(diffs8)),
                         "argmax_agree": f"{agree8}/{len(Xte)}"} if int8_ok else "转换失败"),
    "note": "部署主线路径: h5 或 float32 tflite 直接进 X-CUBE-AI; int8 仅作体积参考",
}
json.dump(report, open(os.path.join(BASE, "stage_a_conversion_report.json"), "w"),
          ensure_ascii=False, indent=2)
log("报告已保存 stage_a_conversion_report.json")
print("DONE")
