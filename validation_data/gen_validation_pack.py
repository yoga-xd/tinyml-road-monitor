# -*- coding: utf-8 -*-
"""
gen_validation_pack.py —— 生成板端验证数据包 board_val_pack.npz
内容（75 条 Gonzalez 留出测试事件，四类全覆盖）:
  raw        (75,128) float32  原始 m/s² 窗口（板上采集对齐量）
  net_sig    (75,128) float32  网络输入 1：逐位置标准化信号 = (raw-pos_mean)/pos_sd
  net_feat   (75,24)  float32  网络输入 2：特征标准化 = (features(raw)-feat_mean)/feat_std
  expected   (75,4)   float32  PC 端 TFLite 参考输出（板上比对基准）
  labels     (75,)    int      真实类别 0正常/1坑洞/2减速带/3磨损
用法: 板子到手后由 compare_board.py 消费。
"""
import os
import json
import numpy as np

BASE = r"D:/tinyml_project"
EPS = 1e-6


def features_batch(Xb):
    """与训练端一字不差（float64 计算）"""
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
    os.chdir(BASE)
    sc = json.load(open("model_gonzalez_crop_scalers.json"))
    pos_mean = np.array(sc["pos_mean"], np.float64)
    pos_std = np.maximum(np.array(sc["pos_std"], np.float64), EPS)
    feat_mean = np.array(sc["feat_mean"], np.float64)
    feat_std = np.maximum(np.array(sc["feat_std"], np.float64), EPS)

    d = np.load(r"data/gonzalez.npz", allow_pickle=False)
    raw = d["X_test"].astype(np.float64)          # (75,128)
    labels = d["y_test"].astype(np.int64)
    print(f"测试事件: {raw.shape[0]} 条 | 类别分布: {np.bincount(labels, minlength=4).tolist()} (正常/坑洞/减速带/磨损)")

    # 网络双输入
    net_sig = (raw - pos_mean[None, :]) / pos_std[None, :]
    feats = features_batch(raw)
    net_feat = (feats - feat_mean[None, :]) / feat_std[None, :]
    print(f"net_sig 范围 [{net_sig.min():.2f},{net_sig.max():.2f}] | net_feat 范围 [{net_feat.min():.2f},{net_feat.max():.2f}]")

    # PC 端 TFLite 参考输出
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path="model_gonzalez_crop.tflite")
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    # 输入槽位: 按 shape 识别 (128,1)=信号, (24,)=特征
    sig_idx = next(i for i, x in enumerate(in_det) if x["shape"][-1] == 1 and x["shape"][1] == 128)
    feat_idx = 1 - sig_idx
    expected = np.zeros((raw.shape[0], 4), np.float32)
    for i in range(raw.shape[0]):
        interp.set_tensor(in_det[sig_idx]["index"],
                          net_sig[i].astype(np.float32).reshape(1, 128, 1))
        interp.set_tensor(in_det[feat_idx]["index"],
                          net_feat[i].astype(np.float32).reshape(1, 24))
        interp.invoke()
        expected[i] = interp.get_tensor(out_det[0]["index"])[0]
    pred = expected.argmax(1)
    acc = (pred == labels).mean()
    print(f"PC 端 TFLite 参考输出完成 | 留出测试准确率 {acc*100:.2f}%（应≈84%）")
    print(f"参考输出概率范围 [{expected.min():.4f},{expected.max():.4f}]")

    np.savez_compressed(
        os.path.join("validation_data", "board_val_pack.npz"),
        raw=raw.astype(np.float32), net_sig=net_sig.astype(np.float32),
        net_feat=net_feat.astype(np.float32), expected=expected,
        labels=labels,
        meta=json.dumps({"n": int(raw.shape[0]), "classes": sc["classes"],
                         "model": "model_gonzalez_crop.tflite",
                         "scalers": "model_gonzalez_crop_scalers.json",
                         "sample_rate_hz": sc["sample_rate_hz"],
                         "window_points": sc["window_points"]}, ensure_ascii=False))
    print("数据包已写入 validation_data/board_val_pack.npz")


if __name__ == "__main__":
    main()
