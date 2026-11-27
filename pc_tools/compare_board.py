# -*- coding: utf-8 -*-
"""
compare_board.py —— 板子到手后的第一验证工具
用官方 stm_ai_runner 把 board_val_pack.npz 里的双输入逐条发到板上，
收集板上输出与 PC 端 TFLite 参考输出比对。

用法:
  C:\miniconda3\envs\tinyml\python.exe compare_board.py --port COM5
  （COM 口在设备管理器里看"USB-SERIAL CH340"对应的号；波特率默认 115200）
前提: 板上已烧录 tinyml_f407 验证固件（Device Application 生成的 aiValidation）。
"""
import os
# 必须在导入 stm_ai_runner 之前设置：X-CUBE-AI 10.2.1 生成的 pb2 与新版 protobuf 不兼容
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import sys
import json
import argparse
import numpy as np

RUNNER_PATH = (r"D:\ST\STM32Cube\Repository\Packs\STMicroelectronics"
               r"\X-CUBE-AI\10.2.1\scripts\ai_runner")
PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "validation_data", "board_val_pack.npz")
NAMES = ["正常", "坑洞", "减速带", "磨损"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="CH340 串口号, 如 COM5")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    sys.path.insert(0, RUNNER_PATH)
    from stm_ai_runner.ai_runner import AiRunner  # noqa: E402

    pack = np.load(PACK, allow_pickle=False)
    meta = json.loads(str(pack["meta"]))
    net_sig = pack["net_sig"]      # (75,128)
    net_feat = pack["net_feat"]    # (75,24)
    expected = pack["expected"]    # (75,4)
    labels = pack["labels"]
    n = meta["n"]
    print(f"验证包: {n} 条 | 模型 {meta['model']}")

    runner = AiRunner()
    desc = f"serial:{args.port}:{args.baud}"
    print(f"连接 {desc} ...（若无响应，按一下板上 RESET 键）")
    if not runner.connect(desc):
        print("连接失败:", runner.last_error if hasattr(runner, 'last_error') else "?")
        print("排查（按命中率排序）:")
        print("  ① 串口被占用——关闭所有串口助手/IDE 终端后重试")
        print("  ② 探索者板 USART1 跳线帽未接到 CH340 一侧")
        print("  ③ COM 号不对（设备管理器看 'USB-SERIAL CH340'）")
        print("  ④ 按一下板上 RESET 键（验证固件复位后才进入待机监听）再重试")
        print("  ⑤ 波特率必须是 115200（固件 usart.c 固定值，不要改）")
        return 1
    print("已连接:", runner.short_desc())

    ses = runner.session()
    diffs, match, pc_pred_ok = [], 0, 0
    for i in range(n):
        inputs = [net_sig[i].reshape(1, 128, 1, 1).astype(np.float32),
                  net_feat[i].reshape(1, 24, 1, 1).astype(np.float32)]
        outs, _ = ses.invoke(inputs)
        out = np.asarray(outs[0]).flatten()
        exp = expected[i]
        diffs.append(np.abs(out - exp))
        if out.argmax() == exp.argmax():
            match += 1
        if out.argmax() == labels[i]:
            pc_pred_ok += 1
        if (i + 1) % 25 == 0:
            print(f"  进度 {i+1}/{n}")

    diffs = np.array(diffs)
    print("\n========== 板端 vs PC 一致性报告 ==========")
    print(f"判类一致率（板 argmax == PC argmax）: {match}/{n} = {match/n*100:.1f}%")
    print(f"PC 参考输出对真实标签准确率: {pc_pred_ok}/{n} = {pc_pred_ok/n*100:.1f}%")
    print(f"输出概率逐元素: 最大绝对差 {diffs.max():.3e} | 平均绝对差 {diffs.mean():.3e}")
    print("判定: 最大绝对差 < 1e-4 → 板上推理与 PC 完全一致（float32 舍入级）")
    runner.disconnect()
    return 0 if diffs.max() < 1e-4 else 2


if __name__ == "__main__":
    sys.exit(main())
