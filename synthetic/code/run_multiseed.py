# -*- coding: utf-8 -*-
"""
run_multiseed.py — 三配置 × 多随机种子的重复验证

目的：中期/学位论文的关键结论必须在同一硬件、同一数据、同一超参下重复多次仍成立，
      而不是单次运行的偶然结果。本脚本跑 3 个随机种子（0/1/2）× 3 个配置 = 9 次训练，
      输出跨种子的均值±标准差，并自动判定核心结论是否 3/3 稳定。

输出：
    out_seeds/seed{0,1,2}/metrics.json     每次运行的完整指标
    out_seeds/summary.json                 跨种子汇总
    out_seeds/report.md                    判定表
"""
import os
import json
import time
import numpy as np
import torch

import train_3dgs as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_seeds")
os.makedirs(OUT, exist_ok=True)

SEEDS = [0, 1, 2]
MODES = ["naive", "dyn-mask", "no-dyn-data"]
ITERS = 2500
POINTS = 5000

KEYS = ["psnr_all_mean", "psnr_static_mean", "psnr_dyn_mean",
        "ssim_mean", "ghost_residual_mean"]


def main():
    meta, K, frames = T.load_scene()
    print(f"[data] {meta['frames']} frames {meta['W']}x{meta['H']} device={T.DEV}", flush=True)
    print(f"[plan] seeds={SEEDS} modes={MODES} iters={ITERS} points={POINTS}", flush=True)

    all_runs = {}
    t_all = time.time()

    for seed in SEEDS:
        sdir = os.path.join(OUT, f"seed{seed}")
        os.makedirs(sdir, exist_ok=True)
        mfile = os.path.join(sdir, "metrics.json")
        if os.path.exists(mfile):
            with open(mfile, encoding="utf-8") as f:
                all_runs[seed] = json.load(f)["configs"]
            print(f"[skip] seed{seed} 已完成，直接读取", flush=True)
            continue

        runs = {}
        for mode in MODES:
            print(f"\n===== seed={seed} mode={mode} =====", flush=True)
            model, hist, use_frames, tt = T.train(
                mode, frames, K, meta, iters=ITERS, max_points=POINTS, seed=seed)
            rows, agg = T.evaluate(model, frames, K, meta)
            agg["train_time_s"] = round(tt, 2)
            agg["n_gauss"] = int(model.means.shape[0])
            agg["mode"] = mode
            agg["seed"] = seed
            runs[mode] = {"agg": agg, "per_frame": rows}
            print(f"[seed{seed}/{mode}] {json.dumps(agg, ensure_ascii=False)}", flush=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with open(mfile, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "iters": ITERS, "init_points": POINTS,
                       "configs": runs}, f, ensure_ascii=False, indent=2)
        all_runs[seed] = runs

    # ---------- 跨种子统计 ----------
    stats = {}
    for mode in MODES:
        stats[mode] = {}
        for k in KEYS + ["train_time_s"]:
            vals = [all_runs[s][mode]["agg"][k] for s in SEEDS if mode in all_runs.get(s, {})]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            stats[mode][k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                              "vals": [float(v) for v in arr]}

    # ---------- 核心结论的稳定性判定 ----------
    checks = []
    # C1: B(dyn-mask) 静态区 PSNR 在每次种子中都高于 A(naive)
    c1 = all(all_runs[s]["dyn-mask"]["agg"]["psnr_static_mean"] > all_runs[s]["naive"]["agg"]["psnr_static_mean"]
             for s in SEEDS)
    c1n = sum(1 for s in SEEDS if all_runs[s]["dyn-mask"]["agg"]["psnr_static_mean"] >
              all_runs[s]["naive"]["agg"]["psnr_static_mean"])
    checks.append(("C1 静态区 PSNR：B(掩码加权) > A(朴素)", c1, f"{c1n}/{len(SEEDS)}"))

    # C2: B 动态残影在每次种子中都低于 A 与 C
    c2n = 0
    for s in SEEDS:
        gb = all_runs[s]["dyn-mask"]["agg"]["ghost_residual_mean"]
        ga = all_runs[s]["naive"]["agg"]["ghost_residual_mean"]
        gc = all_runs[s]["no-dyn-data"]["agg"]["ghost_residual_mean"]
        if gb < ga and gb < gc:
            c2n += 1
    checks.append(("C2 动态残影：B 同时低于 A 与 C", c2n == len(SEEDS), f"{c2n}/{len(SEEDS)}"))

    # C3: B 残影相对 A 的下降幅度至少 10 倍
    ratios = [all_runs[s]["naive"]["agg"]["ghost_residual_mean"] /
              max(all_runs[s]["dyn-mask"]["agg"]["ghost_residual_mean"], 1e-9) for s in SEEDS]
    checks.append((f"C3 残影下降倍数 ≥10×（实测 {min(ratios):.1f}–{max(ratios):.1f}×）",
                   min(ratios) >= 10, f"{sum(1 for r in ratios if r >= 10)}/{len(SEEDS)}"))

    # C4: 仅初始化滤波(C)不足以压住残影 —— C 残影显著高于 B
    c4n = sum(1 for s in SEEDS if all_runs[s]["no-dyn-data"]["agg"]["ghost_residual_mean"] >
              5 * all_runs[s]["dyn-mask"]["agg"]["ghost_residual_mean"])
    checks.append(("C4 仅初始化滤波(C)残影 > B 的 5 倍", c4n == len(SEEDS), f"{c4n}/{len(SEEDS)}"))

    # C5: 排序稳定 —— 每次种子中静态区 PSNR 的排序都是 B > C > A
    c5n = 0
    for s in SEEDS:
        order = sorted(MODES, key=lambda m: -all_runs[s][m]["agg"]["psnr_static_mean"])
        if order == ["dyn-mask", "no-dyn-data", "naive"]:
            c5n += 1
    checks.append(("C5 静态区排序稳定为 B > C > A", c5n == len(SEEDS), f"{c5n}/{len(SEEDS)}"))

    # ---------- 报告 ----------
    lines = ["# 三配置重复验证报告", "",
             f"- 随机种子：{SEEDS}　每配置重复 {len(SEEDS)} 次，共 {len(SEEDS) * len(MODES)} 次训练",
             f"- 超参：iters={ITERS}　init_points={POINTS}",
             f"- 硬件：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
             f"- 总耗时：{round(time.time() - t_all, 1)} s", "",
             "## 一、跨种子均值 ± 标准差", "",
             "| 配置 | 全图 PSNR | 静态区 PSNR | 动态区 PSNR | SSIM | 残影 |",
             "|---|---|---|---|---|---|"]
    name_cn = {"naive": "A 朴素", "dyn-mask": "B 掩码加权", "no-dyn-data": "C 仅初始化滤波"}
    for mode in MODES:
        st = stats[mode]
        cells = []
        for k in KEYS:
            v = st.get(k)
            cells.append(f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "-")
        lines.append(f"| {name_cn[mode]} | " + " | ".join(cells) + " |")

    lines += ["", "## 二、核心结论稳定性判定", "",
              "| 判定项 | 结论 | 命中 |", "|---|---|---|"]
    for name, ok, hit in checks:
        lines.append(f"| {name} | {'✅ 成立' if ok else '❌ 不成立'} | {hit} |")

    lines += ["", "## 三、逐次原始值", "",
              "| 种子 | 配置 | 静态区 PSNR | 残影 | SSIM | 耗时(s) |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        for mode in MODES:
            a = all_runs[s][mode]["agg"]
            lines.append(f"| {s} | {name_cn[mode]} | {a['psnr_static_mean']:.4f} | "
                         f"{a['ghost_residual_mean']:.4f} | {a['ssim_mean']:.4f} | {a['train_time_s']:.1f} |")

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "iters": ITERS, "init_points": POINTS,
                   "stats": stats,
                   "checks": [{"name": n, "pass": bool(o), "hit": h} for n, o, h in checks]},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
