# -*- coding: utf-8 -*-
"""
render_compare.py — 加载三个训练好的高斯模型，渲染同一批视角做定性对比

输出：
    assets/fig_repro_render.png   （GT / A naive / B dyn-mask / C 仅静态帧 四行对照）
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import train_3dgs as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ASSETS = os.path.join(HERE, "..", "中期汇报PPT", "中期汇报PPT", "assets")
os.makedirs(ASSETS, exist_ok=True)

FRAMES = [0, 6, 11]
MODE_ORDER = [
    ("no-dyn-data", "C 仅静态帧训练"),
    ("dyn-mask", "B 掩码加权损失"),
    ("naive", "A 朴素训练"),
]


@torch.no_grad()
def main():
    meta, K, frames = T.load_scene()
    H, W = meta["H"], meta["W"]

    rows = []
    for mode, label in MODE_ORDER:
        mp = os.path.join(OUT, f"model_{mode}.pt")
        if not os.path.exists(mp):
            print(f"!! 缺少 {mp}，跳过 {mode}")
            continue
        n = torch.load(mp, map_location=T.DEV)["means"].shape[0]
        pts, cols = T.backproject(frames, K, n, exclude_dynamic=(mode != "naive"))
        model = T.Gaussians(pts, cols, scale_init=0.05).to(T.DEV)
        model.load_state_dict(torch.load(mp, map_location=T.DEV))
        imgs = []
        for fi in FRAMES:
            f = frames[fi]
            img, _ = T.rasterize(model, f["R"], f["t"], K, H, W)
            imgs.append(img.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy())
        rows.append((label, imgs))
        print(f"rendered {mode}")

    if not rows:
        print("!! 没有可用模型")
        return

    ncol = len(FRAMES)
    nrow = len(rows) + 1
    # 固定画布 + 手动留白，避免 tight_layout 把行标签压成重叠
    fig = plt.figure(figsize=(8.6, 7.5), dpi=210)
    left, right, top, bottom = 0.135, 0.992, 0.945, 0.015
    hspace, wspace = 0.055, 0.035
    gs = fig.add_gridspec(nrow, ncol, left=left, right=right, top=top,
                          bottom=bottom, hspace=hspace, wspace=wspace)

    row_labels = ["真值 GT"] + [lb for lb, _ in rows]

    for c, fi in enumerate(FRAMES):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(frames[fi]["rgb"].cpu().numpy(), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"第 {fi} 帧", fontsize=9.0, color="#8B97A8", pad=4)

    for r, (label, imgs) in enumerate(rows, start=1):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(imgs[c], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])

    # 行标签：按每行第一个 axes 的纵向中心竖排
    for r in range(nrow):
        ax0 = fig.axes[r * ncol]
        bb = ax0.get_position()
        fig.text(0.018, (bb.y0 + bb.y1) / 2.0, row_labels[r],
                 rotation=90, va="center", ha="left",
                 fontsize=9.2,
                 color="#1A2230" if r == 0 else "#1E4FA8",
                 fontweight="bold")

    for ax in fig.axes:
        for s in ax.spines.values():
            s.set_color("#D6DCE5")
            s.set_linewidth(0.6)

    p = os.path.join(ASSETS, "fig_repro_render.png")
    fig.savefig(p, dpi=210, facecolor="white")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
