# -*- coding: utf-8 -*-
"""
make_figs.py — 中期汇报 PPT 用图（全部由真实数据 / 已发表数值绘制）

数据来源：
  fig_traj / fig_ablation  -> Dy3DGS-SLAM 论文表 6 / 表 8（arXiv:2506.05965）
  fig_repro_*              -> 本机 3DGS 动态场景复刻实验 out/metrics.json（实跑）
  fig_pipeline             -> 方法结构示意（依据论文第 5 章方法描述绘制）
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1E4FA8"
MBLUE = "#3D7BD9"
LBLUE = "#E8EFF8"
GREY = "#8B97A8"
DARK = "#1A2230"
RED = "#D9534F"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
REPRO = os.path.join(HERE, "..", "..", "实验复刻_3DGS", "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 流程图
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=210)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec=BLUE, fs=8.2, tc=DARK, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                           linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=MBLUE):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=1.0, color=color,
                                     shrinkA=0, shrinkB=0))

    # 输入
    box(1, 40, 15, 9, "单目 RGB\n序列 I_t", fc=LBLUE, fs=8.0)
    box(1, 24, 15, 9, "前一帧\nI_{t-1}", fc=LBLUE, fs=8.0)

    # 三条证据支路
    box(22, 42, 20, 8, "RAFT 稠密光流\nF_t", fs=8.0)
    box(22, 29, 20, 8, "UniDepth 单目深度\nD_t（相对尺度）", fs=7.6)
    box(22, 15, 20, 8, "6D 运动特征\n(位移+深度差)", fs=7.8)

    box(46, 42, 17, 8, "运动残差 r_flow\n稳健重投影", fs=7.8)
    box(46, 29, 17, 8, "几何残差 r_geo\n深度-光流一致性", fs=7.6)
    box(46, 15, 17, 8, "K-Means 聚类\n运动簇标签 c", fs=7.8)

    # 贝叶斯融合
    box(66.0, 18, 18.5, 25, "贝叶斯后验融合\n\nP(d|e) ∝\nP(e|d)·P(d)\n\n三支证据独立\n条件假设", fc=BLUE, ec=BLUE, fs=7.2, tc="#FFFFFF", bold=True)

    # 输出
    box(87, 26, 12, 11, "像素级\n动态掩码\nM_t", fc=LBLUE, fs=8.0, bold=True)

    for y in (46, 33, 19):
        arrow(16.5, 44.5 if y == 46 else (28.5 if y == 33 else 28.5), 21.5, y)
    arrow(42.5, 46, 45.5, 46)
    arrow(42.5, 33, 45.5, 33)
    arrow(42.5, 19, 45.5, 19)
    arrow(63.5, 46, 66.0, 39)
    arrow(63.5, 33, 66.0, 31)
    arrow(63.5, 19, 66.0, 23)
    arrow(85.0, 31, 87.5, 31)

    # 下游
    ax.text(50, 7.5, "下游：位姿优化外点降权 · 高斯掩码加权渲染 · 高斯生命周期初始化滤波",
            ha="center", va="center", fontsize=7.6, color=GREY)
    ax.add_patch(FancyArrowPatch((75.5, 18.5), (75.5, 10.5), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.0, color=GREY,
                                 linestyle="--", shrinkA=0, shrinkB=0))

    ax.text(1, 52.5, "无预训练语义分割 · 三支几何证据 + 概率融合", fontsize=8.6,
            color=BLUE, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=210,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_pipeline.png")


# ---------------------------------------------------------------- ATE 对比
def fig_traj():
    """平均 ATE RMSE（m），数值取自论文表 6"""
    data = [
        ("ORB-SLAM3", 0.562, GREY),
        ("SplaTAM", 0.029, MBLUE),
        ("本研究", 0.012, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(4.9, 3.2), dpi=210)
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, height=0.52, color=cols, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5, color=DARK)
    ax.set_xscale("log")
    ax.set_xlim(0.008, 1.2)
    ax.set_xlabel("平均 ATE RMSE（m，对数刻度）", fontsize=8.6, color=DARK)
    ax.tick_params(axis="x", labelsize=8, colors=GREY)
    for b, v in zip(bars, vals):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=9.5, color=DARK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D6DCE5")
    ax.spines["bottom"].set_color("#D6DCE5")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_traj.png"), dpi=210, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig_traj.png")


# ---------------------------------------------------------------- 消融
def fig_ablation():
    """七配置消融，数值取自论文表 8"""
    rows = [
        ("完整系统", 0.012, 35.2, True),
        ("w/o 动态初始化滤波", 0.038, 31.8, False),
        ("w/o 高斯生命周期", 0.045, 33.1, False),
        ("w/o 掩码加权损失", 0.052, 32.4, False),
        ("w/o 贝叶斯融合", 0.068, 31.5, False),
        ("w/o 尺度感知跟踪", 0.091, 30.8, False),
        ("基线（无动态处理）", 0.562, 24.6, False),
    ]
    names = [r[0] for r in rows]
    ate = [r[1] for r in rows]
    psnr = [r[2] for r in rows]
    cols = [BLUE if r[3] else MBLUE for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 3.7), dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    x = np.arange(len(names))

    ax = axes[0]
    bars = ax.bar(x, ate, width=0.56, color=cols)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.4)
    ax.set_ylabel("ATE RMSE (m)", fontsize=8.6, color=DARK)
    for b, v in zip(bars, ate):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.16, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=DARK)
    ax.tick_params(axis="y", labelsize=7.5, colors=GREY)
    ax.set_title("跟踪精度（越低越好，对数刻度）", loc="left", fontsize=8,
                 color=BLUE, fontweight="bold", pad=5)

    ax = axes[1]
    bars = ax.bar(x, psnr, width=0.56, color=cols)
    ax.set_ylim(20, 38)
    ax.set_ylabel("PSNR (dB)", fontsize=8.6, color=DARK)
    for b, v in zip(bars, psnr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, color=DARK)
    ax.tick_params(axis="y", labelsize=7.5, colors=GREY)
    ax.set_title("渲染质量（越高越好）", loc="left", fontsize=8,
                 color=BLUE, fontweight="bold", pad=5)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#D6DCE5")
        ax.spines["bottom"].set_color("#D6DCE5")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8.2, rotation=16, ha="right", color=DARK)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig_ablation.png")


# ---------------------------------------------------------------- 本机复刻
def fig_repro():
    mpath = os.path.join(REPRO, "metrics.json")
    if not os.path.exists(mpath):
        print("!! metrics.json 不存在，跳过复刻图")
        return
    meta = json.load(open(mpath, encoding="utf-8"))
    cfgs = meta["configs"]

    # (a) 训练曲线（含移动平均平滑，抑制单帧采样噪声）
    fig, ax = plt.subplots(figsize=(5.0, 3.05), dpi=210)
    label = {"naive": "A  朴素：动态像素参与初始化与优化",
             "no-dyn-data": "C  仅初始化滤波：动态点已剔除，训练仍用全图损失",
             "dyn-mask": "B  初始化滤波 + 掩码加权损失"}
    color = {"naive": RED, "no-dyn-data": MBLUE, "dyn-mask": BLUE}
    for k in ("naive", "no-dyn-data", "dyn-mask"):
        fp = os.path.join(REPRO, f"hist_{k}.npy")
        if not os.path.exists(fp):
            continue
        h = np.load(fp)
        y = h[:, 2]
        w = 5
        ys = np.convolve(y, np.ones(w) / w, mode="valid") if len(y) > w else y
        xs = h[w - 1:, 0] if len(y) > w else h[:, 0]
        ax.plot(xs, ys, color=color[k], linewidth=1.7, label=label[k])
    ax.set_xlabel("训练迭代次数", fontsize=8.4, color=DARK)
    ax.set_ylabel("渲染 PSNR (dB，5 点滑动平均)", fontsize=8.2, color=DARK)
    ax.set_ylim(14, 26)
    ax.tick_params(labelsize=7.6, colors=GREY)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D6DCE5")
    ax.spines["bottom"].set_color("#D6DCE5")
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_repro_curve.png"), dpi=210,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_repro_curve.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_traj()
    fig_ablation()
    fig_repro()
