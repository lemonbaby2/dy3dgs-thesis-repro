# REPRODUCE — 从零复现全过程

本文件给出可逐条执行复现的完整命令序列。**不使用任何第三方数据集**，场景由代码合成，因此任何人可在任意机器上得到同等结论（浮点级差异除外）。

## 一、环境要求

| 项 | 最低 | 本次实测 |
|---|---|---|
| Python | 3.10 | 3.13（Anaconda 基座） |
| PyTorch | 2.0 | 2.11.0+cu128 |
| GPU | 支持 CUDA 即可（显存 ≥ 4 GB） | RTX 5060 Ti 16 GB（sm_120） |
| 其他 | numpy | numpy、matplotlib（仅绘图与对比图需要） |

> 本实现是**纯 PyTorch** 的可微高斯光栅化，**不需要**编译 CUDA 扩展（无 `diff-gaussian-rasterization` / `simple-knn` 依赖），因此在 sm_120 这类新架构上无需等待第三方适配。

安装：

```bash
conda create -n dy3dgs python=3.11 -y
conda activate dy3dgs
pip install torch numpy matplotlib --index-url https://download.pytorch.org/whl/cu128
```

## 二、目录约定

```
dy3dgs-thesis-repro/
├── synthetic/code/      scene_dyn.py  train_3dgs.py  run_multiseed.py  render_compare.py  make_figs.py
├── synthetic/data/      data_dyn/（已随仓库提供；也可由 scene_dyn.py 重新生成）
├── synthetic/results/   metrics.json  out_seeds/
└── synthetic/figures/
```

`synthetic/code/` 内的脚本以**同级工作目录**为基准，运行时先进入仓库根目录。

## 三、步骤

### 步骤 0（可选）重新生成合成场景

```bash
cd dy3dgs-thesis-repro
python synthetic/code/scene_dyn.py
```

输出 12 帧到 `synthetic/code/data_dyn/`。若已使用仓库自带数据，可跳过。
预期：12 帧，`dyn_ratio` 在 0.028–0.095 之间，球体全程在视野内。

### 步骤 1 单次三配置对照（约 22 分钟）

```bash
python synthetic/code/train_3dgs.py --iters 2500 --points 5000
```

预期控制台输出：

```
[data] 12 frames 160x120  device=cuda
[gpu ] NVIDIA GeForce RTX 5060 Ti
===== training config: naive =====
[naive] it     1/2500  loss 0.0712  psnr  16.29 dB  N=5000     0.5s
...
```

结束后打印对比表，并写入 `synthetic/results/metrics.json`。

**单次参考结果**（seed=0）：

| 配置 | PSNR 全图 | PSNR 静态区 | PSNR 动态区 | SSIM | 残影 | 耗时 |
|---|---|---|---|---|---|---|
| naive | 22.585 | 30.777 | 10.307 | 0.9069 | 0.0816 | 441 s |
| dyn-mask | 21.103 | **34.351** | 8.295 | 0.9185 | **0.0048** | 441 s |
| no-dyn-data | 22.271 | 31.444 | 9.937 | 0.9077 | 0.0642 | 441 s |

### 步骤 2 三次重复验证（3 种子 × 3 配置 = 9 次，约 66 分钟）

```bash
python synthetic/code/run_multiseed.py
```

种子固定为 `[0, 1, 2]`，超参与步骤 1 完全一致。若 `synthetic/results/out_seeds/seedN/metrics.json` 已存在则自动跳过，可断点续跑。

输出：

- `synthetic/results/out_seeds/seed{0,1,2}/metrics.json`
- `synthetic/results/out_seeds/summary.json`（跨种子 mean ± std、5 项稳定性判定）
- `synthetic/results/out_seeds/report.md`（人类可读）

**判定标准**：五个核心结论必须在 3/3 个种子上全部成立，否则该结论视为不稳健，不得写入论文。

### 步骤 3 生成定性对比图

```bash
python synthetic/code/render_compare.py
```

输出 4 行 × 3 列对照图（GT / A / B / C × 第 0、6、11 帧）。

### 步骤 4 生成汇报图表

```bash
python synthetic/code/make_figs.py
```

输出 5 张图到 `synthetic/figures/`。其中 `fig_repro_curve.png` 需要 `synthetic/results/hist_*.npy`；若只跑了步骤 2，曲线图可用 `synthetic/results/out_seeds/` 的数据替代绘制。

## 四、常见问题

**Q：结果与表中数字差几个百分点？**
A：不同 GPU 架构的浮点累加顺序不同，PSNR 可能出现 ±0.05 dB 级差异。但**结论排序（B 的静态区最高、残影最低）不应改变**——这正是步骤 2 要验证的。

**Q：能换成真实数据集（TUM / Replica）吗？**
A：本仓库刻意不依赖外部数据集，目的是让"掩码加权损失 vs 初始化滤波"这个机制在**完全可控的真值条件**下被单独验证。真实数据集上的端到端结果见学位论文正文与 Dy3DGS-SLAM 论文；本仓库的结果**不与论文正文的 PSNR 数字横向比较**（场景、分辨率、点数、迭代数均不同）。

**Q：为什么全图 PSNR 反而是 B 最低？**
A：这是设计使然。B 主动放弃拟合动态像素（损失权重为 0），因此动态区 PSNR 必然低，全图指标被拖低。**判据应看静态区 PSNR 与动态残影**，而不是全图 PSNR。这一点在论文与答辩中必须主动说明，否则容易被误读为"方法更差"。
