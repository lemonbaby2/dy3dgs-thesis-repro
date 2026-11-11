# 动态环境三维高斯语义建图 — 复现与验证仓库

本仓库是硕士学位论文《动态环境下视觉语言模型辅助三维高斯语义建图与机器人目标导航》的**可复现实验资产**，包含两条互相独立的证据链：

| 部分 | 回答什么问题 | 数据来源 | 规模 |
|---|---|---|---|
| `official/` | 公开数据集上，3DGS / 4DGS 官方实现能否在本机跑通并达到可核对的指标 | NeRF Synthetic、D-NeRF、Tanks & Temples（公开） | 3DGS 30k iter / 4DGS 20k iter / Truck 7k iter |
| `synthetic/` | **动态掩码加权损失** 与 **动态点初始化滤波** 各自贡献多少，两者是否可互相替代 | 自建合成动态场景（真值位姿、深度、动态掩码全已知） | 3 配置 × 3 随机种子 = 9 次训练 |

两部分的数字**不可横向比较**（场景、分辨率、迭代数、评价划分完全不同），各自回答不同层级的问题。

---

## 一、公开数据集复现结果（`official/`）

官方实现：`graphdeco-inria/gaussian-splatting` @ `54c035f`，`hustvl/4DGaussians` @ `843d5ac`。

| 算法 | 场景 | 迭代 | PSNR | SSIM | LPIPS-VGG | 高斯数 | 峰值显存 | 耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3DGS | NeRF Synthetic Lego（100 held-out val） | 30,000 | **37.41 dB** | 0.9861 | 0.0130 | 297,556 | 6,955 MiB | 18.2 min |
| 4DGS | D-NeRF Bouncing Balls（20 test） | 20,000 | **40.65 dB** | 0.9943 | 0.0149 | 27,864 | 4,459 MiB | 13.6 min |
| 3DGS | Tanks & Temples Truck（32 held-out test） | 7,000 | 23.98 dB | 0.8529 | 0.1950 | 1,684,067 | 7,404 MiB | 10.6 min |

> 第三行是**官方真实场景管线的补充验证**，仅训到 7k checkpoint，不等同于 30k 主结果，不得当成论文数字引用。

### 预处理消融（同一场景，仅改预处理）

| 变体 | PSNR | 结论 |
|---|---:|---|
| Raw RGBA hidden-RGB + 白底 | 2.38 dB | 失败 |
| Raw RGBA hidden-RGB + 黑底 | 2.82 dB | 失败 |
| **确定性 alpha 合成到白底** | **37.41 dB** | 正确做法 |

预处理错了，PSNR 差 **13 倍**。这是本仓库最容易被忽略、也最容易踩的坑。

硬件：RTX 4060 8GB（Windows 10），PyTorch 2.1.2+cu118。
完整指标：`official/metrics/experiment_summary.json`、`official/metrics/preprocessing_ablation.json`。
交付验收：`official/metrics/final_verification.json`（41 项检查，全通过，2026-07-30）。

---

## 二、机制验证结果（`synthetic/`）

合成动态场景（12 帧、160×120、含运动球体、动态像素占比 2.8%–9.5%），真值位姿/深度/掩码全部已知，因此可以**把"掩码加权损失"这一个变量单独隔离出来**。

三个配置：

| 配置 | 初始化 | 损失 |
|---|---|---|
| A `naive` | 全部像素反投影 | 全图 L1 |
| B `dyn-mask` | 仅静态像素反投影 | **掩码加权 L1**（动态像素权重 0） |
| C `no-dyn-data` | 仅静态像素反投影 | 全图 L1 |

3 次重复（种子 0/1/2）的均值 ± 标准差，2500 iter，5000 高斯基元：

| 配置 | PSNR 全图 | **PSNR 静态区** | 动态残影 |
|---|---:|---:|---:|
| A 朴素 | 22.53 ± 0.14 | 30.90 ± 0.42 | 0.0795 ± 0.0027 |
| **B 掩码加权** | 21.11 ± 0.01 | **34.42 ± 0.09** | **0.0049 ± 0.0001** |
| C 仅初始化滤波 | 22.45 ± 0.16 | 31.52 ± 0.66 | 0.0719 ± 0.0082 |

**核心结论**：B 比 A 静态区高 3.52 dB、残影降至 1/16；但只做初始化滤波（C）残影仍有 0.0719，说明**掩码加权损失不可替代，两个环节缺一不可**。三个种子上该排序（B > C > A）**无一例外**。

> ⚠️ **读图须知**：B 的全图 PSNR 最低是设计使然——它主动放弃拟合动态像素。判据必须看静态区 PSNR 与动态残影，否则会把方法读成"更差"。

### 重复验证：这些数字能不能信

| 判据 | 数值 | 含义 |
|---|---|---|
| 官方 3DGS 三次 7k 训练 PSNR | 34.8445 ± **0.0187** dB（极差 0.035） | 代码在本机的固有波动 |
| 合成场景静态区提升 | **+3.52 dB**（30.90±0.42 → 34.42±0.09） | 是固有波动的 **188 倍** |
| 合成场景残影 | 0.0795 → 0.0049（**1/16**） | 量级变化，非噪声 |
| 稳定性判定 | **5/5 成立，每项 3/3 命中** | 结论不依赖随机种子 |

**本机上 < 0.02 dB 的 PSNR 差异与噪声无法区分**，不得解释为方法改进。三配置的静态区区间互不重叠（C 最好 32.21 < B 最差 34.35），比"均值略高"强得多。明细见 `verification/README.md`。

---

## 三、快速开始

```bash
# 机制验证（无需任何外部数据集，纯 PyTorch，不编译 CUDA 扩展）
cd synthetic/code
python train_3dgs.py --iters 2500 --points 5000     # 单次三配置，约 22 min
python run_multiseed.py                              # 3 种子 × 3 配置，约 66 min

# 公开数据集复现：需按 official/README 准备官方仓库与数据集
```

环境：Python ≥ 3.10、PyTorch ≥ 2.0（CUDA）。详见 `REPRODUCE.md`。

## 四、目录

```
├── official/                 公开数据集复现（指标、脚本说明、精选视频）
│   ├── metrics/              experiment_summary / preprocessing_ablation / final_verification
│   ├── videos/               GT-vs-render 对比、4DGS 连续时间、RViz2、Gazebo
│   └── figures/
├── synthetic/                机制验证（自包含，可直接跑）
│   ├── code/                 scene_dyn / train_3dgs / run_multiseed / render_compare / make_figs
│   │   └── data_dyn/         12 帧合成场景真值数据
│   ├── results/              归档指标
│   └── figures/              论文与汇报用图
├── verification/             三次重复验证报告与跨种子统计
└── docs/                     全过程记录、环境修复说明
```

## 五、这个仓库不宣称什么

1. **不宣称复现了论文 SOTA**。`synthetic/` 的数字与论文正文的 35.2 dB 不可比（场景、分辨率、点数、迭代数均不同）。
2. **不宣称完成 LiDAR SLAM 精度实验**。`official/` 的 RViz2/Gazebo 视频只证明高斯数据能通过 ROS2 发布并可视化，**不构成** ATE/RPE 精度证据。
3. **不宣称 D-NeRF 上的 4DGS 结果等于动态 SLAM 结果**。前者是动态新视角合成，后者是动态环境下的位姿估计与建图，任务不同。

## 六、许可

- 代码 `synthetic/code/`、`official/` 脚本 — MIT
- 数据、指标、图表 — CC BY 4.0

引用见 `CITATION.cff`。
