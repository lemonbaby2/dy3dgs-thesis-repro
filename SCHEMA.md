# SCHEMA — 数据结构与字段定义

本仓库所有数据为**合成动态场景**下的真值数据，不含任何第三方数据集副本。

## 一、`synthetic/code/data_dyn/` 目录

由 `synthetic/code/scene_dyn.py` 生成，12 帧，分辨率 160×120。

| 文件 | 形状 / 类型 | 含义 |
|---|---|---|
| `rgb_XXX.npy` | `(120, 160, 3)` float32, [0,1] | 解析光线投射渲染的 RGB 图像（含运动球体） |
| `depth_XXX.npy` | `(120, 160)` float32, 单位 m | 逐像素真值深度 |
| `dynmask_XXX.npy` | `(120, 160)` uint8, {0,1} | 动态物体掩码，1 = 运动球体 |
| `gt_clean_XXX.npy` | `(120, 160, 3)` float32 | 剔除动态物体后的静态真值（用于静态区评估） |
| `meta.json` | dict | 相机内参、帧数、生成参数 |

`XXX` 为 `000`–`011`。

### `meta.json` 关键字段

```json
{
  "frames": 12,
  "H": 120,
  "W": 160,
  "fx": 150.0, "fy": 150.0, "cx": 80.0, "cy": 60.0,
  "dyn_ratio": [0.028, 0.033, ...]
}
```

`dyn_ratio[i]` 是第 i 帧动态像素占比，用于判断该帧是否具备有效动态干扰（本次实验 2.8%–9.5%）。

## 二、`synthetic/results/metrics.json`（单次运行）

```json
{
  "device": "NVIDIA GeForce RTX 5060 Ti",
  "iters": 2500,
  "init_points": 5000,
  "configs": {
    "naive":       { "agg": {...}, "per_frame": [...] },
    "dyn-mask":    { "agg": {...}, "per_frame": [...] },
    "no-dyn-data": { "agg": {...}, "per_frame": [...] }
  }
}
```

### `agg` 字段

| 字段 | 含义 |
|---|---|
| `psnr_all_mean` | 全图 PSNR（含动态像素） |
| `psnr_static_mean` | **静态区 PSNR**（仅 `dynmask==0` 像素）——本实验核心判据 |
| `psnr_dyn_mean` | 动态区 PSNR（越低越说明模型主动放弃拟合动态物体） |
| `ssim_mean` | 全图 SSIM |
| `ghost_residual_mean` | **动态残影强度**：动态物体扫过区域在静态重建中的残余能量，越低越好 |
| `n_gauss` | 高斯基元数量 |
| `train_time_s` | 训练耗时（秒） |
| `frames_used` | 参与优化的帧数 |

### 三个配置的语义

| 配置 | 初始化 | 损失函数 | 用途 |
|---|---|---|---|
| `naive` (A) | 全部像素反投影（含动态） | 全图 L1 | 上界对照：证明动态像素污染代价 |
| `dyn-mask` (B) | 仅静态像素反投影 | **掩码加权 L1**（动态像素权重 0） | 本文方法 |
| `no-dyn-data` (C) | 仅静态像素反投影 | 全图 L1 | 消融：只有初始化滤波、无掩码损失 |

## 三、`synthetic/results/out_seeds/`（重复验证）

```
out_seeds/
├── seed0/metrics.json      # 结构同 metrics.json，另含 seed 字段
├── seed1/metrics.json
├── seed2/metrics.json
├── summary.json            # 跨种子 mean/std + 5 项稳定性判定
└── report.md               # 人类可读报告
```

`summary.json` 的 `checks` 数组记录五个核心结论是否在每个种子上都成立：

| ID | 判定内容 |
|---|---|
| C1 | 静态区 PSNR：B > A |
| C2 | 动态残影：B 同时低于 A 与 C |
| C3 | 残影下降倍数 ≥ 10× |
| C4 | 仅初始化滤波（C）残影 > B 的 5 倍 |
| C5 | 静态区 PSNR 排序稳定为 B > C > A |

## 四、许可

- `synthetic/code/` — MIT
- `synthetic/data/`、`synthetic/results/`、`synthetic/figures/` — CC BY 4.0
