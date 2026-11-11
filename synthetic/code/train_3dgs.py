# -*- coding: utf-8 -*-
"""
train_3dgs.py — 从零实现的可微 3D Gaussian Splatting（纯 PyTorch，无 CUDA 扩展）

复刻目标：动态环境下三维高斯语义建图的最小可用内核，用于中期汇报的实测证据。
包含三个可比配置：
    A naive        —— 全部像素参与优化（动态物体会产生鬼影 / 拖影）
    B dyn-mask     —— 用动态掩码剔除动态像素（论文主张的"动态掩码融合"）
    C no-dyn-data  —— 仅用不含动态物体的帧（理想上界对照组）

指标：PSNR / SSIM（全图、静态区、动态区）+ 训练耗时 + 高斯点数
产物：训练曲线、渲染对比图、点云 PLY、metrics.json

作者：小钛   日期：2026-11-11
"""

import os
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_dyn")
OUTDIR = os.path.join(HERE, "out")
os.makedirs(OUTDIR, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def quat_to_rot(q):
    """(N,4) wxyz -> (N,3,3)"""
    q = q / (q.norm(dim=1, keepdim=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def ssim_t(x, y, win=11, as_tensor=False):
    """x,y: (1,1,H,W) in [0,1]. as_tensor=True 时返回张量（训练用，保留梯度）"""
    pad = win // 2
    mu1 = F.avg_pool2d(x, win, 1, pad)
    mu2 = F.avg_pool2d(y, win, 1, pad)
    s1 = F.avg_pool2d(x * x, win, 1, pad) - mu1 ** 2
    s2 = F.avg_pool2d(y * y, win, 1, pad) - mu2 ** 2
    s12 = F.avg_pool2d(x * y, win, 1, pad) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    v = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    m = v.mean()
    return m if as_tensor else float(m.detach())


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_scene():
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    W, H = meta["W"], meta["H"]
    frames = []
    for i in range(meta["frames"]):
        rgb = np.load(os.path.join(DATA, f"rgb_{i:03d}.npy"))
        dep = np.load(os.path.join(DATA, f"depth_{i:03d}.npy"))
        dm = np.load(os.path.join(DATA, f"dynmask_{i:03d}.npy"))
        gt_clean = np.load(os.path.join(DATA, f"gt_clean_{i:03d}.npy"))
        R = np.array(meta["per_frame"][i]["R_wc"], dtype=np.float32)
        t = np.array(meta["per_frame"][i]["t_wc"], dtype=np.float32)
        frames.append({
            "rgb": torch.from_numpy(rgb).float().to(DEV),
            "depth": torch.from_numpy(dep).float().to(DEV),
            "dyn": torch.from_numpy(dm).float().to(DEV),
            "clean": torch.from_numpy(gt_clean).float().to(DEV),
            "R": torch.from_numpy(R).float().to(DEV),
            "t": torch.from_numpy(t).float().to(DEV),
        })
    K = torch.tensor([[meta["fx"], 0, meta["cx"]],
                      [0, meta["fy"], meta["cy"]],
                      [0, 0, 1]], dtype=torch.float32, device=DEV)
    return meta, K, frames


def backproject(frames, K, max_points, exclude_dynamic=True, rng=None):
    """用真值深度把像素反投影为世界点（取代 SfM），作为高斯初始化"""
    fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
    pts, cols = [], []
    for f in frames:
        dep, rgb, dm = f["depth"], f["rgb"], f["dyn"]
        H, W = dep.shape
        v, u = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
        m = dep > 0.1
        if exclude_dynamic:
            m &= dm < 0.5
        z = dep[m]
        x = (u[m].float() - cx) / fx * z
        y = (v[m].float() - cy) / fy * z
        pc = torch.stack([x, y, z], dim=1)
        Rw, tw = f["R"], f["t"]
        pw = (pc - tw) @ Rw          # R^T (p - t)
        pts.append(pw)
        cols.append(rgb[m])
    pts = torch.cat(pts, 0)
    cols = torch.cat(cols, 0)
    n = pts.shape[0]
    if n > max_points:
        idx = torch.randperm(n, device=DEV)[:max_points] if rng is None else \
            torch.from_numpy(rng.permutation(n)[:max_points]).to(DEV)
        pts, cols = pts[idx], cols[idx]
    return pts.contiguous(), cols.contiguous()


# --------------------------------------------------------------------------
# 高斯模型 + 可微光栅化
# --------------------------------------------------------------------------
class Gaussians(nn.Module):
    def __init__(self, pts, cols, scale_init=0.06):
        super().__init__()
        n = pts.shape[0]
        self.means = nn.Parameter(pts.clone())
        self.log_scales = nn.Parameter(torch.log(torch.full((n, 3), scale_init, device=DEV)))
        q = torch.zeros(n, 4, device=DEV)
        q[:, 0] = 1.0
        self.quats = nn.Parameter(q)
        self.logit_opac = nn.Parameter(torch.full((n, 1), 2.5, device=DEV))
        c = torch.clamp(cols.clone(), 1e-3, 1 - 1e-3)
        self.color_logit = nn.Parameter(torch.log(c / (1 - c)))

    def params_lr(self):
        return [
            {"params": [self.means], "lr": 1.6e-4},
            {"params": [self.log_scales], "lr": 3.0e-3},
            {"params": [self.quats], "lr": 1.0e-3},
            {"params": [self.logit_opac], "lr": 3.0e-3},
            {"params": [self.color_logit], "lr": 1.0e-2},
        ]


def rasterize(model, R_wc, t_wc, K, H, W, bg=0.0):
    means = model.means
    p_cam = means @ R_wc.t() + t_wc
    z = p_cam[:, 2]
    zc = torch.clamp(z, min=1e-3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * p_cam[:, 0] / zc + cx
    v = fy * p_cam[:, 1] / zc + cy

    Rq = quat_to_rot(model.quats)
    S = torch.exp(model.log_scales)
    M = Rq * S.unsqueeze(1)
    Sigma = M @ M.transpose(1, 2)

    n = means.shape[0]
    J = torch.zeros(n, 2, 3, device=DEV, dtype=means.dtype)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * p_cam[:, 0] / zc ** 2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * p_cam[:, 1] / zc ** 2
    T = J @ R_wc
    cov = T @ Sigma @ T.transpose(1, 2)

    cov = cov + torch.eye(2, device=DEV).unsqueeze(0) * 0.30      # 抗锯齿低通
    det = torch.clamp(cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2, min=1e-6)
    inv00 = cov[:, 1, 1] / det
    inv01 = -cov[:, 0, 1] / det
    inv11 = cov[:, 0, 0] / det

    gv, gu = torch.meshgrid(torch.arange(H, device=DEV, dtype=means.dtype),
                            torch.arange(W, device=DEV, dtype=means.dtype), indexing="ij")
    px = gu.reshape(-1)
    py = gv.reshape(-1)
    dx = px.unsqueeze(0) - u.unsqueeze(1)
    dy = py.unsqueeze(0) - v.unsqueeze(1)
    power = -0.5 * (inv00.unsqueeze(1) * dx * dx
                    + 2 * inv01.unsqueeze(1) * dx * dy
                    + inv11.unsqueeze(1) * dy * dy)
    power = torch.clamp(power, max=0.0)
    alpha = torch.sigmoid(model.logit_opac) * torch.exp(power)
    valid = (z > 0.05).float().unsqueeze(1)
    alpha = torch.clamp(alpha * valid, 0.0, 0.985)

    order = torch.argsort(z.detach())
    a = alpha[order]
    c = torch.sigmoid(model.color_logit)[order]
    Tp = torch.cumprod(1.0 - a + 1e-6, dim=0)
    Tprev = torch.cat([torch.ones(1, H * W, device=DEV), Tp[:-1]], dim=0)
    w = a * Tprev
    img = torch.einsum("nh,nc->hc", w, c)
    acc = w.sum(0)
    img = img + (1.0 - acc).unsqueeze(-1) * bg
    img = img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)
    return img, acc.reshape(H, W)


def render_frame_full(model, K, H, W, bg=0.0):
    return None  # 占位，渲染统一走 evaluate


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def train(mode, frames, K, meta, iters=3000, max_points=4000, seed=0, log_every=150):
    H, W = meta["H"], meta["W"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    exclude_dyn_init = mode != "naive"
    if mode == "no-dyn-data":
        use_frames = [i for i, f in enumerate(frames) if f["dyn"].mean() < 0.005]
    else:
        use_frames = list(range(len(frames)))
    if len(use_frames) < 4:
        use_frames = list(range(len(frames)))

    pts, cols = backproject([frames[i] for i in use_frames], K, max_points,
                            exclude_dynamic=exclude_dyn_init, rng=rng)
    # 初始高斯半径：经验尺度（体积估计会被远处稀疏点撑大，实测 0.05 最稳）
    scale0 = 0.05
    print(f"[init] points={pts.shape[0]} scale0={scale0}")
    model = Gaussians(pts, cols, scale_init=scale0).to(DEV)
    opt = torch.optim.Adam(model.params_lr(), eps=1e-15)

    lr_init = 1.6e-4
    lr_final = 1.6e-6
    hist = []
    t0 = time.time()
    for it in range(1, iters + 1):
        frac = min(it / max(iters, 1), 1.0)
        opt.param_groups[0]["lr"] = lr_init * ((lr_final / lr_init) ** frac)
        fi = int(rng.integers(0, len(use_frames)))
        f = frames[use_frames[fi]]
        img, acc = rasterize(model, f["R"], f["t"], K, H, W)
        gt = f["rgb"].permute(2, 0, 1).unsqueeze(0)

        if mode == "dyn-mask":
            m = (f["dyn"] < 0.5).unsqueeze(0).unsqueeze(0).float()
            loss = (torch.abs(img - gt) * m).sum() / (m.sum() * 3 + 1e-6)
        else:
            loss = torch.abs(img - gt).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        with torch.no_grad():
            model.log_scales.data.clamp_(math.log(0.005), math.log(1.2))
            model.logit_opac.data.clamp_(-6.0, 6.0)

        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                img_v, _ = rasterize(model, f["R"], f["t"], K, H, W)
            p = psnr(img_v.clamp(0, 1), gt.clamp(0, 1))
            hist.append({"iter": it, "loss": float(loss.item()), "psnr": p,
                         "n_gauss": int(model.means.shape[0])})
            print(f"[{mode}] it {it:5d}/{iters}  loss {loss.item():.4f}  "
                  f"psnr {p:6.2f} dB  N={model.means.shape[0]}  {time.time()-t0:6.1f}s", flush=True)

    train_time = time.time() - t0
    return model, hist, use_frames, train_time


@torch.no_grad()
def evaluate(model, frames, K, meta, split_frames=None):
    H, W = meta["H"], meta["W"]
    rows = []
    for i, f in enumerate(frames):
        img, acc = rasterize(model, f["R"], f["t"], K, H, W)
        img = img.clamp(0, 1)
        gt = f["rgb"].permute(2, 0, 1).unsqueeze(0)
        clean = f["clean"].permute(2, 0, 1).unsqueeze(0)
        m_stat = (f["dyn"] < 0.5).unsqueeze(0).unsqueeze(0).float()
        m_dyn = 1.0 - m_stat
        # 全图 / 静态区 / 动态区 PSNR（对含动态的 GT）
        def masked_psnr(a, b, m):
            den = m.sum() * 3
            if den < 1:
                return None
            mse = (torch.abs(a - b) ** 2 * m).sum() / den
            return 10.0 * math.log10(1.0 / max(float(mse), 1e-10))
        p_all = psnr(img[0].permute(1, 2, 0).cpu().numpy(), f["rgb"].cpu().numpy())
        p_stat = masked_psnr(img, gt, m_stat)
        p_dyn = masked_psnr(img, gt, m_dyn)
        # 对"无动态干净背景"的误差 -> 动态残影强度
        resid = float((torch.abs(img - clean) * m_dyn).sum() / (m_dyn.sum() * 3 + 1e-6))
        s_all = ssim_t(img, gt)
        rows.append({"frame": i, "psnr_all": round(p_all, 3),
                     "psnr_static": round(p_stat, 3) if p_stat else None,
                     "psnr_dyn": round(p_dyn, 3) if p_dyn else None,
                     "ssim": round(s_all, 4),
                     "ghost_residual": round(resid, 4),
                     "dyn_ratio": round(float(f["dyn"].mean()), 4)})
    valid = [r for r in rows if r["psnr_all"] > 0]
    agg = {
        "psnr_all_mean": round(float(np.mean([r["psnr_all"] for r in valid])), 3),
        "psnr_static_mean": round(float(np.mean([r["psnr_static"] for r in valid if r["psnr_static"]])), 3),
        "psnr_dyn_mean": round(float(np.mean([r["psnr_dyn"] for r in valid if r["psnr_dyn"]])), 3),
        "ssim_mean": round(float(np.mean([r["ssim"] for r in valid])), 4),
        "ghost_residual_mean": round(float(np.mean([r["ghost_residual"] for r in valid])), 4),
    }
    return rows, agg


def save_ply(model, path):
    m = model.means.detach().cpu().numpy()
    c = (torch.sigmoid(model.color_logit).detach().cpu().numpy() * 255).astype(np.uint8)
    s = torch.exp(model.log_scales).detach().cpu().numpy()
    o = torch.sigmoid(model.logit_opac).detach().cpu().numpy()
    n = m.shape[0]
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for k in ("x", "y", "z", "red", "green", "blue"):
            f.write(f"property float {k}\n")
        for k in ("sx", "sy", "sz", "opacity"):
            f.write(f"property float {k}\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{m[i,0]:.5f} {m[i,1]:.5f} {m[i,2]:.5f} "
                    f"{c[i,0]} {c[i,1]} {c[i,2]} "
                    f"{s[i,0]:.5f} {s[i,1]:.5f} {s[i,2]:.5f} {o[i,0]:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--points", type=int, default=4000)
    ap.add_argument("--modes", type=str, default="naive,dyn-mask,no-dyn-data")
    args = ap.parse_args()

    meta, K, frames = load_scene()
    print(f"[data] {meta['frames']} frames {meta['W']}x{meta['H']}  device={DEV}")
    print(f"[gpu ] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    summary = {}
    for mode in args.modes.split(","):
        print(f"\n===== training config: {mode} =====", flush=True)
        model, hist, use_frames, tt = train(mode, frames, K, meta,
                                            iters=args.iters, max_points=args.points)
        rows, agg = evaluate(model, frames, K, meta)
        agg["train_time_s"] = round(tt, 2)
        agg["n_gauss"] = int(model.means.shape[0])
        agg["frames_used"] = len(use_frames)
        agg["mode"] = mode
        summary[mode] = {"agg": agg, "per_frame": rows}
        np.save(os.path.join(OUTDIR, f"hist_{mode}.npy"),
                np.array([[h["iter"], h["loss"], h["psnr"], h["n_gauss"]] for h in hist]))
        torch.save(model.state_dict(), os.path.join(OUTDIR, f"model_{mode}.pt"))
        save_ply(model, os.path.join(OUTDIR, f"gaussians_{mode}.ply"))
        print(f"[{mode}] AGG {json.dumps(agg, ensure_ascii=False)}", flush=True)

    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                   "iters": args.iters, "init_points": args.points,
                   "configs": summary}, f, ensure_ascii=False, indent=2)
    print("\n[done] metrics -> out/metrics.json")

    # 对比表
    print("\n| 配置 | PSNR(全图) | PSNR(静态区) | PSNR(动态区) | SSIM | 残影 | 高斯数 | 耗时(s) |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, v in summary.items():
        a = v["agg"]
        print(f"| {mode} | {a['psnr_all_mean']} | {a['psnr_static_mean']} | {a['psnr_dyn_mean']} "
              f"| {a['ssim_mean']} | {a['ghost_residual_mean']} | {a['n_gauss']} | {a['train_time_s']} |")


if __name__ == "__main__":
    main()
