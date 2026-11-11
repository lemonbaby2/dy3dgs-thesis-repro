# -*- coding: utf-8 -*-
"""
scene_dyn.py — 合成动态室内场景 + 解析光线投射真值渲染

用途：为 3D Gaussian Splatting 复刻实验提供带真值位姿 / 深度 / 动态物体掩码的多视图数据。
设计要点：
  * 场景由解析几何体构成（地面、墙、3 个静态盒体、1 个运动球体）
  * 光线投射求交，得到精确深度与物体 ID -> 动态掩码可作为"完美监督"的上界
  * 相机沿轨迹推进，12 帧，内参固定，位姿为真值（无需 SfM）

作者：小钛   日期：2026-11-11
"""

import os
import json
import math
import numpy as np
from dataclasses import dataclass, field

W, H = 160, 120
FX = FY = 150.0
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
NEAR, FAR = 0.15, 40.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_dyn")


# --------------------------------------------------------------------------
# 几何体定义
# --------------------------------------------------------------------------
@dataclass
class Box:
    """轴对齐盒体"""
    name: str
    center: np.ndarray
    half: np.ndarray
    color: np.ndarray
    dynamic: bool = False

    def intersect(self, o, d):
        """slab 法；o (N,3) d (N,3) -> t (N,) 或 inf"""
        inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
        lo = (self.center - self.half - o) * inv
        hi = (self.center + self.half - o) * inv
        tmin = np.minimum(lo, hi)
        tmax = np.maximum(lo, hi)
        t_near = np.max(tmin, axis=1)
        t_far = np.min(tmax, axis=1)
        hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near > NEAR)
        t = np.where(hit, t_near, np.inf)
        return t, np.tile(self.color, (o.shape[0], 1))


@dataclass
class Plane:
    """轴对齐平面，axis=0/1/2 表示法向轴，value 为坐标，side=+1 取正侧"""
    name: str
    axis: int
    value: float
    color: np.ndarray
    lo: np.ndarray = field(default_factory=lambda: np.array([-99.0, -99.0, -99.0]))
    hi: np.ndarray = field(default_factory=lambda: np.array([99.0, 99.0, 99.0]))

    def intersect(self, o, d):
        dd = d[:, self.axis]
        safe = np.where(np.abs(dd) < 1e-9, 1e-9, dd)
        t = (self.value - o[:, self.axis]) / safe
        p = o + t[:, None] * d
        ok = (t > NEAR) & (np.abs(dd) > 1e-9)
        for a in range(3):
            if a == self.axis:
                continue
            ok &= (p[:, a] >= self.lo[a]) & (p[:, a] <= self.hi[a])
        return np.where(ok, t, np.inf), np.tile(self.color, (o.shape[0], 1))


@dataclass
class Sphere:
    name: str
    radius: float
    color: np.ndarray
    dynamic: bool = True
    center_traj: list = None      # 每帧中心

    def intersect(self, o, d, frame=0):
        c = self.center_traj[min(frame, len(self.center_traj) - 1)]
        oc = o - c
        a = np.sum(d * d, axis=1)
        b = 2.0 * np.sum(oc * d, axis=1)
        cc = np.sum(oc * oc, axis=1) - self.radius ** 2
        disc = b * b - 4 * a * cc
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)
        t = np.where(t0 > NEAR, t0, t1)
        ok &= t > NEAR
        return np.where(ok, t, np.inf), np.tile(self.color, (o.shape[0], 1))


def build_scene():
    """构建静态场景 + 动态物体轨迹"""
    static = [
        Plane("floor", 1, 0.0, np.array([0.46, 0.45, 0.43]),
              lo=np.array([-4.0, -99, -3.0]), hi=np.array([4.0, 99, 3.0])),
        Plane("wall_left", 0, -4.0, np.array([0.72, 0.73, 0.70]),
              lo=np.array([-99, 0.0, -3.0]), hi=np.array([99, 2.8, 3.0])),
        Plane("wall_back", 2, -3.0, np.array([0.66, 0.67, 0.72]),
              lo=np.array([-4.0, 0.0, -99]), hi=np.array([4.0, 2.8, 99])),
        Box("cabinet", np.array([-1.25, 0.40, -1.00]), np.array([0.40, 0.40, 0.40]),
            np.array([0.72, 0.24, 0.20])),
        Box("table", np.array([1.05, 0.55, -1.85]), np.array([0.55, 0.55, 0.38]),
            np.array([0.22, 0.34, 0.66])),
        Box("crate", np.array([0.00, 0.32, -2.45]), np.array([0.60, 0.32, 0.28]),
            np.array([0.24, 0.58, 0.30])),
        Box("pillar", np.array([-2.55, 1.10, -2.20]), np.array([0.22, 1.10, 0.22]),
            np.array([0.58, 0.56, 0.32])),
    ]
    # 动态球：模拟行人在通道中横穿（第 0 帧在左，第 11 帧到右），全程保持在相机前方
    traj = []
    for i in range(12):
        s = i / 11.0
        traj.append(np.array([-0.95 + 1.90 * s,
                              0.95 + 0.05 * math.sin(s * math.pi * 3),
                              -1.70 + 0.15 * math.cos(s * math.pi * 2)]))
    dyn = Sphere("walker", 0.40, np.array([0.85, 0.52, 0.12]), dynamic=True, center_traj=traj)
    return static, dyn


def camera_traj():
    """相机沿 z 轴推进并轻微摆动，返回外参（世界->相机）"""
    poses = []
    for i in range(12):
        s = i / 11.0
        # 位置
        cx = 0.10 * math.sin(s * math.pi * 1.2) - 0.05
        cy = 1.20 + 0.03 * math.sin(s * math.pi * 2.0)
        cz = 2.80 - 2.00 * s
        # 朝向：看向场景深处，带一点横摆
        yaw = 0.05 * math.sin(s * math.pi * 1.0)
        pitch = -0.04
        fwd = np.array([math.sin(yaw) * math.cos(pitch),
                        math.sin(pitch),
                        -math.cos(yaw) * math.cos(pitch)])
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        # 相机坐标系：x 右, y 下, z 前
        R_wc = np.stack([right, down, fwd], axis=0)      # (3,3) world->cam
        t_wc = -R_wc @ np.array([cx, cy, cz])
        poses.append((R_wc, t_wc))
    return poses


def rays(H, W, fx, fy, cx, cy, R_wc, t_wc):
    """生成世界坐标光线"""
    v, u = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    xc = (u - cx) / fx
    yc = (v - cy) / fy
    d_cam = np.stack([xc, yc, np.ones_like(xc)], axis=-1).reshape(-1, 3)
    # 相机->世界
    R_cw = R_wc.T
    o_w = -R_cw @ t_wc
    d_w = d_cam @ R_wc          # 等价于 (R_cw @ d_cam^T)^T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    o = np.tile(o_w.reshape(1, 3), (d_w.shape[0], 1))
    return o, d_w


def render_frame(static, dyn, R_wc, t_wc, frame, with_dynamic=True):
    """解析光线投射：返回 RGB(H,W,3) / depth(H,W) / object_id(H,W)"""
    o, d = rays(H, W, FX, FY, CX, CY, R_wc, t_wc)
    n = o.shape[0]
    best_t = np.full(n, np.inf)
    best_rgb = np.zeros((n, 3))
    best_id = np.full(n, -1, dtype=np.int32)

    objs = list(static) + ([dyn] if with_dynamic else [])

    for oid, obj in enumerate(objs):
        if obj is dyn:
            t, col = dyn.intersect(o, d, frame)
            idx_tag = 100
        else:
            t, col = obj.intersect(o, d)
            idx_tag = oid
        m = t < best_t
        best_t = np.where(m, t, best_t)
        best_rgb = np.where(m[:, None], col, best_rgb)
        best_id = np.where(m, idx_tag, best_id)

    # 球面着色：给球加一点明暗，避免纯色
    if with_dynamic:
        c = dyn.center_traj[min(frame, 11)]
        p = o + best_t[:, None] * d
        nrm = (p - c) / np.maximum(np.linalg.norm(p - c, axis=1, keepdims=True), 1e-6)
        light = np.clip(nrm @ np.array([0.4, 0.8, 0.45]) / np.linalg.norm([0.4, 0.8, 0.45]), 0, 1)
        shade = 0.55 + 0.45 * light
        m = best_id == 100
        best_rgb = np.where(m[:, None], np.tile(dyn.color, (n, 1)) * shade[:, None], best_rgb)

    depth = np.where(np.isfinite(best_t), best_t, 0.0).reshape(H, W)
    rgb = np.clip(best_rgb, 0, 1).reshape(H, W, 3)
    oid_map = best_id.reshape(H, W)
    return rgb, depth, oid_map


def export(seed=0):
    os.makedirs(OUT, exist_ok=True)
    static, dyn = build_scene()
    poses = camera_traj()
    rng = np.random.default_rng(seed)

    frames = []
    for i, (R_wc, t_wc) in enumerate(poses):
        rgb, depth, oid = render_frame(static, dyn, R_wc, t_wc, i, with_dynamic=True)
        rgb_clean, depth_clean, oid_clean = render_frame(static, dyn, R_wc, t_wc, i, with_dynamic=False)
        np.save(os.path.join(OUT, f"rgb_{i:03d}.npy"), rgb.astype(np.float32))
        np.save(os.path.join(OUT, f"depth_{i:03d}.npy"), depth.astype(np.float32))
        np.save(os.path.join(OUT, f"dynmask_{i:03d}.npy"), (oid == 100).astype(np.uint8))
        # 静态背景（用于评估动态移除效果）
        np.save(os.path.join(OUT, f"gt_clean_{i:03d}.npy"), rgb_clean.astype(np.float32))
        frames.append({
            "frame": i,
            "R_wc": R_wc.tolist(),
            "t_wc": t_wc.tolist(),
            "dyn_ratio": float((oid == 100).mean()),
            "depth_min": float(depth[depth > 0].min()),
            "depth_max": float(depth.max()),
        })

    meta = {
        "W": W, "H": H, "fx": FX, "fy": FY, "cx": CX, "cy": CY,
        "near": NEAR, "far": FAR, "frames": len(poses),
        "n_static_objects": len(static),
        "dynamic_object": dyn.name,
        "dynamic_radius": dyn.radius,
        "dynamic_traj": [c.tolist() for c in dyn.center_traj],
        "per_frame": frames,
        "generator": "analytic ray casting, axis-aligned boxes/planes/sphere",
        "note": "真值位姿与深度由解析求交给出，动态掩码为物体 ID 完美分割（上界监督）",
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[scene] exported {len(poses)} frames -> {OUT}")
    print(f"[scene] dyn ratio per frame: "
          + ", ".join(f"{f['dyn_ratio']*100:.1f}%" for f in frames))
    return meta


if __name__ == "__main__":
    export()
