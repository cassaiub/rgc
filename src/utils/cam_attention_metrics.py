"""
Grad-CAM localization metrics over NPZ exports under all_cam_output.

Resizes radio RGB and heatmaps to TARGET (default 150) before any metric.
Caches per-file rows to CSV for fast replotting.
"""
from __future__ import annotations

import json
import os
import re
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from scipy.ndimage import uniform_filter, zoom

CSV_FIELDS = [
    "path",
    "stem",
    "family",
    "arch",
    "spurious",
    "fold",
    "grad_cam",
    "run_dir",
    "group_key",
    "precision_top20",
    "recall_top20",
    "entropy_norm",
    "ssim_hm_vs_radio_mask",
    "peak_dist_norm",
    "spread_l2",
    "pr_prec_json",
    "pr_rec_json",
    "n_sigma",
]

TARGET = 150
# Top fraction of CAM activation to sweep for PR curves (high = stricter / smaller mask)
CAM_TOP_FRACS = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])


def resize_to_target(arr: np.ndarray, size: int = TARGET) -> np.ndarray:
    """Resize (H,W) or (H,W,C) array to (size,size[,C]) using linear interpolation."""
    a = np.asarray(arr)
    if a.ndim == 2:
        h, w = a.shape
        if (h, w) == (size, size):
            return a.astype(np.float64, copy=False)
        zh, zw = size / h, size / w
        return zoom(a.astype(np.float64), (zh, zw), order=1)
    if a.ndim == 3:
        h, w, c = a.shape
        if (h, w) == (size, size):
            return a.astype(np.float64, copy=False)
        zh, zw = size / h, size / w
        return zoom(a.astype(np.float64), (zh, zw, 1), order=1)
    raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}")


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = np.asarray(rgb, dtype=np.float64)
    if r.ndim == 2:
        return r
    return 0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]


def load_gray_heatmap_150(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray 150x150, heatmap 150x150 positive)."""
    gray = rgb_to_gray(resize_to_target(npz["image_rgb"], TARGET))
    if "heatmap_150" in npz.files:
        hm = np.asarray(npz["heatmap_150"], dtype=np.float64)
        if hm.shape != (TARGET, TARGET):
            hm = resize_to_target(hm, TARGET)
    else:
        hm = np.asarray(npz["heatmap_native"], dtype=np.float64)
        hm = resize_to_target(hm, TARGET)
    hm = np.maximum(hm, 0.0)
    return gray, hm


def radio_mask_sigma(gray: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    mu = float(np.mean(gray))
    sig = float(np.std(gray))
    if sig <= 1e-12:
        return np.zeros_like(gray, dtype=bool)
    return gray > (mu + n_sigma * sig)


def cam_mask_top_fraction(hm: np.ndarray, top_frac: float) -> np.ndarray:
    """Pixels in the top `top_frac` mass by value (same count as top_frac * N)."""
    top_frac = float(np.clip(top_frac, 1e-6, 1.0))
    thr = np.quantile(hm, 1.0 - top_frac)
    return hm >= thr


def localization_pr(cam: np.ndarray, src: np.ndarray) -> tuple[float, float]:
    cam = cam.astype(bool)
    src = src.astype(bool)
    inter = np.logical_and(cam, src).sum()
    c = cam.sum()
    s = src.sum()
    if c == 0 or s == 0:
        return np.nan, np.nan
    return inter / c, inter / s


def shannon_entropy_norm(hm: np.ndarray, eps: float = 1e-12) -> float:
    p = hm.reshape(-1).astype(np.float64)
    p = p + eps
    p = p / p.sum()
    h = float(-np.sum(p * np.log(p)))
    n = p.size
    return h / np.log(n) if n > 1 else 0.0


def ssim_gaussian_window(
    img1: np.ndarray, img2: np.ndarray, data_range: float | None = None, win: int = 11
) -> float:
    """Mean SSIM map (Wang et al.) using Gaussian weights approximated with uniform_filter."""
    x = np.asarray(img1, dtype=np.float64)
    y = np.asarray(img2, dtype=np.float64)
    if data_range is None:
        data_range = max(x.max() - x.min(), y.max() - y.min(), 1e-12)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ux = uniform_filter(x, win)
    uy = uniform_filter(y, win)
    uxx = uniform_filter(x * x, win)
    uyy = uniform_filter(y * y, win)
    uxy = uniform_filter(x * y, win)
    vx = np.clip(uxx - ux * ux, 0, None)
    vy = np.clip(uyy - uy * uy, 0, None)
    vxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    m = num / np.maximum(den, 1e-12)
    return float(np.mean(m))


def peak_distance_and_spread(hm: np.ndarray, src: np.ndarray) -> tuple[float, float, float, float]:
    """
    Peak location = argmax of heatmap. Reference = center of mass of source mask.
    Spread = sqrt(var_x + var_y) with weights = normalized heatmap on the grid.
    Returns (dist_norm, spread_combined, spread_x, spread_y) with dist_norm in [0,1] if peak exists.
    """
    h, w = hm.shape
    yy, xx = np.indices((h, w))
    hm = np.maximum(hm.astype(np.float64), 0.0)
    s = hm.sum()
    if s <= 1e-18:
        return np.nan, np.nan, np.nan, np.nan
    wts = hm / s
    mx = float(np.sum(wts * xx))
    my = float(np.sum(wts * yy))
    var_x = float(np.sum(wts * (xx - mx) ** 2))
    var_y = float(np.sum(wts * (yy - my) ** 2))
    spread = float(np.sqrt(var_x + var_y))

    if not np.any(src):
        return np.nan, spread, float(np.sqrt(var_x)), float(np.sqrt(var_y))

    sm = float(src.sum())
    rcx = float((xx * src).sum() / sm)
    rcy = float((yy * src).sum() / sm)

    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    dist = float(np.hypot(px - rcx, py - rcy))
    diag = float(np.hypot(w - 1, h - 1))
    dist_norm = dist / diag if diag > 0 else np.nan
    return dist_norm, spread, float(np.sqrt(var_x)), float(np.sqrt(var_y))


_RUN_RE = re.compile(
    r"^(?P<arch>byol|resnet50|vgg16|convnext_base|swin_b|vit_b_16)-(?P<sp>no_spurious|spurious)_fold_(?P<fold>\d+)$"
)


@dataclass(frozen=True)
class RunMeta:
    family: str  # "byol" | "supervised"
    arch: str
    spurious: bool
    fold: int
    grad_cam: str  # "with_sp" | "without_sp"
    run_dir: str
    group_key: str  # for aggregation plots


def parse_npz_path(path: str, base: str) -> RunMeta | None:
    path = os.path.abspath(path)
    base = os.path.abspath(base)
    rel = os.path.relpath(path, base)
    parts = rel.split(os.sep)
    if len(parts) < 5:
        return None
    family_root = parts[0]
    run_dir = parts[1]
    grad = parts[2]
    if grad not in ("grad_cam_with_sp", "grad_cam_without_sp"):
        return None
    m = _RUN_RE.match(run_dir)
    if not m:
        return None
    arch = m.group("arch")
    spurious = m.group("sp") == "spurious"
    fold = int(m.group("fold"))
    if family_root == "byol":
        fam = "byol"
    elif family_root == "supervised_gradcam":
        fam = "supervised"
    else:
        return None
    arch_disp = {
        "byol": "BYOL",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
        "convnext_base": "ConvNeXt-B",
        "swin_b": "Swin-B",
        "vit_b_16": "ViT-B/16",
    }.get(arch, arch)
    sp_label = "spurious" if spurious else "no spurious"
    group_key = f"{arch_disp} · {sp_label}"
    gc = "with_sp" if grad.endswith("with_sp") else "without_sp"
    return RunMeta(
        family=fam,
        arch=arch_disp,
        spurious=spurious,
        fold=fold,
        grad_cam=gc,
        run_dir=run_dir,
        group_key=group_key,
    )


def iter_npz_paths(base: str) -> Iterator[str]:
    for root, _, files in os.walk(base):
        if os.path.basename(root) != "cam_npz":
            continue
        for f in files:
            if f.endswith(".npz"):
                yield os.path.join(root, f)


def compute_row(path: str, base: str, n_sigma: float = 3.0) -> dict | None:
    meta = parse_npz_path(path, base)
    if meta is None:
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    try:
        gray, hm = load_gray_heatmap_150(z)
    except Exception:
        return None
    src = radio_mask_sigma(gray, n_sigma=n_sigma)
    hm_n = hm.copy()
    if hm_n.max() > hm_n.min():
        hm_n = (hm_n - hm_n.min()) / (hm_n.max() - hm_n.min())
    else:
        hm_n = np.zeros_like(hm_n)

    src_f = src.astype(np.float64)
    ssim = ssim_gaussian_window(hm_n, src_f)
    ent = shannon_entropy_norm(hm_n + 1e-12)
    dist_n, spr, _, _ = peak_distance_and_spread(hm, src)

    pr_prec = []
    pr_rec = []
    for f in CAM_TOP_FRACS:
        cam = cam_mask_top_fraction(hm, f)
        p, r = localization_pr(cam, src)
        pr_prec.append(p)
        pr_rec.append(r)

    p20, r20 = localization_pr(cam_mask_top_fraction(hm, 0.20), src)

    stem = os.path.splitext(os.path.basename(path))[0]
    return {
        "path": path,
        "stem": stem,
        "family": meta.family,
        "arch": meta.arch,
        "spurious": meta.spurious,
        "fold": meta.fold,
        "grad_cam": meta.grad_cam,
        "run_dir": meta.run_dir,
        "group_key": meta.group_key,
        "precision_top20": p20,
        "recall_top20": r20,
        "entropy_norm": ent,
        "ssim_hm_vs_radio_mask": ssim,
        "peak_dist_norm": dist_n,
        "spread_l2": spr,
        "pr_prec_json": json.dumps(pr_prec),
        "pr_rec_json": json.dumps(pr_rec),
        "n_sigma": n_sigma,
    }


def build_table(
    base: str,
    limit: int | None = None,
    n_sigma: float = 3.0,
    progress: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = list(iter_npz_paths(base))
    if limit is not None:
        paths = paths[:limit]
    it = paths
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(paths, desc="cam npz")
        except Exception:
            pass
    for p in it:
        row = compute_row(p, base, n_sigma=n_sigma)
        if row is not None:
            rows.append(row)
    return rows


def save_rows_csv(rows: list[dict[str, Any]], path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in CSV_FIELDS:
                v = r.get(k, "")
                if k == "spurious":
                    out[k] = "1" if v else "0"
                else:
                    out[k] = v
            w.writerow(out)


def load_rows_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["spurious"] = row.get("spurious", "0").strip().lower() in ("true", "1", "yes")
            row["fold"] = int(row["fold"])
            row["precision_top20"] = float(row["precision_top20"])
            row["recall_top20"] = float(row["recall_top20"])
            row["entropy_norm"] = float(row["entropy_norm"])
            row["ssim_hm_vs_radio_mask"] = float(row["ssim_hm_vs_radio_mask"])
            row["peak_dist_norm"] = float(row["peak_dist_norm"])
            row["spread_l2"] = float(row["spread_l2"])
            row["n_sigma"] = float(row["n_sigma"])
            rows.append(row)
    return rows


def pr_curve_by_group(rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean precision/recall vs CAM top-fraction index, per group_key. Returns recall, precision arrays."""
    by_g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_g[row["group_key"]].append(row)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g, sub in by_g.items():
        P, R = [], []
        for row in sub:
            P.append(json.loads(row["pr_prec_json"]))
            R.append(json.loads(row["pr_rec_json"]))
        Pm = np.nanmean(np.asarray(P, dtype=np.float64), axis=0)
        Rm = np.nanmean(np.asarray(R, dtype=np.float64), axis=0)
        out[g] = (Rm, Pm)
    return out


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1) / x.size
    return x, y
