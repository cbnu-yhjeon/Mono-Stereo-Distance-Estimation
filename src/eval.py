"""시차 지표(D1-all, EPE)와 거리 지표(MAE, 상대 오차, 구간별 곡선, 교차 거리 Z*)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_background(disp: np.ndarray) -> np.ndarray:
    """KITTI devkit 방식: 행마다 무효 구간을 양옆 유효값 중 작은 값(배경 쪽)으로 채운다."""
    out = disp.copy()
    w = disp.shape[1]
    cols = np.arange(w)
    for r in range(disp.shape[0]):
        row = out[r]
        valid = np.isfinite(row)
        if valid.all():
            continue
        if not valid.any():
            row[:] = 0.0
            continue
        vi = cols[valid]
        pos = np.searchsorted(vi, cols)  # 오른쪽 이웃 인덱스
        left = row[vi[np.clip(pos - 1, 0, len(vi) - 1)]]
        right = row[vi[np.clip(pos, 0, len(vi) - 1)]]
        left = np.where(pos == 0, right, left)
        right = np.where(pos == len(vi), left, right)
        fill = np.minimum(left, right)
        row[~valid] = fill[~valid]
    return out


def disparity_metrics(pred: np.ndarray, gt: np.ndarray, tau_px: float = 3.0, tau_rel: float = 0.05) -> dict:
    """GT가 있는 픽셀에서 D1(%)과 EPE(px).

    `d1_all`/`epe`는 무효 예측을 배경 보간한 KITTI 공식 방식,
    `d1_valid`/`epe_valid`는 예측이 유효한 픽셀만 본 값, `density`는 유효 예측 비율(%).
    """
    mask = np.isfinite(gt)
    g = gt[mask]
    filled = interpolate_background(pred)[mask]
    err = np.abs(filled - g)
    bad = (err > tau_px) & (err > tau_rel * g)

    p = pred[mask]
    pv = np.isfinite(p)
    err_v = np.abs(p[pv] - g[pv])
    bad_v = (err_v > tau_px) & (err_v > tau_rel * g[pv])
    return {
        "d1_all": 100.0 * bad.mean(),
        "epe": float(err.mean()),
        "d1_valid": 100.0 * bad_v.mean() if pv.any() else np.nan,
        "epe_valid": float(err_v.mean()) if pv.any() else np.nan,
        "density": 100.0 * pv.mean(),
    }


def distance_errors(z_pred: np.ndarray, z_gt: np.ndarray) -> dict:
    """유효 예측만으로 MAE·상대 오차를 내고, 못 구한 비율은 fail_rate로 따로 낸다."""
    z_pred = np.asarray(z_pred, dtype=float)
    z_gt = np.asarray(z_gt, dtype=float)
    ok = np.isfinite(z_pred) & (z_pred > 0)
    e = z_pred[ok] - z_gt[ok]
    rel = np.abs(e) / z_gt[ok]
    return {
        "n": int(len(z_gt)),
        "fail_rate": 100.0 * (1 - ok.mean()) if len(z_gt) else np.nan,
        "mae": float(np.abs(e).mean()) if ok.any() else np.nan,
        "bias": float(e.mean()) if ok.any() else np.nan,
        "rel_mean": 100.0 * float(rel.mean()) if ok.any() else np.nan,
        "rel_median": 100.0 * float(np.median(rel)) if ok.any() else np.nan,
    }


def binned_errors(df: pd.DataFrame, pred_cols: list[str], gt_col: str = "z_gt", bin_m: float = 5.0, max_m: float = 80.0) -> pd.DataFrame:
    """GT 거리 구간별 오차 표 (long format: bin_lo, method, n, fail_rate, mae, ...)."""
    edges = np.arange(0.0, max_m + bin_m, bin_m)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = df[(df[gt_col] >= lo) & (df[gt_col] < hi)]
        for col in pred_cols:
            stats = distance_errors(sel[col].to_numpy(), sel[gt_col].to_numpy())
            rows.append({"bin_lo": lo, "bin_hi": hi, "bin_mid": (lo + hi) / 2, "method": col, **stats})
    return pd.DataFrame(rows)


def crossover_distance(binned: pd.DataFrame, a: str, b: str, metric: str = "rel_median", min_n: int = 20) -> float:
    """a가 b보다 나은 구간에서 b가 나아지는 첫 지점 Z* (선형 보간, m). 교차 없으면 NaN.

    `a`는 가까이서 유리한 방식(스테레오), `b`는 멀리서 유리할 것으로 예상하는 방식(단안 핀홀).
    """
    pa = binned[binned.method == a].set_index("bin_mid")
    pb = binned[binned.method == b].set_index("bin_mid")
    mids = [m for m in pa.index if pa.loc[m, "n"] >= min_n and np.isfinite(pa.loc[m, metric]) and np.isfinite(pb.loc[m, metric])]
    diff = np.array([pa.loc[m, metric] - pb.loc[m, metric] for m in mids])  # 음수 = a가 나음
    for i in range(1, len(mids)):
        if diff[i - 1] < 0 <= diff[i]:
            t = -diff[i - 1] / (diff[i] - diff[i - 1])
            return float(mids[i - 1] + t * (mids[i] - mids[i - 1]))
    return float("nan")
