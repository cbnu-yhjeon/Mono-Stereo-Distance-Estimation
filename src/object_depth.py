"""박스 → 거리. A) 단안 핀홀, B) 스테레오 시차, C) 단안 깊이 + 스케일 보정."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .calib import StereoCalib
from .kitti import Object3D

# KITTI training 전체 label_2 평균 높이(m). 실험에서는 짝수 split으로 다시 추정해 쓴다.
DEFAULT_CLASS_HEIGHTS = {"Car": 1.53, "Van": 2.21, "Truck": 3.25, "Pedestrian": 1.76, "Cyclist": 1.74}

# 박스 안 시차 → 객체 시차 집계 방식 (실험 S2)
DISP_AGGREGATORS = ("median", "center_median", "near_p30", "hist_mode")


def estimate_class_heights(objects: Iterable[Object3D]) -> dict[str, float]:
    by_cls: dict[str, list[float]] = {}
    for o in objects:
        by_cls.setdefault(o.cls, []).append(o.h)
    return {c: float(np.mean(v)) for c, v in by_cls.items()}


def pinhole_distance(box_height_px: float, real_height_m: float, fy: float) -> float:
    """Z = H · f / h. 박스가 퇴화했으면 NaN."""
    if box_height_px <= 1.0:
        return float("nan")
    return real_height_m * fy / box_height_px


def _box_pixels(img: np.ndarray, box: tuple[float, float, float, float], shrink: float = 0.0) -> np.ndarray:
    """박스(정수 픽셀로 클리핑) 안 값. shrink는 각 변을 박스 크기 비율만큼 안쪽으로 줄인다."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
    h, w = img.shape[:2]
    c1, c2 = int(np.clip(np.floor(x1 + dx), 0, w - 1)), int(np.clip(np.ceil(x2 - dx), 1, w))
    r1, r2 = int(np.clip(np.floor(y1 + dy), 0, h - 1)), int(np.clip(np.ceil(y2 - dy), 1, h))
    return img[r1:r2, c1:max(c2, c1 + 1)].ravel()


def aggregate_disparity(values: np.ndarray, how: str = "near_p30") -> float:
    """박스 안 시차들을 하나의 객체 시차로 줄인다.

    박스에는 객체(가까움 = 큰 시차)와 배경(멀리 = 작은 시차)이 섞인다.
    - median        : 전체 중앙값
    - near_p30      : 시차 상위 30%(= 깊이 기준 가장 가까운 30%)의 중앙값
    - hist_mode     : 0.5 px 히스토그램 최빈 구간 (가장 큰 표면)
    `center_median`은 같은 median을 shrink된 박스에 쓰는 것으로, 호출 측에서 처리한다.
    """
    v = values[np.isfinite(values) & (values > 0)]
    if v.size < 5:
        return float("nan")
    if how in ("median", "center_median"):
        return float(np.median(v))
    if how == "near_p30":
        return float(np.median(v[v >= np.percentile(v, 70)]))
    if how == "hist_mode":
        bins = np.arange(0.0, v.max() + 1.0, 0.5)
        hist, edges = np.histogram(v, bins=bins)
        k = int(hist.argmax())
        in_bin = v[(v >= edges[k]) & (v < edges[k + 1])]
        return float(np.median(in_bin))
    raise ValueError(f"unknown aggregator {how!r}; one of {DISP_AGGREGATORS}")


def stereo_distance(disp: np.ndarray, box, calib: StereoCalib, how: str = "near_p30") -> float:
    shrink = 0.25 if how == "center_median" else 0.0
    d = aggregate_disparity(_box_pixels(disp, box, shrink), how)
    return calib.disparity_to_depth(d) if np.isfinite(d) else float("nan")


def mono_depth_distance(depth: np.ndarray, box) -> float:
    """미터 깊이 맵에서 박스 중앙 50% 영역의 가까운 30% 중앙값."""
    v = _box_pixels(depth, box, shrink=0.25)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return float("nan")
    return float(np.median(v[v <= np.percentile(v, 30)]))


def scale_to_reference(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """한 영상 안 객체들에 대해 s = median(ref / pred)로 스케일을 맞춘다 (ref = 핀홀 거리)."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(ref) & (pred > 0)
    if not ok.any():
        return np.full_like(pred, np.nan), float("nan")
    s = float(np.median(ref[ok] / pred[ok]))
    return pred * s, s
