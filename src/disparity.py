"""BM · SGBM · SGBM+WLS 시차 계산. 출력은 float32 픽셀 시차, 무효 픽셀은 NaN."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

METHODS = ("bm", "sgbm", "sgbm_wls")


@dataclass(frozen=True)
class StereoParams:
    method: str = "sgbm"
    num_disparities: int = 128  # 16의 배수
    block_size: int = 5  # 홀수. BM은 5 이상
    p1_scale: int = 8  # P1 = p1_scale · 채널 · block²
    p2_scale: int = 32  # P2 = p2_scale · 채널 · block²
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5
    pad_left: bool = True  # 왼쪽에 numDisparities 만큼 패딩해 좌측 띠도 계산

    def with_(self, **kw) -> StereoParams:
        return replace(self, **kw)


def _make_left_matcher(p: StereoParams):
    if p.method == "bm":
        m = cv2.StereoBM_create(numDisparities=p.num_disparities, blockSize=max(p.block_size, 5))
        m.setUniquenessRatio(p.uniqueness_ratio)
        m.setSpeckleWindowSize(p.speckle_window)
        m.setSpeckleRange(p.speckle_range)
        m.setDisp12MaxDiff(p.disp12_max_diff)
        m.setPreFilterCap(min(p.pre_filter_cap, 63))
        return m
    channels = 3
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=p.num_disparities,
        blockSize=p.block_size,
        P1=p.p1_scale * channels * p.block_size**2,
        P2=p.p2_scale * channels * p.block_size**2,
        disp12MaxDiff=p.disp12_max_diff,
        preFilterCap=p.pre_filter_cap,
        uniquenessRatio=p.uniqueness_ratio,
        speckleWindowSize=p.speckle_window,
        speckleRange=p.speckle_range,
        mode=p.mode,
    )


def compute_disparity(left: np.ndarray, right: np.ndarray, params: StereoParams = StereoParams()) -> np.ndarray:
    """좌 영상 기준 시차 맵. OpenCV의 1/16 고정소수점을 픽셀 단위로 바꾼다.

    OpenCV 매처는 왼쪽 numDisparities 열을 계산하지 않는다. pad_left면 두 영상 왼쪽을
    같은 폭으로 복제 패딩해 그 띠에서도 (영상 안에 대응점이 있는 만큼) 시차를 구한 뒤 잘라낸다.
    """
    if params.method not in METHODS:
        raise ValueError(f"unknown method {params.method!r}; one of {METHODS}")
    matcher = _make_left_matcher(params)
    if params.method == "bm":
        l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    else:
        l, r = left, right
    pad = params.num_disparities if params.pad_left else 0
    if pad:
        l = cv2.copyMakeBorder(l, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        r = cv2.copyMakeBorder(r, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    raw_left = matcher.compute(l, r)

    if params.method == "sgbm_wls":
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        raw_right = right_matcher.compute(r, l)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(params.wls_lambda)
        wls.setSigmaColor(params.wls_sigma)
        raw_left = wls.filter(raw_left, l, disparity_map_right=raw_right)

    disp = raw_left[:, pad:].astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp
