"""KITTI calib 파싱: 투영 행렬 P2, P3에서 초점 거리와 스테레오 기준선을 구한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_calib_file(path: str | Path) -> dict[str, np.ndarray]:
    """`KEY: v1 v2 ...` 형식의 KITTI calib 파일을 키별 1차원 배열로 읽는다.

    Object(`P2:`)와 raw/scene-flow(`P_rect_02:`) 두 형식을 모두 받는다.
    숫자가 아닌 값(예: `calib_time`)은 건너뛴다.
    """
    out: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        try:
            out[key.strip()] = np.array([float(v) for v in values.split()])
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class StereoCalib:
    """정렬(rectified)된 좌(cam2)·우(cam3) 카메라 쌍의 내부 파라미터와 기준선."""

    P2: np.ndarray  # 3x4
    P3: np.ndarray  # 3x4

    @property
    def fx(self) -> float:
        return float(self.P2[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P2[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P2[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P2[1, 2])

    @property
    def baseline(self) -> float:
        """B = P2[0,3]/P2[0,0] − P3[0,3]/P3[0,0] (미터). KITTI에서 약 0.54 m."""
        return float(self.P2[0, 3] / self.P2[0, 0] - self.P3[0, 3] / self.P3[0, 0])

    def disparity_to_depth(self, disparity: np.ndarray | float) -> np.ndarray | float:
        """Z = f·B / d. d ≤ 0은 NaN."""
        d = np.asarray(disparity, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(d > 0, self.fx * self.baseline / d, np.nan)
        return float(z) if z.ndim == 0 else z

    def depth_to_disparity(self, depth: np.ndarray | float) -> np.ndarray | float:
        z = np.asarray(depth, dtype=np.float64)
        d = self.fx * self.baseline / z
        return float(d) if d.ndim == 0 else d

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """카메라(rectified cam0) 좌표 Nx3 → 좌 영상 픽셀 Nx2."""
        homo = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        uvw = homo @ self.P2.T
        return uvw[:, :2] / uvw[:, 2:3]


def load_stereo_calib(path: str | Path) -> StereoCalib:
    raw = parse_calib_file(path)
    for k2, k3 in (("P2", "P3"), ("P_rect_02", "P_rect_03")):
        if k2 in raw and k3 in raw:
            return StereoCalib(raw[k2].reshape(3, 4), raw[k3].reshape(3, 4))
    raise KeyError(f"{path}: P2/P3 (또는 P_rect_02/P_rect_03) 없음")
