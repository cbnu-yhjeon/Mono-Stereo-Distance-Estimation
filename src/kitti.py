"""KITTI Stereo 2015 · Object Detection 로더."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calib import StereoCalib, load_stereo_calib

DEFAULT_ROOT = Path(os.environ.get("KITTI_ROOT", Path.home() / "data" / "kitti"))
EVAL_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Cyclist")


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_disparity_png(path: str | Path) -> np.ndarray:
    """KITTI 시차 PNG(uint16, 값/256 = 픽셀 시차, 0 = GT 없음) → float32, 무효는 NaN."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    disp = raw.astype(np.float32) / 256.0
    disp[raw == 0] = np.nan
    return disp


@dataclass(frozen=True)
class Object3D:
    """KITTI label_2 한 줄. 위치는 rectified 카메라 좌표의 3D 박스 바닥 중심(m)."""

    cls: str
    truncated: float
    occluded: int
    alpha: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float

    @property
    def box_height(self) -> float:
        return self.box[3] - self.box[1]

    def corners(self) -> np.ndarray:
        """3D 박스 꼭짓점 8x3 (카메라 좌표)."""
        l, w, h = self.l, self.w, self.h
        xs = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
        ys = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        zs = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
        c, s = np.cos(self.ry), np.sin(self.ry)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return (rot @ np.vstack([xs, ys, zs])).T + np.array([self.x, self.y, self.z])

    @property
    def z_near(self) -> float:
        """카메라에서 가장 가까운 꼭짓점의 깊이 — 스테레오가 실제로 보는 표면에 가깝다."""
        return float(self.corners()[:, 2].min())


def parse_label_line(line: str) -> Object3D:
    f = line.split()
    v = [float(x) for x in f[1:15]]
    return Object3D(
        cls=f[0], truncated=v[0], occluded=int(v[1]), alpha=v[2],
        box=(v[3], v[4], v[5], v[6]), h=v[7], w=v[8], l=v[9],
        x=v[10], y=v[11], z=v[12], ry=v[13],
    )


def read_labels(path: str | Path) -> list[Object3D]:
    return [parse_label_line(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


class Stereo2015:
    """`data_scene_flow.zip`의 training 200 장면 (frame _10만 사용)."""

    def __init__(self, root: str | Path = DEFAULT_ROOT / "stereo2015"):
        self.root = Path(root) / "training"

    def ids(self) -> list[int]:
        return sorted(int(p.name[:6]) for p in (self.root / "disp_occ_0").glob("*_10.png"))

    def load(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(좌, 우, 시차 GT occ, 시차 GT noc)."""
        name = f"{idx:06d}_10.png"
        return (
            read_image(self.root / "image_2" / name),
            read_image(self.root / "image_3" / name),
            read_disparity_png(self.root / "disp_occ_0" / name),
            read_disparity_png(self.root / "disp_noc_0" / name),
        )


class ObjectDataset:
    """`data_object_*.zip`의 training 7481 프레임."""

    def __init__(self, root: str | Path = DEFAULT_ROOT / "object"):
        self.root = Path(root) / "training"

    def ids(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.root / "label_2").glob("*.txt"))

    def left_path(self, idx: int) -> Path:
        return self.root / "image_2" / f"{idx:06d}.png"

    def left(self, idx: int) -> np.ndarray:
        return read_image(self.left_path(idx))

    def right(self, idx: int) -> np.ndarray:
        return read_image(self.root / "image_3" / f"{idx:06d}.png")

    def calib(self, idx: int) -> StereoCalib:
        return load_stereo_calib(self.root / "calib" / f"{idx:06d}.txt")

    def labels(self, idx: int) -> list[Object3D]:
        return read_labels(self.root / "label_2" / f"{idx:06d}.txt")


def split_ids(ids: list[int]) -> tuple[list[int], list[int]]:
    """짝수 id = 통계/튜닝용, 홀수 id = 평가용. 결정적이고 재현 가능하다."""
    return [i for i in ids if i % 2 == 0], [i for i in ids if i % 2 == 1]
