"""마일스톤 1: 좌·우 영상을 나란히 놓고 수평 에피폴라 선을 그어 정렬을 확인한다.

정렬된 쌍이면 같은 점이 같은 행(y)에 있어야 한다. ORB 매칭 쌍의 y 차이 통계도 함께 낸다.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.kitti import ObjectDataset

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, nargs="+", default=[0, 8, 100])
ap.add_argument("--out", default="results/figures")
args = ap.parse_args()

ds = ObjectDataset()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for idx in args.frames:
    left, right = ds.left(idx), ds.right(idx)
    c = ds.calib(idx)
    kl, dl = orb.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    kr, dr = orb.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    matches = bf.match(dl, dr)
    dy = np.array([kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1] for m in matches])
    dx = np.array([kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0] for m in matches])
    good = np.abs(dy) < 5  # 잘못된 매칭 제외
    print(f"{idx:06d}: f={c.fx:.1f}px B={c.baseline:.4f}m  ORB {good.sum()}/{len(dy)} |dy| median={np.median(np.abs(dy[good])):.2f}px "
          f"disparity range {dx[good].min():.1f}..{dx[good].max():.1f}px")

    canvas = np.vstack([left, right])
    h, w = left.shape[:2]
    for y in range(20, h, 30):
        color = tuple(int(v) for v in np.random.default_rng(y).integers(64, 256, 3))
        cv2.line(canvas, (0, y), (w, y), color, 1)
        cv2.line(canvas, (0, y + h), (w, y + h), color, 1)
    cv2.putText(canvas, f"left  f={c.fx:.1f}px  B={c.baseline:.3f}m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"right  ORB |dy| median {np.median(np.abs(dy[good])):.2f}px", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"epipolar_{idx:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
