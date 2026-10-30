"""S1 정성 비교: 좌 영상 · GT 시차 · BM · SGBM · SGBM+WLS를 한 장에 그린다."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.disparity import StereoParams, compute_disparity
from src.eval import disparity_metrics
from src.kitti import Stereo2015

ap = argparse.ArgumentParser()
ap.add_argument("--scene", type=int, default=1)  # 홀수 = 테스트 장면
args = ap.parse_args()

best = {m: StereoParams(**p) for m, p in json.loads(Path("results/best_stereo_params.json").read_text()).items()}
left, right, gt, _ = Stereo2015().load(args.scene)
panels = [("left", cv2.cvtColor(left, cv2.COLOR_BGR2RGB)), ("GT (occ)", gt)]
for m, p in best.items():
    d = compute_disparity(left, right, p)
    panels.append((f"{m}  D1-all {disparity_metrics(d, gt)['d1_all']:.1f}%", d))

vmax = np.nanpercentile(gt, 99)
fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap=None if img.ndim == 3 else "magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(f"results/figures/disparity_{args.scene:06d}.png", dpi=110)
