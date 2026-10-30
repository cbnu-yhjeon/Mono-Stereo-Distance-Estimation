"""S2 · D1 · D2 · D3: 평가 split(홀수 id)의 모든 GT 객체에 대해 방식별 거리를 계산해 한 표로 저장한다.

- 클래스 평균 높이는 짝수 id 라벨에서 추정한다 (평가 데이터 미사용).
- 스테레오 파라미터는 S1의 results/best_stereo_params.json을 쓴다.
- 검출 박스는 results/cache/detections.csv(run_gpu_models.py)가 있으면 IoU ≥ 0.5로 GT와 1:1 매칭한다.
결과: results/objects.csv
"""

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.disparity import StereoParams, compute_disparity
from src.kitti import EVAL_CLASSES, ObjectDataset, split_ids
from src.object_depth import DISP_AGGREGATORS, estimate_class_heights, pinhole_distance, stereo_distance

OUT = Path("results")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, det_boxes, thr=0.5) -> dict[int, int]:
    """IoU 내림차순 greedy 1:1 매칭. {gt 인덱스: det 인덱스}"""
    pairs = sorted(((iou(g, d), i, j) for i, g in enumerate(gt_boxes) for j, d in enumerate(det_boxes)), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_d:
            out[i] = j
            used_g.add(i)
            used_d.add(j)
    return out


def process(idx, params: dict[str, StereoParams], heights: dict[str, float], dets: pd.DataFrame | None):
    cv2.setNumThreads(1)
    ds = ObjectDataset()
    calib = ds.calib(idx)
    objs = [(j, o) for j, o in enumerate(ds.labels(idx)) if o.cls in EVAL_CLASSES]
    if not objs:
        return []
    left, right = ds.left(idx), ds.right(idx)
    disps, times = {}, {}
    for m, p in params.items():
        t0 = time.perf_counter()
        disps[m] = compute_disparity(left, right, p)
        times[m] = time.perf_counter() - t0

    det_list = [] if dets is None else list(dets.itertuples())
    det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in det_list]
    matches = match([o.box for _, o in objs], det_boxes)

    rows = []
    for i, (j, o) in enumerate(objs):
        r = {
            "idx": idx, "k": j, "cls": o.cls, "truncated": o.truncated, "occluded": o.occluded,
            "box_h": o.box_height, "x1": o.box[0], "y1": o.box[1], "x2": o.box[2], "y2": o.box[3],
            "z_gt": o.z, "z_near": o.z_near, "h_true": o.h, "fB": calib.fx * calib.baseline,
            "pin_gt": pinhole_distance(o.box_height, heights[o.cls], calib.fy),
            "pin_gt_trueH": pinhole_distance(o.box_height, o.h, calib.fy),  # 높이를 정확히 알 때 (상한)
        }
        for m in params:
            for agg in DISP_AGGREGATORS:
                r[f"st_{m}_{agg}"] = stereo_distance(disps[m], o.box, calib, agg)
            r[f"time_{m}"] = times[m]
        if i in matches:
            d = det_list[matches[i]]
            dbox = (d.x1, d.y1, d.x2, d.y2)
            r.update({"det": d.det, "det_cls": d.det_cls, "det_iou": iou(o.box, dbox),
                      "pin_det": pinhole_distance(d.y2 - d.y1, heights[d.det_cls], calib.fy)})
            for m in params:
                r[f"st_det_{m}"] = stereo_distance(disps[m], dbox, calib, "near_p30")
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ds = ObjectDataset()
    stat_ids, eval_ids = split_ids(ds.ids())
    if args.limit:
        eval_ids = eval_ids[: args.limit]

    heights = estimate_class_heights(o for i in stat_ids for o in ds.labels(i) if o.cls in EVAL_CLASSES)
    (OUT / "class_heights.json").write_text(json.dumps(heights, indent=2))
    print("class heights (even split):", {k: round(v, 3) for k, v in heights.items()})

    best = json.loads((OUT / "best_stereo_params.json").read_text())
    params = {m: StereoParams(**p) for m, p in best.items()}

    det_path = OUT / "cache" / "detections.csv"
    dets = pd.read_csv(det_path) if det_path.exists() else None
    if dets is None:
        print("! detections.csv 없음 → D2 생략")
    by_frame = {} if dets is None else dict(tuple(dets.groupby("idx")))
    empty = None if dets is None else dets.iloc[0:0]

    jobs = [(i, by_frame.get(i, empty)) for i in eval_ids]
    with Pool(args.workers) as pool:
        fn = partial(_run, params=params, heights=heights)
        rows = [r for rs in tqdm(pool.imap(fn, jobs, chunksize=8), total=len(jobs)) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "objects.csv", index=False)
    print(f"{len(df)} objects from {df.idx.nunique()} frames → results/objects.csv")


def _run(job, params, heights):
    idx, dets = job
    return process(idx, params, heights, dets)


if __name__ == "__main__":
    main()
