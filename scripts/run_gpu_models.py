"""D2 · D4용 GPU 추론을 한 번에 돌려 캐시한다 (평가 split = 홀수 id).

- 검출기: Ultralytics YOLO11 (COCO) → results/cache/detections.csv
- 단안 깊이: Depth Anything V2 Metric-Outdoor-Small (Virtual KITTI 2로만 미세조정, 실제 KITTI 미사용)
  GT 박스와 검출 박스 각각에서 박스 깊이를 뽑아 results/cache/mono_depth.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

from src.kitti import ObjectDataset, split_ids
from src.object_depth import mono_depth_distance

CACHE = Path("results/cache")
COCO_TO_KITTI = {"car": "Car", "truck": "Truck", "bus": "Truck", "person": "Pedestrian"}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="weights/yolo11m.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = ObjectDataset()
    _, ids = split_ids(ds.ids())
    if args.limit:
        ids = ids[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.yolo)
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval()

    det_rows, depth_rows, t_det, t_depth = [], [], [], []
    for idx in tqdm(ids, desc="gpu models"):
        path = ds.left_path(idx)
        t0 = time.perf_counter()
        r = yolo.predict(str(path), conf=0.25, verbose=False, device=0 if dev == "cuda" else "cpu")[0]
        t_det.append(time.perf_counter() - t0)
        dets = []
        for k, (box, c, conf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())):
            name = r.names[int(c)]
            if name not in COCO_TO_KITTI:
                continue
            dets.append((k, COCO_TO_KITTI[name], float(conf), tuple(map(float, box))))
            det_rows.append({"idx": idx, "det": k, "det_cls": COCO_TO_KITTI[name], "conf": float(conf),
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})

        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        with torch.inference_mode():
            inputs = proc(images=img, return_tensors="pt").to(dev)
            out = depth_net(**inputs)
            depth = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]["predicted_depth"]
            depth = depth.float().cpu().numpy()
        if dev == "cuda":
            torch.cuda.synchronize()
        t_depth.append(time.perf_counter() - t0)

        for j, o in enumerate(ds.labels(idx)):
            if o.cls != "DontCare":
                depth_rows.append({"idx": idx, "kind": "gt", "k": j, "z_mono": mono_depth_distance(depth, o.box)})
        for k, _, _, box in dets:
            depth_rows.append({"idx": idx, "kind": "det", "k": k, "z_mono": mono_depth_distance(depth, box)})

    pd.DataFrame(det_rows).to_csv(CACHE / "detections.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(CACHE / "mono_depth.csv", index=False)
    timing = {"detector_ms": 1000 * np.median(t_det), "mono_depth_ms": 1000 * np.median(t_depth)}
    pd.Series(timing).to_csv(CACHE / "gpu_timing.csv")
    print(timing)


if __name__ == "__main__":
    main()
