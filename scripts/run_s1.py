"""S1: BM vs SGBM vs SGBM+WLS (KITTI Stereo 2015).

짝수 장면(100)에서 파라미터를 격자 탐색해 방식별 최적값을 고르고,
홀수 장면(100)에서 D1-all · EPE · 처리 시간을 보고한다.
결과: results/s1_tuning.csv, results/s1_results.csv, results/best_stereo_params.json
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.disparity import StereoParams, compute_disparity
from src.eval import disparity_metrics
from src.kitti import Stereo2015, split_ids

OUT = Path("results")


def grid() -> list[StereoParams]:
    cands = []
    for bs, ur in itertools.product([9, 15, 21], [5, 15]):
        cands.append(StereoParams(method="bm", block_size=bs, uniqueness_ratio=ur))
    for bs, p2, mode in itertools.product([3, 5, 7, 9, 11], [32, 96], [cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH]):
        cands.append(StereoParams(method="sgbm", block_size=bs, p2_scale=p2, mode=mode))
    return cands


def wls_grid(base: StereoParams) -> list[StereoParams]:
    return [base.with_(method="sgbm_wls", wls_lambda=lam, wls_sigma=sig) for lam, sig in itertools.product([4000.0, 8000.0, 16000.0], [0.8, 1.5])]


def eval_one(job):
    params, idx = job
    cv2.setNumThreads(1)
    left, right, gt_occ, gt_noc = Stereo2015().load(idx)
    t0 = time.perf_counter()
    disp = compute_disparity(left, right, params)
    dt = time.perf_counter() - t0
    m_all = disparity_metrics(disp, gt_occ)
    m_noc = disparity_metrics(disp, gt_noc)
    return {"idx": idx, "time_s": dt, **m_all, "d1_noc": m_noc["d1_all"], "epe_noc": m_noc["epe"]}


def evaluate(pool, params_list, ids):
    jobs = [(p, i) for p in params_list for i in ids]
    res = pool.map(eval_one, jobs, chunksize=4)
    rows = []
    for (p, _), r in zip(jobs, res):
        rows.append({**asdict(p), **r})
    df = pd.DataFrame(rows)
    keys = list(asdict(params_list[0]).keys())
    return df.groupby(keys, as_index=False).mean(numeric_only=True).drop(columns="idx")


def row_to_params(row) -> StereoParams:
    return StereoParams(**{k: (type(getattr(StereoParams(), k))(row[k])) for k in asdict(StereoParams())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tune_ids, test_ids = split_ids(Stereo2015().ids())
    print(f"tune {len(tune_ids)} / test {len(test_ids)} scenes")

    with Pool(args.workers) as pool:
        tuning = evaluate(pool, grid(), tune_ids)
        best = {m: row_to_params(tuning[tuning.method == m].sort_values("d1_all").iloc[0]) for m in ("bm", "sgbm")}
        wls = evaluate(pool, wls_grid(best["sgbm"]), tune_ids)
        best["sgbm_wls"] = row_to_params(wls.sort_values("d1_all").iloc[0])
        tuning = pd.concat([tuning, wls], ignore_index=True)
        tuning.to_csv(OUT / "s1_tuning.csv", index=False)

        test = pd.concat([evaluate(pool, [p], test_ids) for p in best.values()], ignore_index=True)

    # 처리 시간은 병렬 부하 없이 단일 프로세스(OpenCV 기본 스레드)로 다시 잰다.
    for i, p in enumerate(best.values()):
        times = []
        for idx in test_ids[:20]:
            left, right, _, _ = Stereo2015().load(idx)
            t0 = time.perf_counter()
            compute_disparity(left, right, p)
            times.append(time.perf_counter() - t0)
        test.loc[i, "time_ms"] = 1000 * float(np.median(times))

    test.to_csv(OUT / "s1_results.csv", index=False)
    (OUT / "best_stereo_params.json").write_text(json.dumps({m: asdict(p) for m, p in best.items()}, indent=2))
    cols = ["method", "block_size", "p2_scale", "mode", "uniqueness_ratio", "wls_lambda", "wls_sigma", "d1_all", "d1_noc", "epe", "epe_noc", "d1_valid", "density", "time_ms"]
    print(test[cols].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
