import numpy as np
import pandas as pd
import pytest

from src.eval import binned_errors, crossover_distance, disparity_metrics, distance_errors, interpolate_background


def test_interpolate_background_takes_smaller_neighbor():
    d = np.array([[np.nan, 10.0, np.nan, np.nan, 4.0, np.nan]])
    assert interpolate_background(d).tolist() == [[10.0, 10.0, 4.0, 4.0, 4.0, 4.0]]


def test_d1_counts_only_large_and_relative_errors():
    gt = np.array([[100.0, 100.0, 10.0, np.nan]])
    pred = np.array([[104.0, 106.0, 13.5, 1.0]])
    m = disparity_metrics(pred, gt)
    # 104: 4px but 4% → ok, 106: 6px & 6% → bad, 13.5: 3.5px & 35% → bad
    assert m["d1_all"] == pytest.approx(200 / 3)
    assert m["density"] == 100.0


def test_d1_all_interpolates_missing_predictions():
    gt = np.array([[20.0, 20.0, 20.0]])
    pred = np.array([[20.0, np.nan, 20.0]])
    m = disparity_metrics(pred, gt)
    assert m["d1_all"] == 0.0 and m["density"] == pytest.approx(200 / 3)


def test_distance_errors_separates_failures():
    s = distance_errors(np.array([11.0, np.nan, 9.0, 20.0]), np.array([10.0, 10.0, 10.0, 20.0]))
    assert s["fail_rate"] == 25.0
    assert s["mae"] == pytest.approx(2 / 3)
    assert s["bias"] == pytest.approx(0.0)


def test_crossover_found_between_bins():
    z = np.repeat([2.5, 7.5, 12.5, 17.5], 30).astype(float)
    df = pd.DataFrame({"z_gt": z})
    df["stereo"] = z * (1 + 0.004 * z)  # 상대 오차 ∝ Z
    df["mono"] = z * 1.05  # 상대 오차 5% 일정
    b = binned_errors(df, ["stereo", "mono"], max_m=20)
    zs = crossover_distance(b, "stereo", "mono")
    assert 10.0 < zs < 15.0  # 실제 교차 12.5 m
