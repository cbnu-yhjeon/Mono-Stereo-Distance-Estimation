import numpy as np
import pytest

from src.calib import StereoCalib
from src.kitti import parse_label_line
from src.object_depth import aggregate_disparity, pinhole_distance, scale_to_reference, stereo_distance

CALIB = StereoCalib(
    np.array([[700.0, 0, 600, 0], [0, 700, 180, 0], [0, 0, 1, 0]]),
    np.array([[700.0, 0, 600, -700 * 0.5], [0, 700, 180, 0], [0, 0, 1, 0]]),
)


def test_pinhole_inverse_of_projection():
    # 높이 1.5 m 물체가 20 m에 있으면 화면 높이 = 1.5·700/20 = 52.5 px
    assert pinhole_distance(52.5, 1.5, 700.0) == pytest.approx(20.0)
    assert np.isnan(pinhole_distance(0.0, 1.5, 700.0))


def test_near_percentile_ignores_background():
    obj = np.full(60, 35.0)
    bg = np.full(40, 5.0)
    v = np.concatenate([obj, bg])
    assert aggregate_disparity(v, "near_p30") == pytest.approx(35.0)
    assert aggregate_disparity(v, "hist_mode") == pytest.approx(35.0)


def test_stereo_distance_on_synthetic_map():
    disp = np.full((100, 200), 3.5, dtype=np.float32)  # 배경 100 m
    disp[40:80, 50:150] = 17.5  # 물체 20 m
    z = stereo_distance(disp, (40, 30, 160, 85), CALIB, "near_p30")
    assert z == pytest.approx(20.0)


def test_label_parse_and_near_corner():
    o = parse_label_line("Car 0.00 0 -1.58 587.0 173.0 614.0 200.0 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59")
    assert o.cls == "Car" and o.box_height == 27.0
    # 정면을 향한 차(ry≈-π/2)는 길이 방향이 z축 → 가까운 면은 z − l/2
    assert o.z_near == pytest.approx(46.70 - 3.64 / 2, abs=0.05)


def test_scale_to_reference_median():
    scaled, s = scale_to_reference(np.array([1.0, 2.0, np.nan]), np.array([10.0, 20.0, 5.0]))
    assert s == pytest.approx(10.0)
    assert scaled[:2].tolist() == [10.0, 20.0]
