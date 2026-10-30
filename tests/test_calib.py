import numpy as np
import pytest

from src.calib import load_stereo_calib, parse_calib_file

# KITTI object training/calib/000000.txt 의 P2, P3
P2 = "7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03"
P3 = "7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03"


@pytest.fixture
def calib_file(tmp_path):
    p = tmp_path / "000000.txt"
    p.write_text(f"P0: {P2}\nP2: {P2}\nP3: {P3}\nR0_rect: 1 0 0 0 1 0 0 0 1\n")
    return p


def test_parse_skips_non_numeric(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("calib_time: 09-Jan-2012 13:57:47\nP_rect_02: " + P2 + "\nP_rect_03: " + P3 + "\n")
    raw = parse_calib_file(p)
    assert "calib_time" not in raw
    assert load_stereo_calib(p).fx == pytest.approx(721.5377)


def test_focal_and_baseline(calib_file):
    c = load_stereo_calib(calib_file)
    assert c.fx == pytest.approx(721.5377)
    assert c.fy == pytest.approx(721.5377)
    assert c.baseline == pytest.approx(44.85728 / 721.5377 + 339.5242 / 721.5377)
    assert 0.52 < c.baseline < 0.55


def test_depth_disparity_roundtrip(calib_file):
    c = load_stereo_calib(calib_file)
    z = np.array([5.0, 20.0, 60.0])
    assert np.allclose(c.disparity_to_depth(c.depth_to_disparity(z)), z)
    assert np.isnan(c.disparity_to_depth(0.0))
