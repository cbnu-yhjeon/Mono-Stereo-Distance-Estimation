import numpy as np
import pytest

from src.disparity import StereoParams, compute_disparity


@pytest.fixture(scope="module")
def pair():
    rng = np.random.default_rng(0)
    h, w, d_true = 120, 320, 24
    tex = rng.integers(0, 256, (h, w + d_true), dtype=np.uint8)
    tex = np.repeat(tex[:, :, None], 3, axis=2)
    left = tex[:, :w].copy()
    right = tex[:, d_true:].copy()  # right[x − d] = left[x]
    return left, right, d_true


@pytest.mark.parametrize("method", ["bm", "sgbm", "sgbm_wls"])
def test_constant_shift_recovered(pair, method):
    left, right, d_true = pair
    disp = compute_disparity(left, right, StereoParams(method=method, num_disparities=64, block_size=7))
    roi = disp[20:-20, 80:-20]
    assert np.isfinite(roi).mean() > 0.9
    assert np.nanmedian(roi) == pytest.approx(d_true, abs=0.5)


def test_left_band_computed_with_padding(pair):
    left, right, d_true = pair
    p = StereoParams(method="sgbm", num_disparities=64, block_size=7)
    band = slice(d_true + 8, 60)  # 대응점이 우 영상 안에 있는 좌측 띠
    assert np.isfinite(compute_disparity(left, right, p.with_(pad_left=False))[20:-20, band]).mean() < 0.1
    padded = compute_disparity(left, right, p)[20:-20, band]
    assert np.isfinite(padded).mean() > 0.9
    assert np.nanmedian(padded) == pytest.approx(d_true, abs=0.5)
