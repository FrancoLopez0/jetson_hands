import numpy as np
import pytest

from jetson_hands.pipeline import render_raw, render_raw_lut


@pytest.mark.parametrize("pattern", ["random", "dark", "gradient", "saturated"])
@pytest.mark.parametrize("output_width", [40, 80])
def test_lookup_renderer_preserves_levels_gamma_and_bayer_order(pattern, output_width):
    if pattern == "random":
        raw = np.random.default_rng(42).integers(0, 65536, (48, 80), dtype=np.uint16)
    elif pattern == "dark":
        raw = np.zeros((48, 80), np.uint16)
    elif pattern == "saturated":
        raw = np.full((48, 80), 65535, np.uint16)
    else:
        raw = np.linspace(0, 65535, 48 * 80).reshape(48, 80).astype(np.uint16)
    original = raw.copy()
    expected = render_raw(raw, output_width)
    actual = render_raw_lut(raw, output_width)
    assert actual.shape == expected.shape and actual.dtype == np.uint8
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1
    np.testing.assert_array_equal(raw, original)
