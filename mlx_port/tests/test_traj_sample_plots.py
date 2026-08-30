"""Plot/JSON helpers must use the generated 64-waypoint pred, not an interpolation."""

import numpy as np

from mlx_port.traj_sample_plot_utils import (
    N_WAYPOINTS,
    _require_full_xy,
    _speed_mps,
    _xy_for_redraw,
    load_path_stamp,
    quant_path_label,
)


def test_require_full_xy_accepts_64_waypoints():
    xy = np.stack([np.arange(N_WAYPOINTS, dtype=np.float64), np.zeros(N_WAYPOINTS)], axis=-1)
    out = _require_full_xy(xy, "pred_xy")
    assert out.shape == (N_WAYPOINTS, 2)


def test_require_full_xy_rejects_endcaps_only():
    xy = np.zeros((5, 2))
    try:
        _require_full_xy(xy, "pred_xy")
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("expected ValueError for a 5-point path")


def test_redraw_uses_saved_64pt_pred_not_plot_cache():
    pred64 = np.stack(
        [np.linspace(0.0, 6.3, N_WAYPOINTS), np.linspace(0.0, 1.0, N_WAYPOINTS)],
        axis=-1,
    )
    rec = {
        "hist_xy_full": [[-1.0, 0.0]],
        "gt_xy_full": pred64.tolist(),
        "pred_xy_full": pred64.tolist(),
        "pred_xy_plot": np.zeros((N_WAYPOINTS, 2)).tolist(),
        "pred_xy": {"first5": pred64[:5].tolist(), "last5": pred64[-5:].tolist()},
    }
    _hist, _gt, pred = _xy_for_redraw(rec, "unused-clip")
    assert pred is not None
    assert pred.shape == (N_WAYPOINTS, 2)
    np.testing.assert_allclose(pred, pred64)


def test_redraw_does_not_interpolate_missing_pred():
    rec = {
        "hist_xy_full": [[-1.0, 0.0]],
        "gt_xy_full": np.zeros((N_WAYPOINTS, 2)).tolist(),
        "pred_xy": {
            "first5": [[0.7, 0.0], [1.4, 0.0], [2.1, 0.0], [2.8, 0.0], [3.5, 0.0]],
            "last5": [[60.0, 8.0], [62.0, 8.5], [64.0, 9.0], [66.0, 9.4], [68.0, 9.8]],
        },
        "pred_xy_plot": np.ones((N_WAYPOINTS, 2)).tolist(),
    }
    _hist, _gt, pred = _xy_for_redraw(rec, "unused-clip")
    assert pred is None


def test_speed_from_full_path_is_not_a_chord_plateau():
    t = np.arange(N_WAYPOINTS, dtype=np.float64) * 0.1
    # Constant accel: x = v0 t + 0.5 a t^2
    xy = np.stack([7.2 * t + 0.5 * 1.6 * t**2, np.zeros(N_WAYPOINTS)], axis=-1)
    v = _speed_mps(xy)
    assert v.shape == (N_WAYPOINTS,)
    # Finite differences of a smooth accel must not sit on one mid-horizon value.
    mid = v[5:-5]
    assert float(mid.std()) > 0.05


def test_html_report_stamps_bf16_and_t31_paths():
    assert quant_path_label({"lm": "bf16"}) == "bf16"
    assert quant_path_label({"lm": "affine-4-gs64"}) == "T3.1 W4 LM"
    bf16 = load_path_stamp(
        {"quant_path": "bf16", "quantized": {"lm": "bf16", "vision": "bf16", "expert": "bf16"}}
    )
    assert bf16["quant_path"] == "bf16"
    assert "lm=bf16" in bf16["flag_txt"]
    t31 = load_path_stamp(
        {
            "quantized": {"lm": "affine-4-gs64", "vision": "bf16", "expert": "bf16"},
        }
    )
    assert t31["quant_path"] == "T3.1 W4 LM"
    assert "lm=affine-4-gs64" in t31["flag_txt"]
