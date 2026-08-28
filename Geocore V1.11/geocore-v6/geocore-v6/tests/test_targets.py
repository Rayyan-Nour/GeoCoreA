"""Target extraction: ranked peaks, min separation, drivers, CSV output."""
import csv

import numpy as np

from geocore.targets import extract_targets, write_targets_csv
from tests.synthetic import make_grid


def _scene():
    """Probability map with two known peaks and one sub-threshold bump."""
    grid = make_grid(100, 100, pixel_m=100.0)
    yy, xx = np.mgrid[0:100, 0:100]
    p = (0.92 * np.exp(-(((yy - 25) ** 2 + (xx - 30) ** 2) / 60.0))
         + 0.80 * np.exp(-(((yy - 70) ** 2 + (xx - 72) ** 2) / 60.0))
         + 0.40 * np.exp(-(((yy - 50) ** 2 + (xx - 10) ** 2) / 60.0)))
    p = np.clip(p, 0, 1)
    unc = np.full_like(p, 0.1)
    cls = np.where(p >= 0.7, 2, np.where(p >= 0.5, 1, 0)).astype(float)
    return grid, p, unc, cls


class TestExtractTargets:
    def test_finds_ranked_peaks_above_threshold(self):
        grid, p, unc, cls = _scene()
        t = extract_targets(p, unc, cls, grid, threshold=0.7,
                            min_separation_px=10)
        assert len(t) == 2                       # 0.4 bump excluded
        assert t[0]["probability"] > t[1]["probability"]
        assert (t[0]["row"], t[0]["col"]) == (25, 30)
        assert (t[1]["row"], t[1]["col"]) == (70, 72)
        assert t[0]["rank"] == 1 and t[1]["rank"] == 2
        # lat/lon populated and plausible for UTM 12N test grid
        assert -115 < t[0]["lon"] < -108 and 30 < t[0]["lat"] < 45

    def test_min_separation_suppresses_shoulder_maxima(self):
        grid, p, unc, cls = _scene()
        # plateau: many tied maxima around peak 1
        p2 = p.copy()
        p2[23:28, 28:33] = p2[25, 30]
        t = extract_targets(p2, unc, cls, grid, threshold=0.7,
                            min_separation_px=12)
        near_peak1 = [x for x in t
                      if abs(x["row"] - 25) < 12 and abs(x["col"] - 30) < 12]
        assert len(near_peak1) == 1

    def test_why_drivers_report_percentiles(self):
        grid, p, unc, cls = _scene()
        H, W = p.shape
        # one feature that is high exactly at peak 1
        f = np.zeros((H * W, 2), dtype=float)
        f[:, 0] = p.ravel() * 100          # "magnetics" tracks probability
        f[:, 1] = 1.0                      # constant dummy
        valid = np.ones(H * W, bool)
        t = extract_targets(p, unc, cls, grid, X_flat=f, valid_flat=valid,
                            feature_names=["magnetics", "dummy"],
                            importances=[("magnetics", 0.9),
                                         ("dummy", 0.1)],
                            threshold=0.7)
        why = t[0]["why"]
        assert why and why[0]["feature"] == "magnetics"
        assert why[0]["percentile"] >= 99

    def test_depth_values_attached(self):
        grid, p, unc, cls = _scene()
        d = np.full_like(p, 750.0)
        t = extract_targets(p, unc, cls, grid, depth_spectral=d,
                            threshold=0.7)
        assert t[0]["depth_spectral_m"] == 750.0
        assert t[0]["depth_euler_m"] is None

    def test_empty_when_nothing_above_threshold(self):
        grid, p, unc, cls = _scene()
        assert extract_targets(p * 0.4, unc, cls, grid,
                               threshold=0.7) == []

    def test_csv_roundtrip(self, tmp_path):
        grid, p, unc, cls = _scene()
        t = extract_targets(p, unc, cls, grid, threshold=0.7)
        out = write_targets_csv(tmp_path / "t.csv", t, "grade statement")
        with open(out, newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0][:4] == ["rank", "latitude", "longitude",
                               "probability"]
        data = [r for r in rows[1:] if r and r[0].strip().isdigit()]
        assert len(data) == len(t)
        assert any("GRADE CONTEXT" in (r[0] if r else "") for r in rows)
