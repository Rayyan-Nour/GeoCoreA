"""
run_tests.py - end-to-end grading of the GeoCore engine on ground-truth worlds.

For each scenario, run the FULL pipeline (the same entry point the GUI uses)
and grade against known truth:

  signal   -> PASS if holdout contrast AUC >= 0.70 and hit@0.50 > 0
  confound -> PASS if the engine's honest metrics EXPOSE the artifact
              (holdout contrast AUC <= 0.60, i.e. no false confidence)
  mixed    -> report both; expect partial recovery (terrain matching on)

Also grades depth-to-source: compares spectral/Euler depth at true stock
locations against the known planted depths.
"""
from __future__ import annotations

import sys, json, traceback
import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.synthworld import build_world, PIX, H, W
from geocore.config import PipelineConfig
from geocore.pipeline import run_pipeline


def run_scenario(name: str, seed=7):
    world = build_world(name, f"/tmp/gc_world_{name}",
                        seed=seed)
    cfg = PipelineConfig(
        dem_path=world["dem"],
        deposit_csv=world["deposits_csv"],
        feature_rasters={"magnetics": world["magnetics"],
                         "gravity": world["gravity"]},
        feature_vectors={"geochem_cu": world["geochem_csv"]},
        commodity="copper",
        analysis_max_px=0,            # native 600px - fast enough
        results_dir=f"/tmp/gc_results_{name}",
        random_seed=42,
        n_trees=200,
        estimate_depth=(name == "signal"),   # grade depth once
    )
    res = run_pipeline(cfg)
    return world, res


def grade_depths(world, res):
    """Compare estimated depth at true stock pixels vs planted depth."""
    rows = []
    ds = res.depth_spectral
    de = res.depth_euler
    for (r, c, depth, s) in world["stocks"]:
        est_s = float(ds[r, c]) if ds is not None and np.isfinite(ds[r, c]) else None
        est_e = float(de[r, c]) if de is not None and np.isfinite(de[r, c]) else None
        rows.append((depth, est_s, est_e))
    def err(pairs):
        vals = [(t, e) for t, e, in pairs if e is not None]
        if not vals:
            return None, 0
        rel = [abs(e - t) / t for t, e in vals]
        return float(np.median(rel)), len(vals)
    spec_pairs = [(t, s_) for t, s_, _ in rows]
    eul_pairs = [(t, e_) for t, _, e_ in rows]
    spec_err, n_s = err(spec_pairs)
    eul_err, n_e = err(eul_pairs)
    return {"spectral_median_rel_err": spec_err, "n_spectral": n_s,
            "euler_median_rel_err": eul_err, "n_euler": n_e}


def target_recovery(world, res, radius_px=8):
    """What fraction of predicted top targets sit near a TRUE deposit/stock?"""
    truth = set()
    for (r, c) in world["deposit_rcs"]:
        truth.add((r, c))
    hits = 0
    for t in res.targets[:15]:
        r, c = t["row"], t["col"]
        if any((r - tr) ** 2 + (c - tc) ** 2 <= radius_px ** 2
               for tr, tc in truth):
            hits += 1
    return hits, min(15, len(res.targets))


def main():
    summary = {}
    for name in ("signal", "confound", "mixed"):
        print("=" * 70)
        print(f"SCENARIO: {name}")
        print("=" * 70)
        try:
            world, res = run_scenario(name)
        except Exception:
            traceback.print_exc()
            summary[name] = {"status": "CRASH"}
            continue

        cv = res.cv_report
        ho = res.holdout_metrics or {}
        print(f"deposits: train={res.n_deposits_train} "
              f"holdout={res.n_deposits_holdout}")
        print(f"spatial CV AUC : {cv.auc_mean:.3f} +/- {cv.auc_std:.3f} "
              f"({cv.n_folds_used} folds)")
        if ho:
            m = ho.get('contrast_auc_matched')
            print(f"holdout        : contrast AUC {ho['contrast_auc']:.3f}"
                  + (f" | MATCHED {m:.3f}" if m is not None else "")
                  + f" | mean@dep {ho['mean']:.3f} vs bg {ho['background_mean']:.3f}"
                  + f" | hit@0.50 {ho['hit_rate_050']*100:.0f}%")
        from geocore.validation import synthesize_verdict
        vd = synthesize_verdict(cv, ho or {})
        print("engine verdict :", vd['verdict'], '|', '; '.join(vd['reasons']))
        print("importances    :", ", ".join(f"{n}={v:.2f}"
                                            for n, v in res.importances[:5]))
        hits, n = target_recovery(world, res)
        print(f"top-target truth hits: {hits}/{n}")

        entry = {
            "cv_auc": round(cv.auc_mean, 3),
            "holdout_auc": round(ho.get("contrast_auc", float("nan")), 3) if ho else None,
            "holdout_auc_matched": (round(ho["contrast_auc_matched"], 3)
                                     if ho and "contrast_auc_matched" in ho else None),
            "engine_verdict": vd["verdict"],
            "hit_050": ho.get("hit_rate_050") if ho else None,
            "importances": [(n, round(v, 3)) for n, v in res.importances[:5]],
            "target_hits": f"{hits}/{n}",
            "warnings": res.warnings,
        }

        if name == "signal":
            key = ho.get("contrast_auc_matched", ho.get("contrast_auc", 0)) if ho else 0
            ok = ho and key >= 0.70 and ho["hit_rate_050"] > 0 and vd["verdict"] == "VALIDATED"
            entry["verdict"] = "PASS (recovers real signal)" if ok else "FAIL"
            if res.depth_spectral is not None or res.depth_euler is not None:
                d = grade_depths(world, res)
                entry["depth"] = d
                print("depth grading  :", d)
        elif name == "confound":
            # Pass = the engine's own headline verdict exposes the artifact
            ok = vd["verdict"] in ("NO_SIGNAL", "ARTIFACT")
            entry["verdict"] = ("PASS (artifact exposed honestly)" if ok
                                 else "FAIL (false confidence!)")
        else:
            entry["verdict"] = "REPORTED (partial recovery expected)"
        print("VERDICT:", entry["verdict"])
        summary[name] = entry

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    with open("/tmp/gc_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
