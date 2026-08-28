"""
Validation report writer. Every number a client sees comes with its method
and its caveats. No claim in this report is one your geostatistician
reviewer can dismantle.
"""
from __future__ import annotations

from typing import Dict, List


def write_report(path: str, cfg, spec, cv, holdout: Dict,
                 importances: List, grade_ctx, n_train: int, n_holdout: int,
                 n_euler: int, warnings: List[str],
                 targets: list = None, verdict: Dict = None) -> None:
    lines = []
    a = lines.append

    a("# GeoCore Analytics — Validation Report")
    a("")
    a(f"**Commodity:** {spec.label} ({spec.deposit_type})")
    a(f"**Training deposits:** {n_train}  |  **Spatial holdout deposits:** {n_holdout}")
    a("")
    if verdict:
        a("## 0. Verdict")
        a("")
        a(f"**{verdict['verdict']}**")
        for reason in verdict.get("reasons", []):
            a(f"- {reason}")
        a("")
        a("*VALIDATED = spatially generalizable signal. ARTIFACT = in-region "
          "skill that fails out-of-region. NO_SIGNAL = honest metrics show no "
          "usable model. WEAK = inconclusive; exploratory only.*")
        a("")
    a("## 1. Spatial block cross-validation")
    a("")
    a("Folds are contiguous geographic blocks (Valavi et al., 2019); every")
    a("sample — positive *and* negative — is assigned by its true location.")
    a("This is the metric to quote. Random-split AUCs on spatial data are")
    a("inflated and are not reported by this software.")
    a("")
    if cv.n_folds_used:
        a(f"- ROC-AUC: **{cv.auc_mean:.3f} ± {cv.auc_std:.3f}** "
          f"({cv.n_folds_used} usable folds)")
        a(f"- Average precision: {cv.ap_mean:.3f}")
    else:
        a("- Insufficient spatial spread for block CV — treat this run as")
        a("  exploratory only.")
    a("")
    a("## 2. Spatially held-out deposit test")
    a("")
    if holdout:
        a("Deposits in one geographic quadrant were hidden from training,")
        a("then scored against random background pixels:")
        a("")
        a(f"- Held-out deposits: {holdout['count']}")
        a(f"- Mean probability at held-out deposits: {holdout['mean']:.3f} "
          f"(background mean: {holdout['background_mean']:.3f})")
        a(f"- Hit rate ≥0.50: {holdout['hit_rate_050']*100:.0f}%  |  "
          f"≥0.70: {holdout['hit_rate_070']*100:.0f}%")
        a(f"- Deposit-vs-background contrast AUC: "
          f"**{holdout['contrast_auc']:.3f}**")
        if "contrast_auc_matched" in holdout:
            a(f"- Terrain-matched contrast AUC: "
              f"**{holdout['contrast_auc_matched']:.3f}** "
              f"(background matched to holdout terrain; the artifact-proof "
              f"number to quote)")
    else:
        a("Not enough deposits for a spatial holdout in this run.")
    a("")
    a("## 3. Feature importance")
    a("")
    for name, imp in importances[:10]:
        a(f"- {name}: {imp:.4f}")
    a("")
    a("## 4. Depth-to-source estimation")
    a("")
    if n_euler or cfg.estimate_depth:
        a("Depths are estimates of **magnetic source bodies** below the")
        a("sensor datum — a drilling-vector constraint, not an ore depth")
        a("measurement.")
        a("")
        a("- Spectral method: Spector & Grant (1970), windowed radially")
        a("  averaged power spectrum.")
        a(f"- Euler deconvolution: Reid et al. (1990), structural index "
          f"{cfg.euler_structural_index} "
          f"({n_euler} accepted solutions).")
        a("- Where the two methods agree, confidence is higher; where they")
        a("  disagree, trust neither without follow-up geophysics.")
    else:
        a("Not computed (no magnetics layer provided).")
    a("")
    a("## 5. Grade context — read carefully")
    a("")
    a(f"Reference: {grade_ctx.reference}")
    a("")
    a(grade_ctx.statement)
    a("")
    a("**This software does not produce grade maps.** Per-pixel grade")
    a("imagery derived from prospectivity probability is not scientifically")
    a("supportable and creates regulatory exposure (cf. NI 43-101 / JORC")
    a("standards on reporting mineralization).")
    a("")
    if targets:
        a("## Ranked targets")
        a("")
        a("Local probability maxima above the high-prospectivity threshold, "
          "minimum-separation filtered. Full table with per-target drivers "
          "in `geocore_targets.csv`.")
        a("")
        a("| # | Lat | Lon | Prob | Unc | Depth (spec) | Depth (Euler) | "
          "Top driver |")
        a("|---|-----|-----|------|-----|--------------|---------------|"
          "------------|")
        for t in targets[:15]:
            drv = (f"{t['why'][0]['feature']} "
                   f"(P{t['why'][0]['percentile']})") if t["why"] else "-"
            ds = ("-" if t["depth_spectral_m"] is None
                  else f"{t['depth_spectral_m']:.0f} m")
            de = ("-" if t["depth_euler_m"] is None
                  else f"{t['depth_euler_m']:.0f} m")
            un = ("-" if t["uncertainty"] is None
                  else f"{t['uncertainty']:.2f}")
            a(f"| {t['rank']} | {t['lat']:.5f} | {t['lon']:.5f} | "
              f"{t['probability']:.3f} | {un} | {ds} | {de} | {drv} |")
        cor = sum(1 for t in targets if t.get("depth_status") == "corroborated")
        sng = sum(1 for t in targets if t.get("depth_status") == "single-method")
        dsc = sum(1 for t in targets if t.get("depth_status") == "discrepant")
        if cor or sng or dsc:
            a("")
            a(f"Depth corroboration across all targets: {cor} corroborated "
              f"(spectral vs Euler within 35%) · {sng} single-method · "
              f"{dsc} discrepant. Corroborated depths carry the highest "
              f"confidence; discrepant ones need follow-up geophysics.")
        a("")

    a("## 6. Limitations (disclose these to every client)")
    a("")
    a("- Training labels come from public databases (e.g. USGS MRDS) which")
    a("  mix producers, prospects, and occurrences; label noise is real")
    a("  (Zuo & Wang, 2020).")
    a("- Prospectivity reflects *data availability* as much as geology:")
    a("  well-explored areas have more labels. Negative samples are")
    a("  unvisited, not proven barren.")
    a("- Predictions are for target *ranking*, not discovery claims. The")
    a("  product of this analysis is a prioritized exploration plan,")
    a("  validated only by drilling.")
    if warnings:
        a("")
        a("## 7. Run warnings")
        a("")
        for w in warnings:
            a(f"- {w}")
    a("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_executive_summary(path: str, cfg, spec, cv, holdout: Dict,
                            grade_ctx, targets: list,
                            n_deposits: int) -> None:
    """Plain-language report for non-technical readers."""
    L = []
    a = L.append
    a(f"# Exploration Targeting Summary - {spec.label}")
    a("")
    a("*Prepared by GeoCore Analytics. The technical companion with full "
      "methodology and validation is geocore_validation_report.md.*")
    a("")
    a("## What was done")
    a("")
    a(f"Public geoscience data (terrain, geophysics, geochemistry) was "
      f"combined with the locations of {n_deposits} known "
      f"{spec.label.lower()} occurrences to train a model that "
      f"ranks ground by how similar it is to places where deposits are "
      f"already known. The output is a map where every location gets a "
      f"score from 0 to 1.")
    a("")
    a("## How much to trust it")
    a("")
    pct = int(round(cv.auc_mean * 100))
    a(f"Tested the honest way - by hiding known deposits from the model "
      f"and checking whether it finds the ground around them - the model "
      f"scores **{cv.auc_mean:.2f} out of 1.0** (AUC). In plain terms: "
      f"shown one random deposit location and one random empty location, "
      f"the model picks the deposit roughly {pct} times out of 100. A "
      f"coin flip would be 50.")
    ho = holdout.get("contrast_auc") if holdout else None
    if ho:
        a("")
        a(f"A second test on deposits in a region the model never saw "
          f"during training scored {ho:.2f}.")
    a("")
    a("## Top targets")
    a("")
    if targets:
        a("| # | Location (lat, lon) | Score | Est. source depth |")
        a("|---|---------------------|-------|-------------------|")
        for t in targets[:10]:
            d = t["depth_spectral_m"] or t["depth_euler_m"]
            ds = f"~{d:,.0f} m" if d else "n/a"
            a(f"| {t['rank']} | {t['lat']:.4f}, {t['lon']:.4f} | "
              f"{t['probability']:.2f} | {ds} |")
    else:
        a("No locations exceeded the high-prospectivity threshold in this "
          "area.")
    a("")
    a("## What the numbers mean - and what they don't")
    a("")
    a("- A high score means *the ground looks like ground that has "
      "produced deposits before*. It is a reason to spend follow-up "
      "budget there, not a guarantee of a discovery.")
    a("- Depth figures estimate how deep the magnetic rocks causing the "
      "anomaly sit. They guide where a drill hole should aim; they are "
      "not the depth of ore.")
    a(f"- If a deposit of this type exists, history says: "
      f"{grade_ctx.statement}")
    a("- Only drilling proves grade. Every mining company knows this; "
      "this report does not pretend otherwise.")
    a("")
    a("## Recommended next steps")
    a("")
    a("1. Field reconnaissance of the top 5 targets (mapping, sampling)")
    a("2. Detailed ground geophysics over targets that survive step 1")
    a("3. Drill testing of the best-supported targets")
    a("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
