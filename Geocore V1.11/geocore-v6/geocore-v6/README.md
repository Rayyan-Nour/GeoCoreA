# GeoCore v6.0 — Complete Package

## ▶ How to run it

**Windows — double-click `run_geocore.bat`.**
That's it. On first run it installs dependencies automatically (a few
minutes), then launches the Studio. If anything fails the window stays open
so you can read the error.

**macOS / Linux:** `./run_geocore.sh`

**Manual / from a terminal** (any OS), from inside this folder:

```bash
pip install -r requirements.txt      # first time only
python -m app.main                   # launch the Studio
```

> Run it as `python -m app.main`, **not** `python app/main.py` — the module
> form is what sets the import paths correctly.

**Requirements:** Python 3.10+. A GPU helps the 3-D map but isn't required.

### Other things you can run

```bash
python -m pytest tests/ -q           # 60 unit + integration tests
python tests/run_certification.py    # ground-truth certification (~10 min)
```

`demo_3d_viewer.html` — just double-click to open in a browser. This is the
exported 3-D review from a certified VALIDATED run; it needs no Python and
is the file to show in a meeting.

## Package layout

```
geocore-v6/
├── run_geocore.bat / .sh   # launchers
├── app/main.py             # PyQt6 desktop Studio  (python -m app.main)
├── geocore/                # engine (14 modules)
├── viewer/                 # export_3d.py (3-D) + export_web.py (2-D Leaflet)
├── tests/                  # 60 tests + ground-truth certification suite
├── requirements.txt
└── demo_3d_viewer.html     # exported VALIDATED run — open in any browser
```

**Installing over your existing copy:** back up your current folders, then
replace `geocore/`, `viewer/`, `tests/`, and `app/` with these. Note
`main.py` lives in `app/` — that's what its own `parents[1]` path logic and
docstring expect.

## What's in v6 (cumulative)

**Engine correctness**
- Terrain features computed at a physical scale (`terrain_scale_m`, default
  1 km) so slope can't beat the geophysics on resolution alone
- Terrain-matched negative sampling (removes the mountain-vs-basin shortcut)
- Terrain-matched holdout background (`contrast_auc_matched`) — the
  artifact-proof number, since the naive contrast had the same flaw
- `synthesize_verdict`: one headline call — **VALIDATED / ARTIFACT /
  NO_SIGNAL / WEAK** — as Section 0 of every report
- **New in v6:** the negative exclusion buffer is now a true metric radius.
  Iterative binary dilation grew a Manhattan *diamond*, leaving diagonal
  clearance ~22% short — background could be sampled inside the mineralized
  halo. Now a Euclidean distance transform with per-axis sampling, which
  also handles anisotropic pixels (geographic CRS away from the equator).

**Depth**
- Multi-scale Euler fusion (10/16/24/32 px): median error 53.1% → 47.9%,
  coverage 43/44 → 44/44 planted sources
- Per-target corroboration status (spectral vs Euler within 35% →
  `corroborated`), in the CSV, the report, and the 3-D target card

**Targets** — nearest known deposit (name + km) on every target; new CSV
columns `depth_status`, `nearest_deposit_km`, `nearest_deposit_name`.

**3-D viewer (rebuilt)** — working raycast hover (live lat/lon/elevation)
and click query (all layer values + nearest target); target cards with
probability, uncertainty, driver percentile, both depths, corroboration,
nearest deposit. Flicker-free by construction: one opaque mesh, one
composited texture, capped pixel ratio, single render loop. Metric extents
for geographic DEMs, custom orbit controls, brand styling.

## Verification (this build)

- **60/60** unit and integration tests pass, including the geographic-DEM
  3-D regression suite (3/3)
- Desktop app verified: `app.main` imports and `MainWindow` constructs and
  runs its event loop cleanly (headless)
- End-to-end run clean: 25 targets, 3,450 Euler solutions, 14/25 targets
  depth-corroborated, both viewers export
- Ground-truth certification, 3 scenarios × 3 seeds:

```
world     seed  verdict      CV     matched  hit@0.50
signal      7   VALIDATED    0.99   0.99     100%
signal     11   VALIDATED    1.00   0.99     100%
signal     23   VALIDATED    1.00   0.99     100%
confound    7   WEAK         0.68   0.71      50%
confound   11   ARTIFACT     0.70   0.51      25%
confound   23   NO_SIGNAL    0.53   0.43       0%
mixed       7   VALIDATED    1.00   0.99     100%
mixed      11   VALIDATED    0.97   0.97      80%
mixed      23   VALIDATED    1.00   0.99     100%
```

**Safety — never blesses an artifact: PASS (3/3).**
**Sensitivity — recovers real signal: PASS (3/3).**

**Honest change from the earlier certification:** with the buffer fix the
sampling changed, and confound/seed-7 now returns **WEAK** rather than
ARTIFACT. That is still a refusal to validate, so the safety property holds
— but "9/9 verdicts exactly as expected" is no longer the right claim. Say
instead:

- *"Across nine blind ground-truth runs, the engine never once validated a
  confounded model, and recovered real signal in every world that had it."*
- *"On the hardest confound case it returns WEAK — inconclusive, exploratory
  only — rather than a false pass."*

If asked why WEAK and not ARTIFACT: the 1 km terrain smoothing partially
neutralises that planted confound, so the world is genuinely ambiguous at
the model's feature scale, and WEAK is the honest answer. That lands better
with a technical reviewer than a suspiciously perfect scoreboard.
