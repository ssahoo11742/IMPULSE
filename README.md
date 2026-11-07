# IMPULSE V1.1: Inferred Microdebris Population from Unmaneuvered LEO State Evolution

## Abstract

The sub-catalog (≤ 10 cm) orbital debris environment is poorly constrained; NASA's ORDEM 3.1 and ESA's MASTER 8 diverge by over two orders of magnitude in critical LEO regimes. IMPULSE tests whether orbits of decommissioned, non-maneuvering satellites contain tiny signals of microdebris impacts that could discriminate between the models. Historical TLE data are processed with a 9-state augmented Extended Kalman Filter and Fraser-Potter smoother (non-singular (h, k) eccentricity) to flag statistically significant orbital deviations as candidate impulses. Measurement covariance is estimated per object from its own TLE stream. Validation sets the design threshold η = 417.94 (99.9th percentile of 18,582 synthetic null trials) and certifies 50% detection at ∆v50 = 1.19 m/s (1.21 m/s on real TLEs). Applied to 3,325 spacecraft (15,760 windows), the pipeline yields 6 candidates against 15.76 expected from the design noise rate: the count lies below that expectation (P ≈ 0.005) yet above what the more realistic permuted null predicts, so no positive excess is certifiable against either background and the one-sided 95% Poisson upper limit, conditional on the design background, is zero. Candidates carry 1.5–15 m/s impulses, compatible only with 3–6 cm head-on strikes. In the decision-relevant 1–10 cm band the models agree within a factor of ∼ 3 and predict essentially no events; zero detections is therefore fully consistent. IMPULSE demonstrates fleet-scale synthetic sensitivity and places a weak bound (≈ 280–570× the model rate) on the ≥ 2–3 cm impact rate that reflects present performance limits rather than discriminating the models.

**Erratum:** an erratum to V1 is included in this record (see below). V1 remains archived at its own DOI; V1.1 supersedes it.

## Paper Links

- **V1.1 DOI:** [10.5281/zenodo.24025405](http://doi.org/10.5281/zenodo.24025405)
- **V1 DOI (superseded):** [10.5281/zenodo.23852319](http://doi.org/10.5281/zenodo.23852319)
- **Erratum to V1:** 
- Full PDFs of V1, V1.1, and the erratum are included in this record alongside the code.

## Contents of This Record


| Item                           | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `IMPULSE_V1.1_FULL.pdf`        | Final V1.1 paper (Oct 2026)                                                 |
| `IMPULSE_V1_FULL.pdf`          | V1 paper (Sept 2026), superseded by V1.1                                    |
| `IMPULSE_V1_Erratum.pdf`       | Correction notice for V1 (measurement-covariance error)                     |
| `phase2_filter/`               | Filter, smoother, validation, and analysis pipeline (this README covers it) |
| `results/`                     | Frozen result JSONs backing every number, figure, and table in the papers   |
| Supporting packages            | `TLE/`, `constants/`, `propagator/` (see Requirements)                      |


## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `sgp4` (python-sgp4)
- Repository packages (must remain importable): `TLE.tle_io` (TLE parsing), `constants.constants` (MU, F107 baseline, Kp, etc.), `propagator.orbital` (mean elements, Brouwer rates), `propagator.debris_impacts` (NSBM fragment masses, VMF impact velocities; needed only for the model-prediction stages)
- TLE histories are **not** redistributed (see Data Prerequisites)

## Data Prerequisites

The pipeline consumes, from outside this record:

1. **TLE histories** — one `.tle` file per NORAD ID (Space-Track / Celestrak), for the 3,325 decommissioned, non-maneuvering spacecraft in the fleet, plus a metadata JSON with `NORAD_CAT_ID`, altitude, inclination, etc.
2. **ORDEM 3.1 and MASTER 8 flux files** — per-object debris flux tables in the band of interest (1–10 cm and ≥ 2–3 cm regimes).
3. **NSBM debris-impact model inputs** — fragment length distributions and impact velocity models used by `propagator.debris_impacts`.

Exact file paths are passed per script via CLI flags; see each script's `--help`.

## Pipeline Overview

```
run_validation.py                (filter sanity ladder)
        │
convention_consistency_test.py   (Gate 1: TLE-convention consistency,
        │                         pooled synthetic null → η = 417.94)
        ├── gate2_confounders.py          (Gate 2: confounder nulls,
        │                                  pooled-null-matched)
        ├── real_tle_strike_injection.py  (Gate 3: real-TLE Δv recovery,
        │                                  Δv50 = 1.21 m/s, 0/149 floor)
        └── find_dv_threshold.py          (Gate 4: synthetic impulse sweep,
                                           Δv50 = 1.19 m/s, ε(Δv) curve)
        │
run_real_fleet.py               (fleet application: 6 candidates / 15,760 windows)
        │
scrambled_tle_null.py           (permuted / scrambled null, residual mode)
        │
interpolate_flux_to_fleet.py ── eval_fleet_efficiency.py
                                (model comparison: expected detectable events)
        │
analysis/v1-x/cov-fig.py     (Figure 5: per-object covariance ratios)
```

Core machinery shared by all stages: `config.py` (tunables), `dynamics.py`, `ekf.py`, `smoother.py` (Fraser-Potter), `test_statistics.py` (Mahalanobis), `linalg_utils.py`, and `validation.py` (truth propagators; also an import target for the consistency test).

## Reproduction

Run from the repository root (`phase2_filter` importable as a package). Every stage writes a JSON under `results/`; all V1.1 numbers trace to those files.

**One-shot driver:** `python pipeline.py` runs every stage below in dependency order with these exact configurations (see `pipeline.py --list-stages`, `--dry-run`, `--stages`, `--eta`). It skips stages whose outputs already exist, and its `report` stage prints a side-by-side comparison against the paper's numbers. If `TLE/real_tles.tle.meta.json` or the durations JSON is absent, its `prepare` stage derives stand-ins from the repo's own data.

**0. Filter validation ladder** (null / bias / impulse / detect demos):

```bash
python -m phase2_filter.run_validation --step all
```

**1. Pooled synthetic null → design threshold** (18,582 trials, SGP4-consistent convention pipeline, jittered geometry, honest per-trial R):

```bash
python -m phase2_filter.convention_consistency_test \
    --mode null --truth-mode sgp4_consistent --n-trials 18582 \
    --fit-bstar 1 --r-from-fit 1 --jitter-geometry 1 \
    --percentile 99.9 --seed-offset 500000 \
    --output results/synthetic_null_eta_honestR.json
```

→ η = 417.94 (99.9th percentile).

**2. Confounder nulls (Gate 2):**

```bash
python -m phase2_filter.gate2_confounders \
    --threshold 417.94 --n-trials 2000 \
    --output results/gate2_confounders_v2.json
```

**3. Real-TLE strike injection (Gate 3):**

```bash
python -m phase2_filter.real_tle_strike_injection \
    --tle-history-dir TLE/histories \
    --meta TLE/real_tles.tle.meta.json \
    --durations TLE/real_tles.tle.durations_2009_2026.json \
    --threshold 417.94 --r-from-fit 1 \
    --output results/real_tle_efficiency.json
```

→ Δv50 = 1.21 m/s; 0/149 at negligible Δv (zero false triggers on real TLE streams).

**4. Synthetic impulse sweep / efficiency curve (Gate 4)** (honest R in impulse mode too, FIX #16):

```bash
python -m phase2_filter.find_dv_threshold \
    --mode impulse --truth-mode sgp4_consistent \
    --threshold 417.94 --n-trials 200 \
    --dv-grid 0.05,0.1,0.2,0.4,0.7,1.0,1.2,1.5,2.0,3.0,5.0,10.0,20.0 \
    --fit-bstar 1 --r-from-fit 1 --jitter-geometry 1 \
    --seed-offset 560000 \
    --output results/gate4_efficiency_sweep.json
```

→ Δv50 = 1.19 m/s; ε(Δv) efficiency curve feeding the model-prediction stage. (`--n-trials` is per grid point.)

**5. Fleet application** (per-object honest R, production 7-day edge trim):

```bash
python -m phase2_filter.run_real_fleet \
    --tle-history-dir TLE/histories \
    --meta TLE/real_tles.tle.meta.json \
    --durations TLE/real_tles.tle.durations_2009_2026.json \
    --threshold 417.94 --r-from-fit 1 --n-workers 4 \
    --output results/real_detections_honestR.json
```

→ 6 candidates in 15,760 valid windows (0.038%), vs 15.76 expected at the design rate.

**6. Permuted null** (residual-mode reference):

```bash
python -m phase2_filter.scrambled_tle_null \
    --tle-history-dir TLE/histories \
    --meta TLE/real_tles.tle.meta.json \
    --durations TLE/real_tles.tle.durations_2009_2026.json \
    --threshold 417.94 --r-from-fit 1 --scramble-mode residual \
    --output results/scrambled_null_residual.json
```

**7. Model comparison** (flux → fleet, then expected detectable events with window-based exposure; 76 searched days per window is the paper convention):

```bash
python flux_analysis/flux_compare.py \
    --ordem-dir flux_analysis/results-ORDEM \
    --ordem-manifest flux_analysis/manifest.txt \
    --master-dir flux_analysis/results-MASTER \
    --output flux_analysis/flux_comparison.csv

python -m phase2_filter.interpolate_flux_to_fleet \
    --flux-csv flux_analysis/flux_comparison.csv \
    --tle-file TLE/real_tles.tle \
    --meta TLE/real_tles.tle.meta.json \
    --durations TLE/real_tles.tle.durations_2009_2026.json \
    --output-ordem results/ordem_flux_per_object.json \
    --output-master results/master_flux_per_object.json

python -m phase2_filter.eval_fleet_efficiency \
    --efficiency results/gate4_efficiency_sweep.json \
    --ordem-flux results/ordem_flux_per_object.json \
    --master-flux results/master_flux_per_object.json \
    --windows-file results/real_detections_honestR.json \
    --window-days 76 \
    --output results/fleet_expected_detected.json
```

**8. Figure 5** (per-object covariance ratios; script `analysis/v1-x/cov-fig.py`, needs matplotlib):

```bash
python analysis/v1-x/cov-fig.py \
    --fleet-file results/real_detections_honestR.json \
    --out fig5_covariance_ratios.png
```

## Key Results at a Glance


| Quantity                          | Value                                                     | Source        |
| --------------------------------- | --------------------------------------------------------- | ------------- |
| Fleet                             | 3,325 spacecraft, 15,760 valid windows                    | Gate 5 output |
| Candidates                        | 6 (0.038% of windows)                                     | Gate 5 output |
| Expected at design rate           | 15.76 (P(≤6) ≈ 0.0047, a deficit)                         | Gate 5 output |
| Design threshold η                | 417.94 (99.9th pct, 18,582-trial pooled synthetic null)   | Stage 1       |
| Δv50 (synthetic / real TLE)       | 1.19 / 1.21 m/s                                           | Gates 4 / 3   |
| Candidate impulses                | 1.5–15 m/s (≥ 3–6 cm head-on strikes)                     | Gate 5 output |
| Certifiable excess vs either null | None (one-sided 95% UL, given design background, is zero) | Gates 5–6     |
| Implied bound                     | ≈ 570× ORDEM, ≈ 280× MASTER rates at ≥ 2–3 cm             | Stage 7       |


## Method Notes for Reuse

- **Per-object ("honest") R:** measurement covariance is estimated per object from its own TLE stream's one-step prediction residuals — median of squared residuals / 0.4549364 (median of χ²₁), floored at the literature `R_DIAG_ELEMENTS` and capped at 100×. This is the central V1.1 correction over V1's population-wide assumed R; see the erratum and V1.1 §3.2.
- **Statistic:** window-max Mahalanobis distance on the a-channel, evaluated on the window interior only (7-day edge trim), matching the synthetic null.
- **Candidates are not detections:** IMPULSE certifies zero debris-impact detections; absence of a certifiable excess is not evidence of absence of impacts.
- **Known code organization quirk:** `convention_consistency_test.py` and `find_dv_threshold.py` are near-duplicates; the honest-R-in-impulse-mode fix (FIX #16) exists only in `find_dv_threshold.py`, while `convention_consistency_test.py` is the import target for Gates 2–3. If refactoring, merge FIX #16 into `convention_consistency_test.py` and keep that module name.

## Citation

```bibtex
@software{sahoo2026impulse_v1_1,
  author  = {Sahoo, Swayam},
  title   = {{IMPULSE V1.1}: Quantifying the Real-Data Limits of Passive Orbital Sensing for Sub-Catalog Debris-Environment Model Discrimination},
  year    = {2026},
  doi     = {10.5281/zenodo.24025405},
  url     = {http://doi.org/10.5281/zenodo.24025405},
  note    = {V1 (doi:10.5281/zenodo.23852319) is superseded by this release;
             an erratum to V1 is included}
}
```

## License



If you use IMPULSE, its data pipeline, or its null-calibration methodology, please cite both the V1.1 paper and this record.
