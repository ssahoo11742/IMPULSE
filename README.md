# IMPULSE V1.1: Inferred Microdebris Population from Unmaneuvered LEO State Evolution

## Abstract

The sub-catalog (≤ 10 cm) orbital debris environment is poorly constrained; NASA's ORDEM 3.1 and ESA's MASTER 8 diverge by over two orders of magnitude in critical LEO regimes. IMPULSE tests whether orbits of decommissioned, non-maneuvering satellites contain tiny signals of microdebris impacts that could discriminate between the models. Historical TLE data are processed with a 9-state augmented Extended Kalman Filter and Fraser-Potter smoother (non-singular (h, k) eccentricity) to flag statistically significant orbital deviations as candidate impulses. Measurement covariance is estimated per object from its own TLE stream. Validation sets the design threshold η = 417.94 (99.9th percentile of 18,582 synthetic null trials) and certifies 50% detection at ∆v50 = 1.19 m/s (1.21 m/s on real TLEs). Applied to 3,325 spacecraft (15,760 windows), the pipeline yields 6 candidates against 15.76 expected from the design noise rate: the count lies below that expectation (P ≈ 0.005) yet above what the more realistic permuted null predicts, so no positive excess is certifiable against either background and the one-sided 95% Poisson upper limit, conditional on the design background, is zero. Candidates carry 1.5–15 m/s impulses, compatible only with 3–6 cm head-on strikes. In the decision-relevant 1–10 cm band the models agree within a factor of ∼ 3 and predict essentially no events; zero detections is therefore fully consistent. IMPULSE demonstrates fleet-scale synthetic sensitivity and places a weak bound (≈ 280–570× the model rate) on the ≥ 2–3 cm impact rate that reflects present performance limits rather than discriminating the models.

**Erratum:** an erratum to V1 is included in this record (see below). V1 remains archived at its own DOI; V1.1 supersedes it.

## Paper Links

- **V1.1 DOI:** [10.5281/zenodo.24025405](http://doi.org/10.5281/zenodo.24025405)
- **V1 DOI (superseded):** [10.5281/zenodo.23852319](http://doi.org/10.5281/zenodo.23852319)
- **Erratum to V1:** 
- Full PDFs of V1, V1.1, and the erratum are included in this record alongside the code.

The final result is scientifically meaningful: IMPULSE V1 demonstrates that passive orbital histories contain detectable structure, but that TLE-based dynamical anomaly detection under the baseline operating regime is fundamentally limited by an irreducible systematic floor. We quantify this floor, identify its sources, and chart a path toward V2 strategies that may break the degeneracy.



### Initial Attempt

The first iteration of this project that was naively implemented and initially seen as a success is below.

https://github.com/ssahoo11742/DRIFTS

