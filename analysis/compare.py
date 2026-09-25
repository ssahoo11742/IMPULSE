"""
Compare an OBSERVED TLE history against ORDEM- and MASTER-predicted histories,
and against the Phase-2 EKF + Fraser–Potter smoother + detection statistics.

Outputs:
  - history_observed.csv, history_ordem.csv, history_master.csv
  - sma_observed.png
  - sma_ordem.png
  - sma_master.png
  - sma_combined.png
  - sma_annotated.png
  - sma_zoomed.png
  - sma_zoomed_forward.png       ← (Forward EKF only)
  - sma_zoomed_backward.png      ← (Backward EKF only)
  - sma_zoomed_smoothed.png
  - detection_zoomed.png          ← (Mahalanobis + McReynolds + threshold)
"""

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Your Phase-2 package imports  ← ADJUST PATH if needed
# ---------------------------------------------------------------------------
from phase2_filter import config                                     # ← ADJUST
from phase2_filter.validation import (                               # ← ADJUST
    _run_filter_pair,
)
from phase2_filter.test_statistics import (                        # ← ADJUST
    compute_mahalanobis_distance,
    compute_mcreynolds,
)
from propagator.orbital import MeanElements

# --- Physical Constants ---
MU = 3.986004418e14
RE_EQ = 6378137.0
R_EARTH = 6371000.0
J2 = 1.08262668355e-3
J4 = -1.61962e-6
CD = 2.2
SECONDS_PER_DAY = 86400.0

ATM_TABLE = [
    (200.0, 2.5e-8, 37.0),
    (300.0, 5.0e-11, 45.0),
    (400.0, 1.0e-12, 53.0),
    (500.0, 5.0e-13, 58.0),
    (600.0, 1.5e-13, 65.0),
    (800.0, 1.0e-14, 76.0),
]

FLUX = {"ORDEM": 1.553, "MASTER": 0.01118}
RHO_MAT = 2800.0
SIZE_LO, SIZE_HI = 1e-3, 1e-2
V_LO, V_HI = 8e3, 14e3


@dataclass
class MeanElements:
    a: float
    ecc: float
    inc: float
    raan: float
    argp: float
    M: float


def filter_sma_outliers(recs, threshold_km=1.0):
    if len(recs) < 3:
        return recs
    valid_recs = [recs[0]]
    for i in range(1, len(recs) - 1):
        prev_a = valid_recs[-1]["el"].a / 1000.0
        curr_a = recs[i]["el"].a / 1000.0
        next_a = recs[i + 1]["el"].a / 1000.0
        if abs(curr_a - prev_a) > threshold_km and abs(curr_a - next_a) > threshold_km:
            continue
        valid_recs.append(recs[i])
    valid_recs.append(recs[-1])
    return valid_recs


def parse_tles(path):
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip()]
    recs, seen, i = [], set(), 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            name = f"OBJ_{lines[i][2:7].strip()}"
            l1 = lines[i]
            l2 = lines[i + 1] if i + 1 < len(lines) else ""
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 "):
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        if not l2.startswith("2 "):
            continue
        ep = l1[18:32].strip()
        if ep in seen:
            continue
        seen.add(ep)
        y2, doy = int(ep[:2]), float(ep[2:])
        yr = 2000 + y2 if y2 < 57 else 1900 + y2
        epoch = datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        el = MeanElements(
            a=(MU / (float(l2[52:63]) * 2 * math.pi / 86400.0) ** 2) ** (1 / 3),
            ecc=float("0." + l2[26:33].strip()),
            inc=math.radians(float(l2[8:16])),
            raan=math.radians(float(l2[17:25])),
            argp=math.radians(float(l2[34:42])),
            M=math.radians(float(l2[43:51])),
        )
        recs.append({"name": name.strip(), "norad": l1[2:7].strip(), "epoch": epoch, "el": el})
    recs.sort(key=lambda r: r["epoch"])
    return filter_sma_outliers(recs, threshold_km=1.0)


def brouwer_rates(a, ecc, inc):
    n = math.sqrt(MU / a**3)
    p = a * (1.0 - ecc**2)
    eta = math.sqrt(1.0 - ecc**2)
    ci, si2 = math.cos(inc), 1.0 - math.cos(inc) ** 2
    re_p2 = (RE_EQ / p) ** 2
    d_raan = -1.5 * n * J2 * re_p2 * ci
    d_argp = 0.75 * n * J2 * re_p2 * (4.0 - 5.0 * si2)
    dn_j2 = 1.5 * n * J2 * re_p2 * eta * (1.0 - 1.5 * si2)
    g4 = -3.0 / 8.0 * J4 * (RE_EQ / p) ** 4
    d_raan += n * g4 * (5.0 / 4.0) * ci * (3.0 * si2 - 4.0)
    d_argp += n * g4 * (35.0 / 8.0) * (1.0 - 11.0 / 5.0 * ci**2 - 8.0 / 7.0 * ci**2**2) / eta
    return n, d_raan, d_argp, dn_j2


def density(alt_m):
    alt_km = alt_m / 1000.0
    rho0, h0, H = ATM_TABLE[0][1], ATM_TABLE[0][0], ATM_TABLE[0][2]
    for h_min, r, sc in ATM_TABLE:
        if alt_km >= h_min:
            rho0, h0, H = r, h_min, sc
        else:
            break
    return rho0 * math.exp(-(alt_km - h0) / H)


def kepler_nu(M, e):
    E = M if e < 0.8 else math.pi
    for _ in range(12):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    return math.atan2(math.sqrt(1 - e**2) * math.sin(E), math.cos(E) - e)


def gauss_vop_kick(el, dR, dS, dW):
    a, e, i = el.a, el.ecc, el.inc
    p = a * (1 - e**2)
    h = math.sqrt(MU * p)
    nu = kepler_nu(el.M % (2 * math.pi), e)
    r = p / (1 + e * math.cos(nu))
    u = (el.argp + nu) % (2 * math.pi)
    b = a * math.sqrt(1 - e**2)
    da = (2 * a**2 / h) * (e * math.sin(nu) * dR + (p / r) * dS)
    de = (p * math.sin(nu) * dR + ((p + r) * math.cos(nu) + r * e) * dS) / h
    di = (r * math.cos(u) / h) * dW
    draan = (r * math.sin(u) / (h * math.sin(i))) * dW
    dargp = (-p * math.cos(nu) * dR + (p + r) * math.sin(nu) * dS) / (h * e) - math.cos(i) * draan
    dM = (b / (a * e * h)) * ((p * math.cos(nu) - 2 * r * e) * dR - (p + r) * math.sin(nu) * dS)
    return da, de, di, draan, dargp, dM


def inject_impulse(recs, t0, impulse_day, dv_mag=0.05):
    """
    Applies an impulse kick to all record mean elements at or after `impulse_day`.
    """
    applied = False
    for rec in recs:
        day = (rec["epoch"] - t0).total_seconds() / SECONDS_PER_DAY
        if day >= impulse_day:
            if not applied:
                # Calculate Gauss-VOP shift at the target record state
                da, de, di, draan, dargp, dM = gauss_vop_kick(rec["el"], dR=0.0, dS=-dv_mag, dW=0.0)
                cum_da, cum_de, cum_di = da, de, di
                cum_draan, cum_dargp, cum_dM = draan, dargp, dM
                applied = True

            el = rec["el"]
            rec["el"] = MeanElements(
                a=el.a + cum_da,
                ecc=max(0.0, el.ecc + cum_de),
                inc=el.inc + cum_di,
                raan=(el.raan + cum_draan) % (2 * math.pi),
                argp=(el.argp + cum_dargp) % (2 * math.pi),
                M=(el.M + cum_dM) % (2 * math.pi),
            )


def poisson_sample(lam, rng):
    n, L = 0, 1.0
    thresh = math.exp(-lam)
    while True:
        L *= rng.random()
        if L <= thresh:
            return n
        n += 1


def draw_impacts(t_days, flux, area, mass, beta_mom, rng):
    lam = flux * area * t_days / 365.25
    n = poisson_sample(lam, rng)
    imps = []
    for _ in range(n):
        t = rng.random() * t_days
        L = 10 ** rng.uniform(math.log10(SIZE_LO), math.log10(SIZE_HI))
        mp = RHO_MAT * math.pi / 6.0 * L**3
        v = rng.uniform(V_LO, V_HI)
        z = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - z * z)
        dvx, dvy, dvz = s * math.cos(ph), s * math.sin(ph), z
        sc = beta_mom * mp * v / mass
        imps.append((t, sc * math.sqrt(dvx**2 + dvy**2 + dvz**2), sc * abs(dvy), L))
    imps.sort()
    return imps


def simulate(recs, el0, t0, beta_drag, impacts):
    el = MeanElements(el0.a, el0.ecc, el0.inc, el0.raan, el0.argp, el0.M)
    eln = MeanElements(el0.a, el0.ecc, el0.inc, el0.raan, el0.argp, el0.M)
    rows, null_rows = [], []
    imps = list(impacts)
    prev_day = 0.0
    for rec in recs:
        day = (rec["epoch"] - t0).total_seconds() / SECONDS_PER_DAY
        t = prev_day
        while t < day - 1e-9:
            dt = min(1.0, day - t) * SECONDS_PER_DAY
            for e in (el, eln):
                n, d_raan, d_argp, dn_j2 = brouwer_rates(e.a, e.ecc, e.inc)
                rho = density(e.a - R_EARTH)
                v = math.sqrt(MU / e.a)
                e.a -= 2.0 * e.a * beta_drag * rho * v * dt
                e.raan = (e.raan + d_raan * dt) % (2 * math.pi)
                e.argp = (e.argp + d_argp * dt) % (2 * math.pi)
                e.M = (e.M + (n + dn_j2) * dt) % (2 * math.pi)
            t += dt
        while imps and imps[0][0] <= day:
            ti, dvt, dvs, L = imps.pop(0)
            rng2 = random.Random(1234)
            dS = dvs
            dR = math.sqrt(max(dvt**2 - dvs**2, 0.0)) * (1 if rng2.random() < 0.5 else -1)
            dW = 0.0
            da, de, di_, draan, dargp, dM = gauss_vop_kick(el, dR, dS, dW)
            el.a += da
            el.ecc = max(0.0, el.ecc + de)
            el.inc += di_
            el.raan = (el.raan + draan) % (2 * math.pi)
            el.argp = (el.argp + dargp) % (2 * math.pi)
            el.M = (el.M + dM) % (2 * math.pi)
        rows.append(MeanElements(el.a, el.ecc, el.inc, el.raan, el.argp, el.M))
        null_rows.append(MeanElements(eln.a, eln.ecc, eln.inc, eln.raan, eln.argp, eln.M))
        prev_day = day
    return rows, null_rows


def calibrate_drag(recs, el0, t0):
    total_days = (recs[-1]["epoch"] - t0).total_seconds() / SECONDS_PER_DAY
    target_da = recs[0]["el"].a - recs[-1]["el"].a
    if target_da <= 0 or total_days <= 0:
        return CD * 0.3 / 5.0
    rho_avg = density(el0.a - R_EARTH)
    v_avg = math.sqrt(MU / el0.a)
    dt_sec = total_days * SECONDS_PER_DAY
    beta_drag = target_da / (2.0 * el0.a * rho_avg * v_avg * dt_sec)
    return max(beta_drag, 1e-8)


def write_csv(path, recs, states):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_iso", "a_km", "ecc", "inc_deg", "raan_deg", "argp_deg", "M_deg"])
        for rec, st in zip(recs, states):
            writer.writerow([
                rec["epoch"].isoformat(),
                st.a / 1000.0,
                st.ecc,
                math.degrees(st.inc),
                math.degrees(st.raan),
                math.degrees(st.argp),
                math.degrees(st.M),
            ])


def get_interpolated_a(days, obs_a, target_day):
    if target_day <= days[0]:
        return obs_a[0]
    if target_day >= days[-1]:
        return obs_a[-1]
    for i in range(len(days) - 1):
        if days[i] <= target_day <= days[i + 1]:
            t = (target_day - days[i]) / (days[i + 1] - days[i])
            return obs_a[i] + t * (obs_a[i + 1] - obs_a[i])
    return obs_a[0]


def datetime_to_jd(dt: datetime) -> float:
    y, m, d = dt.year, dt.month, dt.day
    hh = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1))
          + d + B - 1524.5 + hh / 24.0)
    return jd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare TLE history against ORDEM, MASTER, and Phase-2 smoother + detection."
    )
    parser.add_argument("--tle", required=True, help="Path to TLE history file")
    parser.add_argument("--mass", type=float, default=5.0, help="Spacecraft mass in kg")
    parser.add_argument("--area", type=float, default=0.3, help="Spacecraft cross-section area in m^2")
    parser.add_argument("--start-date", type=float, default=2000.0,
                        help="Start offset day for annotation/zoom")
    parser.add_argument("--end-date", type=float, default=2150.0,
                        help="End offset day for annotation/zoom")
    parser.add_argument("--Cd", type=float, default=2.2, help="Drag coefficient for EKF")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Detection threshold for McReynolds scalar (and Mahalanobis)")
    parser.add_argument("--margin-days", type=float, default=30.0,
                        help="Days of data before/after zoom window for smoother settle")
    parser.add_argument("--impulse", action="store_true",
                        help="Inject an artificial impulse/spike in the zoom window")
    parser.add_argument("--impulse-magnitude", type=float, default=0.05,
                        help="Magnitude of dv kick in m/s (default: 0.05 m/s)")
    args = parser.parse_args()

    records = parse_tles(args.tle)
    if not records:
        print("Error: No valid TLE records found.")
        sys.exit(1)
    print(f"Loaded {len(records)} filtered TLE records from {args.tle}")

    t0 = records[0]["epoch"]

    # Inject artificial impulse into the record timeline if requested
    if args.impulse:
        impulse_day = (args.start_date + args.end_date) / 2.0
        print(f"Injecting artificial impulse event at Day {impulse_day:.1f} "
              f"with dv = {args.impulse_magnitude} m/s ...")
        inject_impulse(records, t0, impulse_day, dv_mag=args.impulse_magnitude)

    el0 = records[0]["el"]
    total_days = (records[-1]["epoch"] - t0).total_seconds() / SECONDS_PER_DAY

    beta_drag = calibrate_drag(records, el0, t0)

    rng_ordem = random.Random(42)
    rng_master = random.Random(42)

    imps_ordem = draw_impacts(total_days, FLUX["ORDEM"], args.area, args.mass,
                              beta_mom=1.0, rng=rng_ordem)
    imps_master = draw_impacts(total_days, FLUX["MASTER"], args.area, args.mass,
                               beta_mom=1.0, rng=rng_master)

    rows_ordem, null_rows = simulate(records, el0, t0, beta_drag, imps_ordem)
    rows_master, _ = simulate(records, el0, t0, beta_drag, imps_master)

    obs_elements = [r["el"] for r in records]
    write_csv("history_observed.csv", records, obs_elements)
    write_csv("history_ordem.csv", records, rows_ordem)
    write_csv("history_master.csv", records, rows_master)

    days = [(r["epoch"] - t0).total_seconds() / SECONDS_PER_DAY for r in records]
    obs_a = [r.a / 1000.0 for r in obs_elements]
    null_a = [r.a / 1000.0 for r in null_rows]
    ordem_a = [r.a / 1000.0 for r in rows_ordem]
    master_a = [r.a / 1000.0 for r in rows_master]

    # ------------------------------------------------------------------
    # Phase-2 EKF + Fraser–Potter  – restricted to zoomed window + margin
    # ------------------------------------------------------------------
    print("Running Phase-2 EKF + Fraser–Potter smoother on zoomed window …")

    MARGIN_DAYS = args.margin_days
    t_start = max(0.0, (args.start_date - MARGIN_DAYS) * 86400.0)
    t_end   = (args.end_date + MARGIN_DAYS) * 86400.0

    meas = []
    for rec in records:
        t = (rec["epoch"] - t0).total_seconds()
        if t_start <= t <= t_end:
            z = np.array([
                rec["el"].a, rec["el"].ecc, rec["el"].inc,
                rec["el"].raan, rec["el"].argp, rec["el"].M,
            ], dtype=float)
            meas.append((t, z))

    if len(meas) < 10:
        print(f"Error: only {len(meas)} measurements in window – increase --margin-days")
        sys.exit(1)

    print(f"  using {len(meas)} measurements "
          f"(t = {t_start/86400:.1f} … {t_end/86400:.1f} days)")

    # Use the first measurement of the window as the filter initial state
    el0_window = MeanElements(
        a=meas[0][1][0],
        ecc=meas[0][1][1],
        inc=meas[0][1][2],
        raan=meas[0][1][3],
        argp=meas[0][1][4],
        M=meas[0][1][5],
    )
    # Shift times so the window starts at t=0 for the filter
    t0_window = meas[0][0]
    meas_shifted = [(t - t0_window, z) for t, z in meas]

    epoch_jd = datetime_to_jd(t0) + t0_window / 86400.0

    result = _run_filter_pair(
        meas_shifted,
        el0_window,
        Cd_base=args.Cd,
        area=args.area,
        mass=args.mass,
        epoch_jd=epoch_jd,
        tau=3e4,
        q=1e-20,
    )
    print("Smoother finished.")

    # Recover absolute day offsets that match the original timeline
    meas_days = [t / 86400.0 for t, _ in meas]
    sm_a_full = [s[0] / 1000.0 for s in result["sm_states"]]
    fwd_a_full = [s[0] / 1000.0 for s in result["fwd_states"]]
    bwd_a_full = [s[0] / 1000.0 for s in result["bwd_states"]]

    start_day = args.start_date
    end_day   = args.end_date

    print(f"len(meas)          = {len(meas)}")
    print(f"len(sm_states)     = {len(result['sm_states'])}")
    print(f"len(fwd_states)    = {len(result['fwd_states'])}")
    print(f"len(bwd_states)    = {len(result['bwd_states'])}")
    print(f"meas_days range    = {min(meas_days):.3f} … {max(meas_days):.3f}")
    print(f"requested window   = {start_day} … {end_day}")

    if len(sm_a_full) != len(meas_days):
        raise RuntimeError(
            f"Length mismatch: sm_states={len(sm_a_full)}  meas_days={len(meas_days)}"
        )

    # Detection statistics on the (shifted) filter output
    d_mh = compute_mahalanobis_distance(
        result["fwd_states"], result["fwd_covs"],
        result["bwd_states"], result["bwd_covs"],
        result["sm_covs"],
    )
    mcr = compute_mcreynolds(
        result["fwd_states"], result["fwd_covs"],
        result["sm_states"], result["sm_covs"],
    )
    mcr_scalar = mcr["scalar"]

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    start_day, end_day = args.start_date, args.end_date

    # --- Plot 1: Observed (TLE) Only
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, obs_a, "k-", label="Observed (TLE)")
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title("Observed Semi-Major Axis History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_observed.png", dpi=200)
    plt.close()

    # --- Plot 2: ORDEM 3.0 Predicted
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, ordem_a, "r-.", label="ORDEM 3.0 Predicted")
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title("ORDEM 3.0 Predicted Semi-Major Axis Decay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_ordem.png", dpi=200)
    plt.close()

    # --- Plot 3: MASTER 8 Predicted
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, master_a, "b:", label="MASTER 8 Predicted")
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title("MASTER 8 Predicted Semi-Major Axis Decay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_master.png", dpi=200)
    plt.close()

    # --- Plot 4: Combined Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(days, obs_a, "k-", label="Observed (TLE)")
    plt.plot(days, ordem_a, "r-.", label="ORDEM 3.0 Predicted")
    plt.plot(days, master_a, "b:", label="MASTER 8 Predicted")
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title("Semi-Major Axis Decay & Impact Discontinuities (Combined)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_combined.png", dpi=200)
    plt.close()

    # --- Plot 5: Observed with Bracket Annotations
    plt.figure(figsize=(9, 4.5))
    plt.plot(days, obs_a, "k-", label="Observed (TLE)")
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title("Observed Semi-Major Axis History")
    plt.grid(True)
    plt.legend()
    a_start = get_interpolated_a(days, obs_a, start_day)
    a_end = get_interpolated_a(days, obs_a, end_day)
    plt.annotate("[", xy=(start_day, a_start), color="red", fontsize=22,
                 fontweight="bold", ha="right", va="center")
    plt.annotate("]", xy=(end_day, a_end), color="red", fontsize=22,
                 fontweight="bold", ha="left", va="center")
    plt.tight_layout()
    plt.savefig("figs/sma_annotated.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Common mask / zoomed series for filter outputs
    # ------------------------------------------------------------------
    eps = 1e-6
    mask = [(start_day - eps <= d <= end_day + eps) for d in meas_days]
    days_z   = [d for d, m in zip(meas_days, mask) if m]
    sm_a_z   = [a for a, m in zip(sm_a_full, mask) if m]
    fwd_a_z  = [a for a, m in zip(fwd_a_full, mask) if m]
    bwd_a_z  = [a for a, m in zip(bwd_a_full, mask) if m]

    obs_mask = [(start_day - eps <= d <= end_day + eps) for d in days]
    days_obs_z = [d for d, m in zip(days, obs_mask) if m]
    obs_a_z    = [a for a, m in zip(obs_a, obs_mask) if m]

    print(f"Observed points in window : {len(obs_a_z)}")
    print(f"Smoothed points in window : {len(sm_a_z)}")
    print(f"Forward  points in window : {len(fwd_a_z)}")
    print(f"Backward points in window : {len(bwd_a_z)}")
    if sm_a_z:
        print(f"Smoothed SMA range        : {min(sm_a_z):.4f} … {max(sm_a_z):.4f} km")
    else:
        print("Smoothed series is EMPTY – check day alignment above")

    # ------------------------------------------------------------------
    # CRITICAL: Compute ONE common y-limit for all zoomed plots
    # so observed / forward / backward look identical except for the lines
    # ------------------------------------------------------------------
    all_zoom_a = obs_a_z + fwd_a_z + bwd_a_z + sm_a_z
    if all_zoom_a:
        y_min = min(all_zoom_a)
        y_max = max(all_zoom_a)
        y_margin = max(0.02, (y_max - y_min) * 0.15)
        common_ylim = (y_min - y_margin, y_max + y_margin)
    else:
        common_ylim = None

    print(f"Common y-limits for all zoomed plots: {common_ylim}")

    # --- Plot 6: Original Zoomed Observed  (now uses common ylim)
    plt.figure(figsize=(9, 4.5))
    plt.plot(days_obs_z, obs_a_z, "k.-", label="Observed (TLE)", markersize=4)
    plt.xlim(start_day, end_day)
    if common_ylim:
        plt.ylim(common_ylim)
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title(f"Observed Semi-Major Axis (Zoomed: Day {start_day:.1f} to {end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_zoomed.png", dpi=200)
    plt.close()

    # --- Plot 7a: Forward EKF only (pre-smoothed)  – same axes
    plt.figure(figsize=(9, 4.5))
    plt.plot(days_obs_z, obs_a_z, "k.-", label="Observed (TLE)", markersize=4, alpha=0.7)
    if fwd_a_z:
        plt.plot(days_z, fwd_a_z, "b-", linewidth=2.0, label="Forward EKF")
    plt.xlim(start_day, end_day)
    if common_ylim:
        plt.ylim(common_ylim)
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title(f"Observed vs Forward EKF (Zoomed: Day {start_day:.1f}–{end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_zoomed_forward.png", dpi=200)
    plt.close()

    # --- Plot 7b: Backward EKF only (pre-smoothed)  – same axes
    plt.figure(figsize=(9, 4.5))
    plt.plot(days_obs_z, obs_a_z, "k.-", label="Observed (TLE)", markersize=4, alpha=0.7)
    if bwd_a_z:
        plt.plot(days_z, bwd_a_z, "g-", linewidth=2.0, label="Backward EKF")
    plt.xlim(start_day, end_day)
    if common_ylim:
        plt.ylim(common_ylim)
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title(f"Observed vs Backward EKF (Zoomed: Day {start_day:.1f}–{end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_zoomed_backward.png", dpi=200)
    plt.close()

    # --- Plot 7c: Smoothed EKF + Fraser–Potter  – same axes
    plt.figure(figsize=(9, 4.5))
    plt.plot(days_obs_z, obs_a_z, "k.-", label="Observed (TLE)", markersize=4, alpha=0.7)
    if sm_a_z:
        plt.plot(days_z, sm_a_z, "r-", linewidth=2.5, label="Smoothed (EKF + Fraser–Potter)")
    plt.xlim(start_day, end_day)
    if common_ylim:
        plt.ylim(common_ylim)
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title(f"Observed vs Smoothed SMA (Zoomed: Day {start_day:.1f}–{end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_zoomed_smoothed.png", dpi=200)
    plt.close()

    # --- Optional: Combined Forward + Backward + Smoothed  – same axes
    plt.figure(figsize=(9, 4.5))
    plt.plot(days_obs_z, obs_a_z, "k.-", label="Observed (TLE)", markersize=3, alpha=0.5)
    if fwd_a_z:
        plt.plot(days_z, fwd_a_z, "b-", linewidth=1.5, alpha=0.8, label="Forward EKF")
    if bwd_a_z:
        plt.plot(days_z, bwd_a_z, "g-", linewidth=1.5, alpha=0.8, label="Backward EKF")
    if sm_a_z:
        plt.plot(days_z, sm_a_z, "r-", linewidth=2.5, label="Smoothed (Fraser–Potter)")
    plt.xlim(start_day, end_day)
    if common_ylim:
        plt.ylim(common_ylim)
    plt.xlabel("Days since epoch")
    plt.ylabel("Semi-major axis a (km)")
    plt.title(f"Forward / Backward / Smoothed Comparison (Day {start_day:.1f}–{end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/sma_zoomed_fwd_bwd_sm.png", dpi=200)
    plt.close()

    # --- Plot 8: Detection statistic (Mahalanobis + McReynolds) for the window
    w_sm = np.array([s[6:9] for s in result["sm_states"]])
    w_norm = np.linalg.norm(w_sm, axis=1)          # |w|
    w_S    = np.abs(w_sm[:, 1])                    # |w_S| only

    # Restrict to the exact zoom window
    mask = [(start_day - 1e-6 <= d <= end_day + 1e-6) for d in meas_days]
    days_z   = [d for d, m in zip(meas_days, mask) if m]
    w_norm_z = [v for v, m in zip(w_norm, mask) if m]
    w_S_z    = [v for v, m in zip(w_S, mask) if m]

    # Simple noise-floor threshold (e.g. 5× median)
    thresh = 5.0 * np.median(w_norm_z)

    plt.figure(figsize=(9, 4.5))
    plt.plot(days_z, w_norm_z, "b-", linewidth=1.5, label=r"$\vert{}\mathbf{w}\vert{}$ (smoothed)")
    plt.plot(days_z, w_S_z,    "g-", linewidth=1.2, alpha=0.8, label=r"$\vert{}w_S\vert{}$")
    plt.axhline(1.2 * 1e-8, color="r", linestyle="--", label=f"Threshold")
    plt.xlim(start_day, end_day)
    plt.xlabel("Days since epoch")
    plt.ylabel(r"Unmodeled acceleration (m/s²)")
    plt.title(f"Smoothed Acceleration (Zoomed: Day {start_day:.1f}–{end_day:.1f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/detection_zoomed.png", dpi=200)
    plt.close()

    print("Saved plots:")
    print("  sma_observed.png")
    print("  sma_ordem.png")
    print("  sma_master.png")
    print("  sma_combined.png")
    print("  sma_annotated.png")
    print("  sma_zoomed.png")
    print("  sma_zoomed_forward.png     ← (Forward EKF)")
    print("  sma_zoomed_backward.png    ← (Backward EKF)")
    print("  sma_zoomed_smoothed.png")
    print("  sma_zoomed_fwd_bwd_sm.png  ← bonus: all three together")
    print("  detection_zoomed.png")