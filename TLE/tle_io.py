# TLE parsing and formatting.
# The bstar and ndot fields have column-position quirks (decimal point assumed,
# compressed exponent notation) - see comments inline.

import math
from datetime import datetime, timezone, timedelta
from propagator.orbital import MeanElements
from constants.constants import (MU, R_EARTH,
                        SIGMA_A_M, SIGMA_ECC, SIGMA_INC,
                        SIGMA_RAAN, SIGMA_ARGP, SIGMA_M)


def parse_bstar(line1):
    """BSTAR is stored compressed, e.g. ' 29591-3' meaning 0.29591 * 10^-3."""
    raw = line1[53:61].strip()
    if not raw:
        return 0.0
    try:
        sign = -1 if raw[0] == '-' else 1
        body = raw[1:] if raw[0] in '+-' else raw
        # exponent is the trailing +/-N, mantissa is everything before it
        for i in range(len(body) - 1, 0, -1):
            if body[i] in '+-':
                return sign * float('0.' + body[:i]) * 10 ** int(body[i:])
        return sign * float('0.' + body)
    except Exception:
        return 0.0


def fmt_bstar(b):
    """Format a float back into TLE compressed BSTAR notation."""
    if b == 0:
        return " 00000-0"
    s = '-' if b < 0 else ' '
    b = abs(b)
    exp = math.floor(math.log10(b)) + 1
    mantissa_digits = f"{b / 10 ** exp:.5f}"[2:]  # strip "0."
    return f"{s}{mantissa_digits}{exp:+d}"


def tle_checksum(line):
    """Mod-10 checksum: digits count as themselves, '-' counts as 1, everything else 0."""
    total = 0
    for c in line[:68]:
        if c.isdigit():
            total += int(c)
        elif c == '-':
            total += 1
    return total % 10


def parse_tle(name, line1, line2):
    """Parse a name + 2-line TLE into a dict of orbital elements and metadata."""
    ep = line1[18:32].strip()
    y2 = int(ep[:2])
    yr = 2000 + y2 if y2 < 57 else 1900 + y2
    doy = float(ep[2:])

    epoch_dt = datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    epoch_jd = (2451545.0
                + (epoch_dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0)

    inc = math.radians(float(line2[8:16]))
    raan = math.radians(float(line2[17:25]))
    ecc = float('0.' + line2[26:33].strip())
    argp = math.radians(float(line2[34:42]))
    M = math.radians(float(line2[43:51]))
    n_rd = float(line2[52:63])
    n_rs = n_rd * 2 * math.pi / 86400
    a = (MU / n_rs ** 2) ** (1.0 / 3.0)

    return {
        "name": name.strip(),
        "norad_id": line1[2:7].strip(),
        "epoch_str": ep,
        "epoch_dt": epoch_dt,
        "epoch_jd": epoch_jd,
        "line1": line1,
        "line2": line2,
        "a": a,
        "ecc": ecc,
        "inc": inc,
        "raan": raan,
        "argp": argp,
        "M": M,
        "n_revday": n_rd,
        "bstar": parse_bstar(line1),
        "ndot_raw": line1[33:43],   # keep raw string - reformatting as float shifts columns
    }


def tle_to_elements(tle):
    return MeanElements(
        a=tle["a"], ecc=tle["ecc"], inc=tle["inc"],
        raan=tle["raan"], argp=tle["argp"], M=tle["M"],
    )


def elements_to_tle(el, orig_tle, elapsed_s):
    """
    Format mean elements as a TLE string at epoch = orig epoch + elapsed_s.
    We propagate entirely in mean-element space (Brouwer secular rates), so
    there's no osculating<->mean conversion needed here - the output IS a mean
    state already, same space SGP4-derived TLEs live in.
    """
    dt = orig_tle["epoch_dt"] + timedelta(seconds=elapsed_s)
    yr2 = dt.year % 100
    doy = (dt - datetime(dt.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400 + 1
    ep = f"{yr2:02d}{doy:012.8f}"

    bs = fmt_bstar(orig_tle["bstar"])
    norad = orig_tle["norad_id"].ljust(5)
    ndot = orig_tle.get("ndot_raw", "  .00000000")
    n_rd = el.mean_motion() * 86400 / (2 * math.pi)
    ecc_s = f"{el.ecc:.7f}"[2:]
    revs = int(orig_tle["line2"][63:68]) + int(elapsed_s / (86400 / n_rd))

    l1 = (f"1 {norad}U {orig_tle['line1'][9:17]} {ep} {ndot}  00000-0 {bs} 0  9990")
    l1 = l1[:68].ljust(68) + str(tle_checksum(l1))

    l2 = (f"2 {norad} {math.degrees(el.inc) % 360:8.4f} {math.degrees(el.raan) % 360:8.4f} "
          f"{ecc_s} {math.degrees(el.argp) % 360:8.4f} {math.degrees(el.M) % 360:8.4f} "
          f"{n_rd:11.8f}{revs:5d}")
    l2 = l2[:68].ljust(68) + str(tle_checksum(l2))

    return orig_tle["name"], l1, l2


def inject_tle_noise(a, ecc, inc, raan, argp, M, rng):
    """Add TLE fit-residual noise directly in mean-element space [Vallado et al 2006]."""
    a_n = max(a + rng.normal(0.0, SIGMA_A_M), R_EARTH + 150e3)
    ecc_n = float(max(0.0, min(0.99, ecc + rng.normal(0.0, SIGMA_ECC))))
    inc_n = inc + rng.normal(0.0, SIGMA_INC)
    raan_n = raan + rng.normal(0.0, SIGMA_RAAN)
    argp_n = argp + rng.normal(0.0, SIGMA_ARGP)
    M_n = M + rng.normal(0.0, SIGMA_M)
    return a_n, ecc_n, inc_n, raan_n, argp_n, M_n


def load_tles_from_file(path):
    """Load TLEs from a standard 2-line or 3-line (name + 2 lines) TLE file."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    tles = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith('1 ') and lines[i + 1].startswith('2 '):
            tles.append((f"OBJECT_{lines[i][2:7].strip()}", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
            tles.append((lines[i], lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1

    return tles