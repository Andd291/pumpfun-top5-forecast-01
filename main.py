"""pumpfun-top5-forecast-01 forecaster (Python 3.13, standard library only).

Reads the house's 10-minute market-cap snapshots at $MAKEFUN_INPUT_DIR (else
/input) and writes one forecast to $MAKEFUN_OUTPUT_DIR (else /output).

Approach
--------
Every coin's log10 market cap (in SOL) is modelled as a drift + diffusion
process whose parameters are fitted from the history itself at run time:

* a pooled regression per venue (pool / curve) of the hourly log-cap change on
  the coin's size and age gives the class drift and volatility;
* each coin's own recent slope (OLS on its last few marks) is shrunk toward the
  class drift, and its residual volatility toward the class volatility;
* a Monte Carlo (fixed seed, so two runs are byte-identical) simulates every
  candidate coin to each of the seven answer reads, with a per-coin death
  hazard (a coin can fade out of the ranking) and, for curve coins, a
  graduation hazard after which pool dynamics take over;
* the simulated place frequencies give the rank probabilities, and the
  simulated k-th largest caps give each place's size distribution;
* a newcomer mass, growing with the horizon, is left unnamed at every place:
  coins born after the history ends (or never ranked in it) regularly take
  top-5 places, and the scorer gives an unnamed place to them through the
  unnamed share, so an honest forecast reserves probability for them.

No network, no installed packages, deterministic.
"""

import json
import math
import os
import random
import sys
from datetime import datetime

# ---------------------------------------------------------------- tunables
SEED = 20261112
N_PATHS = 1200
MAX_CANDIDATES = 500
MAX_ROWS_PER_PLACE = 60
# Fallback newcomer mass at a read h hours after the cut:
# Q_INF * (1 - exp(-h / TAU)). At run time the history's own mark pairs
# refit (Q_INF, TAU); these are used only when too few pairs exist. The fit
# is then shrunk halfway toward a prior (Q_PRIOR, TAU_PRIOR) and capped at
# Q_MAX: the raw fit on a young market lands on the grid corner (Q=0.95,
# fast saturation) and would leave almost nothing to the coins the history
# actually names; the prior keeps it honest at the long horizons the final
# is scored on.
Q_INF = float(os.environ.get("MF_Q_INF", "0.70"))
TAU_HOURS = float(os.environ.get("MF_TAU", "1.0"))
Q_PRIOR = float(os.environ.get("MF_Q_PRIOR", "0.55"))
TAU_PRIOR = float(os.environ.get("MF_TAU_PRIOR", "1.5"))
Q_MAX = float(os.environ.get("MF_Q_MAX", "0.70"))
TAU_MIN = float(os.environ.get("MF_TAU_MIN", "0.5"))
NEWCOMER_SHRINK = float(os.environ.get("MF_NEW_SHRINK", "0.3"))
# Extra log10-size smoothing (sd) applied to each place's size distribution,
# for model error and for newcomers holding the place.
SIZE_SMOOTH_SD = float(os.environ.get("MF_SMOOTH", "0.03"))
# Newcomer size distribution: Normal(sim place mean + NEW_SIZE_SHIFT,
# sqrt(sim var + NEW_SIZE_SD^2)), mixed in with the newcomer mass.
NEW_SIZE_SHIFT = float(os.environ.get("MF_NEW_SHIFT", "0.05"))
NEW_SIZE_SD = float(os.environ.get("MF_NEW_SD", "0.25"))
# Place-level size model: how much of each place's size distribution comes
# from a direct model of that place's cap level (drift + diffusion fitted
# from the marks' place-cap series) rather than the Monte Carlo histogram.
PLACE_BLEND = float(os.environ.get("MF_PLACE_BLEND", "0.85"))
PLACE_DRIFT_DAMP_H = float(os.environ.get("MF_PLACE_DAMP", "6.0"))
# Cap on the fitted place-cap drift (dex/h). The birth phase really does
# rise ~+1.5-2 dex/h at places 2-5, so a 0.5 cap throws most of it away.
PLACE_DRIFT_CAP = float(os.environ.get("MF_PLACE_DRIFT_CAP", "1.5"))
PLACE_SHIFT_LO = float(os.environ.get("MF_PLACE_SHIFT_LO", "-1.0"))
PLACE_SHIFT_HI = float(os.environ.get("MF_PLACE_SHIFT_HI", "1.5"))
# Width of the place-level size forecast at horizon h:
#   sd(h) = sqrt(resid^2 + DIFF_SD^2 * h + min((se * DAMP(h))^2, SE_CAP^2))
# resid is the place series' residual sd around its own fitted trend
# (the honest short-horizon noise: replacement jumps, not the raw
# increment dispersion, which double-counts the trend the center already
# extrapolates); DIFF_SD is slow diffusion; the slope-SE term is the
# centre's own uncertainty, damped the same way as the drift itself.
PLACE_DIFF_SD = float(os.environ.get("MF_PLACE_DIFF_SD", "0.06"))
PLACE_SE_CAP = float(os.environ.get("MF_PLACE_SE_CAP", "0.45"))
PLACE_SD_FLOOR = float(os.environ.get("MF_PLACE_SD_FLOOR", "0.08"))
PLACE_SD_CAP = float(os.environ.get("MF_PLACE_SD_CAP", "0.60"))
# Name hopeless coins at p 0 (they leave the unnamed share to the rest):
# a coin is hopeless at a place when its simulated probability there is
# negligible and its last cap is small. 0 disables the tactic.
PZERO_MAX = int(os.environ.get("MF_PZERO_MAX", "40"))
PZERO_Y_MAX = float(os.environ.get("MF_PZERO_Y_MAX", "2.0"))  # log10 SOL
# Death hazard per hour, by venue (halved for large caps, raised for small).
HAZARD = {"pool": 0.02, "curve": 0.20}
# Graduation hazard per hour for a curve coin; a migrating coin is imminent.
GRAD_HAZARD = 0.08
GRAD_HAZARD_MIGRATING = 1.0
# Mean/sd of the one-off log-cap shock when a curve coin graduates.
# Measured on the dev history's six curve->pool transitions: mean -0.31,
# sd 0.57 (graduating usually reprices down as the pool opens).
GRAD_SHOCK_MEAN = -0.30
GRAD_SHOCK_SD = 0.45
# Per-coin slopes are shrunk toward the class drift by exp(-h_max / TAU):
# a coin's recent momentum is real at 1h out and mostly noise at 13h out.
SLOPE_DAMP_TAU_H = float(os.environ.get("MF_SLOPE_DAMP", "8.0"))
MICRO = 1_000_000


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def bin_of(x, edges):
    """Bin of log10(cap in SOL) x: 0 below edges[0], 81 at/above edges[-1]."""
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges)
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if edges[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def solve_linear(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting (small n)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_class(pairs):
    """Fit dy/dt ~ a + b*(y-3) + c*min(age,12) on consecutive-mark pairs.

    pairs: list of (y_prev, age_hours, dt_hours, dy). Returns (a, b, c, sigma)
    with sigma the per-sqrt-hour residual volatility, or None when too thin.
    """
    if len(pairs) < 20:
        return None
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for (y, age, dt, dy) in pairs:
        x = (1.0, y - 3.0, min(age, 12.0))
        w = dt
        for i in range(3):
            xty[i] += w * x[i] * (dy / dt)
            for j in range(3):
                xtx[i][j] += w * x[i] * x[j]
    for i in range(3):
        xtx[i][i] += 1e-6
    coef = solve_linear(xtx, xty)
    if coef is None:
        return None
    a, b, c = coef
    num = 0.0
    den = 0.0
    for (y, age, dt, dy) in pairs:
        pred = (a + b * (y - 3.0) + c * min(age, 12.0)) * dt
        r = dy - pred
        num += r * r
        den += dt
    sigma = math.sqrt(num / den) if den > 0 else 0.3
    return (a, b, c, min(max(sigma, 0.05), 1.5))

def _bin_center(b, edges):
    """Representative log10 value of bin b (0..81)."""
    if b <= 0:
        return edges[0] - 0.025
    if b >= len(edges):
        return edges[-1] + 0.025
    return (edges[b - 1] + edges[b]) / 2.0


def normal_bins(mean, sd, edges):
    """Discretise Normal(mean, sd) over the 82 size bins; returns bin -> p."""
    out = {}
    n_bins = len(edges) + 1  # bins 0..81
    for b in range(0, n_bins):
        if b == 0:
            lo, hi = -float("inf"), edges[0]
        elif b == n_bins - 1:
            lo, hi = edges[-1], float("inf")
        else:
            lo, hi = edges[b - 1], edges[b]
        pl = 0.0 if lo == -float("inf") else 0.5 * (1 + math.erf((lo - mean) / (sd * math.sqrt(2))))
        ph = 1.0 if hi == float("inf") else 0.5 * (1 + math.erf((hi - mean) / (sd * math.sqrt(2))))
        p = ph - pl
        if p > 0:
            out[b] = p
    tot = sum(out.values())
    if tot > 0:
        for b in out:
            out[b] /= tot
    return out


def smooth_bins(dist, sd):
    """Convolve a bin distribution with a Gaussian of sd (in log10 units).

    Bins are 0.05 wide; bin b (1..80) covers [edges[b-1], edges[b]).
    """
    if sd <= 0:
        return dict(dist)
    width = 0.05
    spread = int(math.ceil(3 * sd / width))
    out = {}
    for b, p in dist.items():
        for d in range(-spread, spread + 1):
            nb = b + d
            if nb < 0 or nb > 81:
                continue
            z0 = (d - 0.5) * width / sd
            z1 = (d + 0.5) * width / sd
            w = 0.5 * (math.erf(z1 / math.sqrt(2)) - math.erf(z0 / math.sqrt(2)))
            if w <= 0:
                continue
            out[nb] = out.get(nb, 0.0) + p * w
    tot = sum(out.values())
    if tot > 0:
        for b in out:
            out[b] /= tot
    return out


def write_fallback(out_dir, output_name, top5=None):
    """A valid forecast that spreads every place evenly: used only when the
    history holds too few coins to model (never in the real final)."""
    top5 = list(top5 or [])
    pad = 0
    while len(top5) < 5:
        top5.append("1" * 31 + chr(ord("2") + pad))
        pad += 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, output_name), "w") as f:
        f.write(json.dumps({"probabilities": [], "top5": top5[:5],
                            "sizes": []}) + "\n")


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    t0 = parse_time(task["t0"])
    reads = [parse_time(r) for r in task["reads"]]
    output_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(in_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    if not marks:
        write_fallback(out_dir, output_name)
        return
    last_time = parse_time(marks[-1]["at"])
    horizons = [max(0.0, (r - last_time).total_seconds() / 3600.0) for r in reads]
    n_reads = len(reads)
    bounds = [0.0] + horizons
    interval_len = [bounds[i + 1] - bounds[i] for i in range(n_reads)]
    sqrt_len = [math.sqrt(max(d, 0.0)) for d in interval_len]

    # ---------------------------------------------------------- build series
    coins = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            try:
                cap = int(r["market_cap_lamports"])
            except (KeyError, TypeError, ValueError):
                continue
            if cap <= 0:
                continue
            mint = r["mint"]
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {"times": [], "ys": [], "venues": [],
                                   "created": None, "mayhem": False}
            c["times"].append(t)
            c["ys"].append(math.log10(cap) - 9.0)
            c["venues"].append(r.get("venue") or "curve")
            if r.get("created_at"):
                try:
                    c["created"] = parse_time(r["created_at"])
                except ValueError:
                    pass
            if r.get("mayhem"):
                c["mayhem"] = True
    if len(coins) < places:
        write_fallback(out_dir, output_name,
                       top5=[c for c in coins][:places])
        return

    # ------------------------------------------- pooled class fits per venue
    pairs = {"pool": [], "curve": []}
    for mint, c in coins.items():
        created = c["created"]
        t_created = (created - t0).total_seconds() / 3600.0 if created else None
        for i in range(len(c["times"]) - 1):
            dt = c["times"][i + 1] - c["times"][i]
            if dt <= 0:
                continue
            v = "pool" if c["venues"][i] == "pool" else "curve"
            age = c["times"][i] - t_created if t_created is not None else 1.0
            pairs[v].append((c["ys"][i], max(0.0, age), dt,
                             c["ys"][i + 1] - c["ys"][i]))
    fits = {}
    for v in ("pool", "curve"):
        f = fit_class(pairs[v])
        if f is None:
            md = (sum(dy / dt for (_, _, dt, dy) in pairs[v]) / len(pairs[v])
                  if pairs[v] else -0.1)
            f = (md, 0.0, 0.0, 0.35)
        fits[v] = f

    def class_drift(venue, y, age):
        a, b, cc, _ = fits[venue]
        return a + b * (y - 3.0) + cc * min(max(age, 0.0), 12.0)

    # ---------------------------------------------- per-coin drift and vol
    last_t = (last_time - t0).total_seconds() / 3600.0
    cand = []
    for mint, c in coins.items():
        n = len(c["times"])
        venue = "pool" if c["venues"][-1] == "pool" else "curve"
        y_last = c["ys"][-1]
        t_last = c["times"][-1]
        created = c["created"]
        t_created = (created - t0).total_seconds() / 3600.0 if created else None
        age_last = max(0.0, t_last - t_created) if t_created is not None else 1.0
        k = min(n, 8)
        slope = None
        resid_sd = None
        ts = ys = None
        if k >= 2:
            ts = c["times"][-k:]
            ys = c["ys"][-k:]
            mt = sum(ts) / k
            my = sum(ys) / k
            sxx = sum((t - mt) ** 2 for t in ts)
            if sxx > 1e-9:
                slope = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sxx
                if k > 2:
                    ss = sum((y - (my + slope * (t - mt))) ** 2
                             for t, y in zip(ts, ys))
                    resid_sd = math.sqrt(ss / (k - 2))
        drift_class = class_drift(venue, y_last, age_last)
        if slope is None:
            drift = drift_class
        else:
            w = k / (k + 3.0)
            drift = w * slope + (1.0 - w) * drift_class
        drift = min(max(drift, -1.2), 1.0)
        # Horizon-adaptive shrinkage: the further the reads, the less a
        # coin's own recent slope is worth relative to its class drift.
        h_max = max(horizons) if horizons else 0.0
        damp = math.exp(-h_max / SLOPE_DAMP_TAU_H)
        drift = drift_class + (drift - drift_class) * damp
        sigma_class = fits[venue][3]
        if resid_sd is None or ts is None:
            sigma = sigma_class
        else:
            dt_typ = (ts[-1] - ts[0]) / max(k - 1, 1)
            sig_own = resid_sd / math.sqrt(max(dt_typ, 1e-6))
            n_eff = min(k - 1, 8)
            sigma = math.sqrt((n_eff * sig_own * sig_own
                               + 6.0 * sigma_class ** 2) / (n_eff + 6.0))
        sigma = min(max(sigma, 0.06), 1.5)
        dt_stale = max(0.0, last_t - t_last)
        if dt_stale > 0:
            y_last = y_last + drift * dt_stale
            sigma = min(math.sqrt(sigma * sigma
                                  + sigma_class * sigma_class * dt_stale), 1.5)
        age_cut = age_last + dt_stale
        hazard = HAZARD[venue]
        if y_last >= 3.0:
            hazard *= 0.5
        elif y_last < 2.0:
            hazard *= 1.8
        hazard = min(max(hazard, 0.01), 0.8)
        grad_h = 0.0
        if venue != "pool":
            grad_h = (GRAD_HAZARD_MIGRATING if c["venues"][-1] == "migrating"
                      else GRAD_HAZARD)
        cand.append({"mint": mint, "venue": venue, "y": y_last,
                     "drift": drift, "sigma": sigma, "age": age_cut,
                     "hazard": hazard, "grad_h": grad_h,
                     "potential": y_last + max(0.0, drift) * 4.0})
    cand.sort(key=lambda c: (-c["potential"], c["mint"]))

    # ----------------- newcomer rate: fit q(h) from the history's own pairs
    # For each recent mark i as a cut and each later mark j, the share of
    # mark j's top 5 that no row up to i had named is a direct measurement of
    # the probability a place is held by a coin the cut's history cannot name
    # (born later, or never ranked in it). Fit q(h) = Q*(1-exp(-h/TAU)).
    # Two guards against the birth phase lying to us: cuts come only from
    # the last half of the marks and only from marks that actually ranked
    # coins (an empty first mark would report every later coin as new), and
    # the fitted curve is shrunk halfway toward a slow prior and capped, so
    # a short noisy history cannot reserve ~everything for newcomers.
    q_inf, tau_h = Q_INF, TAU_HOURS
    pts = []  # (h_hours, frac_newcomers_in_top5, weight)
    n_marks = len(marks)
    known_upto = []
    seen = set()
    for m in marks:
        for r in m["rows"]:
            seen.add(r["mint"])
        known_upto.append(set(seen))
    mark_t = [(parse_time(m["at"]) - t0).total_seconds() / 3600.0 for m in marks]
    # Recency weight: the latest cut counts most; half-life ~ a quarter of
    # the history in marks (at least 3 marks).
    hl = max(2.0, n_marks / 6.0)
    first_cut = max(0, n_marks - max(6, n_marks // 2))
    for i in range(first_cut, n_marks):
        if len(marks[i].get("rows") or []) < 30:
            continue
        w_i = 0.5 ** ((n_marks - 1 - i) / hl)
        for j in range(i + 1, n_marks):
            h = mark_t[j] - mark_t[i]
            if h > 3.0 or h <= 0:
                break
            top5j = [r["mint"] for r in
                     sorted(marks[j]["rows"], key=lambda r: r["rank"])[:places]]
            if not top5j:
                continue
            frac = sum(1 for mint in top5j if mint not in known_upto[i]) / len(top5j)
            pts.append((h, frac, w_i))
    if len(pts) >= 3:
        best = None
        for q_try in [0.20 + 0.05 * k for k in range(11)]:
            for tau_try in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
                err = sum(w * (q_try * (1.0 - math.exp(-h / tau_try)) - f) ** 2
                          for (h, f, w) in pts)
                if best is None or err < best[0]:
                    best = (err, q_try, tau_try)
        if best is not None:
            _, q_inf, tau_h = best

    def q_at(h):
        """Newcomer share h hours after the cut: fitted curve shrunk
        halfway toward the slow prior, capped."""
        if h <= 0:
            return 0.0
        qf = q_inf * (1.0 - math.exp(-h / tau_h))
        qp = Q_PRIOR * (1.0 - math.exp(-h / TAU_PRIOR))
        q = (1.0 - NEWCOMER_SHRINK) * qf + NEWCOMER_SHRINK * qp
        return min(q, Q_MAX)

    if os.environ.get("MF_DEBUG"):
        print(f"[debug] newcomer fit: pts={len(pts)} q_inf={q_inf:.2f} "
              f"tau={tau_h}h keep={[round(1 - q_at(h), 2) for h in horizons]}",
              file=sys.stderr)

    big = [c for c in cand if c["y"] >= 2.5]
    rest = [c for c in cand if c["y"] < 2.5]
    chosen = big[:MAX_CANDIDATES]
    chosen += rest[:max(0, MAX_CANDIDATES - len(chosen))]
    chosen.sort(key=lambda c: (-c["potential"], c["mint"]))
    cand = chosen

    # ------------------------------------------------------------- simulation
    rng = random.Random(SEED)
    gauss = rng.gauss
    rand = rng.random
    # place_counts[read][place]: mint -> count; size_hist[read][place]: bin -> count
    place_counts = [[dict() for _ in range(places)] for _ in range(n_reads)]
    size_hist = [[dict() for _ in range(places)] for _ in range(n_reads)]

    for _ in range(N_PATHS):
        tops = [[(-1e18, None)] * places for _ in range(n_reads)]
        for c in cand:
            y = c["y"]
            venue = c["venue"]
            sigma = c["sigma"]
            age = c["age"]
            death_t = (-math.log(1.0 - rand()) / c["hazard"]
                       if c["hazard"] > 0 else float("inf"))
            grad_t = (-math.log(1.0 - rand()) / c["grad_h"]
                      if c["grad_h"] > 0 else float("inf"))
            graduated = venue == "pool"
            ys = []
            alive = True
            t_now = 0.0
            for j in range(n_reads):
                dh = interval_len[j]
                if alive and death_t <= bounds[j + 1]:
                    alive = False
                if alive:
                    if (not graduated) and grad_t <= bounds[j + 1]:
                        graduated = True
                        y += gauss(GRAD_SHOCK_MEAN, GRAD_SHOCK_SD)
                    v = "pool" if graduated else venue
                    drift = 0.5 * class_drift(v, y, age + t_now) + 0.5 * c["drift"]
                    if dh > 0:
                        y += drift * dh + sigma * sqrt_len[j] * gauss(0.0, 1.0)
                ys.append(y if alive else -1e18)
                t_now = bounds[j + 1]
            for j in range(n_reads):
                top = tops[j]
                yj = ys[j]
                if yj <= top[-1][0]:
                    continue
                pos = places - 1
                while pos > 0 and yj > top[pos - 1][0]:
                    top[pos] = top[pos - 1]
                    pos -= 1
                top[pos] = (yj, c["mint"])
        for j in range(n_reads):
            top = tops[j]
            for k in range(places):
                yk, mk = top[k]
                if mk is not None:
                    d = place_counts[j][k]
                    d[mk] = d.get(mk, 0) + 1
                    b = bin_of(yk, edges)
                    sh = size_hist[j][k]
                    sh[b] = sh.get(b, 0) + 1

    # ------------------------------------------------------- average over reads
    keep = []
    qbar = 0.0
    for j in range(n_reads):
        q = q_at(horizons[j])
        keep.append(1.0 - q)
        qbar += q / n_reads
    prob = [dict() for _ in range(places)]
    for k in range(places):
        acc = {}
        for j in range(n_reads):
            for mint, cnt in place_counts[j][k].items():
                acc[mint] = acc.get(mint, 0.0) + (cnt / N_PATHS) * keep[j] / n_reads
        prob[k] = acc
    # ----------------- place-level size model: each place's cap level
    # The Monte Carlo's k-th-largest-cap histogram is wide: every coin's own
    # volatility feeds it, and the newcomer mixture widens it further. The
    # place cap itself is a much tamer series — fit its drift and volatility
    # directly from the marks and forecast it per read. Drift is damped
    # (it cannot compound forever) and capped; volatility is shrunk toward
    # the pooled value across places.
    place_series = [[] for _ in range(places)]  # (t_hours, log10 cap SOL)
    for m in marks:
        t = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        for k in range(min(places, len(rows))):
            cap = int(rows[k]["market_cap_lamports"])
            if cap > 0:
                place_series[k].append((t, math.log10(cap) - 9.0))
    place_last = []
    place_drift = []
    place_drift_se = []
    place_resid = []
    place_vol = []
    pooled_incr = []
    pooled_resid = []
    for k in range(places):
        s = place_series[k]
        if not s:
            place_last.append(3.0)
            place_drift.append(0.0)
            place_drift_se.append(1.0)
            place_resid.append(0.15)
            place_vol.append(0.2)
            continue
        place_last.append(s[-1][1])
        pts = s[-12:]
        if len(pts) >= 2:
            mt = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            sxx = sum((p[0] - mt) ** 2 for p in pts)
            slope = (sum((p[0] - mt) * (p[1] - my) for p in pts) / sxx
                     if sxx > 1e-9 else 0.0)
            if len(pts) >= 3 and sxx > 1e-9:
                ss = sum((p[1] - (my + slope * (p[0] - mt))) ** 2 for p in pts)
                resid = math.sqrt(ss / (len(pts) - 2))
                se = max(resid, 0.05) / math.sqrt(sxx)
            else:
                resid = 0.15
                se = 1.0
        else:
            slope = 0.0
            resid = 0.15
            se = 1.0
        place_drift.append(min(max(slope, -PLACE_DRIFT_CAP), PLACE_DRIFT_CAP))
        place_drift_se.append(min(se, 2.0))
        place_resid.append(resid)
        pooled_resid.append(resid)
        place_vol.append(None)
        for (t1, y1), (t2, y2) in zip(s, s[1:]):
            if t2 > t1:
                pooled_incr.append((y2 - y1) / math.sqrt(t2 - t1))
    if pooled_incr:
        mu = sum(pooled_incr) / len(pooled_incr)
        vol_pool = math.sqrt(sum((d - mu) ** 2 for d in pooled_incr)
                             / len(pooled_incr))
    else:
        vol_pool = 0.2
    vol_pool = min(max(vol_pool, 0.08), 0.40)
    for k in range(places):
        if place_vol[k] is not None:
            continue
        s = place_series[k]
        diffs = [(y2 - y1) / math.sqrt(t2 - t1)
                 for (t1, y1), (t2, y2) in zip(s, s[1:])
                 if t2 > t1]
        if len(diffs) >= 3:
            mu = sum(diffs) / len(diffs)
            v = math.sqrt(sum((d - mu) ** 2 for d in diffs) / len(diffs))
            v = math.sqrt((len(diffs) * v * v + 4.0 * vol_pool ** 2)
                          / (len(diffs) + 4.0))
        else:
            v = vol_pool
        place_vol[k] = min(max(v, 0.08), 0.40)
    # Residual sd around each place's fitted trend, shrunk toward the pooled
    # value: this is the honest short-horizon noise of the place cap (the
    # raw increment dispersion double-counts the trend the centre already
    # extrapolates, which is why v2's widths were 3x too wide).
    if pooled_resid:
        resid_pool = math.sqrt(sum(r * r for r in pooled_resid) / len(pooled_resid))
    else:
        resid_pool = 0.15
    resid_pool = min(max(resid_pool, 0.06), 0.25)
    for k in range(places):
        n_pts = min(len(place_series[k][-12:]), 12)
        r = place_resid[k]
        place_resid[k] = math.sqrt((n_pts * r * r + 4.0 * resid_pool ** 2)
                                   / (n_pts + 4.0))
        place_resid[k] = min(max(place_resid[k], 0.06), 0.25)

    size_prob = [dict() for _ in range(places)]
    for k in range(places):
        acc = {}
        for j in range(n_reads):
            for b, cnt in size_hist[j][k].items():
                acc[b] = acc.get(b, 0.0) + (cnt / N_PATHS) / n_reads
        # Blend with the place-level forecast: a Normal per read, averaged
        # over reads (the mixture over reads is what widens it honestly).
        if PLACE_BLEND > 0:
            pl = {}
            for j in range(n_reads):
                h = horizons[j]
                shift = (place_drift[k] * PLACE_DRIFT_DAMP_H
                         * (1.0 - math.exp(-h / PLACE_DRIFT_DAMP_H)))
                shift = min(max(shift, PLACE_SHIFT_LO), PLACE_SHIFT_HI)
                mean = place_last[k] + shift
                damp_h = PLACE_DRIFT_DAMP_H * (1.0 - math.exp(-h / PLACE_DRIFT_DAMP_H))
                se_term = min((place_drift_se[k] * damp_h) ** 2, PLACE_SE_CAP ** 2)
                sd = math.sqrt(place_resid[k] ** 2 + PLACE_DIFF_SD ** 2 * h + se_term)
                sd = min(max(sd, PLACE_SD_FLOOR), PLACE_SD_CAP)
                nb = normal_bins(mean, sd, edges)
                for b, p in nb.items():
                    pl[b] = pl.get(b, 0.0) + p / n_reads
            merged = {}
            for b, p in acc.items():
                merged[b] = (1.0 - PLACE_BLEND) * p
            for b, p in pl.items():
                merged[b] = merged.get(b, 0.0) + PLACE_BLEND * p
            acc = merged
        if qbar > 0.01:
            # Newcomers holding the place run a little larger and much wider
            # than the simulated place cap: mix in a widened component.
            mean_k = sum(_bin_center(b, edges) * p for b, p in acc.items())
            var_k = sum((_bin_center(b, edges) - mean_k) ** 2 * p
                        for b, p in acc.items())
            sd_k = math.sqrt(var_k + NEW_SIZE_SD ** 2)
            newc = normal_bins(mean_k + NEW_SIZE_SHIFT, sd_k, edges)
            acc = {b: p * (1.0 - qbar) for b, p in acc.items()}
            for b, p in newc.items():
                acc[b] = acc.get(b, 0.0) + p * qbar
        size_prob[k] = smooth_bins(acc, SIZE_SMOOTH_SD)

    probabilities = []
    # Hopeless coins, for the p 0 rows: small last cap, negligible simulated
    # probability at every place, and not already named anywhere.
    hopeless = []
    if PZERO_MAX > 0:
        y_by_mint = {c["mint"]: c["y"] for c in cand}
        for mint, y in sorted(y_by_mint.items(), key=lambda kv: (-kv[1], kv[0])):
            if y > PZERO_Y_MAX:
                continue
            if all(prob[k].get(mint, 0.0) < 0.0005 for k in range(places)):
                hopeless.append(mint)
    for k in range(places):
        rows = sorted(prob[k].items(), key=lambda kv: (-kv[1], kv[0]))
        total = 0.0
        for mint, p in rows[:MAX_ROWS_PER_PLACE]:
            u = math.floor(p * MICRO) / MICRO
            if u <= 0:
                continue
            if total + u > 1.0:
                u = math.floor((1.0 - total) * MICRO) / MICRO
                if u <= 0:
                    break
            probabilities.append({"mint": mint, "place": k + 1, "p": u})
            total += u
        named_here = {r["mint"] for r in probabilities if r["place"] == k + 1}
        for mint in hopeless:
            if len(named_here) >= MAX_ROWS_PER_PLACE + PZERO_MAX:
                break
            if mint in named_here:
                continue
            probabilities.append({"mint": mint, "place": k + 1, "p": 0})
            named_here.add(mint)
    sizes = []
    for k in range(places):
        total = 0.0
        for b in sorted(size_prob[k]):
            u = math.floor(size_prob[k][b] * MICRO) / MICRO
            if u <= 0:
                continue
            if total + u > 1.0:
                u = math.floor((1.0 - total) * MICRO) / MICRO
                if u <= 0:
                    break
            sizes.append({"place": k + 1, "bin": int(b), "p": u})
            total += u
    top5 = []
    used = set()
    for k in range(places):
        best = None
        for mint, _p in sorted(prob[k].items(), key=lambda kv: (-kv[1], kv[0])):
            if mint not in used:
                best = mint
                break
        if best is None:
            break
        top5.append(best)
        used.add(best)
    pad = 0
    while len(top5) < places:
        top5.append("1" * 31 + chr(ord("2") + pad))
        pad += 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, output_name), "w") as f:
        f.write(json.dumps({"probabilities": probabilities, "top5": top5,
                            "sizes": sizes}) + "\n")


if __name__ == "__main__":
    main()
