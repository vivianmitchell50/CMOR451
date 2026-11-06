import os
import sys
import time
import heapq
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import count
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
HW 5 - four-station serial line with fallout and rework

Arrivals -> S1 -> S2 -> S3 -> S4 -> Good
             |     |     |     |
          fallout fallout fallout rework (back to S1)

Time unit in the code: minutes.
Arrivals: 125 parts/hr, interarrival ~ Exp(mean 1/125 * 60 min = 0.48 min = 28.8 sec)

Station   capacity   service time (sec)     after service (base system)   part (e) system
S1        1          U(10, 20)              fallout w.p. 0.15             0.10
S2        1          Tria(20, 30, 40)       fallout w.p. 0.30             0.25
S3        2          U(60, 84)              fallout w.p. 0.33             0.28
S4        1          Exp(mean 40)           rework  w.p. 0.08 (to S1)     0.08
(service times are divided by 60 in service_time() to convert seconds -> minutes)

Fallouts are discarded.

State:     S_i = number of busy machines at station i (0/1, or 0/1/2 at S3)
Counters:  N_i = number of parts in queue i,  R = number of parts undergoing rework

Performance measures (steady state):
  L        average number of parts in the system
  maxR     max number of parts undergoing rework at the same time
  W        average time a part spends in the system (all parts that leave; also good parts only)
  U_i      utilization of S1..S4 (busy time / (capacity * elapsed time))
  good/hr  rate at which good parts are completed
  fall/hr  rate at which parts fall out at S1, S2, S3

Random numbers: every replication j gets 9 separate streams (arrivals, service at S1..S4,
routing at S1..S4) seeded from (seed, j, k). Running the base and the part (e) system with the
same (seed, j) gives common random numbers, which is what makes the paired-t in (e)(iii) sharp.
"""

INTERARRIVAL_MEAN = 60 / 125  # 0.48 min = 28.8 sec  (125 parts/hr)
CAP = [1, 1, 2, 1]
P_BASE = [0.15, 0.30, 0.33, 0.08]  # fallout prob at S1..S3, rework prob at S4
P_NEW = [0.10, 0.25, 0.28, 0.08]   # part (e): each fallout prob 5 percentage points lower


def service_time(i, rng):
    if i == 0:
        return rng.uniform(1/6, 1/3)
    if i == 1:
        return rng.triangular(1/3, 1/2, 2/3)
    if i == 2:
        return rng.uniform(1, 84/60)
    return rng.exponential(40/60)


class Part:
    def __init__(self, arrival):
        self.arrival = arrival
        self.rework = False  # True once the part has been sent back from S4


def simulate(rep_seed, p, warmup_min, run_min, trace=False):
    """One replication with routing probabilities p. Statistics are cleared at warmup_min and
    collected for run_min more minutes. If trace is True, also returns the time-average number in
    system for every minute from t = 0 (used for the Welch and run-length plots)."""
    g = [np.random.default_rng([*rep_seed, k]) for k in range(9)]
    arr_rng, svc_rng, route_rng = g[0], g[1:5], g[5:9]

    warmup = warmup_min
    end = warmup + run_min

    t = 0.0
    busy = [0, 0, 0, 0]  # S_i
    queue = [deque() for _ in range(4)]  # N_i = len(queue[i])
    in_system = 0
    R = 0

    # statistics (cleared at the warm-up)
    area_N = 0.0
    area_busy = [0.0] * 4
    maxR = 0
    sum_time_all, n_left = 0.0, 0  # every part that leaves (good or fallout)
    sum_time_good, completed = 0.0, 0  # good parts only
    fallouts = [0, 0, 0]
    t_start = 0.0

    # trace for the Welch plot (never cleared)
    area_N_total = 0.0
    last_tick_area = 0.0
    tr = []

    events = []  # heap of (time, tie-breaker, type, station, part)
    seq = count()

    def schedule(time, kind, station=None, part=None):
        heapq.heappush(events, (time, next(seq), kind, station, part))

    def start_service(i, part):
        busy[i] += 1
        schedule(t + service_time(i, svc_rng[i]), "departure", i, part)

    def enter_station(i, part):
        if busy[i] < CAP[i]:
            start_service(i, part)
        else:
            queue[i].append(part)

    schedule(arr_rng.exponential(INTERARRIVAL_MEAN), "arrival")
    schedule(warmup, "warmup")
    schedule(end, "end")
    if trace:
        schedule(1.0, "tick")

    while events:
        time, _, kind, i, part = heapq.heappop(events)

        # advance the clock and update the time-integrals
        dt = time - t
        area_N += in_system * dt
        area_N_total += in_system * dt
        for k in range(4):
            area_busy[k] += busy[k] * dt
        t = time

        if kind == "arrival":
            schedule(t + arr_rng.exponential(INTERARRIVAL_MEAN), "arrival")
            in_system += 1
            enter_station(0, Part(t))

        elif kind == "departure":  # service completion at station i
            busy[i] -= 1
            if queue[i]:
                start_service(i, queue[i].popleft())

            u = route_rng[i].random()
            leaves = False
            if i < 3 and u < p[i]:  # fallout at S1, S2, S3
                fallouts[i] += 1
                leaves = True
            elif i < 3:
                enter_station(i + 1, part)
            elif u < p[3]:  # rework from S4 -> back to S1
                if not part.rework:  # count each part once, even if reworked again
                    part.rework = True
                    R += 1
                    maxR = max(maxR, R)
                enter_station(0, part)
            else:  # good part
                completed += 1
                sum_time_good += t - part.arrival
                leaves = True

            if leaves:
                in_system -= 1
                n_left += 1
                sum_time_all += t - part.arrival
                if part.rework:
                    R -= 1

        elif kind == "warmup":  # clear the statistics, keep the state
            area_N = 0.0
            area_busy = [0.0] * 4
            maxR = R
            sum_time_all, n_left = 0.0, 0
            sum_time_good, completed = 0.0, 0
            fallouts = [0, 0, 0]
            t_start = t

        elif kind == "tick":
            tr.append(area_N_total - last_tick_area)  # 1-minute bin -> time-average
            last_tick_area = area_N_total
            if t + 1 <= end:
                schedule(t + 1, "tick")

        elif kind == "end":
            break

    T = t - t_start  # minutes of data collection
    hrs = T / 60
    out = np.array([
        area_N / T,                       # L
        maxR,                             # max parts in rework at once
        sum_time_all / n_left,            # W, all parts (min)
        sum_time_good / completed,        # W, good parts (min)
        *[area_busy[k] / (CAP[k] * T) for k in range(4)],  # U1..U4
        completed / hrs,                  # good parts per hour
        *[f / hrs for f in fallouts],     # fallout rates S1..S3 per hour
    ])
    if trace:
        return out, np.array(tr)
    return out


def _rep(args):
    return simulate(*args)


def run_reps(pool, seed, n, p, warmup_min, run_min, trace=False):
    """n independent replications (seeds (seed, 0), ..., (seed, n-1)), spread over all cores."""
    jobs = [((seed, j), p, warmup_min, run_min, trace) for j in range(n)]
    res = list(pool.map(_rep, jobs, chunksize=max(1, n // (4 * (os.cpu_count() or 1)))))
    if trace:
        m = min(len(tr) for _, tr in res)
        return np.array([o for o, _ in res]), np.array([tr[:m] for _, tr in res])
    return np.array(res)


MEASURES = [
    ("Avg number in system L", "parts"),
    ("Max parts in rework at once", "parts"),
    ("Avg time in system, all parts", "min"),
    ("Avg time in system, good parts", "min"),
    ("Utilization S1", ""),
    ("Utilization S2", ""),
    ("Utilization S3", ""),
    ("Utilization S4", ""),
    ("Good part rate", "parts/hr"),
    ("Fallout rate S1", "parts/hr"),
    ("Fallout rate S2", "parts/hr"),
    ("Fallout rate S3", "parts/hr"),
]


def ci(Y, alpha=0.05):
    """Columnwise mean, sample std dev, and t CI half-width for an (n x k) array of replication outputs."""
    n = Y.shape[0]
    ybar = Y.mean(axis=0)
    s = Y.std(axis=0, ddof=1)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * s / np.sqrt(n)
    return ybar, s, h


def flow_rates(p):
    """Traffic equations: long-run arrival rate (per min) into each station, and utilizations."""
    lam0 = 1 / INTERARRIVAL_MEAN
    pass_to_S4 = (1 - p[0]) * (1 - p[1]) * (1 - p[2])
    lam1 = lam0 / (1 - pass_to_S4 * p[3])  # external arrivals + rework
    lam = [lam1, lam1 * (1 - p[0]), lam1 * (1 - p[0]) * (1 - p[1]), lam1 * pass_to_S4]
    means = [15 / 60, 30 / 60, 72 / 60, 40 / 60]  # mean service times in minutes
    rho = [lam[k] * means[k] / CAP[k] for k in range(4)]
    return np.array(lam), np.array(rho)


def theory_column(p):
    lam, rho = flow_rates(p)
    theory = np.full(len(MEASURES), np.nan)
    theory[4:8] = rho
    theory[8] = 60 * lam[3] * (1 - p[3])
    theory[9:12] = 60 * lam[:3] * np.array(p[:3])
    return lam, rho, theory


def welch_moving_average(Ybar, w):
    """Welch's moving average Y_i(w), i = 1, ..., m - w."""
    m = len(Ybar)
    c = np.concatenate([[0], np.cumsum(Ybar)])
    i = np.arange(1, w + 1)
    early = c[2 * i - 1] / (2 * i - 1)
    i = np.arange(w + 1, m - w + 1)
    centered = (c[i + w] - c[i - w - 1]) / (2 * w + 1)
    return np.concatenate([early, centered])


def welch_plot(Y, windows, warmup_min, fname, title=""):
    n, m = Y.shape
    Ybar = Y.mean(axis=0)
    fig, axes = plt.subplots(len(windows), 1, figsize=(8, 2.2 * len(windows)), sharex=True)
    for ax, w in zip(axes, windows):
        ax.plot(np.arange(1, m - w + 1), welch_moving_average(Ybar, w), color="#2a78d6", lw=1.2)
        ax.axvline(warmup_min, color="#52514e", ls=":", lw=1, label=f"warm-up = {warmup_min} min")
        ax.set_title(f"w = {w}", loc="left", fontsize=10)
        ax.set_ylabel("avg # in system")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("minute i")
    fig.suptitle(f"Welch plot{title}, 1-minute avg number in system  (n = {n} pilot replications)")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), dpi=150)
    plt.close(fig)
    return fname


def run_length_plot(Y, warmup_min, run_min, n_prod, fname, title="", n_show=10):
    """Running average of L after the warm-up, and the 95% CI half-width n_prod replications would give."""
    n = Y.shape[0]
    X = Y[:, warmup_min:]
    m = np.unique(np.geomspace(50, X.shape[1], 300).astype(int))
    run_L = (np.cumsum(X, axis=1) / np.arange(1, X.shape[1] + 1))[:, m - 1]
    tq = stats.t.ppf(0.975, n_prod - 1)
    h = tq * run_L.std(axis=0, ddof=1) / np.sqrt(n_prod)
    rel = 100 * h / run_L.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for j in range(min(n_show, n)):
        ax1.plot(m, run_L[j], color="#2a78d6", lw=1, alpha=0.5)
    ax1.set_title(f"running average of L after warm-up, {min(n_show, n)} pilot replications",
                  loc="left", fontsize=10)
    ax1.set_ylabel("L estimate (parts)")
    ax2.plot(m, rel, color="#2a78d6", lw=2, label="L")
    ax2.set_yscale("log")
    ax2.set_title(f"95% CI half-width of L with n = {n_prod} replications, % of estimate",
                  loc="left", fontsize=10)
    ax2.set_ylabel("relative half-width (%)")
    ax2.set_xlabel("run length (minutes after warm-up)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(run_min, color="#52514e", ls=":", lw=1, label=f"run length = {run_min} min")
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Run length{title}  (warm-up = {warmup_min} min, n = {n} pilot replications)")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), dpi=150)
    plt.close(fig)
    return fname


def print_table(Y, title, ref=None):
    ybar, s, h = ci(Y)
    print(title)
    head = f"{'Measure':<32}{'Estimate':>10}{'95% CI':>26}{'Half-width':>12}"
    if ref is not None:
        head += f"{'Theory':>9}"
    print(head + "  Units")
    print("-" * (len(head) + 10))
    for k, ((name, unit), m, hw) in enumerate(zip(MEASURES, ybar, h)):
        line = f"{name:<32}{m:>10.4f}   [{m - hw:>9.4f}, {m + hw:>9.4f}]{hw:>12.4f}"
        if ref is not None:
            line += f"{ref[k]:>9.4f}" if not np.isnan(ref[k]) else f"{'':>9}"
        print(line + f"  {unit}")
    print()


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


def design_and_run(pool, t0, label, p, warmup_min, run_min, welch_len, welch_windows,
                   pilot_n, n0, pilot_seed, n0_seed, prod_seed, suffix, target_rel=0.02):
    """Experiment design (Welch plot -> warm-up, run-length plot -> run length, pilot -> n), then
    production runs. Returns the production output array (n x 12)."""
    lam, rho, theory = theory_column(p)
    print(f"===== {label}: p = {p} =====")
    print("Traffic equations: arrival rate into S1..S4 (parts/hr):", np.round(60 * lam, 4))
    print("                   utilization S1..S4:               ", np.round(rho, 4))
    print()

    # Step 1: long pilot runs, 1-minute avg number in system -> Welch plot and run-length plot
    pilot_min = warmup_min + 2 * run_min
    log(f"{label}: {pilot_n} pilot runs of {pilot_min} min for the Welch plot ...", t0)
    _, Yh = run_reps(pool, pilot_seed, pilot_n, p, 0, pilot_min, trace=True)
    print("saved", welch_plot(Yh[:, :welch_len], welch_windows, warmup_min,
                              f"welch_hw5{suffix}.png", f" ({label})"))

    # Step 2: pilot replications with the chosen warm-up and run length -> how many replications?
    log(f"{label}: {n0} pilot replications ...", t0)
    Y0 = run_reps(pool, n0_seed, n0, p, warmup_min, run_min)
    ybar0, s0, _ = ci(Y0)
    z = stats.norm.ppf(0.975)
    n_req = np.ceil((z * s0 / (target_rel * np.abs(ybar0))) ** 2).astype(int)
    print_table(Y0, f"Pilot ({label}): n0 = {n0}, warm-up = {warmup_min} min, "
                    f"run length = {run_min} min", theory)
    print(f"Replications needed for +/-{100 * target_rel:.0f}% relative precision, n = (z s / (r ybar))^2:")
    for (name, _), nr in zip(MEASURES, n_req):
        print(f"  {name:<32}{nr:>6}")
    print()
    n = int(max(n_req.max(), n0))
    print("saved", run_length_plot(Yh, warmup_min, run_min, n, f"runlength_hw5{suffix}.png",
                                   f" ({label})"))
    print()

    # Step 3: production runs with fresh seeds
    log(f"{label}: {n} production replications ...", t0)
    Y = run_reps(pool, prod_seed, n, p, warmup_min, run_min)
    print_table(Y, f"Production ({label}): n = {n} replications, warm-up = {warmup_min} min, "
                   f"run length = {run_min} min", theory)
    return Y


def paired_t(Y_base, Y_new, idx, alpha=0.05):
    """Paired-t CI for E[new - base] on the measures in idx (rows of Y_base and Y_new are paired)."""
    n = Y_base.shape[0]
    D = Y_new[:, idx] - Y_base[:, idx]
    dbar, s, h = ci(D, alpha)
    print(f"Paired-t comparison, new - base, n = {n} pairs, {100 * (1 - alpha):.0f}% confidence "
          f"(t_{{{n - 1}, {1 - alpha / 2}}} = {stats.t.ppf(1 - alpha / 2, n - 1):.4f})")
    head = (f"{'Measure':<32}{'Base':>10}{'New':>10}{'Mean diff':>11}{'95% CI for diff':>26}"
            f"{'s_D':>9}  Conclusion")
    print(head)
    print("-" * (len(head) + 20))
    for j, k in enumerate(idx):
        lo, hi = dbar[j] - h[j], dbar[j] + h[j]
        if lo > 0:
            concl = "new > base"
        elif hi < 0:
            concl = "new < base"
        else:
            concl = "no significant difference"
        print(f"{MEASURES[k][0]:<32}{Y_base[:, k].mean():>10.4f}{Y_new[:, k].mean():>10.4f}"
              f"{dbar[j]:>11.4f}   [{lo:>9.4f}, {hi:>9.4f}]{s[j]:>9.4f}  {concl}")
    print()


if __name__ == "__main__":
    t0 = time.time()
    sys.stdout.reconfigure(line_buffering=True)
    with ProcessPoolExecutor() as pool:

        # ---------------- parts (c), (d): base system ----------------
        # Welch plot levels off within ~100-200 min, so warm-up = 500 min is a safe margin;
        # run length = 10x warm-up.
        BASE_WARMUP, BASE_RUN = 500, 5000
        Y_base = design_and_run(pool, t0, "base system", P_BASE, BASE_WARMUP, BASE_RUN,
                                welch_len=3000, welch_windows=[1, 24, 100, 250],
                                pilot_n=50, n0=20, pilot_seed=1000, n0_seed=2000,
                                prod_seed=451, suffix="")

        # ---------------- part (e)(i), (ii): fallout probabilities reduced by 5 points ----------
        # S2 now runs at rho ~ 0.975, so the system is much slower to reach steady state and the
        # output is much more autocorrelated: the Welch plot levels off around minute 2000-3000,
        # so warm-up = 5000 min is a safe margin, and run length = 50000 min (10x warm-up).
        NEW_WARMUP, NEW_RUN = 5000, 50000
        Y_new = design_and_run(pool, t0, "reduced fallout", P_NEW, NEW_WARMUP, NEW_RUN,
                               welch_len=25000, welch_windows=[100, 500, 1000, 2500],
                               pilot_n=40, n0=30, pilot_seed=1001, n0_seed=2001,
                               prod_seed=452, suffix="_new")

        # ---------------- part (e)(iii): paired-t comparison ----------------
        # Paired-t needs the same number of replications for both systems, with replication j of
        # each system paired. Both systems are run with the same seeds (common random numbers) and
        # the same (longer) warm-up and run length, so replication j of each system sees the same
        # arrivals and service times over the same time window. The longer settings are valid for
        # the base system as well (it reaches steady state within its 500 min warm-up).
        n_pair = max(len(Y_base), len(Y_new))
        log(f"paired comparison: {n_pair} replications of each system (CRN) ...", t0)
        Yp_base = run_reps(pool, 4510, n_pair, P_BASE, NEW_WARMUP, NEW_RUN)
        Yp_new = run_reps(pool, 4510, n_pair, P_NEW, NEW_WARMUP, NEW_RUN)
        paired_t(Yp_base, Yp_new, idx=[0, 2, 3, 8])
        corr = [np.corrcoef(Yp_base[:, k], Yp_new[:, k])[0, 1] for k in (0, 2, 8)]
        print("Correlation between paired outputs (L, W all, good rate):", np.round(corr, 3))
        print("(positive correlation from CRN makes Var(D) < Var(base) + Var(new))")
        log("done", t0)
