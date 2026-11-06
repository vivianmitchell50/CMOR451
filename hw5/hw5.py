import os
import heapq
from collections import deque
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

Station   capacity   service time (sec)     after service
S1        1          U(10, 20)              fallout w.p. 0.15
S2        1          Tria(20, 30, 40)       fallout w.p. 0.30
S3        2          U(60, 84)              fallout w.p. 0.33
S4        1          Exp(mean 40)           rework  w.p. 0.08 (back to S1), else good
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
"""

INTERARRIVAL_MEAN = 60 / 125  # 0.48 min = 28.8 sec  (125 parts/hr)
CAP = [1, 1, 2, 1]
P = [0.15, 0.30, 0.33, 0.08]  # fallout prob at S1..S3, rework prob at S4


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


def make_streams(seed, n):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def simulate(rng, warmup_min, run_min, trace=None):
    """One replication. Statistics are cleared at warmup_min and collected for run_min more minutes.
    If trace is a list, the time-average number in system for every minute (from t = 0) is appended
    to it (used for the Welch plot)."""
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

    events = []  # heap of (time, tie-breaker, type, station, part)
    seq = count()

    def schedule(time, kind, station=None, part=None):
        heapq.heappush(events, (time, next(seq), kind, station, part))

    def start_service(i, part):
        busy[i] += 1
        schedule(t + service_time(i, rng), "departure", i, part)

    def enter_station(i, part):
        if busy[i] < CAP[i]:
            start_service(i, part)
        else:
            queue[i].append(part)

    schedule(rng.exponential(INTERARRIVAL_MEAN), "arrival", part=None)
    schedule(warmup, "warmup")
    schedule(end, "end")
    if trace is not None:
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
            schedule(t + rng.exponential(INTERARRIVAL_MEAN), "arrival")
            in_system += 1
            enter_station(0, Part(t))

        elif kind == "departure":  # service completion at station i
            busy[i] -= 1
            if queue[i]:
                start_service(i, queue[i].popleft())

            leaves = False
            if i < 3 and rng.random() < P[i]:  # fallout at S1, S2, S3
                fallouts[i] += 1
                leaves = True
            elif i < 3:
                enter_station(i + 1, part)
            elif rng.random() < P[3]:  # rework from S4 -> back to S1
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
            trace.append(area_N_total - last_tick_area)  # 1-minute bin -> time-average
            last_tick_area = area_N_total
            if t + 1 <= end:
                schedule(t + 1, "tick")

        elif kind == "end":
            break

    T = t - t_start  # minutes of data collection
    hrs = T / 60
    return np.array([
        area_N / T,                       # L
        maxR,                             # max parts in rework at once
        sum_time_all / n_left,            # W, all parts (min)
        sum_time_good / completed,        # W, good parts (min)
        *[area_busy[k] / (CAP[k] * T) for k in range(4)],  # U1..U4
        completed / hrs,                  # good parts per hour
        *[f / hrs for f in fallouts],     # fallout rates S1..S3 per hour
    ])


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


def flow_rates():
    """Traffic equations: long-run arrival rate (per min) into each station, and utilizations."""
    lam0 = 1 / INTERARRIVAL_MEAN
    pass_to_S4 = (1 - P[0]) * (1 - P[1]) * (1 - P[2])
    lam1 = lam0 / (1 - pass_to_S4 * P[3])  # external arrivals + rework
    lam = [lam1, lam1 * (1 - P[0]), lam1 * (1 - P[0]) * (1 - P[1]), lam1 * pass_to_S4]
    means = [15 / 60, 30 / 60, 72 / 60, 40 / 60]  # mean service times in minutes
    rho = [lam[k] * means[k] / CAP[k] for k in range(4)]
    return np.array(lam), np.array(rho)


def welch_moving_average(Ybar, w):
    """Welch's moving average Y_i(w), i = 1, ..., m - w."""
    m = len(Ybar)
    c = np.concatenate([[0], np.cumsum(Ybar)])
    i = np.arange(1, w + 1)
    early = c[2 * i - 1] / (2 * i - 1)
    i = np.arange(w + 1, m - w + 1)
    centered = (c[i + w] - c[i - w - 1]) / (2 * w + 1)
    return np.concatenate([early, centered])


def welch_plot(Y, windows, warmup_min, fname="welch_hw5.png"):
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
    fig.suptitle(f"Welch plot, 1-minute avg number in system  (n = {n} pilot replications)")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), dpi=150)
    plt.close(fig)
    return fname


def run_length_plot(Y, warmup_min, run_min, n_prod, fname="runlength_hw5.png", n_show=10):
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
    fig.suptitle(f"Run length  (warm-up = {warmup_min} min, n = {n} pilot replications)")
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


if __name__ == "__main__":
    seed = 451

    # long-run flow rates from the traffic equations (sanity check for the simulation)
    lam, rho = flow_rates()
    lam0 = 1 / INTERARRIVAL_MEAN
    theory = np.full(len(MEASURES), np.nan)
    theory[4:8] = rho
    theory[8] = 60 * lam[3] * (1 - P[3])
    theory[9:12] = 60 * lam[:3] * np.array(P[:3])
    print("Traffic equations: arrival rate into S1..S4 (parts/hr):", np.round(60 * lam, 4))
    print("                   utilization S1..S4:               ", np.round(rho, 4))
    print()

    # ---------------- experiment design ----------------
    warmup_min = 500      # chosen from the Welch plot
    run_min = 5000        # chosen from the run-length plot (10x warm-up)
    target_rel = 0.02    # want every 95% CI half-width within +/-2% of its estimate

    # Step 1: pilot runs for the Welch plot and run-length plot (1-minute avg number in system)
    pilot_n = 50
    pilot_min = warmup_min + 2 * run_min
    streams = make_streams(1000, pilot_n)
    traces = []
    for j in range(pilot_n):
        tr = []
        simulate(streams[j], 0, pilot_min, trace=tr)
        traces.append(tr[:pilot_min])
    Yh = np.array(traces)
    print("saved", welch_plot(Yh[:, :3000], [1, 24, 100, 250], warmup_min))

    # Step 2: pilot replications with the chosen warm-up and run length -> how many replications?
    n0 = 20
    streams = make_streams(2000, n0)
    Y0 = np.array([simulate(streams[j], warmup_min, run_min) for j in range(n0)])
    ybar0, s0, h0 = ci(Y0)
    z = stats.norm.ppf(0.975)
    n_req = np.ceil((z * s0 / (target_rel * np.abs(ybar0))) ** 2).astype(int)
    print_table(Y0, f"Pilot: n0 = {n0}, warm-up = {warmup_min} min, run length = {run_min} min", theory)
    print(f"Replications needed for +/-{100 * target_rel:.0f}% relative precision, n = (z s / (r ybar))^2:")
    for (name, _), nr in zip(MEASURES, n_req):
        print(f"  {name:<32}{nr:>6}")
    print()
    n = int(max(n_req.max(), n0))
    print("saved", run_length_plot(Yh, warmup_min, run_min, n))
    print()

    # Step 3: production runs with fresh seeds
    streams = make_streams(seed, n)
    Y = np.array([simulate(streams[j], warmup_min, run_min) for j in range(n)])
    print_table(Y, f"Production: n = {n} replications, warm-up = {warmup_min} min, "
                   f"run length = {run_min} min", theory)
