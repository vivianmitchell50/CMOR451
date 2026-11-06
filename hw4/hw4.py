import os
import numpy as np
from math import factorial
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
HW 4

Poisson arrivals, rate 1 cust/min, single FCFS line

System 1: one fast server, service times ~Exp(mean 0.9 min)
System 2: two slower servers in parallel, service times ~Exp(mean 1.8 min)

both start empty and idle; rho = 0.9 in both systems

performance measures:
W_q = long-run avg waiting time in the queue
P(W_q > 5) = long-run proportion of cust whose wait > 5
U = long-run server utilization (busy time per server / elapsed time)
"""

def make_streams(seed, n):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]

def exp_gen(mean, rng):
    """Inverse transform: X = -mean * ln(U), U ~ Uniform(0, 1)."""
    u = 1 - rng.random()  # rng.random() is in [0, 1), so u is in (0, 1] and ln(u) is finite
    return -mean * np.log(u)


def multi_server_sim(rng, c, inter_arrival_mean, service_mean, warmup, run_length, waits_out=None):
    """c identical servers fed by one FCFS line (c = 1 for System 1, c = 2 for System 2)."""
    t = 0
    arrival_counter = 0
    departure_counter = 0
    start_counter = 0  # customers who have started service (FCFS -> in arrival order)
    in_system = 0
    busy = 0  # number of busy servers

    t0 = exp_gen(inter_arrival_mean, rng)
    next_arrival = t0
    next_departure = [float('inf')] * c  # completion time at each server
    t_warm = 0

    arrivals = {}  # arrival times of customers still waiting in the queue
    area_q = 0  # area under the queue-length curve Q(t)
    area_busy = 0  # area under B(t), the number of busy servers
    sum_wait = 0  # total queue wait of customers who start service after warm-up
    n_over5 = 0  # how many of those waited more than 5 min
    n_wait = 0  # how many customers started service after warm-up

    while departure_counter < warmup + run_length:
        j = min(range(c), key=next_departure.__getitem__)  # server with the next completion

        # case 1: the next event is an arrival
        if next_arrival < next_departure[j]:
            area_q += (next_arrival - t) * (in_system - busy)
            area_busy += (next_arrival - t) * busy
            t = next_arrival
            arrival_counter += 1
            in_system += 1
            arrivals[arrival_counter] = t
            next_arrival = t + exp_gen(inter_arrival_mean, rng)
            if busy < c:
                free = next_departure.index(float('inf'))  # an idle server
                start_counter += 1
                wait = t - arrivals.pop(start_counter)
                if waits_out is not None:
                    waits_out.append(wait)  # W_i in customer order, for the pilot plots
                if start_counter > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                busy += 1
                next_departure[free] = t + exp_gen(service_mean, rng)

        # case 2: the next event is a departure from server j
        else:
            area_q += (next_departure[j] - t) * (in_system - busy)
            area_busy += (next_departure[j] - t) * busy
            t = next_departure[j]
            in_system -= 1
            busy -= 1
            departure_counter += 1
            next_departure[j] = float('inf')
            if departure_counter == warmup:
                area_q = 0
                area_busy = 0
                t_warm = t
            if in_system > busy:  # someone is waiting -> server j takes the next in line
                start_counter += 1
                wait = t - arrivals.pop(start_counter)
                if waits_out is not None:
                    waits_out.append(wait)  # W_i in customer order, for the pilot plots
                if start_counter > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                busy += 1
                next_departure[j] = t + exp_gen(service_mean, rng)

    T = t - t_warm  # length of the observation period after warm-up
    Wq = sum_wait / n_wait
    P_over5 = n_over5 / n_wait
    U = area_busy / (c * T)
    return np.array([Wq, P_over5, U])


def ci(Y, alpha=0.05):
    """Columnwise point estimate, sample std dev, and CI half-width for an (n x 3) array of replication outputs."""
    n = Y.shape[0]
    ybar = Y.mean(axis=0)
    s = Y.std(axis=0, ddof=1)
    h = stats.t.ppf(1 - alpha/2, n - 1) * s / np.sqrt(n)
    return ybar, s, h


def exact(c, lam, service_mean):
    """Steady-state M/M/c values (times in minutes) for [Wq, P(Wq > 5), U]."""
    mu = 1 / service_mean
    a = lam / mu  # offered load
    rho = a / c
    tail = a**c / factorial(c) / (1 - rho)
    C = tail / (sum(a**k / factorial(k) for k in range(c)) + tail)  
    Wq = C / (c * mu - lam)
    P_over5 = C * np.exp(-(c * mu - lam) * 5)
    U = rho
    return np.array([Wq, P_over5, U])


def welch_moving_average(Wbar, w):
    """W_i(w) for i = 1, ..., t - w, from the averaged process Wbar_1, ..., Wbar_t."""
    t = len(Wbar)
    c = np.concatenate([[0], np.cumsum(Wbar)])  # c[m] = Wbar_1 + ... + Wbar_m
    i = np.arange(1, w + 1)
    early = c[2*i - 1] / (2*i - 1)  # i = 1, ..., w: average of Wbar_1 .. Wbar_{2i-1}
    i = np.arange(w + 1, t - w + 1)
    centered = (c[i + w] - c[i - w - 1]) / (2*w + 1)  # i = w+1, ..., t-w: average of Wbar_{i-w} .. Wbar_{i+w}
    return np.concatenate([early, centered])


def pilot_waits(c, lam, service_mean, n, t, seed):
    """Pilot run: queue waits W_i of the first t customers in each of n replications, as an (n x t) array."""
    streams = make_streams(seed, n)
    W = np.empty((n, t))
    for j in range(n):
        waits = []
        multi_server_sim(streams[j], c, 1 / lam, service_mean, 0, t, waits_out=waits)
        W[j] = waits[:t]
    return W


def welch_plot(W, title, tag, exact_Wq, windows):
    """Welch's method for the warm-up: moving averages of the averaged process for several windows w."""
    n, t = W.shape
    Wbar = W.mean(axis=0)  # averaged process Wbar_i over the n replications

    fig, axes = plt.subplots(len(windows), 1, figsize=(8, 2.2 * len(windows)), sharex=True)
    for ax, w in zip(axes, windows):
        ma = welch_moving_average(Wbar, w)
        ax.plot(np.arange(1, t - w + 1), ma, color="#2a78d6", lw=1.2)
        ax.axhline(exact_Wq, color="#52514e", ls="--", lw=1, label="exact steady-state $W_q$")
        ax.set_title(f"w = {w}", loc="left", fontsize=10)
        ax.set_ylabel(r"$\bar{W}_i(w)$ (min)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("customer i")
    fig.suptitle(f"Welch plot, {title}  (n = {n} pilot replications, t = {t} customers)")
    fig.tight_layout()
    fname = f"welch_{tag}.png"
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), dpi=150)
    plt.close(fig)
    return fname


def run_length_plot(W, title, tag, ex, warmup, run_length, n_prod, n_show=10):
    """Choosing the run length: per-replication running averages after warm-up, and the
    95% CI half-width (relative to the exact value) that n_prod replications would give."""
    n = W.shape[0]
    X = W[:, warmup:]  # customers after the warm-up
    m = np.unique(np.geomspace(100, X.shape[1], 400).astype(int))  # run lengths to evaluate
    run_Wq = (np.cumsum(X, axis=1) / np.arange(1, X.shape[1] + 1))[:, m - 1]
    run_P5 = (np.cumsum(X > 5, axis=1) / np.arange(1, X.shape[1] + 1))[:, m - 1]
    tq = stats.t.ppf(0.975, n_prod - 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for j in range(min(n_show, n)):
        ax1.plot(m, run_Wq[j], color="#2a78d6", lw=1, alpha=0.5)
    ax1.axhline(ex[0], color="#52514e", ls="--", lw=1, label="exact steady-state $W_q$")
    ax1.set_title(f"running average of $W_q$ after warm-up, {min(n_show, n)} pilot replications",
                  loc="left", fontsize=10)
    ax1.set_ylabel("$W_q$ estimate (min)")

    for Y, e, name, color in [(run_Wq, ex[0], "$W_q$", "#2a78d6"), (run_P5, ex[1], "P($W_q$ > 5)", "#eb6834")]:
        h = tq * Y.std(axis=0, ddof=1) / np.sqrt(n_prod)
        ax2.plot(m, 100 * h / e, color=color, lw=2, label=name)
    ax2.set_yscale("log")
    ax2.set_title(f"95% CI half-width with n = {n_prod} replications, as % of exact value",
                  loc="left", fontsize=10)
    ax2.set_ylabel("relative half-width (%)")
    ax2.set_xlabel("run length (customers after warm-up)")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(run_length, color="#52514e", ls=":", lw=1, label=f"current run length = {run_length}")
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Run length, {title}  (warm-up = {warmup}, n = {n} pilot replications)")
    fig.tight_layout()
    fname = f"runlength_{tag}.png"
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), dpi=150)
    plt.close(fig)
    return fname


MEASURES = [
    ("Wq  Avg wait in queue", "min"),
    ("P(Wq > 5)", ""),
    ("U   Server utilization", ""),
]


if __name__ == "__main__":
    seed = 451
    n = 100
    lam = 1  # customers per minute
    systems = [
        ("System 1 - single fast server", 1, 0.9),
        ("System 2 - two slower servers", 2, 1.8),
    ]

    # run length and warm-up, in service completions
    # rho = 0.9 sits between the rho = 0.75 and 0.95 cases from HW 3 Q2, so the
    # warm-up and run length are chosen between those values
    warmup = 3000
    run_length = 10000

    # pilot runs for choosing warm-up and run length (separate seeds from the production runs)
    # pilot_n = 50
    # pilot_t = warmup + 2 * run_length  # long enough to see past the current run length
    # welch_t = 30000  # customers shown on the Welch plot
    # welch_w = [20, 100, 500, 2000]
    # for i, (title, c, service_mean) in enumerate(systems):
    #     tag = f"sys{i + 1}"
    #     W = pilot_waits(c, lam, service_mean, pilot_n, pilot_t, seed=1000 + i)
    #     ex = exact(c, lam, service_mean)
    #     print("saved", welch_plot(W[:, :welch_t], title, tag, ex[0], welch_w))
    #     print("saved", run_length_plot(W, title, tag, ex, warmup, run_length, n))
    # print()

    Ys = []
    for i, (title, c, service_mean) in enumerate(systems):
        streams = make_streams(seed + i, n)  # distinct seed for every replication
        Y = np.array([multi_server_sim(streams[j], c, 1 / lam, service_mean, warmup, run_length) for j in range(n)])
        Ys.append(Y)
        ybar, s, h = ci(Y)

        print(f"{title}  (c = {c}, mean service = {service_mean} min), n = {n} replications, "
              f"warm-up = {warmup}, run length = {run_length} completions")
        print(f"{'Measure':<26}{'Estimate':>10}{'95% CI':>24}{'Exact':>10}  Units")
        print("-" * 82)
        for (name, unit), m, hw, ex in zip(MEASURES, ybar, h, exact(c, lam, service_mean)):
            print(f"{name:<26}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{ex:>10.4f}  {unit}")
        print()

    # paired-t CI for System 1 minus System 2: Z_j = Y1_j - Y2_j are iid across replications
    # (the systems use independent streams, so this is valid; n1 = n2 = n)
    Z = Ys[0] - Ys[1]
    zbar, sz, hz = ci(Z)
    ex_diff = exact(1, lam, 0.9) - exact(2, lam, 1.8)

    print(f"Difference System 1 - System 2 (paired-t), n = {n} replications")
    print(f"{'Measure':<26}{'Estimate':>10}{'95% CI':>24}{'Exact':>10}  Units")
    print("-" * 82)
    for (name, unit), m, hw, ex in zip(MEASURES, zbar, hz, ex_diff):
        print(f"{name:<26}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{ex:>10.4f}  {unit}")
