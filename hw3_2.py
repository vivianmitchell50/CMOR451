import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
HW 3, Q2

M/M/1 queue with lambda cust/hr, service times ~Exp(12)

rho = lambda/mu in {0.50, 0.75, 0.95}

performance measures:
W_q = long-run avg waiting time in the queue
P(W_q > 5) = long-run proportion of cust whose wait > 5
L_q = long-run avg number of cust in queue
U = long-run server utilization
"""

def make_streams(seed, n):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]

def inter_arrival_gen(rng, rate):
    return rng.exponential(1 / rate)

def service_time_gen(rng, mean):
    return rng.exponential(mean)


def mm1_sim(rng, inter_arrival_rate, service_mean, warmup, run_length, waits_out=None):
    t = 0
    arrival_counter = 0
    departure_counter = 0
    in_system = 0

    t0 = inter_arrival_gen(rng, inter_arrival_rate)
    next_arrival = t0
    next_departure = float('inf')
    t_warm = 0

    arrivals = {}  # arrival times of customers still in the system
    area_q = 0  # area under the queue-length curve Q(t)
    area_busy = 0  # area under I_busy(t)
    sum_wait = 0  # total queue wait of customers who start service after warm-up
    n_over5 = 0  # how many of those waited more than 5 min
    n_wait = 0  # how many customers started service after warm-up

    while departure_counter < warmup + run_length:
        # case 1: the next event is an arrival
        if next_arrival < next_departure:
            area_q += (next_arrival - t) * max(in_system - 1, 0)
            area_busy += (next_arrival - t) * (in_system > 0)
            t = next_arrival
            arrival_counter += 1
            in_system += 1
            arrivals[arrival_counter] = t
            next_arrival = t + inter_arrival_gen(rng, inter_arrival_rate)
            if in_system == 1:
                k = departure_counter + 1  # customer starting service
                wait = t - arrivals.pop(k)
                if waits_out is not None:
                    waits_out.append(wait)  # W_i in customer order, for the Welch plot
                if k > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                next_departure = t + service_time_gen(rng, service_mean)

        # case 2: the next event is a departure
        else:
            area_q += (next_departure - t) * max(in_system - 1, 0)
            area_busy += (next_departure - t) * (in_system > 0)
            t = next_departure
            in_system -= 1
            departure_counter += 1
            if departure_counter == warmup:
                area_q = 0
                area_busy = 0
                t_warm = t
            if in_system > 0:
                k = departure_counter + 1  # customer starting service
                wait = t - arrivals.pop(k)
                if waits_out is not None:
                    waits_out.append(wait)  # W_i in customer order, for the Welch plot
                if k > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                next_departure = t + service_time_gen(rng, service_mean)
            else:
                next_departure = float('inf')

    T = t - t_warm  # length of the observation period after warm-up
    Wq = sum_wait / n_wait
    P_over5 = n_over5 / n_wait
    Lq = area_q / T
    U = area_busy / T
    return np.array([Wq, P_over5, Lq, U])


def ci(Y, alpha=0.05):
    """Columnwise point estimate, sample std dev, and CI half-width for an (n x 4) array of replication outputs."""
    n = Y.shape[0]
    ybar = Y.mean(axis=0)
    s = Y.std(axis=0, ddof=1)
    h = stats.t.ppf(1 - alpha/2, n - 1) * s / np.sqrt(n)
    return ybar, s, h


def exact(rho, mu):
    """Steady-state M/M/1 values (times in minutes) for [Wq, P(Wq > 5), Lq, U]."""
    lam = rho * mu
    Wq = rho / (mu - lam) * 60
    P_over5 = rho * np.exp(-(mu - lam) * 5 / 60)
    Lq = rho**2 / (1 - rho)
    U = rho
    return np.array([Wq, P_over5, Lq, U])


def welch_moving_average(Wbar, w):
    """W_i(w) for i = 1, ..., t - w, from the averaged process Wbar_1, ..., Wbar_t."""
    t = len(Wbar)
    c = np.concatenate([[0], np.cumsum(Wbar)])  # c[m] = Wbar_1 + ... + Wbar_m
    i = np.arange(1, w + 1)
    early = c[2*i - 1] / (2*i - 1)  # i = 1, ..., w: average of Wbar_1 .. Wbar_{2i-1}
    i = np.arange(w + 1, t - w + 1)
    centered = (c[i + w] - c[i - w - 1]) / (2*w + 1)  # i = w+1, ..., t-w: average of Wbar_{i-w} .. Wbar_{i+w}
    return np.concatenate([early, centered])


def welch_plot(r, mu, n, t, windows, seed):
    """Pilot run for Welch's method: n replications of the first t customers, plotted for several windows w."""
    lam = r * mu
    streams = make_streams(seed, n)
    W = np.empty((n, t))
    for j in range(n):
        waits = []
        mm1_sim(streams[j], lam / 60, 60 / mu, 0, t, waits_out=waits)
        W[j] = waits[:t]
    Wbar = W.mean(axis=0)  # averaged process Wbar_i over the n replications

    fig, axes = plt.subplots(len(windows), 1, figsize=(8, 2.2 * len(windows)), sharex=True)
    for ax, w in zip(axes, windows):
        ma = welch_moving_average(Wbar, w)
        ax.plot(np.arange(1, t - w + 1), ma, color="#2a78d6", lw=1.2)
        ax.axhline(exact(r, mu)[0], color="#52514e", ls="--", lw=1, label="exact steady-state $W_q$")
        ax.set_title(f"w = {w}", loc="left", fontsize=10)
        ax.set_ylabel(r"$\bar{W}_i(w)$ (min)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("customer i")
    fig.suptitle(rf"Welch plot, $\rho$ = {r}  (n = {n} pilot replications, t = {t} customers)")
    fig.tight_layout()
    fname = f"welch_rho{int(100 * r):03d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


MEASURES = [
    ("Wq  Avg wait in queue", "min"),
    ("P(Wq > 5)", ""),
    ("Lq  Avg number in queue", "customers"),
    ("U   Server utilization", ""),
]


if __name__ == "__main__":
    seed = 451
    n = 50
    mu = 12  # customers per hour -> mean service time 60/mu = 5 min
    rho = [0.5, 0.75, 0.95]

    # pilot runs for Welch's method (separate seeds from the production runs)
    pilot_t = {0.5: 3000, 0.75: 10000, 0.95: 60000}
    pilot_w = {0.5: [5, 25, 100, 250], 0.75: [10, 50, 200, 500], 0.95: [50, 200, 1000, 3000]}
    for r in rho:
        print("saved", welch_plot(r, mu, n, pilot_t[r], pilot_w[r], seed=1000 + int(100 * r)))
    print()

    # run length and warm-up, in service completions
    # warm-up k read off the Welch plots (w = 100, 200, 1000): the curves level off near
    # i = 200, 800, 3000; k rounds up with a margin. run length is at least 20k.
    warmup = {0.5: 500, 0.75: 1500, 0.95: 5000}
    run_length = {0.5: 20000, 0.75: 50000, 0.95: 200000}

    for r in rho:
        lam = r * mu
        streams = make_streams(seed + int(100 * r), n)  # distinct seed for every replication
        Y = np.array([mm1_sim(streams[j], lam / 60, 60 / mu, warmup[r], run_length[r]) for j in range(n)])
        ybar, s, h = ci(Y)

        print(f"rho = {r}  (lambda = {lam:g}/hr, mu = {mu}/hr), n = {n} replications, "
              f"warm-up = {warmup[r]}, run length = {run_length[r]} completions")
        print(f"{'Measure':<26}{'Estimate':>10}{'95% CI':>24}{'Exact':>10}  Units")
        print("-" * 82)
        for (name, unit), m, hw, ex in zip(MEASURES, ybar, h, exact(r, mu)):
            print(f"{name:<26}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{ex:>10.4f}  {unit}")
        print()
