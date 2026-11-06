import numpy as np
from math import factorial
from scipy import stats

"""
HW 4, common random numbers

Same two systems as hw4.py, but replication j of System 1 and replication j of
System 2 are driven by the same random numbers.

To keep the random numbers synchronized, each replication gets two streams:
  - an arrival stream: the k-th draw is customer k's inter-arrival time
  - a service stream: the k-th draw is customer k's service time
FCFS means customers start service in arrival order in both systems, so customer k
gets the same uniform U_k in both: service = -0.9 ln(U_k) in System 1 and
-1.8 ln(U_k) in System 2. With a single stream shared by arrivals and services, the
draws would be consumed in a different order in each system (the event order differs)
and the correlation would be mostly lost.
"""

def make_crn_streams(seed, n):
    """(arrival, service) seed pair for each of n replications; both systems reuse the same pair."""
    return [ss.spawn(2) for ss in np.random.SeedSequence(seed).spawn(n)]

def exp_gen(mean, rng):
    """Inverse transform: X = -mean * ln(U), U ~ Uniform(0, 1)."""
    u = 1 - rng.random()  # rng.random() is in [0, 1), so u is in (0, 1] and ln(u) is finite
    return -mean * np.log(u)


def multi_server_sim(arr_rng, svc_rng, c, inter_arrival_mean, service_mean, warmup, run_length):
    """c identical servers fed by one FCFS line (c = 1 for System 1, c = 2 for System 2)."""
    t = 0
    arrival_counter = 0
    departure_counter = 0
    start_counter = 0  # customers who have started service (FCFS -> in arrival order)
    in_system = 0
    busy = 0  # number of busy servers

    t0 = exp_gen(inter_arrival_mean, arr_rng)
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
            next_arrival = t + exp_gen(inter_arrival_mean, arr_rng)
            if busy < c:
                free = next_departure.index(float('inf'))  # an idle server
                start_counter += 1
                wait = t - arrivals.pop(start_counter)
                if start_counter > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                busy += 1
                next_departure[free] = t + exp_gen(service_mean, svc_rng)  # customer start_counter's service time

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
                if start_counter > warmup:
                    sum_wait += wait
                    n_over5 += wait > 5
                    n_wait += 1
                busy += 1
                next_departure[j] = t + exp_gen(service_mean, svc_rng)  # customer start_counter's service time

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

    # run length and warm-up, in service completions (same as hw4.py)
    warmup = 3000
    run_length = 10000

    seeds = make_crn_streams(seed, n)  # one seed pair per replication, shared by both systems
    Ys = []
    for title, c, service_mean in systems:
        # fresh generators from the same seeds -> both systems see identical random numbers
        Y = np.array([multi_server_sim(np.random.default_rng(a), np.random.default_rng(b),
                                       c, 1 / lam, service_mean, warmup, run_length)
                      for a, b in seeds])
        Ys.append(Y)
        ybar, s, h = ci(Y)

        print(f"{title}  (c = {c}, mean service = {service_mean} min), n = {n} replications, "
              f"warm-up = {warmup}, run length = {run_length} completions")
        print(f"{'Measure':<26}{'Estimate':>10}{'95% CI':>24}{'Exact':>10}  Units")
        print("-" * 82)
        for (name, unit), m, hw, ex in zip(MEASURES, ybar, h, exact(c, lam, service_mean)):
            print(f"{name:<26}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{ex:>10.4f}  {unit}")
        print()

    # paired-t CI for System 1 minus System 2: under CRN, Y1_j and Y2_j are positively
    # correlated, but the pairs Z_j = Y1_j - Y2_j are still iid across replications
    Z = Ys[0] - Ys[1]
    zbar, sz, hz = ci(Z)
    ex_diff = exact(1, lam, 0.9) - exact(2, lam, 1.8)

    print(f"Difference System 1 - System 2 (paired-t, CRN), n = {n} replications")
    print(f"{'Measure':<26}{'Estimate':>10}{'95% CI':>24}{'Exact':>10}  Units")
    print("-" * 82)
    for (name, unit), m, hw, ex in zip(MEASURES, zbar, hz, ex_diff):
        print(f"{name:<26}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{ex:>10.4f}  {unit}")
    print()

    # variance reduction: Var(Z) = Var(Y1) + Var(Y2) - 2 Cov(Y1, Y2); with independent
    # streams the covariance term is 0
    s1 = Ys[0].std(axis=0, ddof=1)
    s2 = Ys[1].std(axis=0, ddof=1)
    print("Effect of CRN on the difference")
    print(f"{'Measure':<26}{'Corr(Y1, Y2)':>14}{'Var(Z)':>14}{'Var(Y1)+Var(Y2)':>18}{'Ratio':>8}")
    print("-" * 80)
    for (name, _), j in zip(MEASURES, range(3)):
        r = np.corrcoef(Ys[0][:, j], Ys[1][:, j])[0, 1]
        indep = s1[j]**2 + s2[j]**2
        print(f"{name:<26}{r:>14.4f}{sz[j]**2:>14.3e}{indep:>18.3e}{sz[j]**2 / indep:>8.3f}")
