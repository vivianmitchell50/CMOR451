import numpy as np
from scipy import stats

"""
HW 3, Q1

Single Server Queue

9am-5pm (t=0 -> t=480)
inter-arrival iid variables (lambda = 10, ~Exp(10/60))
service times iid variables (~Exp(5))

events:
A - arrival
B - service start
D - departure

system state S = ()
"""

rng = np.random.default_rng(451)

def inter_arrival_gen(rng, rate):
    return rng.exponential(1 / rate)

def service_time_gen(rng, mean):
    return rng.exponential(mean)


def single_server_sim(rng, inter_arrival_rate, service_mean):
    t = 0
    arrival_counter = 0
    departure_counter = 0
    in_system = 0

    t0 = inter_arrival_gen(rng, inter_arrival_rate)
    next_arrival = t0
    next_departure = float('inf')
    t_final = float('inf')

    arrivals = {}
    service_starts = {}
    departures = {}
    area_q = 0  # area under the queue-length curve Q(t)

    eod = 480

    while True:
        # case 1: the next event is an arrival, before closing
        if next_arrival < next_departure and next_arrival < eod:
            area_q += (next_arrival - t) * max(in_system - 1, 0)
            t = next_arrival
            arrival_counter += 1
            in_system += 1
            next_arrival = t + inter_arrival_gen(rng, inter_arrival_rate)
            if in_system == 1:
                service_starts[departure_counter + 1] = t
                next_departure = t + service_time_gen(rng, service_mean)
            arrivals[arrival_counter] = t

        # case 2: the next event is a departure, before closing
        elif next_departure < next_arrival and next_departure <= eod:
            area_q += (next_departure - t) * max(in_system - 1, 0)
            t = next_departure
            in_system -= 1
            departure_counter += 1
            if in_system > 0:
                service_starts[departure_counter + 1] = t
                next_departure = t + service_time_gen(rng, service_mean)
            else:
                next_departure = float('inf')
            departures[departure_counter] = t

        # case 3: the next event is after closing, customers in system
        elif min(next_arrival, next_departure) > eod and in_system > 0:
            area_q += (next_departure - t) * max(in_system - 1, 0)
            t = next_departure
            in_system -= 1
            departure_counter += 1
            if in_system > 0:
                service_starts[departure_counter + 1] = t
                next_departure = t + service_time_gen(rng, service_mean)
            else:
                next_departure = float('inf')
            departures[departure_counter] = t

        # case 4: past closing, system empty -> stop
        elif min(next_arrival, next_departure) > eod and in_system == 0:
            t_final = t  # time of final departure (overtime = max(t_final - eod, 0))
            break

    return arrivals, service_starts, departures, t_final, area_q


MEASURES = [
    ("Y1  Avg waiting time in queue", "min"),
    ("Y2  Proportion waiting > 5 min", ""),
    ("Y3  Avg number in queue", "customers"),
    ("Y4  Server utilization", ""),
    ("Y5  Customers served", "customers"),
    ("Y6  Final departure", "min"),
]


def day_measures(rng, inter_arrival_rate=10/60, service_mean=5, eod=480):
    """Simulate one day and return [Y1, ..., Y6]."""
    arrivals, service_starts, departures, t_final, area_q = single_server_sim(rng, inter_arrival_rate, service_mean)
    n = len(departures)
    A = np.array([arrivals[i] for i in range(1, n+1)])
    S = np.array([service_starts[i] for i in range(1, n+1)])
    D = np.array([departures[i] for i in range(1, n+1)])
    waits = S - A
    T = max(eod, t_final)  # length of the working day

    y1 = waits.mean()            # average customer waiting time in the queue
    y2 = np.mean(waits > 5)      # proportion waiting > 5 min
    y3 = area_q / T              # time-average number in queue
    y4 = (D - S).sum() / T       # server utilization (busy time / T)
    y5 = n                       # number of customers served
    y6 = t_final                 # final departure
    return np.array([y1, y2, y3, y4, y5, y6])


def ci(Y, alpha=0.05):
    """Columnwise point estimate, sample std dev, and CI half-width for an (n x 6) array of replication outputs."""
    n = Y.shape[0]
    ybar = Y.mean(axis=0)
    s = Y.std(axis=0, ddof=1)
    h = stats.t.ppf(1 - alpha/2, n - 1) * s / np.sqrt(n)
    return ybar, s, h


def print_table(title, Y):
    ybar, s, h = ci(Y)
    print(title)
    print(f"{'Measure':<32}{'Estimate':>10}{'95% CI':>24}{'Rel HW':>9}  Units")
    print("-" * 86)
    for (name, unit), m, hw in zip(MEASURES, ybar, h):
        print(f"{name:<32}{m:>10.4f}   [{m - hw:>8.4f}, {m + hw:>8.4f}]{hw / abs(m):>9.2%}  {unit}")
    print()


if __name__ == "__main__":
    # lambda = 10 per hour -> 10/60 per minute; mean service time 5 min

    # (a) one simulated day
    y = day_measures(rng)
    hh, mm = divmod(9 * 60 + y[5], 60)
    clock = f"{int(hh - 1) % 12 + 1}:{int(mm):02d} {'am' if hh < 12 else 'pm'}"

    print("(a) One simulated day")
    print("-" * 48)
    print(f"{'Y1  Avg waiting time in queue':<32}{y[0]:>10.2f} min")
    print(f"{'Y2  Proportion waiting > 5 min':<32}{y[1]:>10.1%}")
    print(f"{'Y3  Avg number in queue':<32}{y[2]:>10.2f}")
    print(f"{'Y4  Server utilization':<32}{y[3]:>10.1%}")
    print(f"{'Y5  Customers served':<32}{int(y[4]):>10d}")
    print(f"{'Y6  Final departure':<32}{y[5]:>10.2f} min  ({clock})")
    print()

    # (d) pilot run to choose the number of replications
    target = 0.03
    n0 = 100
    Y_pilot = np.array([day_measures(rng) for _ in range(n0)])
    ybar, s, h = ci(Y_pilot)
    z = stats.norm.ppf(0.975)
    n_req = np.ceil((z * s / (target * np.abs(ybar))) ** 2).astype(int)

    print(f"(d) Pilot run with n0 = {n0} replications")
    print(f"{'Measure':<32}{'Rel HW':>9}{'n needed':>10}")
    print("-" * 51)
    for (name, _), m, hw, nr in zip(MEASURES, ybar, h, n_req):
        print(f"{name:<32}{hw / abs(m):>9.2%}{nr:>10d}")
    n = int(n_req.max())
    print(f"-> n = {n} (largest requirement over all six measures)")
    print()

    # (e) fresh, independent replications; add more if any measure still misses the target
    Y = np.array([day_measures(rng) for _ in range(n)])
    ybar, s, h = ci(Y)
    while np.any(h / np.abs(ybar) > target):
        extra = int(0.1 * n)
        Y = np.vstack([Y, [day_measures(rng) for _ in range(extra)]])
        n += extra
        ybar, s, h = ci(Y)

    print_table(f"(e) Results from n = {n} replications", Y)
