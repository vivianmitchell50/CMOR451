import numpy as np
from scipy import stats

# A ~ Triangular(2, 4, 7)
# B ~ Uniform(3, 6)
# C ~ Exponential(15)
# D ~ Triangular(1, 3, 5)
# E ~ Uniform(2, 5)

# create one independent generator per activity. SeedSequence.spawn gives
# statistically independent streams and seed makes results predictable.
def make_streams(seed):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]

# simulate the project.
# inputs: n (number of iterations), streams (list of Generators)
# outputs: a, b, c, d, e, T, boolean array for c finishes after d
def simulate(n, streams):
    ra, rb, rc, rd, re = streams
    a = ra.triangular(2, 4, 7, n)
    b = rb.uniform(3, 6, n)
    c = rc.exponential(15, n)
    d = rd.triangular(1, 3, 5, n)
    e = re.uniform(2, 5, n)

    # C starts when A finishes. D starts when both A and B finish.
    c_finish = a + c
    d_finish = np.maximum(a, b) + d

    # E starts when both C and D finish. project is done when E finishes.
    T = np.maximum(c_finish, d_finish) + e

    return a, b, c, d, e, T, c_finish > d_finish

# (b) three replications: durations and completion time
print("(b) \n")
n_b = 3
streams_b = make_streams(1)
a, b, c, d, e, T, _ = simulate(n_b, streams_b)
for i in range(n_b):
    print(f"Rep {i+1}: A={a[i]:.3f} B={b[i]:.3f} C={c[i]:.3f} D={d[i]:.3f} E={e[i]:.3f}  T={T[i]:.3f}")

# (c) 500,000 replications: simulated vs theoretical mean of each duration
print("\n(c)")
n = 500000
streams = make_streams(2)
a, b, c, d, e, T, c_after_d = simulate(n, streams)

# triangular mean = (min + mode + max) / 3, uniform mean = (min + max) / 2
theory = [13/3, 4.5, 15, 3, 3.5]

print("\n[Name] [Simulated] [Theoretical Mean]")
for name, sim, th in zip("ABCDE", [x.mean() for x in (a, b, c, d, e)], theory):
    print(name, sim, th)

# (a) (i) expected project completion time
print("\n(a)")
est_mean = T.mean()
print(f"\nEstimated project completion time: {est_mean}")

# (a) (ii) P(T > 22) is the mean of the 0/1 indicator
target = 22
print(f"Estimated probability of missing the 22-week deadline: {(T > target).mean()}")

# (a) (iii) P(C finishes after D) is the mean of the 0/1 indicator
print(f"Estimated probability C finishes after D: {c_after_d.mean()}")

# (d) point estimates and 95% CIs for increasing n.
# all three outputs are sample means (the probabilities are means of 0/1 indicators),
# so one helper works: mean +/- t_{n-1, 0.975} * s / sqrt(n)
def mean_ci(x, conf=0.95):
    m = len(x)
    mean = x.mean()
    half = stats.t.ppf(0.5 + conf / 2, m - 1) * x.std(ddof=1) / np.sqrt(m)
    return mean, mean - half, mean + half

def fmt(r):
    return f"{r[0]:.4f} ({r[1]:.4f}, {r[2]:.4f})"

n_d = [10, 50, 100, 1000, 10000, 100000, 500000]

print("\n(d)")
print(f"{'n':>8} | {'E[T]':^26} | {'P(T > 22)':^26} | {'P(C after D)':^26}")
for m in n_d:
    # fresh independent streams for each n
    *_, T_m, c_after_d_m = simulate(m, make_streams(100 + m))

    ci_T = mean_ci(T_m)
    ci_miss = mean_ci((T_m > target).astype(float))
    ci_cd = mean_ci(c_after_d_m.astype(float))

    print(f"{m:>8} | {fmt(ci_T):^26} | {fmt(ci_miss):^26} | {fmt(ci_cd):^26}")
