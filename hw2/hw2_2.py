import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
from scipy.stats import norm

# D_A ~ N(120, 25^2)
# D_B ~ N(100, 20^2)
# Corr(D_A, D_B) = 0.4

def make_streams(seed):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]

def bivariate_normal(n, mu_x, sd_x, mu_y, sd_y, rho, streams):
    r1, r2 = streams
    z = np.vstack([r1.standard_normal(n), r2.standard_normal(n)])

    cov = np.array([[sd_x**2, rho * sd_x * sd_y], 
                      [rho * sd_x * sd_y, sd_y**2]])

    L = np.linalg.cholesky(cov)

    samples = L @ z + np.array([[mu_x], [mu_y]])

    return samples.T

n = 4749
X = bivariate_normal(n, 120, 25, 100, 20, 0.4, make_streams(123))
print(f"n={n}\nmean: {X.mean(0)}, std: {X.std(0, ddof=1)}, correlation: {np.corrcoef(X.T)[0, 1]}")

# (c)
print(f"P(D_A + D_B > 180) = {(X.sum(axis=1) > 180).mean()}")

# LP
model = gp.Model("bivariate_optimization")
model.Params.OutputFlag = 0

xa = model.addVar(lb=0, name="a")
xb = model.addVar(lb=0, name="b")
model.addConstr(xa + xb <= 180)

n0 = 4749
X0 = bivariate_normal(n0, 120, 25, 100, 20, 0.4, make_streams(456))
dA = np.maximum(X0[:, 0], 0)
dB = np.maximum(X0[:, 1], 0)

profit = np.empty(n0)

for i in range(n0):
    xa.UB, xb.UB = dA[i], dB[i]
    model.setObjective(55 * xa + 80 * xb, GRB.MAXIMIZE)
    model.optimize()
    profit[i] = model.ObjVal - 15*dA[i] - 25*dB[i]

ind = (X0.sum(axis=1) > 180).astype(float)

z= 1.96
ybar, s = profit.mean(), profit.std(ddof=1)
p, = [ind.mean()]
n_profit = math.ceil((z*s/50)**2)
n_prob   = math.ceil((z/0.01)**2 * p*(1-p))
n_req    = max(n_profit, n_prob)

print(f"n0={n0}")
print(f"ybar = {ybar}, s = {s}, p_hat = {p}")
print(f"n_profit = {n_profit}, n_prob = {n_prob}, n_req = {n_req}")

# check Gurobi profits against closed form (B first, then A)
xb_cf = np.minimum(dB, 180)
xa_cf = np.minimum(dA, 180 - xb_cf)
profit_cf = 40*xa_cf + 55*xb_cf - 15*(dA - xa_cf) - 25*(dB - xb_cf)
print("Gurobi matches closed form:", np.allclose(profit, profit_cf))

# (e) theoretical numbers (not simulated)
mu_S = 120 + 100
var_S = 25**2 + 20**2 + 2 * 0.4 * 25 * 20
z_thr = (180 - mu_S) / np.sqrt(var_S)
p_theory = norm.sf(z_thr)
print(f"S ~ N({mu_S}, {var_S}), z = {z_thr:.4f}, P = {p_theory:.4f}")

half = 1.96 * np.sqrt(p * (1 - p) / n0)
lo, hi = p - half, p + half
print(f"p_hat = {p:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}], half-width = {half:.4f}")
print("theory inside CI:", lo <= p_theory <= hi)
