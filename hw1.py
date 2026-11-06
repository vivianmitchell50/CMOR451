import numpy

# X ~ Exponential(mean 1), conditioned on X < a

a = 0.05
c = 1.0 - numpy.exp(-a)          # P(X < a)

# (a) Inverse-transform sampler for the conditional distribution.
#     F(x | X < a) = (1 - e^{-x}) / (1 - e^{-a}),  0 < x < a
#     Setting U = F(x | X < a) and inverting:  x = -ln(1 - U*(1 - e^{-a}))
def sample_conditional(n, rng):
    u = rng.random(n)
    return -numpy.log(1.0 - u * c)

# (b) Simulate 1000 variables and estimate E[X | X < a]
rng = numpy.random.default_rng(451)
n = 1000
x = sample_conditional(n, rng)
est = x.mean()
se = x.std(ddof=1) / numpy.sqrt(n)

# (c) Exact value: E[X | X < a] = 1 - a*e^{-a} / (1 - e^{-a})
exact = 1.0 - a * numpy.exp(-a) / c

print(f"max sample                 = {x.max():.6f}  (must be < {a})")
print(f"(b) simulation estimate    = {est:.6f}")
print(f"    standard error         = {se:.6f}")
print(f"    95% CI                 = ({est - 1.96*se:.6f}, {est + 1.96*se:.6f})")
print(f"(c) exact value            = {exact:.6f}")
print(f"    difference (b) - (c)   = {est - exact:+.6f}  ({(est-exact)/se:+.2f} SEs)")
