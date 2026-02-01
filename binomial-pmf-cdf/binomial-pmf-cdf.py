import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")

    if k < 0:
        return 0.0, 0.0
    if k > n:
        return 0.0, 1.0

    # -------- edge cases --------
    if p == 0:
        pmf = 1.0 if k == 0 else 0.0
        cdf = 1.0
        return pmf, cdf

    if p == 1:
        pmf = 1.0 if k == n else 0.0
        cdf = 1.0 if k >= n else 0.0
        return pmf, cdf

    # -------- PMF --------
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # -------- CDF --------
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return pmf, cdf
    