from math import sqrt
import numpy as np

def aggregate_size_ci(sizes, lowers, uppers):
    # 假设：误差独立，各区间为近似正态的对称 95% CI
    total = sum(sizes)
    ses = [(u - l) / (2 * 1.96) for l, u in zip(lowers, uppers)]
    total_se = sqrt(sum(se**2 for se in ses))
    return total, total - 1.96 * total_se, total + 1.96 * total_se


def gini(values):
    x = np.asarray(values, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("Input must be a non-empty 1D array.")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Handle missing values explicitly; values must be non-negative.")
    if x.max() == 0:
        return np.nan

    # Scaling improves numerical stability without changing Gini.
    x = np.sort(x / x.max())
    n = x.size
    ranks = np.arange(1, n + 1, dtype=float)

    g = 2 * np.dot(ranks, x) / (n * x.sum()) - (n + 1) / n
    return float(max(0.0, g))

if __name__ == "__main__":
    result = aggregate_size_ci(
        sizes=[100, 200, 300],
        lowers=[90, 180, 270],
        uppers=[110, 220, 330],
    )
    print(result)  # (600, 562.58..., 637.41...)
