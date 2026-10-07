from math import sqrt


def aggregate_size_ci(sizes, lowers, uppers):
    # 假设：误差独立，各区间为近似正态的对称 95% CI
    total = sum(sizes)
    ses = [(u - l) / (2 * 1.96) for l, u in zip(lowers, uppers)]
    total_se = sqrt(sum(se**2 for se in ses))
    return total, total - 1.96 * total_se, total + 1.96 * total_se


if __name__ == "__main__":
    result = aggregate_size_ci(
        sizes=[100, 200, 300],
        lowers=[90, 180, 270],
        uppers=[110, 220, 330],
    )
    print(result)  # (600, 562.58..., 637.41...)
