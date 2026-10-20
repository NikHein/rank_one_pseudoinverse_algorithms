"""Random m x n matrix of exact rank k with a chosen singular-value regime."""
import numpy as np


def generate_low_rank_matrix(
    m, n, k, singular_value_scale="normal", seed=None, singular_para=1
):
    if k > min(m, n):
        raise ValueError(f"Rank k={k} cannot exceed min(m,n)={min(m, n)}")
    if seed is not None:
        np.random.seed(seed)

    U, _ = np.linalg.qr(np.random.randn(m, k))
    V, _ = np.linalg.qr(np.random.randn(n, k))

    if singular_value_scale == "small":
        singular_values = np.random.uniform(1e-3, 1e-2, size=k)
    elif singular_value_scale == "big":
        singular_values = np.random.uniform(1e2, 1e3, size=k)
    elif singular_value_scale == "very small":
        singular_values = np.random.uniform(1e-5, 1e-4, size=k)
    elif singular_value_scale == "custom":
        singular_values = np.random.uniform(
            10 ** (-singular_para), 10 ** (-(singular_para - 1)), size=k
        )
    elif singular_value_scale == "big small":
        half_k = k // 2
        big_values = np.random.uniform(1e2, 1e3, size=half_k)
        small_values = np.random.uniform(1e-3, 1e-2, size=k - half_k)
        singular_values = np.concatenate((big_values, small_values))
    else:  # "normal"
        singular_values = np.random.uniform(0.5, 1.5, size=k)

    return U @ np.diag(singular_values) @ V.T
