"""Proposed Algorithms I and II from this paper"""

import numpy as np


def qb_decomposition(A, epsilon=1e-12):
    m, n = A.shape

    first_col_idx = 0
    while first_col_idx < n:
        first_col_norm = np.linalg.norm(A[:, first_col_idx])
        if first_col_norm >= epsilon:
            break
        first_col_idx += 1

    if first_col_idx >= n:
        return np.zeros((m, 0)), np.zeros((0, n))

    Q = np.zeros((n, m), dtype=A.dtype)
    B = np.zeros((n, n), dtype=A.dtype)

    Q[0] = A[:, first_col_idx] / first_col_norm
    B[0] = Q[0].conj() @ A
    k = 1

    j = 0
    max_iters = 2 * n
    iter_count = 0

    while k + j + 1 <= n and iter_count < max_iters:
        iter_count += 1
        next_col_idx = k + j
        if next_col_idx >= n:
            break

        q_bar = A[:, next_col_idx].copy()

        proj = Q[:k].conj() @ q_bar
        q_bar = q_bar - proj @ Q[:k]
        proj = Q[:k].conj() @ q_bar
        q_bar = q_bar - proj @ Q[:k]

        q_bar_norm = np.linalg.norm(q_bar)

        if q_bar_norm > epsilon * np.linalg.norm(A[:, next_col_idx]):
            Q[k] = q_bar / q_bar_norm
            B[k, k:] = Q[k].conj() @ A[:, k:]
            k += 1
        else:
            j += 1

    return Q[:k].T, B[:k]


def pseudo_inverse_matrix(A, epsilon=1e-12):
    m, n = A.shape
    B = np.zeros((m, n), dtype=A.dtype)
    Avec = np.zeros((m, m), dtype=A.dtype)
    r = 0
    j = 0
    while j < m:
        col_j = A[j, :]
        if r > 0:
            proj = B[:r].conj() @ col_j
            b_bar = col_j - proj @ B[:r]
            proj2 = B[:r].conj() @ b_bar
            b_bar = b_bar - proj2 @ B[:r]
        else:
            proj = None
            b_bar = col_j.copy()

        b_bar_norm = np.linalg.norm(b_bar)

        if b_bar_norm > epsilon * np.linalg.norm(col_j):
            B[r] = b_bar / b_bar_norm

            a_j = np.zeros(m, dtype=A.dtype)
            a_j[j] = 1.0
            if r > 0:
                a_j = a_j - proj @ Avec[:r]
            Avec[r] = a_j / b_bar_norm
            r += 1

        j += 1

    A_plus = B[:r].conj().T @ Avec[:r]
    return A_plus


def qb_pinv(A, epsilon=1e-6):
    m, n = A.shape
    transposed = n > m
    A_work = A.T if transposed else A
    Q, B = qb_decomposition(A_work, epsilon=epsilon)
    B_inv = pseudo_inverse_matrix(B, epsilon=epsilon)
    result = B_inv @ Q.conj().T
    return result.T if transposed else result


def fisher_yates(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def direct_pseudoinverse_random(A, epsilon=1e-8, seed=None):
    m, n = A.shape
    if n > m:
        A = np.copy(A).T
        m, n = A.shape
        transpose = True
    else:
        transpose = False

    col_order = fisher_yates(n, seed=seed)
    row_order = fisher_yates(m, seed=seed + 1 if seed is not None else None)

    Q = np.zeros((n, m), dtype=A.dtype)
    rank = 0
    for col_idx in col_order:
        col_j = A[:, col_idx]
        q_bar = col_j.copy()
        if rank > 0:
            proj = Q[:rank].conj() @ col_j
            q_bar = q_bar - proj @ Q[:rank]
            proj = Q[:rank].conj() @ q_bar
            q_bar = q_bar - proj @ Q[:rank]
        q_bar_norm = np.linalg.norm(q_bar)
        if q_bar_norm > epsilon * np.linalg.norm(col_j):
            Q[rank] = q_bar / q_bar_norm
            rank += 1

    r = rank
    Qr = Q[:r]

    P = np.zeros((r, n), dtype=A.dtype)
    R = np.zeros((r, m), dtype=A.dtype)
    k = 0
    for row_idx in row_order:
        if k >= r:
            break
        row_j = A[row_idx, :]
        if k > 0:
            proj_row = P[:k].conj() @ row_j
            p_bar = row_j - proj_row @ P[:k]
            proj_pbar = P[:k].conj() @ p_bar
            p_bar = p_bar - proj_pbar @ P[:k]
        else:
            proj_row = None
            p_bar = row_j.copy()
        p_bar_norm = np.linalg.norm(p_bar)
        if p_bar_norm > epsilon * np.linalg.norm(row_j):
            e_j = np.zeros(m, dtype=A.dtype)
            e_j[row_idx] = 1.0
            r_bar = e_j if k == 0 else e_j - proj_row @ R[:k]
            R[k] = r_bar / p_bar_norm
            P[k] = p_bar / p_bar_norm
            k += 1

    if k == 0:
        A_plus = np.zeros((n, m), dtype=A.dtype)
        return A_plus.T if transpose else A_plus

    Pk = P[:k]
    Rk = R[:k]
    A_plus = Pk.conj().T @ ((Rk @ Qr.T) @ Qr.conj())
    return A_plus.T if transpose else A_plus
