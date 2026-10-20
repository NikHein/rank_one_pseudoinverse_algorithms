"""Absolute Penrose residual of every method on a set of classic ill-conditioned NIST/MatrixMarket test matrices."""
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import scipy.linalg as la

from algorithms import qb_pinv, direct_pseudoinverse_random
from qrginv import qrginv

EPS_ALGO = 1e-8
RCOND_REF = 1e-10
NIST_SEED = 42
METHOD_KEYS = ["direct", "qb", "qrginv", "numpy"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def numpy_pinv(A):
    return np.linalg.pinv(A, rcond=RCOND_REF)


def penrose_error_abs(A, X):
    # mean absolute Frobenius residual of the four Moore-Penrose conditions
    AX = A @ X
    XA = X @ A
    r1 = np.linalg.norm(A @ X @ A - A)
    r2 = np.linalg.norm(X @ A @ X - X)
    r3 = np.linalg.norm(AX - AX.conj().T)
    r4 = np.linalg.norm(XA - XA.conj().T)
    return (r1 + r2 + r3 + r4) / 4


# ── test-matrix generators ───────────────────────────────────────────────────
def nist_hilbert(n):
    return la.hilbert(n)


def nist_clement(n, kind=1):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = i + 1
        A[i + 1, i] = n - i - 1
    if kind == 1:
        A = (A + A.T) / 2
    return A


def nist_frank(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                A[i, j] = n - max(i, j)
    return A


def nist_jordan(n, lambda_val=1):
    A = lambda_val * np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = 1
    return A


def nist_kahan(n, theta=1.2, pert=0):
    s, c = np.sin(theta), np.cos(theta)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = s**i
        for j in range(i + 1, n):
            A[i, j] = -(s**i) * (c ** (j - i))
    if pert != 0:
        for i in range(n):
            A[i, i] += pert * 10 ** (-i)
    return A


def nist_lauchli(n, mu=1e-6):
    A = np.ones((n + 1, n))
    A[0, :] = 1
    for i in range(1, n + 1):
        A[i, :] = 0
        A[i, i - 1] = mu
    return A


def nist_lotkin(n):
    A = la.hilbert(n)
    A[0, :] = 1
    return A


def nist_wilkinson(n):
    if n % 2 == 0:
        n += 1
    A = np.zeros((n, n))
    k = (n - 1) // 2
    for i in range(n):
        A[i, i] = abs(i - k)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def nist_dingdong(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 0.5 / (n - i - j - 0.5)
    return A


def nist_condex(n, k=4, theta=100):
    if k == 1:
        A = np.eye(n)
        A[0, :] = 1
        A[:, 0] = 1
        A[0, 0] = 1
    elif k == 2:
        A = np.eye(n)
        A[-1, :] = 1
    elif k == 3:
        A = np.triu(np.ones((n, n)))
        A[0, 0] = 1 + theta
    else:
        A = np.eye(n)
        A[0, -1] = -1
        A[-1, 0] = -1
        A[0, 0] = 1 + 1 / theta
        A[-1, -1] = 1 + theta
    return A


def nist_dorr(n, theta=0.1):
    h = 1.0 / (n + 1)
    m = int(np.floor((n + 1) / 2))
    A = np.zeros((n, n))
    for i in range(n):
        if i < m - 1:
            c = -(1 + theta * h / 2) / (h * h)
            a = -(-1 + theta * h / 2) / (h * h)
        elif i == m - 1:
            c = a = -1 / (h * h)
        else:
            c = -(-1 + theta * h / 2) / (h * h)
            a = -(1 + theta * h / 2) / (h * h)
        b = (2 + theta * theta * h * h) / (h * h)
        A[i, i] = b
        if i > 0:
            A[i, i - 1] = a
        if i < n - 1:
            A[i, i + 1] = c
    return A


def nist_gear(n):
    A = np.eye(n)
    A[0, n - 1] = 1
    return A


def build_bench():
    return {
        "NIST_hilbert_12": nist_hilbert(12),
        "NIST_clement_25_nonsym": nist_clement(25, kind=0),
        "NIST_frank_20": nist_frank(20),
        "NIST_jordan_15_eig1": nist_jordan(15, lambda_val=1),
        "NIST_kahan_25_theta1.5": nist_kahan(25, theta=1.5),
        "NIST_lauchli_30_mu1e-8": nist_lauchli(30, mu=1e-8),
        "NIST_lotkin_15": nist_lotkin(15),
        "NIST_wilkinson_31": nist_wilkinson(31),
        "NIST_dingdong_30": nist_dingdong(30),
        "NIST_condex_25_k4": nist_condex(25, k=4, theta=100),
        "NIST_dorr_30": nist_dorr(30, theta=0.1),
        "NIST_dorr_40": nist_dorr(40, theta=0.01),
        "NIST_gear_30": nist_gear(30),
    }


def main():
    np.random.seed(42)

    # for each matrix: compute every pseudoinverse and measure its Penrose residual
    rows = []
    for name, A in build_bench().items():
        row = {"Matrix": name, "Shape": f"{A.shape[0]}x{A.shape[1]}"}
        try:
            row["Rank"] = int(np.linalg.matrix_rank(A))
        except Exception:
            row["Rank"] = "?"
        try:
            row["Cond"] = f"{np.linalg.cond(A):.2e}"
        except Exception:
            row["Cond"] = "inf"

        pinvs = {}
        methods = [
            ("direct", lambda B: direct_pseudoinverse_random(B, epsilon=EPS_ALGO, seed=NIST_SEED)),
            ("qb", lambda B: qb_pinv(B, epsilon=EPS_ALGO)),
            ("qrginv", lambda A: qrginv(A, tol=RCOND_REF)),
            ("numpy", numpy_pinv),
        ]
        for tag, fn in methods:
            try:
                pinvs[tag] = fn(A)
            except Exception:
                pinvs[tag] = np.zeros((A.shape[1], A.shape[0]))

        for tag in METHOD_KEYS:
            try:
                row[f"mp_{tag}"] = f"{penrose_error_abs(A, pinvs[tag]):.2e}"
            except Exception:
                row[f"mp_{tag}"] = "err"
        rows.append(row)

    # print + save the table
    header = (
        "Matrix", "Shape", "Rank", "Cond",
        "mp_direct", "mp_qb", "mp_qrginv", "mp_numpy",
    )
    col_w = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in header}
    sep = "  ".join("-" * col_w[h] for h in header)
    print("\n" + sep)
    print("  ".join(h.ljust(col_w[h]) for h in header))
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(col_w[h]) for h in header))
    print(sep)

    np.save(os.path.join(OUT_DIR, "nist_stability.npy"), {"header": list(header), "rows": rows})
    print("  Saved nist_stability.npy")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\nDone in {time.time() - t:.1f} s")
