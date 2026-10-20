"""Fast Tikhonov-regularized Moore-Penrose inverse (soto)
P. Soto-Quiros, A fast method to estimate the moore-penrose inverse for well-determined numerical rank matrices based on tikhonov regularization, J Math Comput Sci 37 (1) (2024) 59–81.
doi:10.22436/jmcs.037.01.05
"""

import numpy as np
from scipy.linalg import get_lapack_funcs


def _rcond(T):
    # MATLAB-style reciprocal condition estimate
    getrf, gecon = get_lapack_funcs(("getrf", "gecon"), (T,))
    anorm = np.linalg.norm(T, 1)
    lu, piv, info = getrf(T, overwrite_a=False)
    rcond, info = gecon(lu, anorm)
    return rcond


def soto(A, tol):
    A = np.asarray(A)
    m, n = A.shape
    At = A.conj().T
    if m >= n:
        T = At @ A
        if _rcond(T) > np.sqrt(np.finfo(float).eps):
            X = np.linalg.solve(T, At)
        else:
            s1 = np.linalg.eigvalsh(T)
            s2 = np.sort(s1)[::-1]
            f = n * np.finfo(float).eps * s2[0]
            s = np.sqrt(s1[s1 > f])
            alpha = 0.5 * np.sqrt(tol / np.sum(1.0 / s**6))
            X = np.linalg.solve(T + alpha * np.eye(n), At)
    else:
        T = A @ At
        if _rcond(T) > np.sqrt(np.finfo(float).eps):
            X = np.linalg.solve(T.conj().T, A).conj().T
        else:
            s1 = np.linalg.eigvalsh(T)
            s2 = np.sort(s1)[::-1]
            f = m * np.finfo(float).eps * s2[0]
            s = np.sqrt(s2[s2 > f])
            alpha = 0.5 * np.sqrt(tol / np.sum(1.0 / s**6))
            X = np.linalg.solve((T + alpha * np.eye(m)).conj().T, A).conj().T
    return X
