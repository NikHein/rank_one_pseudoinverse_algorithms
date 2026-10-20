"""Moore-Penrose inverse via QR (qrginv)
V. N. Katsikis, D. Pappas, A. Petralias, An improved method for the computation of the moore–penrose inverse matrix, Applied Mathematics and Computation 217 (23) (2011)
doi:10.1016/j.amc.2011.04.080).
"""

import numpy as np
import scipy.linalg as sla


def qrginv(B, tol=1e-10):
    B = np.asarray(B)
    N, M = B.shape

    Q, R, P = sla.qr(B, pivoting=True)  # B[:, P] = Q @ R
    r = int(np.sum(np.any(np.abs(R) > tol, axis=1)))  # numerical rank

    R1 = R[:r, :]
    R2 = R1.conj().T @ np.linalg.inv(R1 @ R1.conj().T)  # ginv(R1)

    R3 = np.zeros((M, N), dtype=B.dtype)
    R3[:, :r] = R2
    X = np.zeros((M, N), dtype=B.dtype)
    X[P, :] = R3
    return X @ Q.conj().T
