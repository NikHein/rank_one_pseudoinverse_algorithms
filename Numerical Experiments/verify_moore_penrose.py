"""Mean relative residual of the four Moore-Penrose conditions for A and a pseudoinverse X."""
import numpy as np
from numpy.linalg import norm


def verify_moore_penrose(A, X):
    AX = A @ X
    XA = X @ A
    r1 = norm(A @ X @ A - A) / norm(A)          # A X A = A
    r2 = norm(X @ A @ X - X) / norm(X)          # X A X = X
    r3 = norm(AX - AX.conj().T) / norm(AX)      # (A X)^H = A X
    r4 = norm(XA - XA.conj().T) / norm(XA)      # (X A)^H = X A
    return (r1 + r2 + r3 + r4) / 4
