"""DMD of a synthetic two-mode signal: runtime, DMD error, relative Penrose error and forward error"""

import gc
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from algorithms import qb_pinv, direct_pseudoinverse_random
from fast_tikhonov import fast_tikhonov
from qrginv import qrginv

EPS_ALGO = 1e-8
RCOND_REF = 1e-10
FAST_TOL = 1e-15
N_REPS = 3
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 9,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

PLOT_METHODS = [
    ("direct", "Algorithm I", "s-"),
    ("qb", "Algorithm II", "o-"),
    ("fast", "Fast (Tikhonov)", "v-"),
    ("qrginv", "qrginv", "d-"),
    ("numpy", r"\texttt{numpy} pinv", "^-"),
]
METHOD_KEYS = [k for k, _, _ in PLOT_METHODS]
FWD_KEYS = [k for k in METHOD_KEYS if k != "numpy"]

X_LEFT, X_RIGHT = -15.0, 15.0
T_LEFT, T_RIGHT = 0.0, 8.0 * np.pi
K_VALUES = sorted(set(np.round(np.logspace(0, np.log10(64), 16)).astype(int).tolist()))


def numpy_pinv(A):
    return np.linalg.pinv(A, rcond=RCOND_REF)


def save_fig(fig, name):
    fig.savefig(
        os.path.join(OUT_DIR, f"{name}.eps"),
        format="eps",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        os.path.join(OUT_DIR, f"{name}.png"),
        format="png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    print(f"  Saved {name}.eps / {name}.png")


def _snapshots(nx, nt):
    x = np.linspace(X_LEFT, X_RIGHT, nx)
    t = np.linspace(T_LEFT, T_RIGHT, nt)
    xg, tg = np.meshgrid(x, t)
    sech = lambda z: 1.0 / np.cosh(z)
    F = sech(xg + 3.0) * np.exp(2.3j * tg) + 2.0 * sech(xg) * np.tanh(xg) * np.exp(
        2.8j * tg
    )
    return F.T.copy()


def dmd_error(Xp, X, Y):
    return la.norm(Y - Y @ (Xp @ X), "fro") / la.norm(Y, "fro")


def penrose_residual(A, Ap):
    AA = A @ Ap
    ApA = Ap @ A
    e1 = la.norm(AA @ A - A, "fro") / la.norm(A, "fro")
    e2 = la.norm(ApA @ Ap - Ap, "fro") / la.norm(Ap, "fro")
    e3 = la.norm(AA - AA.conj().T, "fro") / la.norm(AA, "fro")
    e4 = la.norm(ApA - ApA.conj().T, "fro") / la.norm(ApA, "fro")
    return (e1 + e2 + e3 + e4) / 4.0


def main():
    nx_nt_axis = []
    runtimes = {m: [] for m in METHOD_KEYS}
    dmd_errors = {m: [] for m in METHOD_KEYS}
    penrose = {m: [] for m in METHOD_KEYS}
    frob_dist = {m: [] for m in FWD_KEYS}

    # sweep the snapshot-matrix size and evaluate every method
    for k in K_VALUES:
        NX, NT = 32 * k, 64 * k
        print(f"\n  k={k}  NX={NX}  NT={NT}  NX*NT={NX * NT}")

        snaps = _snapshots(NX, NT)
        X, Y = snaps[:, :-1], snaps[:, 1:]

        methods = [
            (
                "direct",
                lambda A, s=k: direct_pseudoinverse_random(A, epsilon=EPS_ALGO, seed=s),
            ),
            ("qb", lambda A: qb_pinv(A, epsilon=EPS_ALGO)),
            ("fast", lambda A: fast_tikhonov(A, FAST_TOL)),
            ("qrginv", lambda A: qrginv(A, tol=RCOND_REF)),
            ("numpy", numpy_pinv),
        ]

        Xp_all = {}
        for name, func in methods:
            try:
                reps = []
                for _ in range(N_REPS):
                    gc.collect()
                    t0 = time.process_time()
                    Xp = func(X)
                    reps.append(time.process_time() - t0)
                Xp_all[name] = Xp
                runtimes[name].append(float(np.mean(reps)))
            except Exception as e:
                print(f"    {name} failed: {e}")
                Xp_all[name] = None
                runtimes[name].append(np.nan)

        Xp_numpy = Xp_all["numpy"]
        for name in METHOD_KEYS:
            if Xp_all[name] is None:
                dmd_errors[name].append(np.nan)
                penrose[name].append(np.nan)
                continue
            dmd_errors[name].append(dmd_error(Xp_all[name], X, Y))
            penrose[name].append(penrose_residual(X, Xp_all[name]))

        numpy_norm = la.norm(Xp_numpy, "fro")
        for name in FWD_KEYS:
            if Xp_all[name] is None:
                frob_dist[name].append(np.nan)
                continue
            frob_dist[name].append(la.norm(Xp_all[name] - Xp_numpy, "fro") / numpy_norm)

        nx_nt_axis.append(NX * NT)

    np.savez(
        os.path.join(OUT_DIR, "dmd_twomode_scaling_results.npz"),
        nx_nt_axis=nx_nt_axis,
        k_values=K_VALUES,
        **{f"runtimes_{m}": runtimes[m] for m in METHOD_KEYS},
        **{f"dmd_errors_{m}": dmd_errors[m] for m in METHOD_KEYS},
        **{f"penrose_{m}": penrose[m] for m in METHOD_KEYS},
        **{f"frob_dist_{m}": frob_dist[m] for m in FWD_KEYS},
    )
    print("  Saved dmd_twomode_scaling_results.npz")

    # 2x2 publication figure
    x = np.array(nx_nt_axis)

    def make_panel(ax, data_dict, methods_to_plot, ylabel, title, show_legend=False):
        for key, mlabel, marker in methods_to_plot:
            vals = np.array(data_dict[key], dtype=float)
            vals = np.where(vals > 0, vals, np.nan)
            ax.loglog(x, vals, marker, markersize=3, linewidth=0.8, label=mlabel)
        ax.set_xlabel(r"$N_x \times N_T$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if show_legend:
            ax.legend(loc="best")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)

    plot_methods = [pm for pm in PLOT_METHODS if pm[0] != "fast"]
    fwd_plot_methods = [pm for pm in plot_methods if pm[0] != "numpy"]

    fig, axes = plt.subplots(2, 2, figsize=(5.07, 5.0))
    make_panel(
        axes[0, 0],
        runtimes,
        plot_methods,
        "CPU Time (second)",
        "Runtime",
        show_legend=True,
    )
    make_panel(
        axes[0, 1],
        dmd_errors,
        plot_methods,
        "DMD error",
        r"$\|Y - YX^{+}X\|_F / \|Y\|_F$",
    )
    make_panel(
        axes[1, 0],
        penrose,
        plot_methods,
        "Relative Penrose Error",
        "Moore--Penrose conditions",
    )
    make_panel(
        axes[1, 1],
        frob_dist,
        fwd_plot_methods,
        "Relative Forward Error",
        r"$\|X^+_\mathrm{algo} - X^+_\mathrm{numpy}\|_F \;/\; \|X^+_\mathrm{numpy}\|_F$",
        show_legend=True,
    )

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.45, hspace=0.4)
    save_fig(fig, "dmd_twomode_scaling")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\nDone in {time.time() - t:.1f} s")
