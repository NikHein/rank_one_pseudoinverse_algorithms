"""CPU time of every method for square n x n matrices as the rank sweeps 1%-20%, for n in {5000,10000,15000,20000}."""
import gc
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt

from algorithms import qb_pinv, direct_pseudoinverse_random
from qrginv import qrginv
from soto import soto
from generate_low_rank_matrix import generate_low_rank_matrix

EPS_ALGO = 1e-8
RCOND_REF = 1e-10
SOTO_TOL = 1e-15
SOTO_SIZE = 5000
N_TRIALS = 3
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
    ("qrginv", "qrginv", "d-"),
    ("numpy", r"\texttt{numpy} pinv", "^-"),
    ("soto", "soto", "v-"),
]
METHOD_KEYS = [k for k, _, _ in PLOT_METHODS]


def numpy_pinv(A):
    return np.linalg.pinv(A, rcond=RCOND_REF)


def save_npy(filename_base, x_data, y_dict):
    data = {"x_data": x_data}
    data.update(y_dict)
    np.save(os.path.join(OUT_DIR, f"{filename_base}.npy"), data)
    print(f"  Saved {filename_base}.npy")


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.eps"), format="eps", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {name}.eps / {name}.png")


def _time_call(fn, A):
    gc.collect()
    t0 = time.process_time()
    fn(A)
    return time.process_time() - t0


def main():
    sizes = [5000, 10000, 15000, 20000]

    fig_t, axes_t = plt.subplots(2, 2, figsize=(5.07, 5))

    for idx, size in enumerate(sizes):
        print(f"\n  n = {size}")
        row, col = idx // 2, idx % 2
        rank_percentages = np.arange(1, 21, 1)
        keys = [k for k in METHOD_KEYS if k != "soto" or size == SOTO_SIZE]

        times = {k: [] for k in keys}

        # time every method for each rank percentage
        for rank_percent in rank_percentages:
            actual_k = min(max(1, int(round(size * rank_percent / 100.0))), size)
            print(f"    rank {rank_percent}%  k={actual_k}", flush=True)

            trial_t = {k: [] for k in keys}
            gc.collect()

            for trial in range(N_TRIALS):
                A = generate_low_rank_matrix(size, size, actual_k, "normal", seed=trial)
                methods = [
                    ("direct", lambda A, trial=trial: direct_pseudoinverse_random(A, epsilon=EPS_ALGO, seed=trial)),
                    ("qb", lambda A: qb_pinv(A, epsilon=EPS_ALGO)),
                    ("qrginv", lambda A: qrginv(A, tol=RCOND_REF)),
                    ("numpy", numpy_pinv),
                    ("soto", lambda A: soto(A, SOTO_TOL)),
                ]
                for key, fn in methods:
                    if key not in keys:
                        continue
                    try:
                        dt = _time_call(fn, A)
                    except Exception as e:
                        print(f"      {key} failed: {e}")
                        dt = np.nan
                    trial_t[key].append(dt)

            for k in keys:
                times[k].append(np.nanmean(trial_t[k]))

        save_npy(f"runtime_vs_rank_n{size}", rank_percentages, {f"{k}_times": v for k, v in times.items()})

        ax = axes_t[row, col]
        for key, mlabel, marker in PLOT_METHODS:
            if key not in keys:
                continue
            ax.semilogy(rank_percentages, times[key], marker, label=mlabel, linewidth=0.7, markersize=2)
        ax.set_title(f"$n = {size}$")
        ax.set_xlabel("Rank (\\%)")
        ax.set_ylabel("CPU Time (second)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)
        ax.legend(loc="lower right")

    fig_t.tight_layout(pad=0.5)
    fig_t.subplots_adjust(hspace=0.4, wspace=0.35)
    save_fig(fig_t, "combined_runtime_quad")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\nDone in {time.time() - t:.1f} s")
