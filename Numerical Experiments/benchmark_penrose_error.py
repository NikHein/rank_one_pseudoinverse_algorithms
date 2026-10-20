"""Relative Penrose error and forward error vs numpy as the row count m grows, for well- and ill-conditioned low-rank matrices."""
import gc
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

from algorithms import qb_pinv, direct_pseudoinverse_random
from fast_tikhonov import fast_tikhonov
from qrginv import qrginv
from verify_moore_penrose import verify_moore_penrose
from generate_low_rank_matrix import generate_low_rank_matrix

EPS_ALGO = 1e-8
RCOND_REF = 1e-10
FAST_TOL = 1e-15
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


def main(n=10000, rank=100, seeds=(42, 123, 456)):
    m_values = np.arange(1000, 10001, 500)
    keys = [f"{k}_errors" for k in METHOD_KEYS] + [
        f"{k}_forward_errors" for k in METHOD_KEYS if k != "numpy"
    ]

    # run every method over m, for two conditioning regimes, averaged over seeds
    all_results = {}
    for singular_scale, label in [("normal", "normal"), ("big small", "ill-conditioned")]:
        print(f"\n  Condition: {label}")
        all_seed = {k: [] for k in keys}

        for seed in seeds:
            print(f"    seed {seed} ...", flush=True)
            seed_res = {k: [] for k in keys}

            for m in m_values:
                actual_rank = min(rank, min(m, n))
                gc.collect()
                try:
                    A = generate_low_rank_matrix(m, n, actual_rank, singular_scale, seed=seed)
                    Ap_numpy = numpy_pinv(A)
                    nrm_ref = nla.norm(Ap_numpy, "fro")
                    seed_res["numpy_errors"].append(verify_moore_penrose(A, Ap_numpy))

                    Ap_direct = direct_pseudoinverse_random(A, epsilon=EPS_ALGO, seed=seed)
                    seed_res["direct_errors"].append(verify_moore_penrose(A, Ap_direct))
                    seed_res["direct_forward_errors"].append(nla.norm(Ap_direct - Ap_numpy, "fro") / nrm_ref)

                    Ap_qb = qb_pinv(A, epsilon=EPS_ALGO)
                    seed_res["qb_errors"].append(verify_moore_penrose(A, Ap_qb))
                    seed_res["qb_forward_errors"].append(nla.norm(Ap_qb - Ap_numpy, "fro") / nrm_ref)

                    try:
                        Ap_fast = fast_tikhonov(A, FAST_TOL)
                        seed_res["fast_errors"].append(verify_moore_penrose(A, Ap_fast))
                        seed_res["fast_forward_errors"].append(nla.norm(Ap_fast - Ap_numpy, "fro") / nrm_ref)
                    except Exception as e:
                        print(f"      fast failed at m={m}: {e}")
                        seed_res["fast_errors"].append(np.nan)
                        seed_res["fast_forward_errors"].append(np.nan)

                    try:
                        Ap_qrginv = qrginv(A, tol=RCOND_REF)
                        seed_res["qrginv_errors"].append(verify_moore_penrose(A, Ap_qrginv))
                        seed_res["qrginv_forward_errors"].append(nla.norm(Ap_qrginv - Ap_numpy, "fro") / nrm_ref)
                    except Exception as e:
                        print(f"      qrginv failed at m={m}: {e}")
                        seed_res["qrginv_errors"].append(np.nan)
                        seed_res["qrginv_forward_errors"].append(np.nan)

                except Exception as e:
                    print(f"      Error at m={m}: {e}")
                    for k in keys:
                        seed_res[k].append(np.nan)

            for k in keys:
                all_seed[k].append(seed_res[k])

        results = {
            k: [np.nanmean([all_seed[k][si][mi] for si in range(len(seeds))]) for mi in range(len(m_values))]
            for k in keys
        }
        save_npy(f"penrose_error_n{n}_rank{rank}_{label}", m_values, results)
        all_results[label] = results

    # 2x2 figure: Penrose error (top) and forward error (bottom), per regime
    fig, axes = plt.subplots(2, 2, figsize=(5.07, 5))
    for idx, (cond, title) in enumerate(
        zip(["normal", "ill-conditioned"], ["Well-conditioned matrices", "Ill-conditioned matrices"])
    ):
        results = all_results[cond]

        ax = axes[0, idx]
        for key, mlabel, marker in PLOT_METHODS:
            ax.semilogy(m_values, results[f"{key}_errors"], marker, label=mlabel, linewidth=0.7, markersize=2)
        ax.set_title(title)
        ax.set_xlabel("Matrix Size (n)")
        ax.set_ylabel("Relative Penrose Error")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)
        ax.legend(loc="upper left")

        ax2 = axes[1, idx]
        for key, mlabel, marker in PLOT_METHODS:
            if key == "numpy":
                continue
            ax2.semilogy(m_values, results[f"{key}_forward_errors"], marker, label=mlabel, linewidth=0.7, markersize=2)
        ax2.set_title(title)
        ax2.set_xlabel("Matrix Size (n)")
        ax2.set_ylabel("Relative Forward Error vs numpy")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.3)
        ax2.legend(loc="upper left")

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    save_fig(fig, "combined_mp_error")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\nDone in {time.time() - t:.1f} s")
