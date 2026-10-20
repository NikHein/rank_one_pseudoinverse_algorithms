# Python Code for the Numerical Experiments of the Paper "Rank-One Update Algorithms for Fast Pseudoinverse Computation in Large Low-Rank Data Problems"

Submitted to the Journal of Computational and Applied Mathematics.

## Authors

Thomas Hochrainer (hochrainer@tugraz.at), Niko Heinemann (niko.heinemann@tugraz.at)

Institut für Festigkeitslehre, Graz University of Technology,
Kopernikusgasse 24, Graz, 8010, Austria

## Description

This repository reproduces all numerical experiments of the paper.

The paper presents simple rank-one update algorithms for the Moore–Penrose pseudoinverse whose cost scales with the matrix rank rather than its dimension. They are fast for low-rank and large matrices. Algorithm II is accurate and robust throughout, Algorithm I is the fastest at low rank. Typical uses are time-critical low-rank pseudoinverse computations such as reduced-order modeling and dynamic mode decomposition.

`algorithms.py` contains the two proposed algorithms; `qrginv.py` and `fast_tikhonov.py` are the reference methods used for comparison.

Supporting modules:

- `generate_low_rank_matrix.py`: random rank-`r` test matrices with a prescribed singular-value regime
- `verify_moore_penrose.py`: mean relative residual of the four Penrose equations

Experiment scripts:

- `benchmark_runtime_vs_rank.py`: CPU time vs. relative rank for square `n × n` matrices (Figure 1)
- `benchmark_runtime_rectangular.py`: CPU time vs. size for `n/2 × n` matrices (Figure 2)
- `benchmark_penrose_error.py`: relative Penrose and forward error for `n × 10000` rank-100 matrices, well- and ill-conditioned (Figure 3)
- `benchmark_nist_stability.py`: absolute Penrose residuals on classic ill-conditioned test matrices from NIST Matrix Market (Table 1)
- `benchmark_dmd_twomode.py`: runtime and accuracy of the dynamic mode decomposition example (Figure 4)

Runtimes are measured single-threaded (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`).

Licensed under GPL-3.0, see `LICENSE`.
