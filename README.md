# Qubit Tune-up — M5 (Adaptive Grid + EIG + DRAG)

ZI EQH Hackathon 2026, single-qubit tune-up track.

This submission contains our **best method (M5)** — a Bayesian estimator that
recovers `(f_qubit, amp_pi, T1, T2*)` of a virtual qubit in **~3 250 shots**.

The other four methods (M1 Classical, M2 Sequential-Bayes, M3 Joint-Chirp,
M4 Joint-Bayes-Spec) are explained in the accompanying slide deck and the
`methods_comparison.pdf` matrix.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# either run from the command line ...
python optimized/joint_estimator_grid.py

# ... or step through the notebook. Remember intall jupyterlab. ("pip install jupyterlab")
jupyter lab run_m5.ipynb
```

## What M5 does

1. **Chirp spectroscopy** (Landau–Zener) — coarse `f_qubit` from the
   plateau of a single linear chirp; also yields a free Ω prior for Rabi.
2. **Bayesian Rabi** — SMC particle filter on `(amp_pi, decay)`.
3. **Joint 3D grid posterior** over `(δω, T1, T2*)`:
   - constraint mask `T2* ≤ 2·T1` baked into the grid
   - **EIG-greedy** experiment selector with log-spaced T1 / Ramsey τ candidates
   - **DRAG-corrected π/2** pulse — β optimised by maximising 2 × π/2 contrast
   - **Pulse-contrast calibration** for `C_T1`, `C_ramsey`
   - **Two-stage adaptive grid refinement** at steps 180 / 380 (3 σ zoom)
   - **Convergence**: relative σ < 5 % for T1 and T2*, σ_δω < 80 kHz
4. **Active-reset verification** + π-pulse fidelity check.

### Shot budget (~3 250 total)

| Stage                       | Shots         |
| --------------------------- | ------------- |
| Chirp spectroscopy          | ~700          |
| Bayesian Rabi               | ~200          |
| DRAG calibration (7 × 120)  | 840           |
| Pulse-contrast cal (2 × 300)| 600           |
| Adaptive grid loop (≤ 600)  | ~600          |
| Active reset + π-verify     | ~310          |
| **Total**                   | **~3 250**    |

## Typical result (seed = 42)

```
truth         f_q = 5.7350 GHz   T1 = 27.40 µs   T2* = 18.00 µs
M5 estimate   f_err 0.003 MHz   T1 err 4.0%    T2* err 5.0%   shots ~3 250
```

## Files

```
qubit.py                       VirtualQubit / VirtualQubitPair simulator
pyproject.toml                 dependencies (laboneq, numpy, scipy, matplotlib)
LICENSE                        Apache-2.0
run_m5.ipynb                   one-cell-per-step walkthrough of M5
raw_data_comparison.txt        RAW DATA COMPARISON — M5 vs Classical Pipeline
test_main.py                   CI entry point: Automated validation of qubit frequency and coherence ($T_1$, $T_2^*$)
.github/
   workflows/                  Contains python-tests.yml for automated CI/CD testing on every push.
optimized/
  optimized_tuneup.py          chirp_spectroscopy, bayesian_rabi, active_reset
  joint_estimator.py           ground-truth helper
  joint_estimator_grid.py      ★ M5 entry point: run_joint_grid()
figures/
  binary_search_spec.png       Lorentzian-narrowing spectroscopy figure
  ramsey_fringe.png            Ramsey decay + 3D convergence trajectory
  grid_convergence.png         3D grid posterior collapsing to truth
  budget_breakdown.png         shot-accounting bar / pie chart
  methods_comparison.pdf       M1..M5 technique matrix
```

## Dependencies

- Python ≥ 3.11
- `laboneq` (Zurich Instruments)
- `numpy`, `scipy`, `matplotlib`

All listed in `pyproject.toml`.
