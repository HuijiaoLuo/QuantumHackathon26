"""Grid-based Bayesian joint estimator with EIG and DRAG  (M7 / Grid).

Architectural alternative to the particle-filter pipelines in
joint_estimator.py:

  * 3D grid posterior over (delta_omega, T1, T2*)  --  no resampling noise
  * Constraint  T2* <= 2 T1  is enforced as a zero-prior mask
  * Each shot: pick the (exp_type, tau) with maximum expected information
    gain (EIG)  -- no fixed schedule, no hard tau-cap
  * DRAG-corrected pi/2 pulse (Gaussian + scaled derivative on Q channel)
  * Late-stage grid refinement: zoom +-3.5 sigma after step 250
  * Adaptive convergence stop on relative T1, T2* and absolute delta_omega

This module reuses our existing primitives only for spectroscopy and
amplitude Rabi  (chirp_spectroscopy, bayesian_rabi)  and adds its own
Gaussian/DRAG pulse compilation so the DRAG quadrature has a non-zero
derivative.

Public entry point:
    run_joint_grid(seed=42, ...) -> dict   compatible with print_five_way()
"""
from __future__ import annotations
import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "optimized"))

from qubit import VirtualQubit
from optimized_tuneup import (
    ShotCounter,
    chirp_spectroscopy,
    bayesian_rabi,
    active_reset,
)
from joint_estimator import _get_ground_truth


# ──────────────────────────────────────────────────────────────────────
#  Pulse compilation: Gaussian + DRAG
# ──────────────────────────────────────────────────────────────────────

def lorentzian_spec(qubit, counter, f_lo=5.0e9, f_hi=6.2e9,
                    n_coarse=21, n_fine=15, shots_coarse=200, shots_fine=300,
                    pulse_length=2e-6, amp=0.5):
    """Two-stage Lorentzian-fit spectroscopy.

    Stage 1: 21-point coarse scan over [5, 6.2] GHz @ 200 shots → peak.
    Stage 2: 15-point fine scan over peak±50 MHz @ 300 shots, Lorentzian fit.

    Total shots: 21*200 + 15*300 = 4200 + 4500 = 8700.
    Compared to Luis's 101*2000 = 202000 (23× more shots) but achieves
    similar sub-MHz accuracy because the Lorentzian fit on the fine
    window is the dominant precision source."""
    from scipy.optimize import curve_fit
    # rectangular spec pulse
    n_samples = max(64, int(pulse_length * 4e9))
    t_spec = np.linspace(0.0, pulse_length, n_samples)
    wf_spec = np.full(n_samples, amp, dtype=np.complex128)

    # ── Stage 1: coarse ────────────────────────────────────────────
    f_coarse = np.linspace(f_lo, f_hi, n_coarse)
    p1_coarse = np.zeros(n_coarse)
    for i, f in enumerate(f_coarse):
        qubit.reset()
        qubit.evolve(t_spec, wf_spec, drive_freq=float(f))
        p1_coarse[i] = float(qubit.measure(shots=shots_coarse).mean())
        counter.total += shots_coarse
    f0_coarse = float(f_coarse[int(np.argmax(p1_coarse))])

    # ── Stage 2: fine, ±50 MHz around peak ─────────────────────────
    f_fine = np.linspace(f0_coarse - 50e6, f0_coarse + 50e6, n_fine)
    p1_fine = np.zeros(n_fine)
    for i, f in enumerate(f_fine):
        qubit.reset()
        qubit.evolve(t_spec, wf_spec, drive_freq=float(f))
        p1_fine[i] = float(qubit.measure(shots=shots_fine).mean())
        counter.total += shots_fine

    def _lor(f, f0, gamma, A, offset):
        return offset + A * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2)

    f0_guess = float(f_fine[int(np.argmax(p1_fine))])
    try:
        popt, _ = curve_fit(
            _lor, f_fine, p1_fine,
            p0=[f0_guess, 5e6, p1_fine.max() - p1_fine.min(), p1_fine.min()],
            bounds=([f_fine[0], 0.5e6, 0.0, 0.0],
                    [f_fine[-1], 200e6, 1.0, 0.5]),
            maxfev=5000,
        )
        f_res = float(popt[0])
    except (RuntimeError, ValueError):
        f_res = f0_guess
    return f_res


def compile_gauss(length: float, amp: float = 1.0,
                  sigma_frac: float = 0.25, n_samples: int = 96):
    """Return (t, waveform) for a Gaussian pulse with the SAME pulse-area
    as a constant-amplitude rectangular pulse  rect(length, amp).

    Reasoning: qubit rotation angle  ~  integral of envelope.  A bare
    Gaussian with peak == `amp` rotates much less than a rectangle of
    equal peak; we therefore scale the Gaussian peak up so its integral
    over `length` equals  amp * length  (= rect integral)."""
    t = np.linspace(-length / 2, length / 2, n_samples)
    sigma = length * sigma_frac
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    g -= g.min()                # zero at the edges
    g /= g.max()                # peak == 1 before scaling
    # Match pulse area to a unit-amplitude rectangle of the same length.
    area_unit_rect  = length
    area_unit_gauss = float(np.trapezoid(g, t))
    scale_to_match  = area_unit_rect / area_unit_gauss
    return t + length / 2, (g * amp * scale_to_match).astype(np.complex128)


def make_drag(t: np.ndarray, wf: np.ndarray, beta: float) -> np.ndarray:
    """Add a DRAG quadrature  Q(t) = beta * dI/dt  to a real-valued
    Gaussian pulse.  Returns complex waveform  I + 1j * beta * dI/dt."""
    I = wf.real.copy()
    dt = np.diff(t)
    dI = np.zeros_like(I)
    dI[:-1] = np.diff(I) / dt
    dI[-1] = dI[-2]
    return I + 1j * beta * dI


def calibrate_drag_beta(qubit, drive_freq, t_half, wf_half_base,
                        n_betas=15, beta_range=(-2e-7, 2e-7),
                        shots_per_beta=400):
    """Scan beta, find the value that maximises  P(1)  for a back-to-back
    pi/2 + pi/2 pulse pair.  Returns beta_opt and the scan trace."""
    betas = np.linspace(*beta_range, n_betas)
    p1 = np.zeros(n_betas)
    for i, b in enumerate(betas):
        wf_h = make_drag(t_half, wf_half_base, b)
        qubit.reset()
        qubit.evolve(t_half, wf_h, drive_freq=drive_freq)
        qubit.evolve(t_half, wf_h, drive_freq=drive_freq)
        p1[i] = float(qubit.measure(shots=shots_per_beta).mean())
    return float(betas[int(np.argmax(p1))]), betas, p1


# ──────────────────────────────────────────────────────────────────────
#  3D grid Bayesian estimator
# ──────────────────────────────────────────────────────────────────────

class GridEstimator:
    """Joint posterior on a regular grid of  (delta_omega, T1, T2*)."""

    def __init__(self,
                 dw_half: float = 2 * np.pi * 2e6,
                 T1_range: tuple = (5e-6, 80e-6),
                 T2_range: tuple = (5e-6, 50e-6),
                 n_dw: int = 200, n_T1: int = 40, n_T2: int = 35,
                 ro_err: float = 0.02,
                 C_T1: float = 1.0, C_ramsey: float = 1.0):
        self.dw_vals = np.linspace(0.0, dw_half, n_dw)
        self.T1_vals = np.linspace(*T1_range, n_T1)
        self.T2_vals = np.linspace(*T2_range, n_T2)
        self.ro_err = ro_err
        self.C_T1 = C_T1
        self.C_ramsey = C_ramsey

        dw, T1, T2 = np.meshgrid(self.dw_vals, self.T1_vals, self.T2_vals,
                                 indexing="ij")
        self._dw = dw
        self._T1 = T1
        self._T2 = T2
        self._p = np.ones_like(dw)
        self._p[T2 > 2.0 * T1] = 0.0     # physical constraint
        self._p /= self._p.sum()

    # ── likelihoods ──────────────────────────────────────────────────
    def _p1_T1(self, tau):
        ideal = (1.0 - self.C_T1) / 2.0 + self.C_T1 * np.exp(-tau / self._T1)
        return self.ro_err + (1.0 - 2.0 * self.ro_err) * ideal

    def _p1_ramsey(self, tau):
        ideal = 0.5 * (1.0 + self.C_ramsey
                       * np.exp(-tau / self._T2)
                       * np.cos(self._dw * tau))
        return self.ro_err + (1.0 - 2.0 * self.ro_err) * ideal

    # ── posterior update (single shot) ──────────────────────────────
    def update(self, exp_type: str, tau: float, m: int):
        p1 = self._p1_T1(tau) if exp_type == "T1" else self._p1_ramsey(tau)
        p1 = np.clip(p1, 1e-10, 1.0 - 1e-10)
        self._p *= p1 if m == 1 else (1.0 - p1)
        s = self._p.sum()
        if s > 0:
            self._p /= s

    # ── batched binomial update (k ones out of n shots at one tau) ──
    def update_batch(self, exp_type: str, tau: float, k: int, n: int):
        """Apply binomial likelihood for n shots with k ones at the same tau.
        Reduces single-shot Bernoulli noise — same total info per shot but
        far less stochasticity in the posterior trajectory."""
        p1 = self._p1_T1(tau) if exp_type == "T1" else self._p1_ramsey(tau)
        p1 = np.clip(p1, 1e-10, 1.0 - 1e-10)
        log_lik = k * np.log(p1) + (n - k) * np.log(1.0 - p1)
        log_lik -= log_lik.max()
        self._p *= np.exp(log_lik)
        s = self._p.sum()
        if s > 0:
            self._p /= s

    # ── EIG-based experiment selector ────────────────────────────────
    @staticmethod
    def _entropy(w):
        s = float(w.sum())
        if s < 1e-15:
            return 0.0
        q = w / s
        m = q > 0
        return float(-np.sum(q[m] * np.log(q[m])))

    def _eig(self, p1):
        p1 = np.clip(p1, 1e-10, 1.0 - 1e-10)
        P1 = float(np.sum(self._p * p1))
        H0 = self._entropy(self._p)
        H1 = self._entropy(self._p * p1)
        H0_post = self._entropy(self._p * (1.0 - p1))
        return H0 - P1 * H1 - (1.0 - P1) * H0_post

    def select(self, n_cands: int = 8):
        """EIG-greedy experiment selector — Luis-style.
        T1 candidates log-spaced in [0.2, 3] × T1_est.
        Ramsey candidates log-spaced in [0.1, 2] × T2_est."""
        est = self.estimates()
        T1_e = max(est["T1"][0], 1e-6)
        T2_e = max(est["T2"][0], 1e-6)
        T1_cand  = np.logspace(np.log10(T1_e * 0.2),
                               np.log10(T1_e * 3.0), n_cands)
        ram_cand = np.logspace(np.log10(T2_e * 0.1),
                               np.log10(T2_e * 2.0), n_cands)
        best = (-np.inf, "T1", T1_e)
        for tau in T1_cand:
            ig = self._eig(self._p1_T1(tau))
            if ig > best[0]:
                best = (ig, "T1", float(tau))
        for tau in ram_cand:
            ig = self._eig(self._p1_ramsey(tau))
            if ig > best[0]:
                best = (ig, "ramsey", float(tau))
        return best[1], best[2]

    # ── adaptive grid refinement around posterior peak ──────────────
    def refine(self, n_sigma: float = 3.5):
        from scipy.interpolate import RegularGridInterpolator
        est = self.estimates()
        old_dw, old_T1, old_T2 = self.dw_vals, self.T1_vals, self.T2_vals
        n_dw, n_T1, n_T2 = len(old_dw), len(old_T1), len(old_T2)

        dw_m, dw_s = est["dw"]
        T1_m, T1_s = est["T1"]
        T2_m, T2_s = est["T2"]

        dw_hw = max(n_sigma * dw_s, 2 * np.pi * 200e3)
        T1_hw = max(n_sigma * T1_s, 2e-6)
        T2_hw = max(n_sigma * T2_s, 3e-6)

        new_dw = np.linspace(max(0.0, dw_m - dw_hw), dw_m + dw_hw, n_dw)
        new_T1 = np.linspace(max(old_T1[0],  T1_m - T1_hw),
                             min(old_T1[-1], T1_m + T1_hw), n_T1)
        new_T2 = np.linspace(max(old_T2[0],  T2_m - T2_hw),
                             min(old_T2[-1], T2_m + T2_hw), n_T2)

        interp = RegularGridInterpolator(
            (old_dw, old_T1, old_T2), self._p,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        dw_g, T1_g, T2_g = np.meshgrid(new_dw, new_T1, new_T2, indexing="ij")
        pts = np.stack([dw_g.ravel(), T1_g.ravel(), T2_g.ravel()], axis=-1)
        new_p = interp(pts).reshape(n_dw, n_T1, n_T2)
        new_p[T2_g > 2.0 * T1_g] = 0.0
        new_p = np.clip(new_p, 0.0, None)
        s = new_p.sum()
        if s > 0:
            new_p /= s
        self.dw_vals, self.T1_vals, self.T2_vals = new_dw, new_T1, new_T2
        self._dw, self._T1, self._T2 = dw_g, T1_g, T2_g
        self._p = new_p

    # ── marginals & summary ──────────────────────────────────────────
    def marginals(self):
        return (self._p.sum(axis=(1, 2)),
                self._p.sum(axis=(0, 2)),
                self._p.sum(axis=(0, 1)))

    def estimates(self):
        p_dw, p_T1, p_T2 = self.marginals()
        dw_m = float(np.sum(p_dw * self.dw_vals))
        T1_m = float(np.sum(p_T1 * self.T1_vals))
        T2_m = float(np.sum(p_T2 * self.T2_vals))
        dw_s = float(np.sqrt(max(np.sum(p_dw * (self.dw_vals - dw_m) ** 2), 0)))
        T1_s = float(np.sqrt(max(np.sum(p_T1 * (self.T1_vals - T1_m) ** 2), 0)))
        T2_s = float(np.sqrt(max(np.sum(p_T2 * (self.T2_vals - T2_m) ** 2), 0)))
        return {"dw": (dw_m, dw_s), "T1": (T1_m, T1_s), "T2": (T2_m, T2_s)}


# ──────────────────────────────────────────────────────────────────────
#  Adaptive joint loop
# ──────────────────────────────────────────────────────────────────────

def adaptive_joint_grid(qubit, counter, drive_freq, amp_pi, amp_pi_half,
                        pulse_length=48e-9,
                        max_steps=600,
                        rel_thresh=0.05,
                        dw_std_thresh=2 * np.pi * 80e3,
                        verbose=False):
    """Run the adaptive grid filter.  Returns (estimator, history,
    f_precise, T1, T2_star, T_phi, n_shots)."""
    # Build Gaussian waveforms (separately for pi and pi/2).
    t_pi,   wf_pi_base   = compile_gauss(pulse_length, amp=amp_pi)
    t_half, wf_half_base = compile_gauss(pulse_length, amp=amp_pi_half)

    # ── DRAG calibration on the pi/2 pulse (lean budget) ────────────
    beta_opt, _, _ = calibrate_drag_beta(qubit, drive_freq,
                                         t_half, wf_half_base,
                                         n_betas=7, shots_per_beta=120)
    counter.total += 7 * 120
    wf_half = make_drag(t_half, wf_half_base, beta_opt)
    wf_pi   = wf_pi_base                # plain Gaussian for pi (DRAG overcorrects)

    # ── pulse-contrast calibration (lean) ───────────────────────────
    CAL = 300
    qubit.reset()
    qubit.evolve(t_pi, wf_pi, drive_freq=drive_freq)
    p1_pi = float(qubit.measure(shots=CAL).mean())
    counter.total += CAL
    C_T1 = float(np.clip((p1_pi - 0.5) / (1.0 - 0.5 - 0.02), 0.01, 1.0))

    qubit.reset()
    qubit.evolve(t_half, wf_half, drive_freq=drive_freq)
    qubit.evolve(t_half, wf_half, drive_freq=drive_freq)
    p1_2half = float(qubit.measure(shots=CAL).mean())
    counter.total += CAL
    C_ram = float(np.clip((p1_2half - 0.5) / (1.0 - 0.5 - 0.02), 0.01, 1.0))

    if verbose:
        print(f"  [Cal] beta={beta_opt*1e9:.2f}ns  C_T1={C_T1:.3f}  C_ram={C_ram:.3f}")

    est = GridEstimator(C_T1=C_T1, C_ramsey=C_ram)
    history = []
    # Two refinement zooms: early after δω locks, late after envelope settles.
    refine_at = {180, 380}

    for step in range(max_steps):
        exp_type, tau = est.select(n_cands=8)

        qubit.reset()
        if exp_type == "T1":
            qubit.evolve(t_pi, wf_pi, drive_freq=drive_freq)
            qubit.wait(tau)
        else:
            qubit.evolve(t_half, wf_half, drive_freq=drive_freq)
            n_w = max(50, int(tau / 200e-9))
            t_w = np.linspace(0.0, tau, n_w)
            qubit.evolve(t_w, np.zeros(n_w, dtype=np.complex128),
                         drive_freq=drive_freq)
            qubit.evolve(t_half, wf_half, drive_freq=drive_freq)
        m = int(qubit.measure(shots=1)[0])
        counter.total += 1
        est.update(exp_type, tau, m)

        s = est.estimates()
        history.append({"step": step, "type": exp_type, "tau": tau,
                        "m": m,
                        "T1": s["T1"][0], "T1_std": s["T1"][1],
                        "T2": s["T2"][0], "T2_std": s["T2"][1],
                        "dw": s["dw"][0], "dw_std": s["dw"][1]})

        # Two-stage refinement (180/380) if δω is well-localised.
        if (step + 1) in refine_at and s["dw"][1] < 2 * np.pi * 500e3:
            est.refine(n_sigma=3.5)
            if verbose:
                print(f"  ── refined at step {step+1}")

        rel_T1 = s["T1"][1] / max(s["T1"][0], 1e-9)
        rel_T2 = s["T2"][1] / max(s["T2"][0], 1e-9)
        if (rel_T1 < rel_thresh and rel_T2 < rel_thresh
                and s["dw"][1] < dw_std_thresh):
            if verbose:
                print(f"  Converged at step {step + 1}")
            break

    s = est.estimates()
    dw_m, _ = s["dw"]; T1_m, _ = s["T1"]; T2_m, _ = s["T2"]
    f_precise = drive_freq + dw_m / (2 * np.pi)
    inv_Tphi = 1.0 / T2_m - 1.0 / (2.0 * T1_m)
    T_phi = (1.0 / inv_Tphi) if inv_Tphi > 0 else float("inf")
    return est, history, f_precise, T1_m, T2_m, T_phi, len(history)


# ──────────────────────────────────────────────────────────────────────
#  High-level pipeline (compatible with print_five_way)
# ──────────────────────────────────────────────────────────────────────

def run_joint_grid(seed: int = 42, verbose: bool = False) -> dict:
    """Run the full pipeline:  spec  ->  Rabi  ->  joint adaptive grid
    ->  active reset.  Returns a result dict with the same key set as
    run_joint_optimized() so it can drop into print_five_way."""
    t0 = time.time()
    q = VirtualQubit(seed=seed)
    c = ShotCounter()

    # 1. spectroscopy (chirp; cheap).
    f_q_spec, _ = chirp_spectroscopy(q, c)
    shots_spec = c.total

    # 2. Bayesian Rabi for amp_pi / amp_pi_half.
    amp_pi, amp_pi_half = bayesian_rabi(
        q, c, f_q_spec, n_iterations=10, shots_per_iter=20, n_particles=4000,
    )
    shots_rabi = c.total - shots_spec

    # 3. adaptive grid joint estimator with DRAG.
    # Drive offset -1 MHz centres the one-sided posterior (0..dw_half=2MHz)
    # exactly like Luis's reference (his spec is binary-search accurate so
    # he uses zero offset; ours is chirp-based with ~1 MHz typical error).
    drive_freq = f_q_spec - 1.0e6
    pre_grid = c.total
    _, history, f_precise, T1, T2_star, T_phi, n_steps = adaptive_joint_grid(
        q, c, drive_freq, amp_pi, amp_pi_half, verbose=verbose,
    )
    shots_grid = c.total - pre_grid

    # 4. active reset verification (30 trials, count successes within
    #    max_attempts).
    pre_ar = c.total
    t_pi_ar, wf_pi_ar = compile_gauss(48e-9, amp=amp_pi)
    AR_TRIALS = 30
    ar_success = 0
    for _ in range(AR_TRIALS):
        # prepare an excited state then try to reset
        q.reset()
        q.evolve(t_pi_ar, wf_pi_ar, drive_freq=f_precise)
        c.total += 1
        att = active_reset(q, t_pi_ar, wf_pi_ar, f_precise)
        c.total += att
        # final measurement to verify ground state
        if int(q.measure(shots=1)[0]) == 0:
            ar_success += 1
        c.total += 1
    ar_rate = ar_success / AR_TRIALS
    shots_ar = c.total - pre_ar

    # 5. pi-pulse verification.
    q.reset()
    q.evolve(t_pi_ar, wf_pi_ar, drive_freq=f_precise)
    p1_pi_verify = float(q.measure(shots=200).mean())
    c.total += 200

    elapsed = time.time() - t0

    return {
        "method":         "M7_GridDRAG",
        "f_qubit":        f_precise,
        "f_qubit_spec":   f_q_spec,
        "amp_pi":         amp_pi,
        "amp_pi_half":    amp_pi_half,
        "T1":             T1,
        "T2_star":        T2_star,
        "T_phi":          T_phi,
        "p1_pi_verify":   p1_pi_verify,
        "ar_success_rate": ar_rate,
        "shots_total":    int(c.total),
        "shots_spec":     int(shots_spec),
        "shots_rabi":     int(shots_rabi),
        "shots_t1":       int(sum(1 for h in history if h["type"] == "T1")),
        "shots_ramsey":   int(sum(1 for h in history if h["type"] == "ramsey")),
        "shots_ar":       int(shots_ar),
        "wall_time":      elapsed,
        "n_steps":        int(n_steps),
        "history":        history,
    }


# ──────────────────────────────────────────────────────────────────────
#  Standalone smoke test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    truth = _get_ground_truth(args.seed)
    print(f"ground truth (seed {args.seed}):")
    print(f"  f_q  = {truth['f_qubit']/1e9:.6f} GHz")
    print(f"  T1   = {truth['T1']*1e6:.2f} us")
    print(f"  T2*  = {truth['T2_star']*1e6:.2f} us")
    print(f"  T_phi= {truth['T_phi']*1e6:.2f} us")
    print()

    r = run_joint_grid(seed=args.seed, verbose=args.verbose)
    print()
    print(f"M7 GridDRAG results:")
    print(f"  f_q  = {r['f_qubit']/1e9:.6f} GHz   "
          f"(err {(r['f_qubit']-truth['f_qubit'])/1e6:+.3f} MHz)")
    print(f"  T1   = {r['T1']*1e6:.2f} us       "
          f"(err {abs(r['T1']-truth['T1'])/truth['T1']*100:.1f}%)")
    print(f"  T2*  = {r['T2_star']*1e6:.2f} us       "
          f"(err {abs(r['T2_star']-truth['T2_star'])/truth['T2_star']*100:.1f}%)")
    print(f"  shots total = {r['shots_total']}  ({r['n_steps']} adaptive)")
    print(f"  AR success rate = {r['ar_success_rate']*100:.1f}%")
    print(f"  pi verify P1 = {r['p1_pi_verify']:.3f}")
    print(f"  wall time = {r['wall_time']:.1f}s")
