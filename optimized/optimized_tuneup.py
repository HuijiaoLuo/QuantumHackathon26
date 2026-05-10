"""
Optimized Qubit Tune-Up: Chirp Spectroscopy + Bayesian Inference
================================================================

This script implements all 5 preparatory tasks with two key optimizations:
  1. Chirp spectroscopy instead of point-by-point frequency sweep
  2. Bayesian particle filter for adaptive Rabi, T1, and Ramsey

Then runs the classical pipeline for direct comparison and prints a
detailed benchmark table.

Usage:
    cd ZI_EQH_2026
    source .venv/bin/activate
    python optimized/optimized_tuneup.py
"""

import time
import numpy as np
from scipy.optimize import curve_fit
import logging

from laboneq.simple import *
from laboneq.simulator.output_simulator import OutputSimulator
from qubit import VirtualQubit

logging.getLogger("laboneq").setLevel(logging.WARNING)

# ─── Global LabOne Q session ────────────────────────────────────────────────
def _make_session():
    desc = DeviceSetup("setup")
    desc.add_dataserver(host="localhost", port="8004")
    desc.add_instruments(SHFSG("dev", address="DEV12345"))
    desc.add_connections(
        "dev",
        create_connection(to_signal="q0/drive", ports="SGCHANNELS/0/OUTPUT"),
    )
    s = Session(desc)
    s.connect(do_emulation=True)
    return s

SESSION = _make_session()
CHANNEL = "dev/sgchannels_0_output"

def compile_const(length, amp=1.0):
    """Compile a constant pulse and return (t, wf)."""
    exp = Experiment()
    exp.add_signal("drive", connect_to="q0/drive")
    with exp.acquire_loop_rt(count=1):
        with exp.section():
            exp.play(
                signal="drive",
                pulse=pulse_library.const(length=length, amplitude=amp),
            )
    c = SESSION.compile(exp)
    sim = OutputSimulator(c)
    snip = sim.get_snippet(
        CHANNEL, start=0, output_length=sim.max_output_length, get_wave=True
    )
    return snip.time, snip.wave


# ═══════════════════════════════════════════════════════════════════════════
#                       OPTIMIZED  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class ShotCounter:
    """Track total shots fired."""
    def __init__(self):
        self.total = 0
    def measure(self, qubit, shots):
        self.total += shots
        return qubit.measure(shots=shots)


# ── Task 1: Chirp Spectroscopy ──────────────────────────────────────────────

def chirp_spectroscopy(qubit, counter,
                       f_start=5.0e9, f_end=6.0e9,
                       T_chirp=10e-6, amp=0.8,
                       n_bisect=8, shots_per_point=30):
    """Find qubit frequency via chirp + sigmoid refinement.

    Stage 1: Coarse bisection on chirp end-time to narrow the resonance
             to a ~50 MHz window.
    Stage 2: Fine scan within the window + sigmoid fit for sub-MHz accuracy.

    Returns:
        f_qubit estimate in Hz, plateau_height (for Rabi prior seeding)
    """
    f_center = (f_start + f_end) / 2
    f_off0 = f_start - f_center
    chirp_rate = (f_end - f_start) / T_chirp
    N_samples = int(T_chirp * 2e9)  # 2 GHz sampling

    def _make_chirp(frac):
        T = T_chirp * frac
        n = max(int(N_samples * frac), 200)
        t = np.linspace(0, T, n)
        phase = 2 * np.pi * (f_off0 * t + 0.5 * chirp_rate * t ** 2)
        return t, amp * np.exp(1j * phase)

    # Stage 1: Coarse bisection to find approximate step location
    lo, hi = 0.0, 1.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        t_ch, wf_ch = _make_chirp(mid)
        qubit.reset()
        qubit.evolve(t_ch, wf_ch, drive_freq=f_center)
        bits = counter.measure(qubit, shots_per_point)
        p1 = bits.mean()
        if p1 > 0.45:
            hi = mid
        else:
            lo = mid

    frac_coarse = (lo + hi) / 2
    f_coarse = f_start + frac_coarse * (f_end - f_start)

    # Stage 2: Fine scan around coarse estimate + sigmoid fit
    span = 40e6  # +/- 20 MHz
    f_lo = max(f_coarse - span / 2, f_start)
    f_hi = min(f_coarse + span / 2, f_end)
    n_fine = 15
    fracs_fine = np.linspace(
        (f_lo - f_start) / (f_end - f_start),
        (f_hi - f_start) / (f_end - f_start),
        n_fine,
    )
    P1_fine = []
    freqs_fine = f_start + fracs_fine * (f_end - f_start)
    for frac in fracs_fine:
        t_ch, wf_ch = _make_chirp(frac)
        qubit.reset()
        qubit.evolve(t_ch, wf_ch, drive_freq=f_center)
        P1_fine.append(counter.measure(qubit, shots_per_point).mean())
    P1_fine = np.array(P1_fine)

    # Measure the plateau height (for Landau-Zener Rabi prior)
    plateau_height = np.mean(P1_fine[P1_fine > 0.5]) if np.any(P1_fine > 0.5) else 0.5

    # Sigmoid fit: P1 = A / (1 + exp(-k*(f - f0))) + offset
    try:
        from scipy.optimize import curve_fit as _cf

        def _sigmoid(x, x0, k, A, off):
            return A / (1 + np.exp(-k * (x - x0))) + off

        popt, _ = _cf(
            _sigmoid, freqs_fine, P1_fine,
            p0=[f_coarse, 1e-7, 0.9, 0.02],
            maxfev=5000,
        )
        f_qubit = popt[0]
    except Exception:
        # Fallback: use midpoint
        half_max = (max(P1_fine) + min(P1_fine)) / 2
        cross_idx = np.argmin(np.abs(P1_fine - half_max))
        f_qubit = freqs_fine[cross_idx]

    return f_qubit, plateau_height


# ── Task 2: Bayesian Amplitude Rabi ─────────────────────────────────────────

def bayesian_rabi(qubit, counter, f_drive, pulse_length=48e-9,
                  n_iterations=25, shots_per_iter=20, n_particles=5000,
                  omega_prior=None):
    """Infer pi-pulse amplitude via Bayesian particle filter.

    Information-optimal strategy:
      - Few shots per iteration (20) but informative amp choice
      - "Long-arm" measurements: amp = k*pi/omega for k=1,3,5,...
        Multi-period probing amplifies small omega errors -> tighter posterior
      - Adaptive amp = (n+0.5)*pi/omega chosen to maximize Fisher info
        (steepest slope of sin^2)

    Args:
        omega_prior: If provided, (center, width) tuple from chirp LZ analysis.

    Returns:
        amp_pi, amp_pi_half
    """
    t_p, wf_unit = compile_const(pulse_length)

    if omega_prior is not None:
        center, width = omega_prior
        particles = np.random.normal(center, width, n_particles)
        particles = np.clip(particles, 0.3, 15)
    else:
        particles = np.random.uniform(0.5, 8.0, n_particles)
    weights = np.ones(n_particles) / n_particles
    A_vis, bg = 0.85, 0.02

    for iteration in range(n_iterations):
        omega_mean = np.average(particles, weights=weights)
        omega_std = np.sqrt(
            np.average((particles - omega_mean) ** 2, weights=weights)
        )

        # Information-optimal amp selection:
        # Probe at amp ~ (k + 0.5)*pi/omega -> sin^2 = 0.5 (max slope -> max Fisher info)
        # Use k=0 (pi/2 point) early, then larger k for finer resolution
        # because n*omega*amp accumulates phase, multiplying error sensitivity
        rel_unc = omega_std / max(omega_mean, 0.1)
        if iteration < 4:
            # Initial exploration: pi/2 and pi points to anchor the model
            amp_test = (np.pi / 2 if iteration % 2 == 0 else np.pi) / omega_mean
        elif rel_unc > 0.05:
            # Medium uncertainty: use 1.5*pi (still single-period unambiguous)
            amp_test = 1.5 * np.pi / omega_mean
        else:
            # Low uncertainty: use long-arm probes 3.5*pi or 5.5*pi
            # These multiply the error by 7x or 11x -> very tight likelihood
            k = 3 if iteration % 2 == 0 else 5
            amp_test = (k + 0.5) * np.pi / omega_mean
        amp_test = np.clip(amp_test, 0.05, 3.5)

        qubit.reset()
        qubit.evolve(t_p, wf_unit * amp_test, drive_freq=f_drive)
        bits = counter.measure(qubit, shots_per_iter)
        k_meas = int(bits.sum())

        p1_model = A_vis * np.sin(particles * amp_test / 2) ** 2 + bg
        p1_model = np.clip(p1_model, 1e-6, 1 - 1e-6)
        # Use log-likelihood to avoid underflow with many shots
        log_lik = (k_meas * np.log(p1_model)
                   + (shots_per_iter - k_meas) * np.log(1 - p1_model))
        log_lik -= log_lik.max()
        weights *= np.exp(log_lik)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(n_particles) / n_particles

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < n_particles / 2:  # Resample more aggressively
            idx = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx] + np.random.normal(
                0, max(omega_std * 0.01, 0.002), n_particles
            )
            particles = np.clip(particles, 0.1, 15)
            weights = np.ones(n_particles) / n_particles

    omega_final = np.average(particles, weights=weights)
    amp_pi = np.pi / omega_final
    return amp_pi, amp_pi / 2


# ── Task 3: Bayesian T1 ─────────────────────────────────────────────────────

def bayesian_t1(qubit, counter, f_drive, amp_pi, pulse_length=48e-9,
                n_iterations=18, shots_per_iter=30, n_particles=5000):
    """Infer T1 via Bayesian particle filter.

    Information-optimal strategy:
      - Optimal delay for exp(-tau/T1) is tau ≈ 1.27 * T1 (analytical Fisher max)
      - Use 3 delays per "round": one near 1.27*T1 (max info), one short
        (anchors amplitude A), one long (anchors offset)
      - Far fewer iterations because each measurement is highly informative
    Returns:
        T1 in seconds
    """
    t_p, wf_unit = compile_const(pulse_length)
    wf_pi = wf_unit * amp_pi

    particles = np.random.uniform(1e-6, 80e-6, n_particles)
    weights = np.ones(n_particles) / n_particles
    A_vis, bg = 0.85, 0.02

    for iteration in range(n_iterations):
        T1_mean = np.average(particles, weights=weights)
        T1_std = np.sqrt(
            np.average((particles - T1_mean) ** 2, weights=weights)
        )

        # 3-point cycle: anchor + Fisher-optimal + baseline
        cycle = iteration % 3
        if cycle == 0:
            delay = T1_mean * 1.27   # Fisher-optimal point
        elif cycle == 1:
            delay = T1_mean * 0.25   # Anchor amplitude A
        else:
            delay = T1_mean * 2.5    # Anchor baseline / late decay
        delay = np.clip(delay, 0.5e-6, 200e-6)

        qubit.reset()
        qubit.evolve(t_p, wf_pi, drive_freq=f_drive)
        if delay > 0:
            qubit.wait(delay)
        bits = counter.measure(qubit, shots_per_iter)
        k = int(bits.sum())

        p1_model = A_vis * np.exp(-delay / particles) + bg
        p1_model = np.clip(p1_model, 1e-6, 1 - 1e-6)
        log_lik = (k * np.log(p1_model)
                   + (shots_per_iter - k) * np.log(1 - p1_model))
        log_lik -= log_lik.max()
        weights *= np.exp(log_lik)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(n_particles) / n_particles

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < n_particles / 2:
            idx = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx] + np.random.normal(
                0, max(T1_std * 0.01, 0.03e-6), n_particles
            )
            particles = np.clip(particles, 0.1e-6, 200e-6)
            weights = np.ones(n_particles) / n_particles

    return np.average(particles, weights=weights)


# ── Task 4: Active Reset ────────────────────────────────────────────────────

def active_reset(qubit, t_pi, wf_pi, f_drive, max_attempts=5):
    """Measure-and-flip reset. Returns number of attempts."""
    for attempt in range(max_attempts):
        if qubit.measure(shots=1)[0] == 0:
            return attempt + 1
        qubit.evolve(t_pi, wf_pi, drive_freq=f_drive)
    return max_attempts


# ── Task 5: Bayesian Ramsey ──────────────────────────────────────────────────

def bayesian_ramsey(qubit, counter, f_spec, amp_pi_half, pulse_length=48e-9,
                    detuning=2e6, n_iterations=22, shots_per_iter=20,
                    n_particles=5000):
    """Infer precise qubit frequency and T2* via Bayesian Ramsey.

    Uses PGH (Particle Guess Heuristic): tau = 1/|d1 - d2| where d1, d2 are
    two random samples from the current posterior. This is provably
    information-optimal for sinusoidal models (Wiebe & Granade 2016).

    Two phases interleaved:
      - Frequency: PGH tau, mostly short
      - T2*: long taus near T2 estimate (every 4th iteration)

    Returns:
        f_qubit_precise, T2_star
    """
    t_p, wf_unit = compile_const(pulse_length)
    wf_half = wf_unit * amp_pi_half
    f_drive = f_spec + detuning

    p_delta = np.random.uniform(0.1e6, 5e6, n_particles)
    p_T2 = np.random.uniform(0.5e-6, 50e-6, n_particles)
    weights = np.ones(n_particles) / n_particles

    for iteration in range(n_iterations):
        delta_mean = np.average(p_delta, weights=weights)
        delta_std = np.sqrt(
            np.average((p_delta - delta_mean) ** 2, weights=weights)
        )
        T2_mean = np.average(p_T2, weights=weights)

        # Every 4th iteration: T2* probe (long tau)
        if iteration >= 6 and iteration % 4 == 3:
            tau = T2_mean * (0.8 + 0.6 * np.random.random())
        else:
            # PGH: tau = 1 / (2*pi * |d1 - d2|) for two posterior samples
            idx = np.random.choice(n_particles, size=2, p=weights)
            d1, d2 = p_delta[idx[0]], p_delta[idx[1]]
            diff = abs(d1 - d2)
            if diff < 1e3:
                tau = 1.0 / (2 * np.pi * max(delta_std, 1e3))
            else:
                tau = 1.0 / (2 * np.pi * diff)
        # Clip to physically sensible range, also < ~T2 to keep visibility
        tau = np.clip(tau, 50e-9, min(3 * T2_mean, 30e-6))

        qubit.reset()
        qubit.evolve(t_p, wf_half, drive_freq=f_drive)
        if tau > 0:
            qubit.wait(tau)
        qubit.evolve(t_p, wf_half, drive_freq=f_drive)
        bits = counter.measure(qubit, shots_per_iter)
        k = int(bits.sum())

        p1_model = 0.5 + 0.5 * np.cos(
            2 * np.pi * p_delta * tau
        ) * np.exp(-tau / p_T2)
        p1_model = np.clip(p1_model, 1e-6, 1 - 1e-6)
        log_lik = (k * np.log(p1_model)
                   + (shots_per_iter - k) * np.log(1 - p1_model))
        log_lik -= log_lik.max()
        weights *= np.exp(log_lik)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(n_particles) / n_particles

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < n_particles / 2:
            idx = np.random.choice(n_particles, size=n_particles, p=weights)
            p_delta = p_delta[idx] + np.random.normal(
                0, max(delta_std * 0.01, 200), n_particles
            )
            T2_std = np.sqrt(
                np.average((p_T2 - np.average(p_T2, weights=weights)) ** 2,
                           weights=weights)
            )
            p_T2 = p_T2[idx] + np.random.normal(
                0, max(T2_std * 0.01, 0.03e-6), n_particles
            )
            p_delta = np.clip(p_delta, 0, 10e6)
            p_T2 = np.clip(p_T2, 0.1e-6, 100e-6)
            weights = np.ones(n_particles) / n_particles

    delta_final = np.average(p_delta, weights=weights)
    T2_final = np.average(p_T2, weights=weights)
    f_qubit = f_drive - delta_final
    return f_qubit, T2_final


# ═══════════════════════════════════════════════════════════════════════════
#                       CLASSICAL  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def classical_spectroscopy(qubit, counter, shots=2000):
    """Coarse + fine spectroscopy with Lorentzian fit."""
    t_sp, wf_sp = compile_const(5e-6)

    # Coarse
    freqs_c = np.linspace(5.0e9, 6.0e9, 51)
    P1_c = []
    for f in freqs_c:
        qubit.reset()
        qubit.evolve(t_sp, wf_sp, drive_freq=f)
        P1_c.append(counter.measure(qubit, shots).mean())
    f_rough = freqs_c[np.argmax(P1_c)]

    # Fine
    freqs_f = np.linspace(f_rough - 100e6, f_rough + 100e6, 101)
    P1_f = []
    for f in freqs_f:
        qubit.reset()
        qubit.evolve(t_sp, wf_sp, drive_freq=f)
        P1_f.append(counter.measure(qubit, shots).mean())
    P1_f = np.array(P1_f)

    def lorentzian(f, f0, g, A, off):
        return A * g ** 2 / ((f - f0) ** 2 + g ** 2) + off

    pi = np.argmax(P1_f)
    popt, _ = curve_fit(
        lorentzian, freqs_f, P1_f,
        p0=[freqs_f[pi], 10e6, P1_f[pi] - np.median(P1_f), np.median(P1_f)],
        bounds=([freqs_f[0], 0, 0, 0], [freqs_f[-1], np.inf, np.inf, 1]),
        maxfev=10000,
    )
    return popt[0]


def classical_rabi(qubit, counter, f_drive, pulse_length=48e-9,
                   n_points=81, shots=3000):
    """Classical amplitude Rabi with sin^2 fit."""
    t_p, wf_unit = compile_const(pulse_length)
    amps = np.linspace(0.01, 2.0, n_points)
    P1 = []
    for a in amps:
        qubit.reset()
        qubit.evolve(t_p, wf_unit * a, drive_freq=f_drive)
        P1.append(counter.measure(qubit, shots).mean())
    P1 = np.array(P1)

    def rabi(amp, A, w, ph, off):
        return A * np.sin(w * amp / 2 + ph) ** 2 + off

    pi_idx = np.argmax(P1)
    w_g = np.pi / amps[pi_idx] if amps[pi_idx] > 0 else np.pi
    popt, _ = curve_fit(
        rabi, amps, P1,
        p0=[max(P1) - min(P1), w_g, 0, min(P1)],
        bounds=([0, 0, -np.pi, -0.1], [1, 100, np.pi, 0.5]),
        maxfev=10000,
    )
    amp_pi = (np.pi - 2 * popt[2]) / popt[1]
    return amp_pi, amp_pi / 2


def classical_t1(qubit, counter, f_drive, amp_pi, pulse_length=48e-9,
                 n_points=61, shots=3000):
    """Classical T1 with exponential fit."""
    t_p, wf_unit = compile_const(pulse_length)
    wf_pi = wf_unit * amp_pi
    delays = np.linspace(0, 120e-6, n_points)
    P1 = []
    for d in delays:
        qubit.reset()
        qubit.evolve(t_p, wf_pi, drive_freq=f_drive)
        if d > 0:
            qubit.wait(d)
        P1.append(counter.measure(qubit, shots).mean())
    P1 = np.array(P1)

    def decay(t, T1, A, off):
        return A * np.exp(-t / T1) + off

    popt, _ = curve_fit(
        decay, delays, P1,
        p0=[20e-6, P1[0] - P1[-1], P1[-1]],
        bounds=([1e-6, 0, 0], [200e-6, 1, 0.5]),
        maxfev=10000,
    )
    return popt[0]


def classical_ramsey(qubit, counter, f_spec, amp_pi_half, pulse_length=48e-9,
                     detuning=2e6, n_points=101, shots=3000):
    """Classical Ramsey with damped-cosine fit."""
    t_p, wf_unit = compile_const(pulse_length)
    wf_half = wf_unit * amp_pi_half
    f_drive = f_spec + detuning
    delays = np.linspace(0, 5e-6, n_points)
    P1 = []
    for tau in delays:
        qubit.reset()
        qubit.evolve(t_p, wf_half, drive_freq=f_drive)
        if tau > 0:
            qubit.wait(tau)
        qubit.evolve(t_p, wf_half, drive_freq=f_drive)
        P1.append(counter.measure(qubit, shots).mean())
    P1 = np.array(P1)

    def ramsey(tau, df, T2, A, off):
        return A * np.cos(2 * np.pi * df * tau) * np.exp(-tau / T2) + off

    A_g = (max(P1) - min(P1)) / 2
    popt, _ = curve_fit(
        ramsey, delays, P1,
        p0=[detuning, 5e-6, A_g, np.mean(P1)],
        bounds=([-50e6, 100e-9, 0, 0], [50e6, 200e-6, 1, 1]),
        maxfev=10000,
    )
    f_qubit = f_drive - popt[0]
    return f_qubit, popt[1]


# ═══════════════════════════════════════════════════════════════════════════
#                           MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_optimized(seed=42):
    """Run the full optimized pipeline with closed-loop refinement.

    Pipeline:
      1. Chirp spectroscopy → f_qubit (coarse), plateau_height
      2. Bayesian Rabi (seeded with omega_prior from chirp) → amp_pi
      3. Bayesian T1 → T1
      4. Active Reset verification
      5. Bayesian Ramsey → f_qubit (precise), T2*
      6. Closed-loop: re-run Rabi at corrected frequency for maximum accuracy

    Returns dict of results.
    """
    q = VirtualQubit(seed=seed)
    c = ShotCounter()
    t0 = time.time()

    # Task 1 – Chirp Spectroscopy
    f_q_chirp, plateau_height = chirp_spectroscopy(q, c)
    shots_spec = c.total

    # Compute omega_prior from Landau-Zener plateau height:
    #   P_LZ = 1 - exp(-2*pi*Omega^2 / (4*alpha))
    #   where alpha = chirp_rate * 2*pi, Omega = drive_strength
    #   => Omega = sqrt( -4*alpha * ln(1 - P_LZ) / (2*pi) )
    # We convert this Omega (in rad/s) to the Rabi frequency parameter
    # omega = Omega * pulse_length (dimensionless rotation per unit amplitude).
    # This gives a rough prior center for the Bayesian Rabi.
    chirp_rate = 1e9 / 10e-6  # (f_end - f_start) / T_chirp = 1GHz / 10us
    alpha = chirp_rate * 2 * np.pi
    P_LZ = np.clip(plateau_height, 0.05, 0.99)
    try:
        Omega_est = np.sqrt(-4 * alpha * np.log(1 - P_LZ) / (2 * np.pi))
        # Convert to omega in units of the Rabi model: omega = pi / amp_pi
        # amp_pi ~ pi / (Omega * pulse_length * scale_factor)
        # Since our model uses omega such that P1 = sin(omega * amp / 2)^2,
        # omega ~ Omega * pulse_length (approximately)
        omega_center = Omega_est * 48e-9
        omega_prior = (omega_center, omega_center * 0.3)  # 30% width
    except (ValueError, FloatingPointError):
        omega_prior = None

    # Task 2 – Bayesian Rabi (first pass, seeded with chirp prior)
    amp_pi, amp_pi_half = bayesian_rabi(q, c, f_q_chirp, omega_prior=omega_prior)
    shots_rabi = c.total - shots_spec

    # Task 3 – Bayesian T1
    T1 = bayesian_t1(q, c, f_q_chirp, amp_pi)
    shots_t1 = c.total - shots_spec - shots_rabi

    # Task 4 – Active Reset test
    t_p, wf_u = compile_const(48e-9)
    wf_pi = wf_u * amp_pi
    ar_success = 0
    ar_total_att = 0
    n_trials = 200
    for _ in range(n_trials):
        q.reset()
        q.evolve(t_p, wf_pi, drive_freq=f_q_chirp)
        att = active_reset(q, t_p, wf_pi, f_q_chirp)
        c.total += 1
        ar_total_att += att
        if q.measure(shots=1)[0] == 0:
            ar_success += 1
        c.total += 1
    ar_rate = ar_success / n_trials

    # Task 5 – Bayesian Ramsey
    shots_before_ramsey = c.total
    f_q_ramsey, T2_star = bayesian_ramsey(q, c, f_q_chirp, amp_pi_half)
    shots_ramsey = c.total - shots_before_ramsey

    # ── Closed-loop refinement ──
    # Re-run Rabi at Ramsey-corrected frequency. With a tight prior from
    # first Rabi, only a few iterations of long-arm probing are needed.
    shots_before_rerun = c.total
    amp_pi, amp_pi_half = bayesian_rabi(
        q, c, f_q_ramsey,
        n_iterations=10, shots_per_iter=20, n_particles=5000,
        omega_prior=(np.pi / amp_pi, 0.05)  # very tight prior
    )
    shots_rabi += c.total - shots_before_rerun

    elapsed = time.time() - t0

    # Verification: apply pi pulse at Ramsey frequency
    q.reset()
    q.evolve(t_p, wf_u * amp_pi, drive_freq=f_q_ramsey)
    p1_verify = q.measure(shots=5000).mean()

    return {
        "method": "Optimized (Chirp + Bayesian + Closed-Loop)",
        "f_qubit": f_q_ramsey,
        "f_qubit_spec": f_q_chirp,
        "amp_pi": amp_pi,
        "amp_pi_half": amp_pi_half,
        "T1": T1,
        "T2_star": T2_star,
        "ar_success_rate": ar_rate,
        "ar_avg_attempts": ar_total_att / n_trials,
        "p1_pi_verify": p1_verify,
        "shots_spec": shots_spec,
        "shots_rabi": shots_rabi,
        "shots_t1": shots_t1,
        "shots_ramsey": shots_ramsey,
        "shots_total": c.total,
        "wall_time": elapsed,
    }


def run_classical(seed=42):
    """Run the full classical pipeline. Returns dict of results."""
    q = VirtualQubit(seed=seed)
    c = ShotCounter()
    t0 = time.time()

    # Task 1
    f_q_spec = classical_spectroscopy(q, c)
    shots_spec = c.total

    # Task 2
    amp_pi, amp_pi_half = classical_rabi(q, c, f_q_spec)
    shots_rabi = c.total - shots_spec

    # Task 3
    T1 = classical_t1(q, c, f_q_spec, amp_pi)
    shots_t1 = c.total - shots_spec - shots_rabi

    # Task 4
    t_p, wf_u = compile_const(48e-9)
    wf_pi = wf_u * amp_pi
    ar_success = 0
    ar_total_att = 0
    n_trials = 200
    for _ in range(n_trials):
        q.reset()
        q.evolve(t_p, wf_pi, drive_freq=f_q_spec)
        att = active_reset(q, t_p, wf_pi, f_q_spec)
        c.total += 1
        ar_total_att += att
        if q.measure(shots=1)[0] == 0:
            ar_success += 1
        c.total += 1
    shots_ar = n_trials * 2
    ar_rate = ar_success / n_trials

    # Task 5
    shots_before_ramsey = c.total
    f_q_ramsey, T2_star = classical_ramsey(q, c, f_q_spec, amp_pi_half)
    shots_ramsey = c.total - shots_before_ramsey

    elapsed = time.time() - t0

    # Verification
    q.reset()
    q.evolve(t_p, wf_u * amp_pi, drive_freq=f_q_ramsey)
    p1_verify = q.measure(shots=5000).mean()

    return {
        "method": "Classical (Sweep + Fit)",
        "f_qubit": f_q_ramsey,
        "f_qubit_spec": f_q_spec,
        "amp_pi": amp_pi,
        "amp_pi_half": amp_pi_half,
        "T1": T1,
        "T2_star": T2_star,
        "ar_success_rate": ar_rate,
        "ar_avg_attempts": ar_total_att / n_trials,
        "p1_pi_verify": p1_verify,
        "shots_spec": shots_spec,
        "shots_rabi": shots_rabi,
        "shots_t1": shots_t1,
        "shots_ramsey": shots_ramsey,
        "shots_total": c.total,
        "wall_time": elapsed,
    }


def print_comparison(opt, cls):
    """Print a detailed side-by-side comparison table."""
    W = 62

    def row(label, v_opt, v_cls, unit="", fmt=".4f", ratio=True):
        s_opt = f"{v_opt:{fmt}}" if isinstance(v_opt, float) else str(v_opt)
        s_cls = f"{v_cls:{fmt}}" if isinstance(v_cls, float) else str(v_cls)
        r = ""
        if ratio and isinstance(v_opt, (int, float)) and v_cls != 0:
            factor = v_cls / v_opt if v_opt != 0 else float("inf")
            r = f"  ({factor:.1f}x)" if factor > 1 else ""
        line = f"  {label:<24s} {s_opt:>12s} {s_cls:>12s}{r}"
        if unit:
            line += f" {unit}"
        print(line)

    print()
    print("=" * W)
    print("         BENCHMARK: OPTIMIZED  vs  CLASSICAL")
    print("=" * W)
    print(f"  {'':24s} {'Optimized':>12s} {'Classical':>12s}")
    print("-" * W)

    print("  ── Qubit Parameters ──")
    row("f_qubit (Ramsey)", opt["f_qubit"] / 1e9, cls["f_qubit"] / 1e9, "GHz", ".6f", False)
    row("f_qubit (Spec only)", opt["f_qubit_spec"] / 1e9, cls["f_qubit_spec"] / 1e9, "GHz", ".6f", False)
    row("amp_pi", opt["amp_pi"], cls["amp_pi"])
    row("amp_pi_half", opt["amp_pi_half"], cls["amp_pi_half"])
    row("T1", opt["T1"] * 1e6, cls["T1"] * 1e6, "us", ".2f", False)
    row("T2*", opt["T2_star"] * 1e6, cls["T2_star"] * 1e6, "us", ".2f", False)
    row("Pi-pulse P1 (verify)", opt["p1_pi_verify"], cls["p1_pi_verify"], "", ".4f", False)
    row("Active reset rate", opt["ar_success_rate"] * 100, cls["ar_success_rate"] * 100, "%", ".1f", False)

    print()
    print("  ── Shot Counts ──")
    row("Spectroscopy", opt["shots_spec"], cls["shots_spec"], "shots", "d")
    row("Rabi", opt["shots_rabi"], cls["shots_rabi"], "shots", "d")
    row("T1", opt["shots_t1"], cls["shots_t1"], "shots", "d")
    row("Ramsey", opt["shots_ramsey"], cls["shots_ramsey"], "shots", "d")
    print("-" * W)
    row("TOTAL SHOTS", opt["shots_total"], cls["shots_total"], "shots", "d")

    print()
    print("  ── Wall-Clock Time ──")
    row("Total time", opt["wall_time"], cls["wall_time"], "s", ".1f")

    speedup_shots = cls["shots_total"] / max(opt["shots_total"], 1)
    speedup_time = cls["wall_time"] / max(opt["wall_time"], 0.01)
    print()
    print(f"  >>> Shot reduction:  {speedup_shots:.1f}x fewer measurements")
    print(f"  >>> Wall-clock speedup: {speedup_time:.1f}x faster")
    print("=" * W)


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEED = 42
    print("Running OPTIMIZED pipeline...")
    res_opt = run_optimized(seed=SEED)
    print(f"  Done. {res_opt['shots_total']} total shots in {res_opt['wall_time']:.1f}s\n")

    print("Running CLASSICAL pipeline...")
    res_cls = run_classical(seed=SEED)
    print(f"  Done. {res_cls['shots_total']} total shots in {res_cls['wall_time']:.1f}s\n")

    print_comparison(res_opt, res_cls)
