from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import qutip as qt


# An Embedder lifts a single-qubit operator into a larger (multi-qubit) Hilbert space.
Embedder = Callable[[qt.Qobj], qt.Qobj]

_IDENTITY_EMBED: Embedder = lambda O: O

# Default time resolution for wait() free-evolution segments.
_WAIT_STEPS = 200


class VirtualQubit:
    """A single-qubit simulator whose physical parameters are hidden from the caller.

    Hidden parameters (set at construction and never exposed directly):
        _fq     : qubit resonance frequency [Hz]
        _T1     : energy-relaxation time [s]
        _T2     : dephasing time [s]
        _omega  : drive-to-rotation calibration constant [rad / (V·s)]
        _ro_err : single-shot readout bit-flip probability

    The qubit state is stored in ``self.state`` (a density matrix) and updated
    in-place by `evolve`, `wait`, and `reset`.  Calls chain:

        qubit.reset()
        qubit.evolve(t1, wave1, drive_freq)   # |0⟩ → ρ₁
        qubit.wait(200e-9)                    # ρ₁ → ρ₂  (free decay)
        qubit.evolve(t2, wave2, drive_freq)   # ρ₂ → ρ₃
        bits = qubit.measure(shots=1000)      # sample from ρ₃

    Call `reset` to explicitly return the qubit to |0⟩ before starting
    a new experiment.
    """

    # Ground state |0⟩ as a density matrix — shared across instances.
    _GROUND_STATE: qt.Qobj = qt.basis(2, 0).proj()

    def __init__(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self._fq: float = rng.uniform(5.0e9, 6.0e9)
        self._T1: float = rng.uniform(20e-6, 40e-6)
        self._T2: float = rng.uniform(15e-6, 25e-6)
        self._omega: float = rng.uniform(0.9, 1.1) * (np.pi / 6.27e-8)
        self._ro_err: float = 0.02

        # Current quantum state; starts in |0⟩.
        self.state: qt.Qobj = self._GROUND_STATE

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the qubit to the ground state |0⟩.

        Call this at the beginning of each new experiment to discard any
        previously accumulated state.
        """
        self.state = self._GROUND_STATE

    # ------------------------------------------------------------------
    # Hamiltonian / Lindblad helpers
    # ------------------------------------------------------------------

    def hamiltonian_terms(
        self,
        wave: npt.NDArray[np.complex128],
        drive_freq: float,
        embed: Embedder = _IDENTITY_EMBED,
    ) -> list:
        """Build the rotating-frame Hamiltonian as a QuTiP time-dependent list.

        The drive rotates at ``drive_freq``, so the qubit detuning enters as a
        static Z term.  The in-phase (I) and quadrature (Q) components of the
        waveform drive σx and σy respectively, scaled by ``_omega``.

        Args:
            wave:       Complex envelope sampled on the same time grid used by
                        `evolve`.  ``wave.real`` is the I component;
                        ``wave.imag`` is the Q component.
            drive_freq: LO + IF frequency of the drive signal [Hz].
            embed:      Optional operator embedding for multi-qubit contexts.
                        Defaults to the identity (single-qubit simulation).

        Returns:
            A list in QuTiP's ``[H0, [H1, c1(t)], ...]`` format.
        """
        detuning = 2 * np.pi * (self._fq - drive_freq)
        drive_x = wave.real * self._omega  # I component → σx drive
        drive_y = wave.imag * self._omega  # Q component → σy drive
        return [
            0.5 * detuning * embed(qt.sigmaz()),
            [0.5 * embed(qt.sigmax()), drive_x],
            [0.5 * embed(qt.sigmay()), drive_y],
        ]

    def collapse_operators(self, embed: Embedder = _IDENTITY_EMBED) -> list[qt.Qobj]:
        """Return the Lindblad collapse operators for T1 and T2 decay.

        Args:
            embed: Optional operator embedding for multi-qubit contexts.

        Returns:
            ``[√(1/T1) σ₋,  √(1/(2T2)) σz]``
        """
        return [
            np.sqrt(1 / self._T1) * embed(qt.sigmap()),
            np.sqrt(1 / (2 * self._T2)) * embed(qt.sigmaz()),
        ]

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def evolve(
        self,
        t: npt.NDArray[np.float64],
        wave: npt.NDArray[np.complex128],
        drive_freq: float,
    ) -> None:
        """Propagate the qubit forward under a drive pulse, updating ``self.state``.

        The propagation starts from the current ``self.state``, so successive
        calls chain automatically.  Use `reset` to start fresh from |0⟩.

        The time grid ``t`` should start at 0 — it is interpreted as relative
        time within the pulse, not absolute lab time.

        Args:
            t:          Time grid [s], shape ``(N,)``, starting at 0.
            wave:       Complex drive envelope on the same grid, shape ``(N,)``.
            drive_freq: Drive frequency [Hz].
        """
        H = self.hamiltonian_terms(wave, drive_freq)
        c_ops = self.collapse_operators()
        result = qt.mesolve(H, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    def wait(self, duration: float, n_steps: int = _WAIT_STEPS) -> None:
        """Let the qubit decay freely for ``duration`` seconds under zero drive.

        No pulse is applied — only the T1 and T2 Lindblad channels act.  This
        is useful for modelling idle time between gates, readout latency, or
        any period where the qubit is left undriven.

        Like `evolve`, this updates ``self.state`` in-place so calls chain.

        Args:
            duration: Idle time [s].
            n_steps:  Number of time steps for the mesolve integration.
                      More steps give higher accuracy for long idles; the
                      default of 200 is sufficient for durations up to ~T1.
        """
        t = np.linspace(0, duration, n_steps)
        H_zero = qt.Qobj(np.zeros((2, 2)))
        c_ops = self.collapse_operators()
        result = qt.mesolve(H_zero, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, shots: int = 1) -> npt.NDArray[np.uint8]:
        """Sample computational-basis outcomes from the current state.

        Reads ``self.state`` at call time, so the result always reflects the
        most recent `evolve` or `wait`.  Can be called multiple
        times after a single evolution without side effects on the state.

        Args:
            shots: Number of single-shot measurements to draw.

        Returns:
            Array of shape ``(shots,)`` with values in ``{0, 1}``.
        """
        p1 = float(np.clip((qt.basis(2, 1).proj() * self.state).tr().real, 0, 1))
        bits = (np.random.rand(shots) < p1).astype(np.uint8)
        return self._apply_readout_error(bits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_readout_error(
        self, bits: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Randomly flip each bit with probability ``_ro_err``."""
        flips = (np.random.rand(*bits.shape) < self._ro_err).astype(np.uint8)
        return bits ^ flips
