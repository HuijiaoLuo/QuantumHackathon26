# test_main.py
import sys
import os
import pytest

# Ensure the 'optimized' and root directory are in the path
sys.path.insert(0, os.path.abspath('optimized'))
sys.path.insert(0, os.path.abspath('.'))

from joint_estimator_grid import run_joint_grid
from joint_estimator import _get_ground_truth

def test_joint_grid_estimation():
    """
    Test the M5 joint estimation grid to ensure the estimated values 
    fall within an acceptable error margin compared to the ground truth.
    """
    SEED = 42
    truth = _get_ground_truth(SEED)
    result = run_joint_grid(seed=SEED, verbose=False)

    # 1. Check Frequency error (should be within +/- 5 MHz as an example)
    freq_error_mhz = (result['f_qubit'] - truth['f_qubit']) / 1e6
    assert abs(freq_error_mhz) < 5.0, f"Frequency error too high: {freq_error_mhz:.3f} MHz"

    # 2. Check T1 error (should be within 20% relative error)
    t1_rel_error = abs(result['T1'] - truth['T1']) / truth['T1']
    assert t1_rel_error < 0.20, f"T1 relative error too high: {t1_rel_error*100:.1f}%"

    # 3. Check Active Reset success rate (should be reasonably high)
    assert result['ar_success_rate'] > 0.80, f"Active Reset success rate too low: {result['ar_success_rate']}"

    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_joint_grid_estimation()