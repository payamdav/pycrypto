"""
kalman_fast.py
==============
High-performance 1D (scalar) Kalman Filter implemented with Numba JIT compilation.

Standard control-theory notation used throughout:
    z_k        — raw measurement at step k
    x̂_k        — posteriori state estimate at step k
    P_k        — posteriori error covariance at step k
    Q          — process noise covariance (constant)
    R          — measurement noise covariance (constant)
    K_k        — Kalman gain at step k
    x̂_k^-     — prior (predicted) state estimate at step k
    P_k^-      — prior (predicted) error covariance at step k
"""

import numpy as np
from numba import njit


@njit
def kalman_filter_step(
    measurement: float,
    prev_estimate: float,
    prev_error_cov: float,
    process_variance: float,
    measurement_variance: float,
) -> tuple:
    """
    Executes a single, stateless recursive step of a 1D Kalman Filter.
    All parameters are strictly required (no default values).

    Parameters
    ----------
    measurement : float
        The current raw observation/measurement ($z_k$).
    prev_estimate : float
        The posteriori state estimate from the previous step ($\\hat{x}_{k-1}$).
    prev_error_cov : float
        The posteriori error covariance from the previous step ($P_{k-1}$).
    process_variance : float
        The process noise covariance ($Q$).
    measurement_variance : float
        The measurement noise covariance ($R$).

    Returns
    -------
    tuple[float, float]
        current_estimate ($\\hat{x}_k$), current_error_cov ($P_k$)
    """

    # ------------------------------------------------------------------
    # PREDICT STAGE
    # ------------------------------------------------------------------

    # Prior state estimate: x̂_k^- = x̂_{k-1}
    # (identity state-transition model: F = 1, no control input)
    prior_estimate = prev_estimate

    # Prior error covariance: P_k^- = P_{k-1} + Q
    prior_error_cov = prev_error_cov + process_variance

    # ------------------------------------------------------------------
    # UPDATE STAGE — Kalman Gain
    # ------------------------------------------------------------------

    # Kalman gain: K_k = P_k^- / (P_k^- + R)
    kalman_gain = prior_error_cov / (prior_error_cov + measurement_variance)

    # ------------------------------------------------------------------
    # CORRECT STAGE
    # ------------------------------------------------------------------

    # Posteriori state estimate: x̂_k = x̂_k^- + K_k * (z_k - x̂_k^-)
    current_estimate = prior_estimate + kalman_gain * (measurement - prior_estimate)

    # Posteriori error covariance: P_k = (1 - K_k) * P_k^-
    current_error_cov = (1.0 - kalman_gain) * prior_error_cov

    return current_estimate, current_error_cov


@njit
def kalman_filter_batch(
    measurements: np.ndarray,
    initial_estimate: float,
    initial_error_cov: float,
    process_variance: float,
    measurement_variance: float,
) -> tuple:
    """
    Processes an array of sequential measurements by internally calling
    kalman_filter_step in a loop. Returns pre-allocated NumPy arrays.
    All parameters are strictly required (no default values).

    Parameters
    ----------
    measurements : np.ndarray
        1D array of sequential observations ($z$).
    initial_estimate : float
        The initial state guess at step zero ($\\hat{x}_0$).
    initial_error_cov : float
        The initial error covariance at step zero ($P_0$).
    process_variance : float
        The constant process noise covariance ($Q$) applied across the batch.
    measurement_variance : float
        The constant measurement noise covariance ($R$) applied across the batch.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        estimates (array of $\\hat{x}$), error_covariances (array of $P$)
    """

    n = len(measurements)

    # Pre-allocate output arrays — O(1) memory overhead within the compiled loop
    estimates = np.empty(n, dtype=np.float64)
    error_covariances = np.empty(n, dtype=np.float64)

    # Running state variables: est = x̂_{k-1},  cov = P_{k-1}
    est = initial_estimate
    cov = initial_error_cov

    # Sequential loop — each step is a full predict + update + correct cycle
    for k in range(n):
        # z_k is measurements[k]
        est, cov = kalman_filter_step(
            measurements[k],   # z_k
            est,               # x̂_{k-1}
            cov,               # P_{k-1}
            process_variance,  # Q
            measurement_variance,  # R
        )
        estimates[k] = est        # x̂_k
        error_covariances[k] = cov  # P_k

    return estimates, error_covariances


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("1D Kalman Filter — Numba JIT warm-up & functional test")
    print("=" * 60)

    # ---- parameters (no defaults — all explicit) ----
    Q = 1e-4   # process variance
    R = 1e-2   # measurement variance
    x0 = 0.0  # initial state estimate  (x̂_0)
    P0 = 1.0  # initial error covariance (P_0)

    # ---- mock measurement sequence ----
    rng = np.random.default_rng(42)
    true_signal = np.linspace(0.0, 1.0, 500, dtype=np.float64)
    noise = rng.normal(0.0, 0.1, size=500).astype(np.float64)
    z = true_signal + noise  # z_k sequence

    # ------------------------------------------------------------------
    # Test kalman_filter_step (single step, triggers JIT compilation)
    # ------------------------------------------------------------------
    print("\n[1] kalman_filter_step — single step (first call triggers JIT)...")
    t0 = time.perf_counter()
    x_hat, P = kalman_filter_step(z[0], x0, P0, Q, R)
    t1 = time.perf_counter()
    print(f"    z_0={z[0]:.6f}  =>  x̂_0={x_hat:.6f},  P_0={P:.6f}  [{(t1-t0)*1e3:.1f} ms incl. JIT]")

    # Second call — compiled path
    t0 = time.perf_counter()
    x_hat2, P2 = kalman_filter_step(z[1], x_hat, P, Q, R)
    t1 = time.perf_counter()
    print(f"    z_1={z[1]:.6f}  =>  x̂_1={x_hat2:.6f},  P_1={P2:.6f}  [{(t1-t0)*1e6:.1f} µs compiled]")

    # ------------------------------------------------------------------
    # Test kalman_filter_batch (full sequence, triggers JIT compilation)
    # ------------------------------------------------------------------
    print("\n[2] kalman_filter_batch — full sequence (first call triggers JIT)...")
    t0 = time.perf_counter()
    estimates, error_covariances = kalman_filter_batch(z, x0, P0, Q, R)
    t1 = time.perf_counter()
    print(f"    Processed {len(z)} measurements  [{(t1-t0)*1e3:.1f} ms incl. JIT]")

    # Second call — compiled path
    t0 = time.perf_counter()
    estimates2, _ = kalman_filter_batch(z, x0, P0, Q, R)
    t1 = time.perf_counter()
    print(f"    Re-ran on same data  [{(t1-t0)*1e6:.1f} µs compiled]")

    # Sanity: final estimate should be close to true_signal[-1] = 1.0
    print(f"\n    true_signal[-1] = {true_signal[-1]:.4f}")
    print(f"    x̂_N            = {estimates[-1]:.4f}  (should be near 1.0)")
    print(f"    P_N             = {error_covariances[-1]:.6f}  (should be small)")

    # Consistency check between step-wise and batch outputs
    assert abs(estimates[0] - x_hat) < 1e-12, "Step and batch disagree on index 0"
    assert abs(estimates[1] - x_hat2) < 1e-12, "Step and batch disagree on index 1"
    print("\n    Consistency check passed: step-wise == batch at indices 0 and 1.")

    print("\nDone.")
