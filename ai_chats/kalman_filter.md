# Task Specification: High-Performance 1D Kalman Filter Package (with Standard Notation)

You are an expert software engineer specializing in high-performance computational finance and quantitative analysis tools. Your task is to develop a lightweight, production-grade Python module for a 1D (scalar) Kalman Filter optimized for speed using **Numba**.

You must explicitly incorporate standard control theory/signal processing symbols ($z_k, \hat{x}_k, P_k, Q, R, K_k$) within the documentation, docstrings, and internal mathematical comments of the code.

---

## Architectural Requirements

1. **Performance:** Both functions must be decorated with Numba's `@njit` to ensure compilation to pure machine code, minimizing execution latency.
2. **Memory Management:** In the batch processing function, pre-allocate NumPy arrays for the outputs to guarantee $O(1)$ memory overhead and maximum loop speed within the compiled space.
3. **Generalization:** Avoid domain-specific terminology (e.g., do not use "price"). Use generalized state-estimation terminology (e.g., `measurement`).
4. **No Defaults:** To ensure explicit configuration in algorithmic pipelines, **do not** include default values for any parameters.

---

## Technical Specifications & Prototypes

### 1. The Recursive Step Function

This function must be completely stateless, executing a single discrete-time prediction and update step of the Kalman filter loop.

#### Function Signature

```python
from numba import njit

@njit
def kalman_filter_step(
    measurement: float,
    prev_estimate: float,
    prev_error_cov: float,
    process_variance: float,
    measurement_variance: float
) -> tuple[float, float]:
    """
    Executes a single, stateless recursive step of a 1D Kalman Filter.
    All parameters are strictly required (no default values).

    Parameters:
    -----------
    measurement : float
        The current raw observation/measurement ($z_k$).
    prev_estimate : float
        The posteriori state estimate from the previous step ($\hat{x}_{k-1}$).
    prev_error_cov : float
        The posteriori error covariance from the previous step ($P_{k-1}$).
    process_variance : float
        The process noise covariance ($Q$).
    measurement_variance : float
        The measurement noise covariance ($R$).

    Returns:
    --------
    tuple[float, float]
        current_estimate ($\hat{x}_k$), current_error_cov ($P_k$)
    """

```

#### Mathematical Implementation Inside the Step

Your code comments must reflect these standard equations:

* **Predict Stage:**
* Prior State Estimate ($\hat{x}_k^-$): `prior_estimate = prev_estimate`
* Prior Error Covariance ($P_k^-$): `prior_error_cov = prev_error_cov + process_variance`  *(i.e., $P_k^- = P_{k-1} + Q$)*


* **Update Stage (Kalman Gain):**
* Kalman Gain ($K_k$): `kalman_gain = prior_error_cov / (prior_error_cov + measurement_variance)` *(i.e., $K_k = P_k^- / (P_k^- + R)$)*


* **Correct Stage:**
* Posteriori State Estimate ($\hat{x}_k$): `current_estimate = prior_estimate + kalman_gain * (measurement - prior_estimate)` *(i.e., $\hat{x}_k = \hat{x}_k^- + K_k(z_k - \hat{x}_k^-)$)*
* Posteriori Error Covariance ($P_k$): `current_error_cov = (1.0 - kalman_gain) * prior_error_cov` *(i.e., $P_k = (1 - K_k)P_k^-$)*


* **Return:** `(current_estimate, current_error_cov)`

---

### 2. The Batch Processing Function

This function processes an entire sequence of data points. It must initialize the state tracker variables and internally loop through the input array, calling the recursive `kalman_filter_step` function at each iteration.

#### Function Signature

```python
import numpy as np
from numba import njit

@njit
def kalman_filter_batch(
    measurements: np.ndarray,
    initial_estimate: float,
    initial_error_cov: float,
    process_variance: float,
    measurement_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes an array of sequential measurements by internally calling 
    kalman_filter_step in a loop. Returns pre-allocated NumPy arrays.
    All parameters are strictly required (no default values).

    Parameters:
    -----------
    measurements : np.ndarray
        1D array of sequential observations ($z$).
    initial_estimate : float
        The initial state guess at step zero ($\hat{x}_0$).
    initial_error_cov : float
        The initial error covariance at step zero ($P_0$).
    process_variance : float
        The constant process noise covariance ($Q$) applied across the batch.
    measurement_variance : float
        The constant measurement noise covariance ($R$) applied across the batch.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        estimates (array of $\hat{x}$), error_covariances (array of $P$)
    """

```

#### Implementation Requirements Inside the Batch

* Pre-allocate two float64 NumPy arrays (`estimates` and `error_covariances`) matching the exact length of the incoming `measurements` array using `np.empty()` or `np.zeros()`.
* Maintain running state variables `est` ($\hat{x}_{k-1}$) and `cov` ($P_{k-1}$), initializing them using `initial_estimate` and `initial_error_cov`.
* Execute a fast sequential `for` loop over the length of the data.
* In each iteration, update the running state variables by passing them into `kalman_filter_step()`.
* Assign the outputs to their respective index positions in the pre-allocated `estimates` and `error_covariances` arrays.
* **Return:** `(estimates, error_covariances)`

---

## Deliverables

Please output a single, self-contained Python file (`kalman_fast.py`) containing:

1. The necessary imports (`numpy` and `numba.njit`).
2. The `@njit` decorated `kalman_filter_step` function containing the standard math symbols in its docstring/comments.
3. The `@njit` decorated `kalman_filter_batch` function.
4. A quick execution example under an `if __name__ == "__main__":` block that tests both functions with a mock dummy array to trigger the initial Numba compilation pass.