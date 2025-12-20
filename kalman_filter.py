# kalman_filter.py
import numpy as np
import sys

# === GPU ACCELERATION TOGGLE ===
USE_GPU = True

try:
    if USE_GPU:
        import cupy as xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.fft as fft
        print(f"[{__name__}] 🚀 GPU Acceleration ENABLED (CuPy)")
    else:
        raise ImportError
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndimage
    import scipy.fft as fft
    print(f"[{__name__}] 🐢 GPU Acceleration DISABLED (NumPy)")


class KalmanFilter:
    def __init__(self, nx, ny, process_noise, measurement_noise):
        """
        Initializes an array of independent Kalman filters for a 2D grid.
        """
        print("Kalman Filter initialized for data assimilation.")
        # State is [u, v] for each grid point
        self.state_estimate = xp.zeros((nx, ny, 2))
        # Estimate covariance (our uncertainty in the state)
        self.estimate_covariance = xp.zeros((nx, ny, 2, 2))
        for i in range(nx):
            for j in range(ny):
                self.estimate_covariance[i, j] = xp.eye(2)

        # Model parameters
        self.transition_matrix = xp.eye(2) # A
        self.process_noise_covariance = xp.eye(2) * process_noise # Q
        self.measurement_noise_covariance = measurement_noise # R

    def predict(self):
        """
        Predicts the next state.
        """
        # --- FIX: Corrected einsum strings for batch matrix multiplication ---
        # State prediction: x_hat = A @ x_hat
        # 'ij,...j->...i' means: for each item in the batch (...), multiply matrix A (ij) by vector x (j)
        self.state_estimate = xp.einsum('ij,...j->...i', self.transition_matrix, self.state_estimate)
        
        # Covariance prediction: P = A @ P @ A.T + Q
        # Step 1: P_temp = A @ P
        P_temp = xp.einsum('ik,...kl->...il', self.transition_matrix, self.estimate_covariance)
        # Step 2: P_final = P_temp @ A.T
        self.estimate_covariance = xp.einsum('...ik,jk->...ij', P_temp, self.transition_matrix.T) + self.process_noise_covariance

    def update(self, measurement):
        """
        Updates the state estimate with a new measurement (from ERA5).
        """
        # (The update logic was correct and remains the same)
        residual_covariance = self.estimate_covariance + self.measurement_noise_covariance
        kalman_gain = xp.einsum('...ik,...kj->...ij', self.estimate_covariance, xp.linalg.inv(residual_covariance))
        
        residual = measurement - self.state_estimate
        self.state_estimate += xp.einsum('...ik,...k->...i', kalman_gain, residual)
        
        update_factor = xp.eye(2) - kalman_gain
        self.estimate_covariance = xp.einsum('...ik,...kj->...ij', update_factor, self.estimate_covariance)