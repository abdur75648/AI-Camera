# src/tracker/core/kalman_filter.py

import numpy as np
import scipy.linalg
from typing import Tuple

# Chi-squared distribution percent point function (inverse of cdf)
# For 0.95 confidence, used as Mahalanobis gating threshold.
# Degrees of freedom: N
# For position (x, y) only: N=2
# For position and size (x, y, a, h): N=4
CHI2INV95 = {
    1: 3.841458820694124,
    2: 5.991464547107979,
    3: 7.814727903251179,
    4: 9.487729036781154,
    5: 11.070497693516351,
    6: 12.591587243743977,
    7: 14.067140449349192,
    8: 15.50731305586545,
    9: 16.918977604620448
}


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space is (cx, cy, a, h, v_cx, v_cy, v_a, v_h),
    where (cx, cy) is the center of the bounding box, 'a' is the aspect ratio (w/h),
    'h' is the height, and 'v_' denotes the respective velocities.
    """

    def __init__(self, dt: float = 1.0):
        """
        Args:
            dt (float): Time step interval.
        """
        ndim = 4  # Number of state variables to track (cx, cy, a, h)
        
        # State transition matrix (F)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix (H)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion and observation uncertainty weights.
        # These control how much uncertainty is added in the predict/update steps.
        # Values are heuristic and might need tuning.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement_xyah: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an unassociated measurement.

        Args:
            measurement_xyah (np.ndarray): Bounding box in (cx, cy, a, h) format.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mean (np.ndarray): Initial state mean (8-dimensional).
                - covariance (np.ndarray): Initial state covariance (8x8 matrix).
        """
        mean_pos = measurement_xyah
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.concatenate((mean_pos, mean_vel))

        # Initial uncertainty (standard deviations)
        std = [
            2 * self._std_weight_position * measurement_xyah[3],  # cx
            2 * self._std_weight_position * measurement_xyah[3],  # cy
            1e-2,                                                # a
            2 * self._std_weight_position * measurement_xyah[3],  # h
            10 * self._std_weight_velocity * measurement_xyah[3], # v_cx
            10 * self._std_weight_velocity * measurement_xyah[3], # v_cy
            1e-5,                                                # v_a
            10 * self._std_weight_velocity * measurement_xyah[3]  # v_h
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): Current state mean (8-dimensional).
            covariance (np.ndarray): Current state covariance (8x8 matrix).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - predicted_mean (np.ndarray): Predicted state mean.
                - predicted_covariance (np.ndarray): Predicted state covariance.
        """
        # Process noise (Q) - uncertainty in motion model
        std_pos = [
            self._std_weight_position * mean[3],      # cx proportional to height
            self._std_weight_position * mean[3],      # cy proportional to height
            1e-2,                                     # a (aspect ratio)
            self._std_weight_position * mean[3]       # h
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],      # v_cx
            self._std_weight_velocity * mean[3],      # v_cy
            1e-5,                                     # v_a
            self._std_weight_velocity * mean[3]       # v_h
        ]
        motion_cov_diag = np.square(np.concatenate((std_pos, std_vel)))
        motion_cov = np.diag(motion_cov_diag).astype(np.float32)

        # Predict: x_pred = F * x
        predicted_mean = np.dot(self._motion_mat, mean)
        # Predict: P_pred = F * P * F_T + Q
        predicted_covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return predicted_mean, predicted_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.

        Args:
            mean (np.ndarray): State mean (8-dimensional).
            covariance (np.ndarray): State covariance (8x8 matrix).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - projected_mean (np.ndarray): Projected state mean in measurement space (4-dim: cx, cy, a, h).
                - projected_covariance (np.ndarray): Projected state covariance in measurement space (4x4 matrix).
        """
        # Measurement noise (R) - uncertainty in measurement
        std = [
            self._std_weight_position * mean[3], # cx
            self._std_weight_position * mean[3], # cy
            1e-1,                                # a
            self._std_weight_position * mean[3]  # h
        ]
        innovation_cov_diag = np.square(std)
        innovation_cov = np.diag(innovation_cov_diag).astype(np.float32)

        # Project mean: y_pred = H * x_pred
        projected_mean = np.dot(self._update_mat, mean)
        # Project covariance: S = H * P_pred * H_T + R
        projected_covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) + innovation_cov
            
        return projected_mean, projected_covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement_xyah: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.

        Args:
            mean (np.ndarray): Predicted state mean (8-dimensional).
            covariance (np.ndarray): Predicted state covariance (8x8 matrix).
            measurement_xyah (np.ndarray): Current measurement (4-dim: cx, cy, a, h).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - updated_mean (np.ndarray): Corrected state mean.
                - updated_covariance (np.ndarray): Corrected state covariance.
        """
        projected_mean, projected_covariance_S = self.project(mean, covariance)

        # Kalman gain: K = P_pred * H_T * S^-1
        # Using Cholesky decomposition for S^-1: S = L * L_T
        # K = P_pred * H_T * (L * L_T)^-1
        # K = P_pred * H_T * L_T^-1 * L^-1
        # Let B = P_pred * H_T. Solve L * L_T * K_T = B_T
        # Solve L * M = B_T for M, then L_T * K_T = M for K_T.
        
        # More directly using scipy.linalg.cho_solve:
        # K_transpose = cho_solve(cho_factor(S), (P_pred * H_T)_T )
        # K = K_transpose_T
        
        # Equivalent form from original:
        # kalman_gain = P_pred * H_T * S_inv
        # K = (covariance @ self._update_mat.T) @ np.linalg.inv(projected_covariance_S) # Less stable
        
        # Using original's approach with cho_solve for stability
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_covariance_S, lower=True, check_finite=False)
        
        kalman_gain_numerator = np.dot(covariance, self._update_mat.T)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), kalman_gain_numerator.T, check_finite=False).T

        # Innovation (residual): y_tilde = z - H * x_pred
        innovation = measurement_xyah - projected_mean

        # Update mean: x_new = x_pred + K * y_tilde
        updated_mean = mean + np.dot(kalman_gain, innovation)
        # Update covariance: P_new = (I - K * H) * P_pred
        # More stable form: P_new = P_pred - K * S * K_T
        # Original DeepSORT form: P_new = P_pred - K * H * P_pred
        # P_new = covariance - K @ self._update_mat @ covariance # (I-KH)P form
        updated_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance_S, kalman_gain.T))
            
        return updated_mean, updated_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements_xyah: np.ndarray, only_position: bool = False) -> np.ndarray:
        """
        Compute gating distance (squared Mahalanobis distance) between state distribution and measurements.

        Args:
            mean (np.ndarray): State mean (8-dimensional).
            covariance (np.ndarray): State covariance (8x8 matrix).
            measurements_xyah (np.ndarray): An Nx4 matrix of N measurements (cx, cy, a, h).
            only_position (bool): If True, consider only (cx, cy) for distance calculation.

        Returns:
            np.ndarray: An array of length N with squared Mahalanobis distances.
        """
        projected_mean, projected_covariance = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_covariance = projected_covariance[:2, :2]
            measurements_xyah_filtered = measurements_xyah[:, :2]
        else:
            measurements_xyah_filtered = measurements_xyah

        # Mahalanobis distance: d^2 = (z - H*x_pred)^T * S^-1 * (z - H*x_pred)
        # z = measurement, H*x_pred = projected_mean, S = projected_covariance
        delta = measurements_xyah_filtered - projected_mean
        
        try:
            # Using Cholesky decomposition for S^-1
            cholesky_factor, lower = scipy.linalg.cho_factor(
                projected_covariance, lower=True, check_finite=False)
            # Solve L*y = delta_T for y, then Mahalanobis distance is y_T*y
            z_intermediate = scipy.linalg.solve_triangular(
                cholesky_factor, delta.T, lower=True, check_finite=False, overwrite_b=False)
            squared_mahalanobis = np.sum(z_intermediate * z_intermediate, axis=0)
        except np.linalg.LinAlgError:
            # Covariance matrix might be singular if track uncertainty is too low or dimensions are redundant
            print("Warning: KalmanFilter.gating_distance encountered LinAlgError (possibly singular covariance). "
                  "Returning large distances.")
            # Fallback: Use Euclidean distance or return infinity if matrix is singular
            # For now, return large values to effectively reject these measurements
            return np.full(measurements_xyah_filtered.shape[0], np.inf, dtype=np.float32)
            
        return squared_mahalanobis


if __name__ == '__main__':
    print("--- Testing src/tracker/core/kalman_filter.py ---")
    kf = KalmanFilter()

    # Dummy measurement (cx, cy, aspect_ratio, height)
    # Example: A box at (100, 150) of size 30x60 (w, h)
    # cx = 100 + 30/2 = 115
    # cy = 150 + 60/2 = 180
    # a = 30/60 = 0.5
    # h = 60
    measurement1 = np.array([115, 180, 0.5, 60], dtype=np.float32)

    # Test 1: Initiation
    try:
        mean, covariance = kf.initiate(measurement1)
        assert mean.shape == (8,)
        assert covariance.shape == (8, 8)
        assert np.allclose(mean[:4], measurement1) # Initial position should match measurement
        assert np.all(mean[4:] == 0) # Initial velocities should be zero
        print("Test 1 PASSED: KalmanFilter initiation.")
        print(f"  Initial mean: {mean.round(2)}")
        print(f"  Initial covariance (diag): {np.diag(covariance).round(2)}")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        raise

    # Test 2: Prediction
    try:
        predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        assert predicted_mean.shape == (8,)
        assert predicted_covariance.shape == (8, 8)
        # Position should change if initial velocity was non-zero (here it's zero, so cx,cy,a,h remain same if dt=1)
        # Velocities remain same as no acceleration. Uncertainty should increase.
        print(f"  Predicted mean: {predicted_mean.round(2)}")
        assert np.all(np.diag(predicted_covariance) >= np.diag(covariance)) # Uncertainty should not decrease
        print("Test 2 PASSED: KalmanFilter prediction.")
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        raise

    # Test 3: Update
    # New measurement, slightly shifted
    measurement2 = np.array([118, 183, 0.51, 62], dtype=np.float32)
    try:
        # Use the predicted state from Test 2 for update
        updated_mean, updated_covariance = kf.update(predicted_mean, predicted_covariance, measurement2)
        assert updated_mean.shape == (8,)
        assert updated_covariance.shape == (8, 8)
        # Updated mean should be somewhere between predicted_mean and measurement2
        print(f"  Updated mean: {updated_mean.round(2)}")
        # Uncertainty might decrease after update if measurement is good
        # print(f"  Updated covariance (diag): {np.diag(updated_covariance).round(2)}")
        print("Test 3 PASSED: KalmanFilter update.")
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        raise
    
    # Test 4: Gating distance
    # measurements_batch: (N, 4)
    measurements_batch = np.array([
        [118, 183, 0.51, 62],  # Close to updated_mean
        [10, 10, 0.4, 50],    # Far from updated_mean
        [117, 182, 0.50, 61]   # Reasonably close
    ], dtype=np.float32)
    try:
        # Use the state after last update
        mean_for_gating, cov_for_gating = updated_mean, updated_covariance
        
        distances_full = kf.gating_distance(mean_for_gating, cov_for_gating, measurements_batch, only_position=False)
        distances_pos_only = kf.gating_distance(mean_for_gating, cov_for_gating, measurements_batch, only_position=True)
        
        assert distances_full.shape == (3,)
        assert distances_pos_only.shape == (3,)
        print(f"  Gating distances (full state): {distances_full.round(2)}")
        print(f"  Gating distances (pos only): {distances_pos_only.round(2)}")
        assert distances_full[0] < distances_full[1] # First measurement should be closer
        assert distances_pos_only[0] < distances_pos_only[1]
        
        # Check against chi-squared threshold (example)
        # Threshold for 4 DoF (full state gating)
        threshold_4dof = CHI2INV95[4]
        print(f"  Chi2Inv95 (4 DoF): {threshold_4dof:.2f}")
        assert distances_full[0] < threshold_4dof # Expect first to be a valid match
        
        print("Test 4 PASSED: KalmanFilter gating_distance.")
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        raise

    print("--- Finished testing src/tracker/core/kalman_filter.py ---")