# src/tracker/core/track.py

import traceback
import numpy as np
from typing import List, Optional

from .kalman_filter import KalmanFilter
from .detection import Detection

class TrackState:
    """Enumeration for the state of a track."""
    Tentative = 1  # Newly initiated, not yet confirmed
    Confirmed = 2  # Confirmed after enough evidence
    Deleted = 3    # Marked for deletion

class Track:
    """
    A single target track with state managed by a Kalman Filter.
    The state includes (center_x, center_y, aspect_ratio, height) and their velocities.
    """
    _next_id: int = 1 # Class variable to generate unique track IDs

    def __init__(self,
                 initial_mean: np.ndarray,
                 initial_covariance: np.ndarray,
                 initial_detection: 'Detection', # Use string literal for forward declaration type hint
                 n_init: int,
                 max_age: int,
                 feature_budget: Optional[int] = None
                ):
        """
        Initializes a new track.

        Args:
            initial_mean: Initial state mean from Kalman Filter's initiate method.
            initial_covariance: Initial state covariance from Kalman Filter.
            initial_detection: The `Detection` object that initiated this track.
            n_init: Number of consecutive updates required to confirm the track.
            max_age: Maximum number of consecutive frames a track can be missed before deletion.
            feature_budget: Maximum number of features to store in the gallery. None for unlimited.
        """
        self.track_id: int = Track._next_id
        Track._next_id += 1

        self.mean: np.ndarray = initial_mean
        self.covariance: np.ndarray = initial_covariance
        
        self.class_name: str = initial_detection.class_name
        self.confidence: float = initial_detection.confidence # Confidence of the last associated detection

        self.hits: int = 1             # Number of successful updates (matches)
        self.age: int = 1              # Total frames since track initiation
        self.time_since_update: int = 0 # Frames since last successful update/match

        self.state: int = TrackState.Tentative

        # Configuration for track lifecycle
        self._n_init: int = n_init
        self._max_age: int = max_age # This track's specific max_age, set at creation

        self.features: List[np.ndarray] = []
        self._feature_budget: Optional[int] = feature_budget
        if initial_detection.feature is not None:
            self._add_feature(initial_detection.feature)
            
        # Store the most recent detection that successfully updated this track
        self.last_successful_detection: 'Detection' = initial_detection


    def _add_feature(self, feature: np.ndarray):
        """Adds a feature to the track's gallery, managing the budget (FIFO)."""
        self.features.append(feature)
        if self._feature_budget is not None and len(self.features) > self._feature_budget:
            self.features.pop(0) # Remove the oldest feature

    def predict(self, kf: 'KalmanFilter'):
        """Propagates the state distribution to the current time step using Kalman Filter."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: 'KalmanFilter', detection: 'Detection'):
        """
        Performs Kalman Filter measurement update and updates track attributes.
        This method is called when a track is successfully matched with a detection.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())

        if detection.feature is not None:
            self._add_feature(detection.feature)
        
        self.hits += 1
        self.time_since_update = 0 # Reset as it was just updated
        self.confidence = detection.confidence
        self.class_name = detection.class_name # Usually remains the same
        self.last_successful_detection = detection

        # Update track state to Confirmed if it meets the criteria
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        # If a track was somehow marked Deleted but then gets an update (should be rare if managed well)
        elif self.state == TrackState.Deleted:
            # print(f"Warning: Track {self.track_id} was Deleted but received an update. Reviving to Confirmed.")
            self.state = TrackState.Confirmed # Revive the track

    def mark_missed(self):
        """
        Marks this track as missed (no association at the current time step).
        Updates state to Deleted if conditions are met (e.g., too many misses).
        This is typically called after `predict()` if a track is not matched.
        """
        if self.state == TrackState.Tentative:
            # Tentative tracks are deleted immediately if missed before confirmation
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed:
            # Confirmed tracks are deleted if missed for too long
            if self.time_since_update > self._max_age:
                self.state = TrackState.Deleted
        # If already Deleted, no change.

    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Returns True if this track is marked for deletion."""
        return self.state == TrackState.Deleted

    def to_tlwh(self) -> np.ndarray:
        """
        Converts current KF state mean to (top-left-x, top-left-y, width, height).
        State mean is (cx, cy, aspect_ratio, height, ...velocities...).
        """
        mean_pos = self.mean[:4].copy() # cx, cy, aspect_ratio, height
        
        # Calculate width from aspect_ratio and height
        if mean_pos[3] > 0:  # height > 0
            width = mean_pos[2] * mean_pos[3]  # aspect_ratio * height
        else: # Handle zero or negative height to prevent issues
            width = 0
            mean_pos[3] = max(0, mean_pos[3]) # Ensure height is not negative
        
        # Convert (cx, cy, w, h) to (tl_x, tl_y, w, h)
        tl_x = mean_pos[0] - width / 2.0
        tl_y = mean_pos[1] - mean_pos[3] / 2.0
        
        return np.array([tl_x, tl_y, width, mean_pos[3]], dtype=np.float32)

    def to_tlbr(self) -> np.ndarray:
        """Converts current KF state mean to (x1, y1, x2, y2) format."""
        tlwh = self.to_tlwh()
        tlbr = tlwh.copy()
        tlbr[2:] += tlbr[:2] # x2 = x1 + w, y2 = y1 + h
        return tlbr
    
    @staticmethod
    def reset_id_counter(start_id: int = 1):
        """Resets the global track ID counter. Useful for testing."""
        Track._next_id = start_id

    def __repr__(self) -> str:
        state_map = {TrackState.Tentative: "Tentative", TrackState.Confirmed: "Confirmed", TrackState.Deleted: "Deleted"}
        state_str = state_map.get(self.state, "UnknownState")
        return (f"Track(ID={self.track_id}, Cls='{self.class_name}', State='{state_str}', "
                f"Age={self.age}, Hits={self.hits}, MissesTSU={self.time_since_update}, " # TSU = Time Since Update
                f"Conf={self.confidence:.2f}, MaxAge={self._max_age}, "
                f"LastPosTLWH={self.to_tlwh().round(1)})")

# --- Test Suite ---
if __name__ == '__main__':

    print("--- Testing src/tracker/core/track.py ---")
    Track.reset_id_counter()

    kf_test = KalmanFilter()
    dummy_tlwh_init = np.array([50, 50, 20, 40], dtype=np.float32)
    
    det_xyah_init = Detection(dummy_tlwh_init, 0.9, "person", None).to_xyah()
    mean_init, cov_init = kf_test.initiate(det_xyah_init)
    
    initial_detection_obj = Detection(
        tlwh=dummy_tlwh_init, 
        confidence=0.9, 
        class_name="person", 
        feature=np.random.rand(128).astype(np.float32)
    )

    # Test 1: Track Initialization
    print("\n--- Test 1: Initialization ---")
    try:
        track1 = Track(mean_init, cov_init, initial_detection_obj, n_init=3, max_age=5, feature_budget=10)
        assert track1.track_id == 1
        assert track1.state == TrackState.Tentative
        assert track1.hits == 1
        assert track1.age == 1
        assert track1.time_since_update == 0
        assert len(track1.features) == 1
        assert track1.class_name == "person"
        print(f"  {track1}")
        print("Test 1 PASSED.")
    except Exception as e:
        print(f"Test 1 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 2: Track Prediction
    print("\n--- Test 2: Prediction ---")
    try:
        track1.predict(kf_test)
        assert track1.age == 2
        assert track1.time_since_update == 1
        print(f"  {track1}")
        print("Test 2 PASSED.")
    except Exception as e:
        print(f"Test 2 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 3: Track Update and Confirmation
    print("\n--- Test 3: Update & Confirmation ---")
    update_tlwh = np.array([52, 52, 20, 40], dtype=np.float32) # Slightly moved
    update_detection_obj = Detection(
        tlwh=update_tlwh, 
        confidence=0.92, 
        class_name="person", 
        feature=np.random.rand(128).astype(np.float32)
    )
    try:
        # First update (hits=2, n_init=3 -> still Tentative)
        track1.update(kf_test, update_detection_obj)
        assert track1.time_since_update == 0
        assert track1.hits == 2
        assert track1.state == TrackState.Tentative
        assert len(track1.features) == 2
        assert track1.confidence == 0.92
        print(f"  After 1st update: {track1}")

        # Second update (hits=3, n_init=3 -> Confirmed)
        track1.predict(kf_test) # Age=3, tsu=1 before update
        track1.update(kf_test, update_detection_obj) # Hits=3, tsu=0
        assert track1.hits == 3
        assert track1.state == TrackState.Confirmed
        print(f"  After 2nd update (now Confirmed): {track1}")
        print("Test 3 PASSED.")
    except Exception as e:
        print(f"Test 3 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 4: Feature Budget
    print("\n--- Test 4: Feature Budget ---")
    try:
        Track.reset_id_counter() # Reset for this test
        budget_track = Track(mean_init, cov_init, initial_detection_obj, n_init=1, max_age=5, feature_budget=2)
        assert len(budget_track.features) == 1
        
        # Add 1st new feature (total 2, budget not exceeded)
        det_feat2 = Detection(update_tlwh, 0.9, "person", np.random.rand(128).astype(np.float32))
        budget_track.update(kf_test, det_feat2)
        assert len(budget_track.features) == 2
        
        # Add 2nd new feature (total 3, oldest should be popped to maintain budget of 2)
        det_feat3 = Detection(update_tlwh, 0.9, "person", np.random.rand(128).astype(np.float32))
        budget_track.update(kf_test, det_feat3)
        assert len(budget_track.features) == 2 
        print(f"  {budget_track}, Features len: {len(budget_track.features)}")
        print("Test 4 PASSED.")
    except Exception as e:
        print(f"Test 4 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 5: Mark Missed and Deletion (Confirmed Track)
    print("\n--- Test 5: Mark Missed & Deletion (Confirmed Track) ---")
    # track1 is Confirmed, hits=3, tsu=0, age=3, max_age=5 initially
    try:
        track1._max_age = 2 # Modify internal _max_age for this specific test
        print(f"  Initial state for Test 5: {track1}")

        # Miss 1
        track1.predict(kf_test) # age=4, tsu=1
        track1.mark_missed()    # tsu(1) not > _max_age(2). State Confirmed.
        assert track1.state == TrackState.Confirmed
        assert track1.time_since_update == 1
        print(f"  After 1st miss: {track1}")

        # Miss 2
        track1.predict(kf_test) # age=5, tsu=2
        track1.mark_missed()    # tsu(2) not > _max_age(2). State Confirmed.
        assert track1.state == TrackState.Confirmed
        assert track1.time_since_update == 2
        print(f"  After 2nd miss: {track1}")
        
        # Miss 3
        track1.predict(kf_test) # age=6, tsu=3
        track1.mark_missed()    # tsu(3) IS > _max_age(2). State Deleted.
        assert track1.state == TrackState.Deleted
        assert track1.time_since_update == 3
        print(f"  After 3rd miss (should be Deleted): {track1}")
        print("Test 5 PASSED.")
    except Exception as e:
        print(f"Test 5 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 6: Tentative Track Deletion on First Miss
    print("\n--- Test 6: Tentative Track Deletion ---")
    try:
        Track.reset_id_counter()
        tentative_track = Track(mean_init, cov_init, initial_detection_obj, n_init=3, max_age=5)
        assert tentative_track.state == TrackState.Tentative
        
        tentative_track.predict(kf_test) # age=2, tsu=1
        tentative_track.mark_missed()    # Tentative track, missed -> Deleted
        assert tentative_track.state == TrackState.Deleted
        print(f"  {tentative_track}")
        print("Test 6 PASSED.")
    except Exception as e:
        print(f"Test 6 FAILED: {e}\n{traceback.format_exc()}")
        raise

    # Test 7: to_tlwh() and to_tlbr() conversion
    print("\n--- Test 7: BBox Conversions ---")
    try:
        Track.reset_id_counter()
        conv_track = Track(mean_init, cov_init, initial_detection_obj, n_init=3, max_age=5)
        # Initial mean was based on det_xyah_init from dummy_tlwh_init [50,50,20,40]
        # xyah = [60, 70, 0.5, 40] (cx, cy, a, h)
        expected_tlwh = dummy_tlwh_init 
        
        tlwh_output = conv_track.to_tlwh()
        assert tlwh_output.shape == (4,)
        assert np.allclose(tlwh_output, expected_tlwh, atol=1e-1), f"Expected TLWH {expected_tlwh}, got {tlwh_output}"
        print(f"  to_tlwh() output: {tlwh_output.round(2)}")
        
        tlbr_output = conv_track.to_tlbr()
        expected_tlbr = np.array([50,50, 50+20, 50+40], dtype=np.float32)
        assert tlbr_output.shape == (4,)
        assert np.allclose(tlbr_output, expected_tlbr, atol=1e-1), f"Expected TLBR {expected_tlbr}, got {tlbr_output}"
        print(f"  to_tlbr() output: {tlbr_output.round(2)}")
        print("Test 7 PASSED.")
    except Exception as e:
        print(f"Test 7 FAILED: {e}\n{traceback.format_exc()}")
        raise

    print("--- Finished testing src/tracker/core/track.py ---")