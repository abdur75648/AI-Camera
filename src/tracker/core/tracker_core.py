# src/tracker/core/tracker_core.py

import numpy as np
from typing import List, Tuple, Optional

from .kalman_filter import KalmanFilter
from .track import Track, TrackState
from .detection import Detection
from . import matching
from . import linear_assignment

class TrackerCore:
    """
    The core multi-target tracker usingSORT-like principles with appearance features (DeepSORT-like).

    It manages a list of active tracks, performs prediction, association, and updates.
    """
    def __init__(self,
                 max_cosine_distance: float = 0.2, # From original DeepSORT config
                 nn_budget: Optional[int] = 100,   # Feature gallery budget per track
                 max_iou_distance: float = 0.7,    # Max IoU distance for IoU matching
                 max_age: int = 70,                # Max frames a track is kept without updates
                 n_init: int = 3                   # Min consecutive detections to confirm a track
                ):
        """
        Args:
            max_cosine_distance: Max cosine distance for appearance-based matching.
            nn_budget: Max number of features to store in each track's gallery.
                       If None, all features are stored.
            max_iou_distance: Max IoU distance (1 - IoU) for IoU-based matching.
            max_age: Maximum number of frames a track can be missed before deletion.
            n_init: Number of consecutive updates to confirm a new track.
        """
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        Track.reset_id_counter() # Ensure fresh IDs for new tracker instance

    def predict(self):
        """
        Propagates all active track states one time step forward using Kalman Filter.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: List[Detection]):
        """
        Performs measurement update and track management.

        Args:
            detections (List[Detection]): A list of `Detection` objects for the current frame.
                                          These detections should already have ReID features.
        """
        # 1. Perform matching
        matches, unmatched_track_indices, unmatched_detection_indices = self._match(detections)

        # 2. Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 3. Handle unmatched tracks (mark as missed or delete)
        for track_idx in unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        # 4. Initiate new tracks for unmatched detections
        for detection_idx in unmatched_detection_indices:
            self._initiate_track(detections[detection_idx])

        # 5. Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Note: Original DeepSORT had a 'metric.partial_fit' step here to update
        # the nearest neighbor index for appearance features. In our simplified approach,
        # features are stored directly in Track.features. If a more complex NN search
        # structure was used (like Faiss), it would be updated here.
        # For now, matching.appearance_cost_metric directly uses Track.features.

    def _match(self, detections: List[Detection]
              ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associates detections with existing tracks.
        Implements a matching cascade:
        1. Appearance matching for confirmed tracks.
        2. IoU matching for remaining unmatched tracks/detections and unconfirmed tracks.
        """

        # --- Define the appearance distance metric function with Mahalanobis gating ---
        def gated_appearance_metric(
            tracks: List[Track], 
            dets: List[Detection], 
            trk_indices: List[int], 
            det_indices: List[int]
        ) -> np.ndarray:
            
            # Calculate appearance costs (e.g., cosine distance)
            cost_matrix_app = matching.appearance_cost_metric(
                tracks, dets, trk_indices, det_indices, metric_type="cosine"
            )
            
            # Apply Mahalanobis distance gating to the appearance cost matrix
            cost_matrix_gated = linear_assignment.gate_cost_matrix_by_mahalanobis(
                self.kf, cost_matrix_app, tracks, dets, trk_indices, det_indices
            )
            return cost_matrix_gated

        # --- Split tracks into confirmed and unconfirmed ---
        confirmed_track_indices = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_track_indices = [
            i for i, t in enumerate(self.tracks) if t.is_tentative()
        ]

        # --- Stage 1: Matching cascade with appearance metric for confirmed tracks ---
        # This matches confirmed tracks, prioritizing those with fewer misses (smaller time_since_update).
        # The cascade_depth is effectively self.max_age.
        matches_app, unmatched_confirmed_track_indices, unmatched_detection_indices_after_app = \
            linear_assignment.matching_cascade(
                gated_appearance_metric,
                self.max_cosine_distance, # Max appearance distance
                self.max_age,             # Cascade depth (max age of tracks)
                self.tracks,
                detections,
                confirmed_track_indices   # Only try to match confirmed tracks by appearance
            )

        # --- Stage 2: IoU matching for remaining tracks and detections ---
        # Tracks to consider for IoU matching:
        # - Unconfirmed (tentative) tracks.
        # - Confirmed tracks that were not matched by appearance AND were recently updated (time_since_update == 1).
        #   (Original DeepSORT behavior: these are considered for IoU if appearance match failed)
        
        iou_candidate_track_indices = unconfirmed_track_indices + [
            idx for idx in unmatched_confirmed_track_indices 
            if self.tracks[idx].time_since_update == 1 # Only if missed just one frame
        ]
        
        # Filter out tracks that are too old from unmatched_confirmed_track_indices
        # These will not participate in IoU matching if they missed more than 1 frame.
        remaining_unmatched_track_indices = [
            idx for idx in unmatched_confirmed_track_indices
            if self.tracks[idx].time_since_update > 1
        ]

        if iou_candidate_track_indices and unmatched_detection_indices_after_app:
            # Define the IoU cost metric (already includes 1-IoU)
            # No Mahalanobis gating here, as IoU itself is a spatial constraint.
            
            matches_iou, unmatched_iou_track_indices, unmatched_detection_indices_after_iou = \
                linear_assignment.min_cost_matching(
                    matching.iou_cost,
                    self.max_iou_distance, # Max IoU distance (1 - min_iou)
                    self.tracks,
                    detections,
                    iou_candidate_track_indices,
                    unmatched_detection_indices_after_app
                )
        else: # No tracks or no detections for IoU matching
            matches_iou = []
            unmatched_iou_track_indices = iou_candidate_track_indices # All IoU candidates remain unmatched
            unmatched_detection_indices_after_iou = unmatched_detection_indices_after_app


        # --- Consolidate results ---
        all_matches = matches_app + matches_iou
        
        # Unmatched tracks are those in remaining_unmatched_track_indices (too old for IoU)
        # PLUS those from iou_candidate_track_indices that didn't get an IoU match.
        final_unmatched_track_indices = remaining_unmatched_track_indices + unmatched_iou_track_indices
        final_unmatched_detection_indices = unmatched_detection_indices_after_iou
        
        return all_matches, final_unmatched_track_indices, final_unmatched_detection_indices


    def _initiate_track(self, detection: Detection):
        """
        Initializes a new track from a detection.
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        new_track = Track(
            initial_mean=mean,
            initial_covariance=covariance,
            initial_detection=detection, # Pass the full detection object
            n_init=self.n_init,
            max_age=self.max_age,
            feature_budget=self.nn_budget
        )
        self.tracks.append(new_track)
        # print(f"Initiated new {new_track}") # For debugging

    def get_active_tracks(self) -> List[Track]:
        """Returns a list of currently active (confirmed or tentative and not old) tracks."""
        return [t for t in self.tracks if not t.is_deleted()] # Or more specific filtering if needed


if __name__ == '__main__':
    print("--- Testing src/tracker/core/tracker_core.py ---")
    
    # Test parameters
    MAX_COS_DIST = 0.5
    NN_BUDGET = 5
    MAX_IOU_DIST = 0.8 # Corresponds to min IoU of 0.2
    MAX_AGE_TEST = 3
    N_INIT_TEST = 2

    tracker = TrackerCore(
        max_cosine_distance=MAX_COS_DIST,
        nn_budget=NN_BUDGET,
        max_iou_distance=MAX_IOU_DIST,
        max_age=MAX_AGE_TEST,
        n_init=N_INIT_TEST
    )

    # --- Frame 1: Initiate two tracks ---
    print("\n--- Frame 1 ---")
    det1_f1_tlwh = np.array([10, 10, 20, 40], dtype=np.float32)
    det1_f1_feat = np.random.rand(128).astype(np.float32); det1_f1_feat /= np.linalg.norm(det1_f1_feat)
    detection1_f1 = Detection(det1_f1_tlwh, 0.95, "person", det1_f1_feat)

    det2_f1_tlwh = np.array([100, 100, 30, 60], dtype=np.float32)
    det2_f1_feat = np.random.rand(128).astype(np.float32); det2_f1_feat /= np.linalg.norm(det2_f1_feat)
    detection2_f1 = Detection(det2_f1_tlwh, 0.90, "car", det2_f1_feat)
    
    current_detections_f1 = [detection1_f1, detection2_f1]
    tracker.predict() # Predicts for empty list of tracks, does nothing
    tracker.update(current_detections_f1)

    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].track_id == 1 and tracker.tracks[0].state == TrackState.Tentative
    assert tracker.tracks[1].track_id == 2 and tracker.tracks[1].state == TrackState.Tentative
    print(f"  Tracks after Frame 1: {[str(t) for t in tracker.tracks]}")
    print("Frame 1 Test PASSED.")

    # --- Frame 2: Update tracks, one should confirm ---
    print("\n--- Frame 2 ---")
    # Detection for track 1 (slightly moved, similar feature)
    det1_f2_tlwh = np.array([12, 12, 20, 40], dtype=np.float32)
    det1_f2_feat = det1_f1_feat * 0.95 + np.random.rand(128).astype(np.float32)*0.05 # Similar feature
    det1_f2_feat /= np.linalg.norm(det1_f2_feat)
    detection1_f2 = Detection(det1_f2_tlwh, 0.96, "person", det1_f2_feat)

    # Detection for track 2 (very different feature, should rely on IoU or create new)
    det2_f2_tlwh = np.array([102, 102, 30, 60], dtype=np.float32) # Good IoU
    det2_f2_feat_diff = np.random.rand(128).astype(np.float32); det2_f2_feat_diff /= np.linalg.norm(det2_f2_feat_diff) # Different feature
    detection2_f2 = Detection(det2_f2_tlwh, 0.88, "car", det2_f2_feat_diff)
    
    current_detections_f2 = [detection1_f2, detection2_f2]
    tracker.predict()
    tracker.update(current_detections_f2)

    assert len(tracker.tracks) == 2 # Should re-associate based on IoU primarily for track 2 if appearance fails
    track1_f2 = next(t for t in tracker.tracks if t.track_id == 1)
    track2_f2 = next(t for t in tracker.tracks if t.track_id == 2)
    
    assert track1_f2.hits == 2 and track1_f2.state == TrackState.Confirmed # N_INIT_TEST = 2
    assert track2_f2.hits == 2 and track2_f2.state == TrackState.Confirmed # N_INIT_TEST = 2
    print(f"  Tracks after Frame 2: {[str(t) for t in tracker.tracks]}")
    print("Frame 2 Test PASSED.")

    # --- Frame 3: One track missed, one new track ---
    print("\n--- Frame 3 ---")
    # Detection for track 1 continues
    det1_f3_tlwh = np.array([15, 15, 20, 40], dtype=np.float32)
    det1_f3_feat = det1_f1_feat * 0.9 + np.random.rand(128).astype(np.float32)*0.1 # Similar feature
    det1_f3_feat /= np.linalg.norm(det1_f3_feat)
    detection1_f3 = Detection(det1_f3_tlwh, 0.97, "person", det1_f3_feat)
    
    # New detection far away
    det3_f3_tlwh = np.array([200, 200, 25, 50], dtype=np.float32)
    det3_f3_feat = np.random.rand(128).astype(np.float32); det3_f3_feat /= np.linalg.norm(det3_f3_feat)
    detection3_f3 = Detection(det3_f3_tlwh, 0.92, "person", det3_f3_feat)
    
    current_detections_f3 = [detection1_f3, detection3_f3]
    tracker.predict()
    tracker.update(current_detections_f3)
    
    assert len(tracker.tracks) == 3 # Track 1 updated, Track 2 missed, Track 3 initiated
    track1_f3 = next(t for t in tracker.tracks if t.track_id == 1)
    track2_f3 = next(t for t in tracker.tracks if t.track_id == 2) # Should still exist but time_since_update > 0
    track3_f3 = next(t for t in tracker.tracks if t.track_id == 3)

    assert track1_f3.hits == 3 and track1_f3.time_since_update == 0
    assert track2_f3.time_since_update == 1 # Missed this frame
    assert track3_f3.state == TrackState.Tentative and track3_f3.hits == 1
    print(f"  Tracks after Frame 3: {[str(t) for t in tracker.tracks]}")
    print("Frame 3 Test PASSED.")

    # --- Frame 4, 5, 6: Track 2 continues to be missed, should be deleted ---
    print("\n--- Frame 4, 5, 6 (Track 2 deletion) ---")
    # MAX_AGE_TEST = 3. So, if time_since_update becomes > 3, it's deleted.
    # Frame 3: t2.time_since_update = 1
    # Frame 4: predict -> t2.time_since_update = 2. update (no match) -> mark_missed -> still Confirmed
    # Frame 5: predict -> t2.time_since_update = 3. update (no match) -> mark_missed -> still Confirmed
    # Frame 6: predict -> t2.time_since_update = 4. update (no match) -> mark_missed -> Deleted
    
    empty_detections = []
    for i in range(3): # Simulate 3 more frames where track 2 is missed
        print(f"  Simulating Frame {4+i} (empty detections for track 2)")
        tracker.predict()
        # Provide only detection for track 1 to keep it alive and test track 2 deletion
        # If we provide empty_detections, track 1 will also be marked missed.
        # Let's simulate track 1 continuing to be detected to isolate track 2's deletion.
        det1_cont_tlwh = np.array([18+i, 18+i, 20, 40], dtype=np.float32)
        det1_cont_feat = det1_f1_feat # Reuse feature
        det1_cont = Detection(det1_cont_tlwh, 0.9, "person", det1_cont_feat)
        tracker.update([det1_cont]) # Track 3 will also be missed now.
        
        track2_f_current = next((t for t in tracker.tracks if t.track_id == 2), None)
        if track2_f_current:
            print(f"    Track 2: {track2_f_current}")
        else:
            print(f"    Track 2 (ID:2) deleted at frame {4+i}.")
            break
    
    active_track_ids = [t.track_id for t in tracker.tracks]
    assert 2 not in active_track_ids # Track 2 should be deleted
    assert 1 in active_track_ids # Track 1 should be alive
    # Track 3 was initiated in Frame 3, missed in 4,5,6. n_init=2, max_age=3.
    # F3: hits=1, tsu=0, Tentative
    # F4: predict (age=2, tsu=1). update (miss) -> mark_missed -> Deleted (tentative + miss)
    assert 3 not in active_track_ids # Track 3 (tentative) should also be deleted after first miss.
    
    print(f"  Tracks after Frame 6: {[str(t) for t in tracker.tracks]}")
    print("Frame 4,5,6 Test PASSED (Track deletion).")
    
    print("--- Finished testing src/tracker/core/tracker_core.py ---")