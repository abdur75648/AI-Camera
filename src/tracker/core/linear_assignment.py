# src/tracker/core/linear_assignment.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Callable, Optional
from .track import TrackState # For dummy track creation

# Define a large cost to represent infeasible assignments
INFTY_COST = 1e5

# Forward declare Track and Detection for type hinting if they are in the same 'core' package
# This avoids circular imports if other modules in 'core' import this one.
# However, since Track and Detection are already defined, we can import them directly.
from .track import Track 
from .detection import Detection
from .kalman_filter import KalmanFilter, CHI2INV95 # For gating


def min_cost_matching(
    distance_metric: Callable[[List[Track], List[Detection], List[int], List[int]], np.ndarray],
    max_distance: float,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves the linear assignment problem (minimum cost matching).

    Args:
        distance_metric: A callable that computes the cost matrix.
            It takes (tracks, detections, track_indices, detection_indices)
            and returns an NxM cost matrix.
        max_distance: Threshold for assignments. Costs > max_distance are disallowed.
        tracks: A list of active tracks.
        detections: A list of current detections.
        track_indices: Indices of tracks to consider for matching. Defaults to all.
        detection_indices: Indices of detections to consider. Defaults to all.

    Returns:
        A tuple containing:
            - matches (List[Tuple[int, int]]): List of (track_idx, detection_idx) pairs.
            - unmatched_track_indices (List[int]): List of indices of unmatched tracks.
            - unmatched_detection_indices (List[int]): List of indices of unmatched detections.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if not detection_indices or not track_indices:
        return [], track_indices, detection_indices  # Nothing to match

    # Compute the cost matrix using the provided metric
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    
    # Mark costs exceeding max_distance as infeasible for the assignment algorithm
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5 # Slightly above to ensure they are not picked if others are valid

    # Use scipy's linear_sum_assignment (Hungarian algorithm)
    # It finds row_ind[k] is assigned to col_ind[k]
    row_assign_indices, col_assign_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(track_indices) # Start with all tracks as unmatched
    unmatched_detections = list(detection_indices) # Start with all detections as unmatched

    for r_idx, c_idx in zip(row_assign_indices, col_assign_indices):
        # r_idx is an index into the rows of cost_matrix (which correspond to track_indices_for_metric)
        # c_idx is an index into the columns of cost_matrix (which correspond to detection_indices_for_metric)
        
        actual_track_idx = track_indices[r_idx]
        actual_detection_idx = detection_indices[c_idx]

        # Check if the assignment cost is valid (not exceeding original max_distance)
        if cost_matrix[r_idx, c_idx] <= max_distance:
            matches.append((actual_track_idx, actual_detection_idx))
            if actual_track_idx in unmatched_tracks:
                unmatched_tracks.remove(actual_track_idx)
            if actual_detection_idx in unmatched_detections:
                unmatched_detections.remove(actual_detection_idx)
        # else: Assignments with cost > max_distance are effectively unassigned due to the gating
        # and will remain in unmatched_tracks/unmatched_detections lists if not handled by
        # the previous removal. This 'else' branch may not be strictly necessary if
        # cost_matrix was properly gated before linear_sum_assignment.
        # However, the check ensures robustness.

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable[[List[Track], List[Detection], List[int], List[int]], np.ndarray],
    max_distance: float,
    cascade_depth: int,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
    kf: Optional[KalmanFilter] = None, # For Mahalanobis gating if distance_metric is appearance
    gated_cost_weight: float = 1.0 # Weight for the Mahalanobis distance in combined cost
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs matching in a cascade, prioritizing tracks with smaller 'time_since_update'.
    This is useful for appearance-based matching where older tracks are less reliable.

    Args:
        distance_metric: Cost metric function.
        max_distance: Max allowed cost for an assignment.
        cascade_depth: Max 'time_since_update' to consider. Corresponds to max_age of tracks.
        tracks: List of active tracks.
        detections: List of current detections.
        track_indices: Indices of tracks to consider.
        detection_indices: Indices of detections to consider.
        kf: KalmanFilter instance, required if Mahalanobis gating is used within the cascade.
        gated_cost_weight: (Not directly used here, but could be passed to metric)

    Returns:
        Tuple: (matches, unmatched_tracks, unmatched_detections)
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    current_unmatched_detections = list(detection_indices)
    all_matches = []
    
    # Iterate from most recent tracks (time_since_update=1) up to cascade_depth
    for level in range(cascade_depth):
        if not current_unmatched_detections:
            break # No detections left to match

        # Select tracks at the current cascade level (age/time_since_update)
        # In DeepSORT's original cascade, level 0 corresponds to time_since_update == 1
        tracks_at_level_indices = [
            idx for idx in track_indices if tracks[idx].time_since_update == (level + 1)
        ]

        if not tracks_at_level_indices:
            continue # No tracks at this cascade level

        # Perform matching for this level
        level_matches, _, current_unmatched_detections = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            tracks_at_level_indices,
            current_unmatched_detections # Only match against currently unmatched detections
        )
        all_matches.extend(level_matches)

    # Determine tracks that were not matched in any cascade level
    matched_track_indices_set = {trk_idx for trk_idx, _ in all_matches}
    all_unmatched_tracks = [idx for idx in track_indices if idx not in matched_track_indices_set]

    return all_matches, all_unmatched_tracks, current_unmatched_detections


def gate_cost_matrix_by_mahalanobis(
    kf: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: List[int],
    detection_indices: List[int],
    only_position: bool = False,
    gating_threshold_override: Optional[float] = None
) -> np.ndarray:
    """
    Invalidates infeasible entries in a cost matrix using Mahalanobis distance gating.
    Modifies the cost_matrix in-place by setting infeasible entries to INFTY_COST.

    Args:
        kf: The KalmanFilter instance.
        cost_matrix: The NxM cost matrix to be gated.
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Row indices in cost_matrix map to these track indices.
        detection_indices: Col indices in cost_matrix map to these detection indices.
        only_position: If True, use only (cx, cy) for gating.
        gating_threshold_override: Optional custom Mahalanobis threshold.

    Returns:
        np.ndarray: The gated cost matrix (modified in-place).
    """
    gating_dim = 2 if only_position else 4
    default_threshold = CHI2INV95.get(gating_dim, INFTY_COST) # Fallback to INFTY if dim not in dict
    gating_threshold = gating_threshold_override if gating_threshold_override is not None else default_threshold

    # Prepare all measurements once
    # Ensure detections are converted to xyah for KF
    measurements_xyah = np.asarray([detections[det_idx].to_xyah() for det_idx in detection_indices])
    if measurements_xyah.ndim == 1 and len(detection_indices) == 1: # Single detection case
        measurements_xyah = measurements_xyah.reshape(1, -1)


    for row_idx, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if not measurements_xyah.size: # No detections to gate against
            cost_matrix[row_idx, :] = INFTY_COST
            continue

        # Calculate Mahalanobis distances for the current track against all (selected) detections
        gating_distances = kf.gating_distance(
            track.mean, track.covariance, measurements_xyah, only_position
        )
        
        # Invalidate entries in cost_matrix where gating_distance exceeds threshold
        cost_matrix[row_idx, gating_distances > gating_threshold] = INFTY_COST
        
    return cost_matrix


if __name__ == '__main__':
    # Test the linear assignment functions
    print("--- Testing src/tracker/core/linear_assignment.py ---")
    Track.reset_id_counter() # Reset track IDs for consistent testing

    # Dummy data
    # Create some dummy tracks
    kf_test = KalmanFilter()
    tracks_list = []
    detections_list = []

    # Track 1: recently updated
    meas1_tlwh = np.array([10,10,20,40]); meas1_xyah = Detection(meas1_tlwh,0.9,'p',None).to_xyah()
    m1,c1 = kf_test.initiate(meas1_xyah); det1_init = Detection(meas1_tlwh,0.9,'p',np.random.rand(5))
    t1 = Track(m1,c1,det1_init,n_init=1,max_age=3); t1.state = TrackState.Confirmed; t1.time_since_update = 1
    tracks_list.append(t1)

    # Track 2: missed once
    meas2_tlwh = np.array([100,100,30,60]); meas2_xyah = Detection(meas2_tlwh,0.8,'c',None).to_xyah()
    m2,c2 = kf_test.initiate(meas2_xyah); det2_init = Detection(meas2_tlwh,0.8,'c',np.random.rand(5))
    t2 = Track(m2,c2,det2_init,n_init=1,max_age=3); t2.state = TrackState.Confirmed; t2.time_since_update = 2 # Missed once
    tracks_list.append(t2)
    
    # Track 3: very old, should not match in cascade easily
    meas3_tlwh = np.array([200,200,25,50]); meas3_xyah = Detection(meas3_tlwh,0.7,'b',None).to_xyah()
    m3,c3 = kf_test.initiate(meas3_xyah); det3_init = Detection(meas3_tlwh,0.7,'b',np.random.rand(5))
    t3 = Track(m3,c3,det3_init,n_init=1,max_age=3); t3.state = TrackState.Confirmed; t3.time_since_update = 3 # Missed twice
    tracks_list.append(t3)

    # Detections
    # Detection A: good match for Track 1
    det_a = Detection(np.array([12,12,18,38]), 0.95, 'p', np.random.rand(5))
    detections_list.append(det_a)
    # Detection B: good match for Track 2
    det_b = Detection(np.array([102,102,28,58]), 0.85, 'c', np.random.rand(5))
    detections_list.append(det_b)
    # Detection C: new detection
    det_c = Detection(np.array([300,300,40,80]), 0.9, 'n', np.random.rand(5))
    detections_list.append(det_c)

    # Dummy distance metric (e.g., simple sum of TLWH differences, lower is better)
    def dummy_metric(tracks, detections, t_indices, d_indices):
        cost = np.zeros((len(t_indices), len(d_indices)))
        for i, trk_i in enumerate(t_indices):
            for j, det_i in enumerate(d_indices):
                # Simple difference, not a real metric
                cost[i,j] = np.sum(np.abs(tracks[trk_i].to_tlwh() - detections[det_i].tlwh))
        return cost

    # Test 1: min_cost_matching
    print("\n--- Test 1: min_cost_matching ---")
    try:
        matches, unmatched_t, unmatched_d = min_cost_matching(
            dummy_metric, max_distance=50, tracks=tracks_list, detections=detections_list
        )
        print(f"  Matches: {matches}") # Expect (0,0), (1,1) roughly
        print(f"  Unmatched Tracks: {unmatched_t}") # Expect [2]
        print(f"  Unmatched Detections: {unmatched_d}") # Expect [2]
        assert (0,0) in matches or (tracks_list[0].track_id-1, detections_list.index(det_a)) in matches # Check Track 1 with Det A
        assert (1,1) in matches or (tracks_list[1].track_id-1, detections_list.index(det_b)) in matches # Check Track 2 with Det B
        assert 2 in unmatched_t or tracks_list[2].track_id-1 in unmatched_t
        assert 2 in unmatched_d or detections_list.index(det_c) in unmatched_d
        print("Test 1 PASSED (qualitative check).")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        raise

    # Test 2: matching_cascade
    print("\n--- Test 2: matching_cascade ---")
    try:
        # For cascade, tracks with time_since_update=1 are matched first.
        # Max distance 50 should allow t1-det_a and t2-det_b
        cascade_matches, cascade_unmatched_t, cascade_unmatched_d = matching_cascade(
            dummy_metric, max_distance=50, cascade_depth=3, 
            tracks=tracks_list, detections=detections_list
        )
        print(f"  Cascade Matches: {cascade_matches}")
        print(f"  Cascade Unmatched Tracks: {cascade_unmatched_t}")
        print(f"  Cascade Unmatched Detections: {cascade_unmatched_d}")
        # t1 (idx 0) should match det_a (idx 0)
        # t2 (idx 1) should match det_b (idx 1)
        # t3 (idx 2) should be unmatched
        # det_c (idx 2) should be unmatched
        assert (0,0) in cascade_matches or (tracks_list[0].track_id-1, detections_list.index(det_a)) in cascade_matches
        assert (1,1) in cascade_matches or (tracks_list[1].track_id-1, detections_list.index(det_b)) in cascade_matches
        assert 2 in cascade_unmatched_t or tracks_list[2].track_id-1 in cascade_unmatched_t
        assert 2 in cascade_unmatched_d or detections_list.index(det_c) in cascade_unmatched_d
        print("Test 2 PASSED (qualitative check).")

    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        raise

    # Test 3: gate_cost_matrix_by_mahalanobis
    print("\n--- Test 3: gate_cost_matrix_by_mahalanobis ---")
    try:
        # Create a cost matrix (e.g., from an appearance metric)
        # Rows: tracks_list[0], tracks_list[1] (indices 0, 1)
        # Cols: detections_list[0], detections_list[1], detections_list[2] (indices 0, 1, 2)
        trk_indices_for_gate = [0, 1] # t1, t2
        det_indices_for_gate = [0, 1, 2] # det_a, det_b, det_c
        
        # Initial dummy cost matrix (e.g., all costs are low, indicating potential matches)
        appearance_cost_matrix = np.array([
            [0.1, 0.8, 0.9],  # Costs for track t1 vs det_a, det_b, det_c
            [0.7, 0.2, 1.0]   # Costs for track t2 vs det_a, det_b, det_c
        ], dtype=np.float32)
        
        gated_matrix = gate_cost_matrix_by_mahalanobis(
            kf_test, appearance_cost_matrix.copy(), tracks_list, detections_list,
            trk_indices_for_gate, det_indices_for_gate
        )
        print(f"  Original Appearance Costs:\n{appearance_cost_matrix}")
        print(f"  Gated Costs (Mahalanobis):\n{gated_matrix}")

        # Expected behavior (qualitative):
        # - t1 (idx 0) vs det_a (idx 0) should have low Mahalanobis dist, so cost[0,0] remains 0.1
        # - t1 (idx 0) vs det_c (idx 2) should have high Mahalanobis dist, so cost[0,2] becomes INFTY_COST
        # - t2 (idx 1) vs det_b (idx 1) should have low Mahalanobis dist, so cost[1,1] remains 0.2
        # - t2 (idx 1) vs det_c (idx 2) should have high Mahalanobis dist, so cost[1,2] becomes INFTY_COST

        # Check if some costs became INFTY_COST as expected for distant detections
        assert gated_matrix[0, 2] == INFTY_COST, "t1 vs det_c should be gated out by Mahalanobis"
        assert gated_matrix[1, 2] == INFTY_COST, "t2 vs det_c should be gated out by Mahalanobis"
        # Check if good matches remained ungated (their original low cost)
        assert gated_matrix[0, 0] < INFTY_COST 
        assert gated_matrix[1, 1] < INFTY_COST
        print("Test 3 PASSED (qualitative check on gating effect).")

    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        raise
        
    print("--- Finished testing src/tracker/core/linear_assignment.py ---")