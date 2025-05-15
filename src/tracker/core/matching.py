# src/tracker/core/matching.py

import numpy as np
from typing import List, Optional

# Forward declare Track and Detection for type hinting if they are in the same 'core' package
# from .track import Track (already imported in linear_assignment, but good practice here too if used directly)
# from .detection import Detection

# Import INFTY_COST if needed for some specific logic, though usually gating handles this
from .linear_assignment import INFTY_COST 

def iou(bbox_tlwh: np.ndarray, candidates_tlwh: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) between a bounding box and candidate boxes.

    Args:
        bbox_tlwh (np.ndarray): A single bounding box (top-left-x, top-left-y, width, height).
        candidates_tlwh (np.ndarray): An array of N candidate bounding boxes (N, 4)
                                      in the same (tlwh) format.

    Returns:
        np.ndarray: An array of N IoU scores, one for each candidate.
    """
    if candidates_tlwh.size == 0:
        return np.array([], dtype=np.float32)

    # Convert to (x1, y1, x2, y2) format
    box_tl, box_br = bbox_tlwh[:2], bbox_tlwh[:2] + bbox_tlwh[2:]
    candidates_tl = candidates_tlwh[:, :2]
    candidates_br = candidates_tlwh[:, :2] + candidates_tlwh[:, 2:]

    # Calculate intersection areas
    # Top-left corner of intersection
    inter_tl_x = np.maximum(box_tl[0], candidates_tl[:, 0])
    inter_tl_y = np.maximum(box_tl[1], candidates_tl[:, 1])
    # Bottom-right corner of intersection
    inter_br_x = np.minimum(box_br[0], candidates_br[:, 0])
    inter_br_y = np.minimum(box_br[1], candidates_br[:, 1])

    # Width and height of intersection
    inter_w = np.maximum(0., inter_br_x - inter_tl_x)
    inter_h = np.maximum(0., inter_br_y - inter_tl_y)
    
    area_intersection = inter_w * inter_h

    # Calculate union areas
    area_box = bbox_tlwh[2] * bbox_tlwh[3]
    area_candidates = candidates_tlwh[:, 2] * candidates_tlwh[:, 3]
    area_union = area_box + area_candidates - area_intersection

    # IoU: intersection / union. Avoid division by zero.
    iou_scores = area_intersection / np.maximum(area_union, 1e-7) # Add epsilon for stability
    return iou_scores


def iou_cost(
    tracks: List['Track'], # Use string literal for forward declaration if Track is not yet fully defined/imported
    detections: List['Detection'],
    track_indices: List[int],
    detection_indices: List[int]
) -> np.ndarray:
    """
    Computes the IoU-based cost matrix for matching tracks and detections.
    Cost is defined as 1 - IoU.

    Args:
        tracks: List of active tracks.
        detections: List of current detections.
        track_indices: Indices of tracks to compute costs for.
        detection_indices: Indices of detections to compute costs for.

    Returns:
        np.ndarray: An M x N cost matrix, where M is len(track_indices)
                    and N is len(detection_indices). Cost = 1 - IoU.
    """
    num_tracks = len(track_indices)
    num_detections = len(detection_indices)

    if num_tracks == 0 or num_detections == 0:
        return np.empty((num_tracks, num_detections), dtype=np.float32)

    cost_matrix = np.full((num_tracks, num_detections), INFTY_COST, dtype=np.float32)

    for row_idx, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        
        # IoU matching is typically used for tracks that were recently updated or are tentative.
        # The original DeepSORT applies IoU matching for tracks with time_since_update == 1
        # or unconfirmed tracks. This logic is usually handled by the calling Tracker.
        # Here, we just compute the cost. If a track is too old, its cost might be set high
        # by the gating mechanism or by specific tracker logic.
        # if track.time_since_update > 1 and track.is_confirmed(): # Example of such logic
        #     cost_matrix[row_idx, :] = INFTY_COST # Or handled by gating
        #     continue

        track_bbox_tlwh = track.to_tlwh()
        candidate_detections_tlwh = np.asarray(
            [detections[det_idx].tlwh for det_idx in detection_indices], dtype=np.float32
        )

        if candidate_detections_tlwh.size > 0:
            iou_scores = iou(track_bbox_tlwh, candidate_detections_tlwh)
            cost_matrix[row_idx, :] = 1.0 - iou_scores
            
    return cost_matrix


def cosine_distance(features_a: np.ndarray, features_b: np.ndarray, data_is_normalized: bool = False) -> np.ndarray:
    """
    Computes pairwise cosine distance between two sets of feature vectors.
    Distance = 1 - CosineSimilarity.

    Args:
        features_a (np.ndarray): An M x D array of M feature vectors.
        features_b (np.ndarray): An N x D array of N feature vectors.
        data_is_normalized (bool): If True, assumes features are already L2 normalized.

    Returns:
        np.ndarray: An M x N distance matrix.
    """
    if features_a.size == 0 or features_b.size == 0:
        return np.empty((features_a.shape[0], features_b.shape[0]), dtype=np.float32)

    if not data_is_normalized:
        norm_a = np.linalg.norm(features_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(features_b, axis=1, keepdims=True)
        # Handle potential zero norm vectors to avoid division by zero / NaN
        features_a_normalized = features_a / np.maximum(norm_a, 1e-7)
        features_b_normalized = features_b / np.maximum(norm_b, 1e-7)
    else:
        features_a_normalized = features_a
        features_b_normalized = features_b
    
    # Cosine similarity: dot(A, B.T)
    similarity_matrix = np.dot(features_a_normalized, features_b_normalized.T)
    
    # Cosine distance: 1 - similarity
    # Clip to ensure distance is non-negative (due to potential floating point inaccuracies)
    distance_matrix = 1.0 - similarity_matrix
    return np.maximum(distance_matrix, 0.0)


def appearance_cost_metric(
    tracks: List['Track'],
    detections: List['Detection'],
    track_indices: List[int],
    detection_indices: List[int],
    metric_type: str = "cosine" # Could be "euclidean" in future
) -> np.ndarray:
    """
    Computes the appearance-based cost matrix using feature cosine distance.
    Each track has a gallery of features. The cost between a track and a detection
    is the minimum cosine distance between the detection's feature and any feature
    in the track's gallery.

    Args:
        tracks: List of active tracks.
        detections: List of current detections.
        track_indices: Indices of tracks to compute costs for.
        detection_indices: Indices of detections to compute costs for.
        metric_type: The type of distance metric for appearance (currently 'cosine').

    Returns:
        np.ndarray: An M x N cost matrix.
    """
    num_tracks = len(track_indices)
    num_detections = len(detection_indices)

    if num_tracks == 0 or num_detections == 0:
        return np.empty((num_tracks, num_detections), dtype=np.float32)

    cost_matrix = np.full((num_tracks, num_detections), INFTY_COST, dtype=np.float32)

    # Collect all detection features once
    detection_features_list = []
    valid_det_indices_map = {} # Map original detection_indices to their new index in detection_features_list
    current_valid_idx = 0
    for det_idx in detection_indices:
        if detections[det_idx].feature is not None:
            detection_features_list.append(detections[det_idx].feature)
            valid_det_indices_map[det_idx] = current_valid_idx
            current_valid_idx += 1
    
    if not detection_features_list: # No detections have features
        return cost_matrix # All costs remain INFTY_COST

    all_detection_features = np.asarray(detection_features_list, dtype=np.float32)

    for row_idx, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if not track.features: # Track has no features in its gallery
            cost_matrix[row_idx, :] = INFTY_COST # Cannot match by appearance
            continue

        track_gallery_features = np.asarray(track.features, dtype=np.float32)
        
        # Compute distances from all features in track's gallery to all detection features
        # This results in a (num_track_features x num_detection_features) distance matrix
        if metric_type == "cosine":
            dist_matrix_gallery_to_detections = cosine_distance(track_gallery_features, all_detection_features)
        else:
            raise ValueError(f"Unsupported appearance metric_type: {metric_type}")

        # The cost for this track vs each detection is the *minimum* distance found
        # from any feature in its gallery to the detection's feature.
        min_distances_to_detections = np.min(dist_matrix_gallery_to_detections, axis=0)
        
        # Assign these minimum distances to the correct columns in the main cost_matrix
        for original_det_idx, mapped_det_idx in valid_det_indices_map.items():
            # Find where original_det_idx appears in the requested detection_indices list
            # This maps back from the dense `all_detection_features` to the sparse `detection_indices`
            if original_det_idx in detection_indices:
                col_idx_in_cost_matrix = detection_indices.index(original_det_idx)
                cost_matrix[row_idx, col_idx_in_cost_matrix] = min_distances_to_detections[mapped_det_idx]
                
    return cost_matrix


if __name__ == '__main__':
    # Import dummy classes from where they would be for testing
    from .track import Track, TrackState
    from .detection import Detection
    import numpy as np

    print("--- Testing src/tracker/core/matching.py ---")
    Track.reset_id_counter()

    # Test 1: iou() function
    print("\n--- Test 1: iou() ---")
    bbox1_tlwh = np.array([0,0,10,10])
    candidates_tlwh_t1 = np.array([
        [0,0,10,10],      # Perfect overlap
        [5,5,10,10],      # Partial overlap (25% of area of bbox1)
        [0,0,5,5],        # bbox1 contains this (25% of area of bbox1)
        [20,20,10,10]     # No overlap
    ])
    try:
        iou_scores = iou(bbox1_tlwh, candidates_tlwh_t1)
        print(f"  IoU scores: {iou_scores}")
        assert np.isclose(iou_scores[0], 1.0)
        # For [5,5,10,10] vs [0,0,10,10]: Intersection is 5x5=25. Union is 100+100-25=175. IoU=25/175 approx 0.1428
        assert np.isclose(iou_scores[1], 25.0 / (100.0 + 100.0 - 25.0))
        # For [0,0,5,5] vs [0,0,10,10]: Intersection is 5x5=25. Union is 100+25-25=100. IoU=25/100 = 0.25
        assert np.isclose(iou_scores[2], 25.0 / 100.0)
        assert np.isclose(iou_scores[3], 0.0)
        print("Test 1 PASSED.")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        raise

    # Dummy tracks and detections for cost matrix tests
    # Track t_iou (idx 0)
    det_init_iou = Detection(np.array([10,10,20,30]), 0.9, 'p', None) # No feature needed for IoU
    # Note: Track needs mean/cov, but for iou_cost, only to_tlwh() is used from track.
    # We can mock a track or create a simple one.
    class MockTrack:
        def __init__(self, tlwh, time_since_update=0, state=TrackState.Confirmed):
            self.tlwh_val = tlwh
            self.time_since_update = time_since_update
            self.state_val = state
        def to_tlwh(self): return self.tlwh_val
        def is_confirmed(self): return self.state_val == TrackState.Confirmed

    tracks_for_iou = [MockTrack(np.array([10,10,20,30]))]
    detections_for_iou = [
        Detection(np.array([10,10,20,30]), 0.9, 'p', None), # Perfect match
        Detection(np.array([100,100,10,10]), 0.8, 'p', None) # No overlap
    ]

    # Test 2: iou_cost()
    print("\n--- Test 2: iou_cost() ---")
    try:
        cost_mat_iou = iou_cost(tracks_for_iou, detections_for_iou, [0], [0,1])
        print(f"  IoU Cost Matrix:\n{cost_mat_iou}")
        assert np.isclose(cost_mat_iou[0,0], 0.0) # 1 - 1.0
        assert np.isclose(cost_mat_iou[0,1], 1.0) # 1 - 0.0
        print("Test 2 PASSED.")
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        raise

    # Test 3: cosine_distance()
    print("\n--- Test 3: cosine_distance() ---")
    feat_a1 = np.array([[1,0,0], [0,1,0]], dtype=np.float32) # Normalized
    feat_b1 = np.array([[1,0,0], [0,0,1], [0.707, 0.707, 0]], dtype=np.float32) # b1[2] is normalized
    try:
        cos_dist = cosine_distance(feat_a1, feat_b1, data_is_normalized=True)
        print(f"  Cosine Distance Matrix (normalized input):\n{cos_dist}")
        # Expected:
        # [[0.0, 1.0, 0.293],  <- feat_a1[0] vs feat_b1
        #  [1.0, 0.0, 0.293]]  <- feat_a1[1] vs feat_b1
        assert np.isclose(cos_dist[0,0], 0.0)
        assert np.isclose(cos_dist[0,1], 1.0)
        assert np.isclose(cos_dist[0,2], 1.0 - 0.707)
        assert np.isclose(cos_dist[1,0], 1.0)
        assert np.isclose(cos_dist[1,1], 1.0) # [0,1,0] vs [0,0,1] should be 1.0 (orthogonal)
        assert np.isclose(cos_dist[1,2], 1.0 - 0.707)
        
        # Test with unnormalized data
        feat_a2_un = np.array([[2,0,0]], dtype=np.float32)
        feat_b2_un = np.array([[3,0,0],[0,4,0]], dtype=np.float32)
        cos_dist_un = cosine_distance(feat_a2_un, feat_b2_un, data_is_normalized=False)
        print(f"  Cosine Distance Matrix (unnormalized input):\n{cos_dist_un}")
        assert np.isclose(cos_dist_un[0,0], 0.0) # Parallel vectors
        assert np.isclose(cos_dist_un[0,1], 1.0) # Orthogonal vectors
        print("Test 3 PASSED.")
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        raise
        
    # Test 4: appearance_cost_metric() with a track having multiple gallery features
    print("\n--- Test 4: appearance_cost_metric() with gallery ---")
    # Mock Track with features
    class MockTrackWithFeatures(MockTrack):
        def __init__(self, tlwh, features_list, time_since_update=0, state=TrackState.Confirmed):
            super().__init__(tlwh, time_since_update, state)
            self.features = features_list
    track_gallery = MockTrackWithFeatures(np.array([0,0,1,1]), 
                                         [np.array([1,0,0], dtype=np.float32),      # feat1
                                          np.array([0.9,0.1,0], dtype=np.float32)]) # feat2 (slightly different from feat1)
    det_match_feat2 = Detection(np.array([0,0,1,1]), 0.9, 'p', np.array([0.9,0.1,0], dtype=np.float32)) # Matches feat2
    
    try:
        cost_gallery = appearance_cost_metric([track_gallery], [det_match_feat2], [0], [0])
        print(f"  Appearance Cost (gallery vs det_match_feat2): {cost_gallery}")
        # Should pick the minimum distance, which should be close to 0
        assert np.isclose(cost_gallery[0,0], 0.0)
        print("Test 4 PASSED.")
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        raise

    print("--- Finished testing src/tracker/core/matching.py ---")