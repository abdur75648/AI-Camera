# src/tracker/deepsort_tracker.py

import numpy as np
import cv2 # For image cropping
from typing import List, Tuple, Optional
import torch # For torch.device, though TRTEngine internally handles it
import traceback # For detailed error reporting in tests
from collections import defaultdict
from typing import Dict
from .reid_model import ReIDModel
from .core.tracker_core import TrackerCore
from .core.detection import Detection as CoreDetection
from .. import config

class DeepSORT:
    """
    High-level DeepSORT tracker.
    Integrates a Re-Identification (ReID) model for appearance feature extraction
    and a TrackerCore for managing track states, Kalman filtering, and associations.
    """
    def __init__(self,
                 reid_model_path: str = str(config.REID_ENGINE_PATH),
                 reid_input_shape: Tuple[int, int] = config.REID_INPUT_SHAPE,
                 max_cosine_distance: float = config.DEEPSORT_MAX_DIST,
                 nn_budget: Optional[int] = config.DEEPSORT_NN_BUDGET,
                 max_iou_distance: float = config.DEEPSORT_MAX_IOU_DISTANCE,
                 max_age: int = config.DEEPSORT_MAX_AGE,
                 n_init: int = config.DEEPSORT_N_INIT,
                 min_detection_confidence: float = config.DEEPSORT_MIN_CONFIDENCE
                ):
        """
        Initializes the DeepSORT tracker.

        Args:
            reid_model_path: Path to the ReID TensorRT engine.
            reid_input_shape: Input shape (H, W) for the ReID model.
            max_cosine_distance: Threshold for appearance feature matching.
            nn_budget: Maximum number of features to store per track gallery.
            max_iou_distance: Gating threshold for IoU matching (1 - min_iou).
            max_age: Maximum number of frames a track can be missed before deletion.
            n_init: Number of consecutive updates required to confirm a new track.
            min_detection_confidence: Minimum confidence for YOLO detections to be processed.
        """
        self.reid_model = ReIDModel(
            engine_path=reid_model_path,
            input_shape=reid_input_shape
        )
        self.tracker_core = TrackerCore(
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )
        self.min_detection_confidence = min_detection_confidence
        self.frame_count = 0 # For debugging or time-dependent logic

        print("DeepSORT Tracker initialized.")
        print(f"  ReID Model: {reid_model_path} (Input: {reid_input_shape})")
        print(f"  TrackerCore Params: CosDist={max_cosine_distance}, IoUDist={max_iou_distance}, "
              f"MaxAge={max_age}, NInit={n_init}, NNBudget={nn_budget}")

    def update(self,
               yolo_bboxes_xyxy: np.ndarray,
               yolo_confidences: np.ndarray,
               yolo_class_ids: np.ndarray,
               original_frame_bgr: np.ndarray
              ) -> List[Tuple[int, int, int, int, int, str, float]]:
        """
        Processes object detections from YOLO and updates tracks.

        Args:
            yolo_bboxes_xyxy: Detected bounding boxes (N, 4) in (x1, y1, x2, y2) format.
            yolo_confidences: Confidence scores (N,) for each detection.
            yolo_class_ids: Class IDs (N,) for each detection.
            original_frame_bgr: The full BGR frame from which detections were made.

        Returns:
            A list of tuples for currently confirmed and updated tracks:
            (x1, y1, x2, y2, track_id, class_name, track_confidence).
        """
        self.frame_count += 1
        
        # Always predict track states forward for the current frame
        self.tracker_core.predict()

        # 1. Filter YOLO detections based on confidence and class
        process_indices = []
        for i in range(len(yolo_bboxes_xyxy)):
            class_id_int = int(yolo_class_ids[i])
            # Ensure class_id is within bounds of config.CLASSES
            class_name = config.CLASSES[class_id_int] if 0 <= class_id_int < len(config.CLASSES) else "Unknown"
            if (yolo_confidences[i] >= self.min_detection_confidence and
                class_name in config.CLASSES_TO_TRACK):
                process_indices.append(i)

        core_detections: List[CoreDetection] = []
        if process_indices:
            filtered_bboxes_xyxy = yolo_bboxes_xyxy[process_indices]
            filtered_confidences = yolo_confidences[process_indices]
            filtered_class_ids = yolo_class_ids[process_indices]

            # 2. Extract image crops for ReID
            image_crops = self._extract_image_crops(original_frame_bgr, filtered_bboxes_xyxy)

            # 3. Extract ReID features for valid crops
            appearance_features = np.empty((0,0)) # Default if no valid crops
            valid_crop_indices = [i for i, crop in enumerate(image_crops) if crop.size > 0]
            
            if valid_crop_indices:
                valid_image_crops = [image_crops[i] for i in valid_crop_indices]
                if valid_image_crops: # Ensure list is not empty after filtering
                     appearance_features = self.reid_model.extract_features_batched(valid_image_crops)

            # 4. Create CoreDetection objects for TrackerCore
            core_detections = self._create_core_detections(
                filtered_bboxes_xyxy, filtered_confidences, filtered_class_ids,
                appearance_features, valid_crop_indices
            )
        
        # 5. Update TrackerCore with the (potentially empty) list of CoreDetections
        # If core_detections is empty, TrackerCore.update will mark all tracks as missed.
        self.tracker_core.update(core_detections)

        # 6. Format output for confirmed tracks that were updated in this frame
        output_tracks = []
        for track in self.tracker_core.tracks: # .tracks is already cleaned of deleted items by TrackerCore.update
            if track.is_confirmed() and track.time_since_update == 0:
                tlwh = track.to_tlwh()
                x1, y1, w, h = tlwh
                # Ensure w and h are non-negative before calculating x2, y2
                w = max(0, w)
                h = max(0, h)
                x2, y2 = x1 + w, y1 + h
                output_tracks.append(
                    (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
                     track.track_id,
                     track.class_name,
                     float(track.confidence)) # Confidence of the last matched detection
                )
        return output_tracks

    def _extract_image_crops(self, frame_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> List[np.ndarray]:
        """Helper to extract image crops from bounding boxes, ensuring valid coordinates."""
        crops = []
        frame_h, frame_w = frame_bgr.shape[:2]
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            # Clamp coordinates to frame boundaries
            x1_c = max(0, x1)
            y1_c = max(0, y1)
            x2_c = min(frame_w, x2)
            y2_c = min(frame_h, y2)
            
            if x1_c < x2_c and y1_c < y2_c: # Check for valid crop dimensions
                crops.append(frame_bgr[y1_c:y2_c, x1_c:x2_c])
            else:
                crops.append(np.array([])) # Placeholder for invalid/zero-area crop
        return crops

    def _create_core_detections(self,
                                bboxes_xyxy: np.ndarray,
                                confidences: np.ndarray,
                                class_ids: np.ndarray,
                                appearance_features: np.ndarray,
                                valid_crop_indices: List[int] # Indices in bboxes_xyxy that had valid crops
                               ) -> List[CoreDetection]:
        """Helper to create CoreDetection objects, correctly mapping features to detections."""
        core_detections_list: List[CoreDetection] = []
        
        # Create a map from the index in valid_crop_indices to the actual feature vector
        # This assumes appearance_features corresponds 1:1 with valid_image_crops
        feature_for_valid_crop_idx: Dict[int, np.ndarray] = {}
        if appearance_features.ndim == 2 and appearance_features.shape[0] == len(valid_crop_indices):
            feature_for_valid_crop_idx = {
                original_idx: appearance_features[i] 
                for i, original_idx in enumerate(valid_crop_indices)
            }

        for i in range(len(bboxes_xyxy)): # Iterate through all *filtered* YOLO detections
            bbox = bboxes_xyxy[i]
            conf = confidences[i]
            class_id_int = int(class_ids[i])
            class_name = config.CLASSES[class_id_int] if 0 <= class_id_int < len(config.CLASSES) else "Unknown"

            x1, y1, x2, y2 = bbox
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
            
            # Get feature if this detection (index i) corresponded to a valid crop and had a feature extracted
            current_feature = feature_for_valid_crop_idx.get(i, None)
            
            core_det = CoreDetection(
                tlwh=tlwh,
                confidence=float(conf),
                class_name=class_name,
                feature=current_feature # Will be None if no valid crop or feature for this detection
            )
            core_detections_list.append(core_det)
        return core_detections_list


# --- Test Suite ---
if __name__ == '__main__':
    from .core.track import TrackState 
    from .. import config

    print("--- Testing src/tracker/deepsort_tracker.py ---")

    if not config.REID_ENGINE_PATH.exists():
        print(f"Test skipped: ReID engine not found at {config.REID_ENGINE_PATH}")
        print("Please ensure 'scripts/download_models.sh' and 'scripts/export_trt_engines.sh' have been run.")
    else:
        try:
            # Test-specific parameters for faster state changes
            test_n_init = 2
            test_max_age = 1 # Track deleted if missed for *more than* 1 frame (i.e., tsu > 1)
            
            deepsort_tracker = DeepSORT(
                reid_model_path=str(config.REID_ENGINE_PATH),
                n_init=test_n_init,
                max_age=test_max_age 
            )
            print("DeepSORT tracker initialized for test.")

            frame_h, frame_w = 720, 1280
            dummy_frame = np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)

            # --- Frame 1: Initiate two tracks ---
            print("\n--- Test Frame 1 ---")
            yolo_boxes_f1 = np.array([ # (x1, y1, x2, y2)
                [50, 50, 50+20, 50+40],     # Person 1
                [300, 100, 300+30, 100+60], # Car 1
                [500, 500, 550, 600]        # Low conf person (filtered by min_detection_confidence)
            ], dtype=np.float32)
            yolo_scores_f1 = np.array([0.9, 0.85, 0.2], dtype=np.float32) 
            yolo_classes_f1 = np.array([
                config.CLASSES.index('person'), 
                config.CLASSES.index('car'),
                config.CLASSES.index('person')
                ], dtype=np.int32)

            tracked_objects_f1 = deepsort_tracker.update(
                yolo_boxes_f1, yolo_scores_f1, yolo_classes_f1, dummy_frame
            )
            print(f"  Tracked objects Output Frame 1: {tracked_objects_f1}")
            assert len(tracked_objects_f1) == 0, "Frame 1: No tracks should be confirmed and outputted yet."
            assert len(deepsort_tracker.tracker_core.tracks) == 2, "Frame 1: Two high-confidence detections should initiate tracks."
            for track in deepsort_tracker.tracker_core.tracks:
                assert track.state == TrackState.Tentative
            print("  Frame 1 PASSED: Tracks initiated as Tentative, no confirmed output.")

            # --- Frame 2: Update and confirm tracks ---
            print("\n--- Test Frame 2 ---")
            yolo_boxes_f2 = np.array([ # (x1, y1, x2, y2)
                [52, 52, 52+20, 52+40],       # Match for Track ID 1 (Person)
                [302, 102, 302+30, 102+60],   # Match for Track ID 2 (Car)
                [700, 200, 700+25, 200+50]    # New Person (Track ID 3)
            ], dtype=np.float32)
            yolo_scores_f2 = np.array([0.92, 0.88, 0.8], dtype=np.float32)
            yolo_classes_f2 = np.array([
                config.CLASSES.index('person'),
                config.CLASSES.index('car'),
                config.CLASSES.index('person')
                ], dtype=np.int32)

            tracked_objects_f2 = deepsort_tracker.update(
                yolo_boxes_f2, yolo_scores_f2, yolo_classes_f2, dummy_frame
            )
            print(f"  Tracked objects Output Frame 2: {tracked_objects_f2}")
            assert len(tracked_objects_f2) == 2, "Frame 2: Tracks 1 and 2 should be confirmed and outputted."
            
            internal_tracks_f2 = deepsort_tracker.tracker_core.tracks
            assert len(internal_tracks_f2) == 3, "Frame 2: Should have 3 internal tracks (2 confirmed, 1 new tentative)."
            
            confirmed_output_ids = {obj[4] for obj in tracked_objects_f2}
            assert 1 in confirmed_output_ids and 2 in confirmed_output_ids
            
            track1_f2 = next(t for t in internal_tracks_f2 if t.track_id == 1)
            track2_f2 = next(t for t in internal_tracks_f2 if t.track_id == 2)
            track3_f2 = next(t for t in internal_tracks_f2 if t.track_id == 3)
            assert track1_f2.state == TrackState.Confirmed and track1_f2.hits == test_n_init
            assert track2_f2.state == TrackState.Confirmed and track2_f2.hits == test_n_init
            assert track3_f2.state == TrackState.Tentative and track3_f2.hits == 1
            print(f"  Internal tracks state Frame 2: {[str(t) for t in internal_tracks_f2]}")
            print("  Frame 2 PASSED: Tracks confirmed, new track initiated.")

            # --- Frame 3: Track missing; new track (ID 3) gets confirmed ---
            print("\n--- Test Frame 3 (Track Missing & Confirmation) ---")
            # Tracks 1 & 2 are missed. Track 3 (initiated in F2) gets another hit.
            yolo_boxes_f3 = np.array([ # (x1, y1, x2, y2)
                [702, 202, 702+25, 202+50] # Match for Track ID 3 (New Person from F2)
            ], dtype=np.float32)
            yolo_scores_f3 = np.array([0.82], dtype=np.float32)
            yolo_classes_f3 = np.array([config.CLASSES.index('person')], dtype=np.int32)
            
            tracked_objects_f3 = deepsort_tracker.update(
                yolo_boxes_f3, yolo_scores_f3, yolo_classes_f3, dummy_frame
            )
            # Expected after F3 update:
            # T1: predict (tsu=1), update (miss) -> mark_missed. tsu=1. state=Confirmed. _max_age=1.
            # T2: predict (tsu=1), update (miss) -> mark_missed. tsu=1. state=Confirmed. _max_age=1.
            # T3: predict (tsu=1), update (match) -> hits=2, tsu=0. state becomes Confirmed. _max_age=1.
            print(f"  Tracked objects Output Frame 3: {tracked_objects_f3}")
            assert len(tracked_objects_f3) == 1, "Frame 3: Only Track 3 should be outputted as it was updated."
            assert tracked_objects_f3[0][4] == 3, "Frame 3: Track ID 3 should be the one outputted."

            internal_tracks_f3_after_update = deepsort_tracker.tracker_core.tracks
            print(f"  Internal tracks after F3 update: {[str(t) for t in internal_tracks_f3_after_update]}")
            track1_f3 = next(t for t in internal_tracks_f3_after_update if t.track_id == 1)
            track2_f3 = next(t for t in internal_tracks_f3_after_update if t.track_id == 2)
            track3_f3 = next(t for t in internal_tracks_f3_after_update if t.track_id == 3)
            assert track1_f3.time_since_update == 1 and track1_f3.state == TrackState.Confirmed
            assert track2_f3.time_since_update == 1 and track2_f3.state == TrackState.Confirmed
            assert track3_f3.time_since_update == 0 and track3_f3.state == TrackState.Confirmed # Confirmed now

            # --- Frame 4: Simulate empty detections. Tracks 1 & 2 should be deleted. Track 3 missed. ---
            print("  Simulating Frame 4 (empty detections for T1, T2, T3)")
            # T1: predict (tsu=2), update (miss) -> mark_missed. tsu (2) > _max_age (1) -> Deleted
            # T2: predict (tsu=2), update (miss) -> mark_missed. tsu (2) > _max_age (1) -> Deleted
            # T3: predict (tsu=1), update (miss) -> mark_missed. tsu (1) <= _max_age (1) -> Confirmed
            tracked_objects_f4 = deepsort_tracker.update(
                np.array([]), np.array([]), np.array([]), dummy_frame
            )
            print(f"  Tracked objects Output Frame 4: {tracked_objects_f4}")
            assert len(tracked_objects_f4) == 0, "Frame 4: No tracks should be outputted if all missed."
            
            active_internal_ids_f4 = {t.track_id for t in deepsort_tracker.tracker_core.tracks if not t.is_deleted()}
            print(f"  Internal track IDs after F4 update: {active_internal_ids_f4}")
            print(f"  Internal tracks after F4 update: {[str(t) for t in deepsort_tracker.tracker_core.tracks]}")
            
            track1_f4 = next((t for t in deepsort_tracker.tracker_core.tracks if t.track_id == 1), None)
            track2_f4 = next((t for t in deepsort_tracker.tracker_core.tracks if t.track_id == 2), None)
            track3_f4 = next((t for t in deepsort_tracker.tracker_core.tracks if t.track_id == 3), None)

            assert track1_f4 is None or track1_f4.is_deleted(), f"Track 1 should be deleted. State: {track1_f4}"
            assert track2_f4 is None or track2_f4.is_deleted(), f"Track 2 should be deleted. State: {track2_f4}"
            assert 3 in active_internal_ids_f4, "Track 3 should still be active"
            if track3_f4:
                 assert track3_f4.time_since_update == 1 and track3_f4.state == TrackState.Confirmed
            
            print("  Frame 3 & 4 (Track Missing & Deletion) PASSED.")

        except Exception as e:
            print(f"ERROR during DeepSORT tracker test: {e}")
            traceback.print_exc()