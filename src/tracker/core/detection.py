# src/tracker/core/detection.py

import numpy as np

class Detection:
    """
    Represents a single object detection in a frame.

    Attributes:
        tlwh (np.ndarray): Bounding box in (top-left-x, top-left-y, width, height) format.
        confidence (float): Detection confidence score.
        class_name (str): Name of the detected class (e.g., 'person', 'car').
        feature (np.ndarray): Appearance feature vector for the detected object.
    """
    def __init__(self, tlwh: np.ndarray, confidence: float, class_name: str, feature: np.ndarray):
        """
        Args:
            tlwh (np.ndarray): Bounding box (top-left-x, top-left-y, width, height).
            confidence (float): Detection confidence.
            class_name (str): Detected class name.
            feature (np.ndarray): Appearance feature vector.
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_name = class_name # Store class name directly
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None # Feature can be None initially

    def to_tlbr(self) -> np.ndarray:
        """
        Converts bounding box from (tlwh) to (top-left-x, top-left-y, bottom-right-x, bottom-right-y).
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]  # x2 = x1 + w, y2 = y1 + h
        return ret

    def to_xyah(self) -> np.ndarray:
        """
        Converts bounding box from (tlwh) to (center-x, center-y, aspect-ratio, height).
        Aspect ratio is width / height.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0  # cx = x1 + w/2, cy = y1 + h/2
        if ret[3] > 0: # Avoid division by zero if height is 0
            ret[2] /= ret[3]      # a = w / h
        else:
            ret[2] = 0 # Or some other sensible default for aspect ratio if height is 0
        return ret

    def __repr__(self) -> str:
        return (f"Detection(tlwh={self.tlwh}, conf={self.confidence:.2f}, "
                f"cls='{self.class_name}', feat_shape={self.feature.shape if self.feature is not None else 'None'})")

if __name__ == '__main__':
    print("--- Testing src/tracker/core/detection.py ---")

    # Dummy data
    dummy_tlwh = np.array([10, 20, 30, 60], dtype=np.float32) # x, y, w, h
    dummy_confidence = 0.95
    dummy_class_name = "person"
    dummy_feature_dim = 512 # Example feature dimension
    dummy_feature = np.random.rand(dummy_feature_dim).astype(np.float32)

    # Test 1: Create a Detection object
    try:
        det1 = Detection(dummy_tlwh, dummy_confidence, dummy_class_name, dummy_feature)
        print(f"Successfully created Detection object: {det1}")
        assert np.array_equal(det1.tlwh, dummy_tlwh)
        assert det1.confidence == dummy_confidence
        assert det1.class_name == dummy_class_name
        assert det1.feature is not None and det1.feature.shape == (dummy_feature_dim,)
        print("Test 1 PASSED: Object creation and attribute assignment.")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")

    # Test 2: to_tlbr() conversion
    try:
        tlbr = det1.to_tlbr()
        expected_tlbr = np.array([10, 20, 10+30, 20+60], dtype=np.float32) # x1, y1, x2, y2
        assert np.allclose(tlbr, expected_tlbr), f"Expected {expected_tlbr}, got {tlbr}"
        print(f"to_tlbr() output: {tlbr}")
        print("Test 2 PASSED: to_tlbr conversion.")
    except Exception as e:
        print(f"Test 2 FAILED: {e}")

    # Test 3: to_xyah() conversion
    try:
        xyah = det1.to_xyah()
        # cx = 10 + 30/2 = 25
        # cy = 20 + 60/2 = 50
        # a  = 30 / 60 = 0.5
        # h  = 60
        expected_xyah = np.array([25, 50, 0.5, 60], dtype=np.float32)
        assert np.allclose(xyah, expected_xyah), f"Expected {expected_xyah}, got {xyah}"
        print(f"to_xyah() output: {xyah}")
        print("Test 3 PASSED: to_xyah conversion.")
    except Exception as e:
        print(f"Test 3 FAILED: {e}")

    # Test 4: Detection with no feature (e.g., for an unmatched detection before ReID)
    try:
        det_no_feature = Detection(dummy_tlwh, dummy_confidence, dummy_class_name, None)
        print(f"Successfully created Detection object with no feature: {det_no_feature}")
        assert det_no_feature.feature is None
        print("Test 4 PASSED: Object creation with None feature.")
    except Exception as e:
        print(f"Test 4 FAILED: {e}")

    # Test 5: to_xyah() with zero height
    try:
        dummy_tlwh_zero_h = np.array([10, 20, 30, 0], dtype=np.float32)
        det_zero_h = Detection(dummy_tlwh_zero_h, 0.5, "car", None)
        xyah_zero_h = det_zero_h.to_xyah()
        # cx = 10 + 30/2 = 25
        # cy = 20 + 0/2 = 20
        # a  = 0 (due to h=0)
        # h  = 0
        expected_xyah_zero_h = np.array([25, 20, 0, 0], dtype=np.float32)
        assert np.allclose(xyah_zero_h, expected_xyah_zero_h), f"Expected {expected_xyah_zero_h}, got {xyah_zero_h}"
        print(f"to_xyah() with zero height output: {xyah_zero_h}")
        print("Test 5 PASSED: to_xyah with zero height.")
    except Exception as e:
        print(f"Test 5 FAILED: {e}")

    print("--- Finished testing src/tracker/core/detection.py ---")