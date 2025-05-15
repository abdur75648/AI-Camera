import random
import cv2
import numpy as np
from pathlib import Path

# --- Project Root ---
# Assuming this config.py is in src/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Model Configuration ---
# Paths to TensorRT engine files
YOLO_ENGINE_PATH = PROJECT_ROOT / "models/detection/yolov8n.engine"
REID_ENGINE_PATH = PROJECT_ROOT / "models/reid/deepsort_reid.engine"

# YOLOv8 specific
YOLO_INPUT_SHAPE = (640, 640) # H, W
YOLO_CONF_THRESHOLD = 0.3    # Confidence threshold for YOLO detections
YOLO_NMS_THRESHOLD = 0.5     # NMS threshold for YOLO (if NMS is done post-inference)
                               # Note: Ultralytics export often includes NMS in the engine.

# DeepSORT specific
# These values are typically from the original deep_sort.yaml
DEEPSORT_MAX_DIST = 0.2             # Maximum cosine distance for appearance matching
DEEPSORT_MIN_CONFIDENCE = 0.3       # Minimum detection confidence for a ReID feature to be considered
DEEPSORT_NMS_MAX_OVERLAP = 1.0      # NMS overlap for detections passed to DeepSORT (usually high if YOLO already did NMS)
DEEPSORT_MAX_IOU_DISTANCE = 0.7     # Gating threshold for IOU matching
DEEPSORT_MAX_AGE = 70               # Maximum number of frames to keep a track alive without updates
DEEPSORT_N_INIT = 3                 # Number of consecutive detections to confirm a track
DEEPSORT_NN_BUDGET = 100            # Maximum number of features to store per track for ReID

# ReID model specific
REID_INPUT_SHAPE = (128, 64) # H, W for the ReID model input

# --- Class Configuration (COCO for YOLOv8) ---
# List of class names corresponding to the YOLOv8 model's output indices
CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
)

# --- Tracking Configuration ---
# Specify which classes to track (e.g., only 'person')
# Use a set for efficient lookup
CLASSES_TO_TRACK = {'person', 'car', 'bus', 'truck', 'motorcycle'}

# --- Visualization Configuration ---
# Seed for consistent random colors, or remove for fully random colors each run
# random.seed(42)

# Generate a unique color for each class for bounding boxes
CLASS_COLORS = {
    cls_name: [random.randint(0, 255) for _ in range(3)]
    for cls_name in CLASSES
}

# Fallback color for tracks if class-specific color isn't found (should not happen if configured correctly)
DEFAULT_TRACK_COLOR = (0, 255, 0) # Green

# Font for text overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_ID = 0.7
FONT_SCALE_INFO = 0.9
FONT_THICKNESS = 2

# --- Video I/O ---
DEFAULT_OUTPUT_FPS = 30


# --- Helper function to get color for a track ---
def get_track_color(class_name):
    """Returns a color for a given class name, or a default color."""
    return CLASS_COLORS.get(class_name, DEFAULT_TRACK_COLOR)

def get_class_color(class_name):
    """Returns a color for a given class name for general detections."""
    return CLASS_COLORS.get(class_name, (200, 200, 200)) # Light gray for unknown

# --- Sanity Checks (Optional, but good for development) ---
if not YOLO_ENGINE_PATH.exists():
    print(f"Warning: YOLO Engine not found at {YOLO_ENGINE_PATH}")
if not REID_ENGINE_PATH.exists():
    print(f"Warning: ReID Engine not found at {REID_ENGINE_PATH}")