#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define model URLs
YOLO_ONNX_URL="https://raw.githubusercontent.com/nabang1010/YOLO_Object_Tracking_TensorRT/main/models/onnx/yolov8n.onnx"
DEEPSORT_ONNX_URL="https://raw.githubusercontent.com/nabang1010/YOLO_Object_Tracking_TensorRT/main/models/onnx/deepsort.onnx"

# Define output directories
MODELS_DIR="./models" # Relative to the scripts directory
DETECTION_MODEL_DIR="$MODELS_DIR/detection"
REID_MODEL_DIR="$MODELS_DIR/reid"

# Define output filenames
YOLO_ONNX_FILENAME="yolov8n.onnx"
DEEPSORT_ONNX_FILENAME="deepsort_reid.onnx" # New name for clarity

# Create directories if they don't exist
mkdir -p "$DETECTION_MODEL_DIR"
mkdir -p "$REID_MODEL_DIR"

echo "Starting model download..."

# Download YOLOv8 ONNX model
echo "Downloading YOLOv8 ONNX model from $YOLO_ONNX_URL..."
if curl -L "$YOLO_ONNX_URL" -o "$DETECTION_MODEL_DIR/$YOLO_ONNX_FILENAME"; then
    echo "YOLOv8 ONNX model downloaded successfully to $DETECTION_MODEL_DIR/$YOLO_ONNX_FILENAME"
else
    echo "Failed to download YOLOv8 ONNX model. Please check the URL and your internet connection."
    exit 1
fi

# Download DeepSORT ReID ONNX model
echo "Downloading DeepSORT ReID ONNX model from $DEEPSORT_ONNX_URL..."
if curl -L "$DEEPSORT_ONNX_URL" -o "$REID_MODEL_DIR/$DEEPSORT_ONNX_FILENAME"; then
    echo "DeepSORT ReID ONNX model downloaded successfully to $REID_MODEL_DIR/$DEEPSORT_ONNX_FILENAME"
else
    echo "Failed to download DeepSORT ReID ONNX model. Please check the URL and your internet connection."
    exit 1
fi

echo "All models downloaded."