#!/bin/bash

set -e

# --- Configuration ---
# Adjust this path if trtexec is located elsewhere
TRTEXEC_PATH="trtexec" # Assumes trtexec is in PATH.

# NOTE: We're handling both static and dynamic ONNX models:
# - Static models: TensorRT doesn't accept shape parameters (--minShapes, --optShapes, --maxShapes)
# - Dynamic models: TensorRT accepts shape parameters, but we need to ensure the model supports dynamic shapes.

# Model paths (relative to the AICamera project root)
MODELS_DIR="./models"
DETECTION_MODEL_DIR="$MODELS_DIR/detection"
REID_MODEL_DIR="$MODELS_DIR/reid"

YOLO_ONNX_PATH="$DETECTION_MODEL_DIR/yolov8n.onnx"
YOLO_ENGINE_PATH="$DETECTION_MODEL_DIR/yolov8n.engine"

DEEPSORT_ONNX_PATH="$REID_MODEL_DIR/deepsort_reid.onnx"
DEEPSORT_ENGINE_PATH="$REID_MODEL_DIR/deepsort_reid.engine"

# YOLOv8 Input/Output Configuration (Verify with your yolov8n.onnx model using Netron)
YOLO_INPUT_NAME="images" # Common input name for Ultralytics YOLO models
YOLO_INPUT_SHAPE_MIN="1x3x640x640"
YOLO_INPUT_SHAPE_OPT="1x3x640x640" # Using fixed batch size 1, 640x640 for simplicity
YOLO_INPUT_SHAPE_MAX="1x3x640x640"

# DeepSORT ReID Input Configuration (Verify with your deepsort_reid.onnx model)
REID_INPUT_NAME="input" # Common input name for ReID models, might vary
REID_INPUT_SHAPE_MIN="1x3x128x64" # Batch x Channels x Height x Width
REID_INPUT_SHAPE_OPT="1x3x128x64" # Using fixed batch size 1
REID_INPUT_SHAPE_MAX="8x3x128x64" # Can be made dynamic e.g., 8 for max batch

# Common trtexec flags
COMMON_FLAGS="--fp16 --useSpinWait --useCudaGraph --noDataTransfers"
# --int8 could be an option for further optimization if calibration data is available
# Add --verbose for detailed logs if needed

# --- Helper Function ---
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: ONNX file not found at $1. Please run download_models.sh first."
        exit 1
    fi
}

# --- Main Export Logic ---
echo "Starting TensorRT engine export..."

# 1. Export YOLOv8 Engine
echo "--------------------------------------"
echo "Exporting YOLOv8 TensorRT engine..."
check_file_exists "$YOLO_ONNX_PATH"

if "$TRTEXEC_PATH" \
    --onnx="$YOLO_ONNX_PATH" \
    --saveEngine="$YOLO_ENGINE_PATH" \
    $COMMON_FLAGS; then
    echo "YOLOv8 TensorRT engine exported successfully to $YOLO_ENGINE_PATH"
else
    echo "Initial YOLOv8 export failed. Trying alternative method..."
    
    # Try with explicit batch flag for older TensorRT versions
    if "$TRTEXEC_PATH" \
        --onnx="$YOLO_ONNX_PATH" \
        --saveEngine="$YOLO_ENGINE_PATH" \
        --explicitBatch \
        $COMMON_FLAGS; then
        echo "YOLOv8 TensorRT engine exported successfully with explicitBatch to $YOLO_ENGINE_PATH"
    else
        echo "Failed to export YOLOv8 TensorRT engine."
        exit 1
    fi
fi

# 2. Export DeepSORT ReID Engine
echo "--------------------------------------"
echo "Exporting DeepSORT ReID TensorRT engine..."
check_file_exists "$DEEPSORT_ONNX_PATH"

if "$TRTEXEC_PATH" \
    --onnx="$DEEPSORT_ONNX_PATH" \
    --saveEngine="$DEEPSORT_ENGINE_PATH" \
    --minShapes="$REID_INPUT_NAME=$REID_INPUT_SHAPE_MIN" \
    --optShapes="$REID_INPUT_NAME=$REID_INPUT_SHAPE_OPT" \
    --maxShapes="$REID_INPUT_NAME=$REID_INPUT_SHAPE_MAX" \
    $COMMON_FLAGS; then
    echo "DeepSORT ReID TensorRT engine exported successfully to $DEEPSORT_ENGINE_PATH"
else
    echo "Initial DeepSORT ReID export failed. Trying alternative method..."
    
    # Try with explicit batch flag for older TensorRT versions
    if "$TRTEXEC_PATH" \
        --onnx="$DEEPSORT_ONNX_PATH" \
        --saveEngine="$DEEPSORT_ENGINE_PATH" \
        --explicitBatch \
        $COMMON_FLAGS; then
        echo "DeepSORT ReID TensorRT engine exported successfully with explicitBatch to $DEEPSORT_ENGINE_PATH"
    else
        echo "Failed to export DeepSORT ReID TensorRT engine."
        exit 1
    fi
fi

echo "--------------------------------------"
echo "All TensorRT engines exported."