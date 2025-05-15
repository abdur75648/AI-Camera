# src/detector/yolo_detector.py

import torch
import numpy as np
from typing import Tuple, List, Dict

from ..trt_utils.trt_engine import TRTEngine
from ..utils import image_processing
from .. import config

class YOLODetector:
    """
    YOLOv8 Detector class for performing object detection using a TensorRT engine.
    """
    def __init__(self,
                 engine_path: str = str(config.YOLO_ENGINE_PATH),
                 input_shape: Tuple[int, int] = config.YOLO_INPUT_SHAPE, # (H, W)
                 conf_threshold: float = config.YOLO_CONF_THRESHOLD,
                 nms_threshold: float = config.YOLO_NMS_THRESHOLD, # May not be used if NMS is in engine
                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                ):
        """
        Initializes the YOLODetector.

        Args:
            engine_path (str): Path to the YOLOv8 TensorRT engine file.
            input_shape (Tuple[int, int]): Target input shape (H, W) for the model.
            conf_threshold (float): Confidence threshold for detections.
            nms_threshold (float): NMS threshold (if NMS is performed post-TRT).
            device (torch.device): Device to run inference on.
        """
        self.engine_path = engine_path
        self.input_shape = input_shape # Expected (H, W)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold # Note: NMS might be part of the TRT engine
        self.device = device

        self.trt_engine = TRTEngine(engine_path, device=self.device)
        
        # Assuming the Ultralytics exported ONNX model (and thus the TRT engine)
        # has a single image input. Verify its name.
        self.input_name = self.trt_engine.get_input_details()[0].name
        
        # Expected output names from an Ultralytics YOLOv8 TensorRT engine
        # (after export with NMS included, common with `yolo export ... simplify=True`)
        # These typically are: 'num_dets', 'bboxes', 'scores', 'labels'
        # Verify these names using Netron on your ONNX or by inspecting engine outputs.
        # Let's assume these are the standard names.
        self.output_names = {
            'num_dets': 'num_dets', # Or whatever it's named in your engine
            'bboxes': 'bboxes',     # Or 'boxes'
            'scores': 'scores',
            'labels': 'labels'      # Or 'classes'
        }
        # Check if all expected output names are present in the engine
        engine_output_names = [info.name for info in self.trt_engine.get_output_details()]
        for key, name in self.output_names.items():
            if name not in engine_output_names:
                print(f"Warning: Expected output tensor '{name}' for '{key}' not found in engine. "
                      f"Available outputs: {engine_output_names}. Check your ONNX export and engine.")
                # You might want to raise an error here or adapt based on actual output names
        
        print(f"YOLODetector initialized with engine: {engine_path}")
        print(f"  Input name: {self.input_name}, Input shape: {self.input_shape}")
        print(f"  Expected output names: {self.output_names.values()}")


    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs object detection on a single BGR frame.

        Args:
            frame_bgr (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - bboxes_xyxy (np.ndarray): Detected bounding boxes (N, 4) in (x1, y1, x2, y2) format,
                                            scaled to original frame coordinates.
                - scores (np.ndarray): Confidence scores (N,) for each detection.
                - class_ids (np.ndarray): Class IDs (N,) for each detection.
                - filtered_indices (np.ndarray): Indices of detections that passed confidence filtering.
        """
        original_shape = frame_bgr.shape[:2] # H, W

        # Preprocess the image
        img_tensor_chw, ratios, padding_wh = image_processing.preprocess_yolo_input(
            frame_bgr, target_shape=self.input_shape
        )
        
        # Convert NumPy array to PyTorch tensor
        input_torch_tensor = torch.from_numpy(img_tensor_chw).to(self.device)

        # Prepare inputs for TRTEngine (dictionary format)
        inputs_dict = {self.input_name: input_torch_tensor}

        # Perform inference
        outputs_dict = self.trt_engine.infer(inputs_dict)

        # --- Post-process outputs ---
        # The structure of outputs depends on how the YOLOv8 model was exported to ONNX/TRT.
        # Typically, with Ultralytics export (including NMS):
        # - num_dets: (1, 1) tensor, number of detections after NMS.
        # - bboxes: (1, num_dets_val, 4) tensor, (x1, y1, x2, y2) in letterboxed space.
        # - scores: (1, num_dets_val) tensor.
        # - labels: (1, num_dets_val) tensor.

        try:
            num_dets = outputs_dict[self.output_names['num_dets']][0].item() # Get the scalar value
            # Squeeze batch dimension if present
            bboxes_letterboxed = outputs_dict[self.output_names['bboxes']][0, :num_dets, :].cpu().numpy() # (N, 4)
            scores = outputs_dict[self.output_names['scores']][0, :num_dets].cpu().numpy() # (N,)
            class_ids = outputs_dict[self.output_names['labels']][0, :num_dets].cpu().numpy().astype(np.int32) # (N,)
        except KeyError as e:
            print(f"Error: Output tensor name '{e}' not found in TRT engine outputs. Check self.output_names.")
            print(f"Available output names from engine: {list(outputs_dict.keys())}")
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0, dtype=int)
        except Exception as e:
            print(f"Error processing TRT engine outputs: {e}")
            print(f"Output dictionary keys: {list(outputs_dict.keys())}")
            for name, tensor in outputs_dict.items():
                print(f"  Output '{name}' shape: {tensor.shape}, dtype: {tensor.dtype}")
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0, dtype=int)


        if num_dets == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0, dtype=int)

        # Filter by confidence threshold (TRT NMS might already do this, but an explicit check is good)
        # The `score_threshold` in `trtexec` usually handles this.
        # If not, or for an additional filter:
        confident_indices = scores >= self.conf_threshold
        
        bboxes_letterboxed = bboxes_letterboxed[confident_indices]
        scores = scores[confident_indices]
        class_ids = class_ids[confident_indices]
        
        if bboxes_letterboxed.shape[0] == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0, dtype=int)
            
        # Scale bounding boxes back to original image coordinates
        bboxes_original_xyxy = image_processing.scale_bboxes(
            bboxes_letterboxed,
            original_shape=original_shape,
            letterbox_shape=self.input_shape,
            ratio=ratios,
            padding=padding_wh
        )
        
        return bboxes_original_xyxy, scores, class_ids, np.where(confident_indices)[0]


if __name__ == '__main__':
    # This is a placeholder for a simple test
    # Ensure you have a 'yolov8n.engine' in 'AICamera/models/detection/'
    # and a sample image.
    
    print("Running YOLODetector test...")
    if not config.YOLO_ENGINE_PATH.exists():
        print(f"Test skipped: YOLO engine not found at {config.YOLO_ENGINE_PATH}")
        print("Please run 'scripts/download_models.sh' and 'scripts/export_trt_engines.sh' first.")
    else:
        try:
            detector = YOLODetector(engine_path=str(config.YOLO_ENGINE_PATH))
            
            # Create a dummy image or load a sample image
            # sample_image_path = config.PROJECT_ROOT / "sample_input/sample_image.jpg" # You'll need to add a sample image
            # if sample_image_path.exists():
            #     dummy_frame = cv2.imread(str(sample_image_path))
            # else:
            dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            
            print(f"Detecting objects in a dummy frame of shape {dummy_frame.shape}...")
            
            import time
            start_time = time.time()
            bboxes, scores, class_ids, _ = detector.detect(dummy_frame)
            end_time = time.time()
            
            print(f"Detection completed in {end_time - start_time:.4f} seconds.")
            
            if bboxes.shape[0] > 0:
                print(f"Detected {bboxes.shape[0]} objects.")
                for i in range(min(5, bboxes.shape[0])): # Print first 5 detections
                    print(f"  Box: {bboxes[i]}, Score: {scores[i]:.2f}, Class ID: {class_ids[i]} ({config.CLASSES[class_ids[i]]})")
            else:
                print("No objects detected.")

            # Example of drawing (requires opencv-python, not headless, and a display environment)
            # frame_with_dets = image_processing.draw_detections(dummy_frame.copy(), bboxes, scores, class_ids, config.CLASSES)
            # cv2.imshow("YOLO Detections", frame_with_dets)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during YOLODetector test: {e}")
            import traceback
            traceback.print_exc()