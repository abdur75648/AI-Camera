# src/tracker/reid_model.py

import torch
import traceback
import numpy as np
from typing import List, Tuple, Dict, Optional # Added Optional
import os # For checking engine existence in test

from ..trt_utils.trt_engine import TRTEngine
from ..utils import image_processing
from .. import config

class ReIDModel:
    """
    Re-Identification (ReID) model class for extracting appearance features
    from image crops using a TensorRT engine.
    """
    def __init__(self,
                 engine_path: str = str(config.REID_ENGINE_PATH),
                 input_shape: Tuple[int, int] = config.REID_INPUT_SHAPE, # (H, W)
                 device: Optional[torch.device] = None # Made device optional for easier testing if CUDA not available
                ):
        """
        Initializes the ReIDModel.

        Args:
            engine_path (str): Path to the ReID TensorRT engine file.
            input_shape (Tuple[int, int]): Target input shape (H, W) for the ReID model.
            device (torch.device, optional): Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.engine_path = engine_path
        self.input_shape = input_shape # Expected (H, W)
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Only initialize TRTEngine if an engine path is provided and we are not on CPU for actual TRT usage
        if os.path.exists(self.engine_path) and self.device.type == 'cuda':
            self.trt_engine = TRTEngine(engine_path, device=self.device)
            # Assuming the ReID engine has a single input. Verify its name.
            self.input_name = self.trt_engine.get_input_details()[0].name
            # Assuming the ReID engine has a single output (the feature vector). Verify its name.
            self.output_name = self.trt_engine.get_output_details()[0].name
            self.feature_dim = self.trt_engine.get_output_details()[0].shape[-1] # Get feature dimension

            print(f"ReIDModel initialized with engine: {engine_path}")
            print(f"  Input name: {self.input_name}, Input shape: {self.input_shape}")
            print(f"  Output name: {self.output_name}, Feature dim: {self.feature_dim}")
        elif self.device.type == 'cpu' and not os.path.exists(self.engine_path):
            print(f"ReIDModel initialized in CPU mode for testing (no engine at {engine_path}). Feature extraction will be mocked.")
            self.trt_engine = None
            self.input_name = "input" # Dummy name
            self.output_name = "output" # Dummy name
            self.feature_dim = 512 # Dummy feature dim for CPU mock
        elif not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"ReID engine not found at {self.engine_path} and not in CPU mock mode.")
        else: # CPU mode but engine exists (should not happen for TRT)
             print(f"Warning: ReIDModel on CPU but engine exists at {self.engine_path}. TRT Engine will not be used.")
             self.trt_engine = None
             self.input_name = "input"
             self.output_name = "output"
             self.feature_dim = 512


    def extract_features_batched(self, image_crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Extracts appearance features for a batch of image crops.

        Args:
            image_crops_bgr (List[np.ndarray]): A list of image crops,
                                               each as a BGR NumPy array (H, W, C).

        Returns:
            np.ndarray: A NumPy array of shape (N, feature_dim) containing
                        the extracted features, where N is the number of crops.
                        Returns an empty array if no valid crops are provided or an error occurs.
        """
        if not image_crops_bgr:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        valid_preprocessed_crops = []
        for i, crop in enumerate(image_crops_bgr):
            if not isinstance(crop, np.ndarray) or crop.ndim != 3 or \
               crop.shape[0] == 0 or crop.shape[1] == 0 or crop.shape[2] != 3:
                print(f"Warning: Invalid image crop at index {i} received in ReIDModel. Shape: {crop.shape if isinstance(crop, np.ndarray) else type(crop)}. Skipping.")
                continue
            
            # Preprocess each crop (resize, normalize, etc.)
            tensor_chw_batch_dim = image_processing.preprocess_reid_input(
                crop, target_shape=self.input_shape
            ) # Returns (1, C, H, W)
            valid_preprocessed_crops.append(tensor_chw_batch_dim)
        
        if not valid_preprocessed_crops: # All crops were invalid
             return np.empty((0, self.feature_dim), dtype=np.float32)

        # Concatenate preprocessed crops into a single batch tensor
        input_batch_tensor_np = np.concatenate(valid_preprocessed_crops, axis=0)
        input_torch_tensor = torch.from_numpy(input_batch_tensor_np).to(self.device)

        # --- MOCK FOR CPU TESTING if TRT engine is not available ---
        if self.trt_engine is None or self.device.type == 'cpu':
            print("  (ReIDModel Mock/CPU Mode: Returning dummy features)")
            num_valid_crops = input_torch_tensor.shape[0]
            return np.random.rand(num_valid_crops, self.feature_dim).astype(np.float32)
        # --- END MOCK ---

        # Prepare inputs for TRTEngine
        inputs_dict = {self.input_name: input_torch_tensor}

        # Perform inference
        try:
            outputs_dict = self.trt_engine.infer(inputs_dict)
            features_tensor = outputs_dict[self.output_name]
        except KeyError as e:
            print(f"Error: Output tensor name '{e}' not found in ReID TRT engine outputs. Check self.output_name.")
            print(f"Available output names from engine: {list(outputs_dict.keys()) if 'outputs_dict' in locals() else 'Error before output dict created'}")
            return np.empty((0, self.feature_dim), dtype=np.float32) # Fallback
        except Exception as e:
            print(f"Error during ReID feature extraction with TRT engine: {e}")
            return np.empty((0, self.feature_dim), dtype=np.float32) # Fallback

        # Convert features to NumPy array and detach from graph
        return features_tensor.detach().cpu().numpy()

if __name__ == '__main__':
    print("--- Testing src/tracker/reid_model.py ---")
    
    # --- Test Configuration ---
    # Attempt to use CUDA if available and engine exists, otherwise fallback to CPU mock
    use_cuda_if_available = True 
    device_for_test = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    engine_exists = config.REID_ENGINE_PATH.exists()
    
    print(f"  Test Device: {device_for_test}")
    print(f"  ReID Engine Path: {config.REID_ENGINE_PATH}")
    print(f"  ReID Engine Exists: {engine_exists}")

    reid_model_instance: Optional[ReIDModel] = None

    # Test 1: Initialization
    print("\n--- Test 1: ReIDModel Initialization ---")
    try:
        if device_for_test.type == 'cuda' and not engine_exists:
            print("  Skipping CUDA TRT engine initialization test as engine file is missing.")
            print("  Proceeding with CPU mock initialization test.")
            # Force CPU mode for initialization test if engine is missing but CUDA was requested
            reid_model_instance = ReIDModel(engine_path=str(config.REID_ENGINE_PATH), device=torch.device('cpu'))
        else:
            reid_model_instance = ReIDModel(engine_path=str(config.REID_ENGINE_PATH), device=device_for_test)
        
        assert reid_model_instance is not None
        print(f"  ReIDModel successfully initialized. Feature dim: {reid_model_instance.feature_dim}")
        if reid_model_instance.trt_engine is not None:
            assert reid_model_instance.input_name is not None
            assert reid_model_instance.output_name is not None
        print("Test 1 PASSED.")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        traceback.print_exc()
        # If init fails, subsequent tests might not run or might error out.
        reid_model_instance = None # Ensure it's None so other tests might skip

    if reid_model_instance: # Proceed only if initialization was successful (or mocked)
        # Test 2: Feature extraction with valid crops
        print("\n--- Test 2: Feature extraction with valid crops ---")
        try:
            dummy_crop1 = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8) # H, W, C
            dummy_crop2 = np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8)
            image_crops_valid = [dummy_crop1, dummy_crop2]
            
            print(f"  Extracting features for {len(image_crops_valid)} valid dummy crops...")
            features = reid_model_instance.extract_features_batched(image_crops_valid)
            
            assert features.ndim == 2, f"Features ndim unexpected: {features.ndim}"
            assert features.shape[0] == len(image_crops_valid), f"Expected {len(image_crops_valid)} feature vectors, got {features.shape[0]}"
            assert features.shape[1] == reid_model_instance.feature_dim, f"Expected feature_dim {reid_model_instance.feature_dim}, got {features.shape[1]}"
            assert features.dtype == np.float32
            print(f"  Extracted features shape: {features.shape}")
            if features.size > 0 : print(f"  First 5 elements of first feature: {features[0, :5]}")
            print("Test 2 PASSED.")
        except Exception as e:
            print(f"Test 2 FAILED: {e}")
            traceback.print_exc()

        # Test 3: Feature extraction with empty list of crops
        print("\n--- Test 3: Feature extraction with empty list ---")
        try:
            features_empty_list = reid_model_instance.extract_features_batched([])
            assert features_empty_list.shape == (0, reid_model_instance.feature_dim)
            print(f"  Features from empty list shape: {features_empty_list.shape}")
            print("Test 3 PASSED.")
        except Exception as e:
            print(f"Test 3 FAILED: {e}")
            traceback.print_exc()

        # Test 4: Feature extraction with some invalid crops
        print("\n--- Test 4: Feature extraction with mixed valid/invalid crops ---")
        try:
            dummy_crop_valid = np.random.randint(0, 255, (config.REID_INPUT_SHAPE[0], config.REID_INPUT_SHAPE[1], 3), dtype=np.uint8)
            crop_invalid_shape1 = np.random.randint(0, 255, (50, 3), dtype=np.uint8) # Wrong ndim
            crop_invalid_shape2 = np.zeros((0, 50, 3), dtype=np.uint8) # Empty dimension
            crop_invalid_type = "not_an_array"

            mixed_crops = [dummy_crop_valid, crop_invalid_shape1, crop_invalid_type, crop_invalid_shape2, dummy_crop_valid]
            print(f"  Extracting features for {len(mixed_crops)} mixed crops (expecting 2 valid)...")
            features_mixed = reid_model_instance.extract_features_batched(mixed_crops)
            
            assert features_mixed.shape[0] == 2, f"Expected 2 feature vectors from mixed list, got {features_mixed.shape[0]}"
            assert features_mixed.shape[1] == reid_model_instance.feature_dim
            print(f"  Features from mixed list shape: {features_mixed.shape}")
            print("Test 4 PASSED.")
        except Exception as e:
            print(f"Test 4 FAILED: {e}")
            traceback.print_exc()
            
        # Test 5: Feature extraction with all invalid crops
        print("\n--- Test 5: Feature extraction with all invalid crops ---")
        try:
            all_invalid_crops = [
                np.zeros((0,0,3), dtype=np.uint8), 
                np.random.randint(0,255, (10,10,1), dtype=np.uint8) # Wrong channels
            ]
            features_all_invalid = reid_model_instance.extract_features_batched(all_invalid_crops)
            assert features_all_invalid.shape == (0, reid_model_instance.feature_dim)
            print(f"  Features from all-invalid list shape: {features_all_invalid.shape}")
            print("Test 5 PASSED.")
        except Exception as e:
            print(f"Test 5 FAILED: {e}")
            traceback.print_exc()
    else:
        print("Skipping further ReIDModel tests due to initialization failure.")

    print("--- Finished testing src/tracker/reid_model.py ---")