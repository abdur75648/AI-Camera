# src/trt_utils/trt_engine.py

import tensorrt as trt
import torch
from typing import List, Optional, Tuple, Union, Dict, NamedTuple
from pathlib import Path
import time # For warm-up timing
from .. import config # For accessing example input shapes for warm-up

# Define a named tuple for tensor information
TensorInfo = NamedTuple('TensorInfo', [('name', str), ('dtype', torch.dtype), ('shape', Tuple[int, ...]), ('is_dynamic', bool)])

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Or trt.Logger.ERROR for less verbosity

class TRTEngine(torch.nn.Module):
    """
    TensorRT Engine wrapper using PyTorch for simplified inference,
    aligned with modern TensorRT Python APIs.
    """
    dtype_mapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }

    def __init__(self, engine_path: Union[str, Path], device: Optional[torch.device] = None):
        super().__init__()
        self.engine_path = Path(engine_path) if isinstance(engine_path, str) else engine_path
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cpu':
            print("Warning: TRTEngine is intended for CUDA devices, but CPU device was selected.")
            self.engine = None
            self.context = None
            self.input_info_list: List[TensorInfo] = []
            self.output_info_list: List[TensorInfo] = []
            return

        self._init_engine()
        self._init_bindings_info() # Store info, don't pre-allocate here
        self._warm_up()

    def _init_engine(self):
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

    def _init_bindings_info(self):
        self.input_info_list: List[TensorInfo] = []
        self.output_info_list: List[TensorInfo] = []

        for i in range(self.engine.num_io_tensors): # type: ignore
            name = self.engine.get_tensor_name(i) # type: ignore
            # Note: get_tensor_shape gives profile shape, get_binding_shape gives context shape (after set_input_shape)
            # For initial info, profile shape is more relevant.
            # However, to check for dynamism, profile min/max/opt shapes are better.
            # Let's use get_tensor_profile_shape to determine dynamism based on profile.
            # A binding is dynamic if its dimensions in min_shape != max_shape for any profile.
            # For simplicity here, we check if -1 is in the current context's binding shape,
            # which is an indicator after an engine is built with dynamic axes.
            
            # Use get_tensor_shape for initial shape (might be profile dependent if multiple profiles)
            # For dynamic axes, this shape might contain -1s.
            shape = tuple(self.engine.get_tensor_shape(name)) # type: ignore
            dtype = self.dtype_mapping[self.engine.get_tensor_dtype(name)] # type: ignore
            is_dynamic = any(dim == -1 for dim in shape) # A simple check for dynamism

            tensor_details = TensorInfo(name=name, dtype=dtype, shape=shape, is_dynamic=is_dynamic)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: # type: ignore
                self.input_info_list.append(tensor_details)
            else:
                self.output_info_list.append(tensor_details)
            
    def _get_example_input_shape(self, tensor_info: TensorInfo) -> Tuple[int, ...]:
        """ Helper to create a concrete shape for dynamic warm-up inputs. """
        # Try to get a shape from config based on engine name convention
        is_yolo = 'yolo' in self.engine_path.name.lower()
        is_reid = 'reid' in self.engine_path.name.lower()

        if is_yolo:
            example_config_shape_hw = getattr(config, 'YOLO_INPUT_SHAPE', (640, 640))
        elif is_reid:
            example_config_shape_hw = getattr(config, 'REID_INPUT_SHAPE', (128, 64))
        else: # Fallback for other models
            example_config_shape_hw = (256, 256) # A generic fallback

        # Construct a plausible shape, typically (Batch, Channels, Height, Width)
        # This logic is heuristic and might need refinement for very diverse models
        current_profile_shape = list(tensor_info.shape) # This might contain -1s
        final_shape = []

        if len(current_profile_shape) == 4: # Assuming NCHW
            final_shape.append(1 if current_profile_shape[0] == -1 else current_profile_shape[0]) # Batch
            final_shape.append(current_profile_shape[1] if current_profile_shape[1] != -1 else 3) # Channels
            final_shape.append(example_config_shape_hw[0] if current_profile_shape[2] == -1 else current_profile_shape[2]) # Height
            final_shape.append(example_config_shape_hw[1] if current_profile_shape[3] == -1 else current_profile_shape[3]) # Width
            return tuple(final_shape)
        elif len(current_profile_shape) > 0 : # Other shapes, make a simple guess
             return tuple(dim if dim != -1 else 1 for dim in current_profile_shape)
        else: # No shape info, very unlikely
            return (1,3,256,256)


    def _warm_up(self, iterations=5):
        if self.device.type == 'cpu':
            print("Skipping warm-up on CPU.")
            return
            
        dummy_inputs_dict: Dict[str, torch.Tensor] = {}
        any_dynamic = False
        for info in self.input_info_list:
            if info.is_dynamic:
                any_dynamic = True
                # For dynamic inputs, create a representative shape for warm-up
                # This uses the _get_example_input_shape helper
                concrete_shape = self._get_example_input_shape(info)
                print(f"  Warm-up: Dynamic input '{info.name}' using example shape {concrete_shape}.")
                dummy_inputs_dict[info.name] = torch.randn(concrete_shape, dtype=info.dtype, device=self.device)
            else: # Static input
                dummy_inputs_dict[info.name] = torch.randn(info.shape, dtype=info.dtype, device=self.device)
        
        if any_dynamic:
             print(f"Warning: Engine '{self.engine_path.name}' has dynamic inputs. Warm-up with example shapes.")
        
        print(f"Warming up engine '{self.engine_path.name}' for {iterations} iterations...")
        start_time = time.time()
        for _ in range(iterations):
            _ = self.infer(dummy_inputs_dict) # Pass as dict
        
        # Ensure all CUDA operations are done before measuring time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        end_time = time.time()
        print(f"Warm-up for '{self.engine_path.name}' finished in {end_time - start_time:.3f} seconds.")

    @torch.no_grad()
    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device.type == 'cpu':
            raise RuntimeError("TRTEngine inference cannot be run on CPU.")
            
        # Prepare input tensors and set input shapes for dynamic axes
        for info in self.input_info_list:
            input_tensor = inputs.get(info.name)
            if input_tensor is None:
                raise ValueError(f"Missing input: '{info.name}'")
            
            if input_tensor.device != self.device:
                input_tensor = input_tensor.to(self.device)
            if input_tensor.dtype != info.dtype:
                 print(f"Warning: Input tensor '{info.name}' dtype mismatch. "
                       f"Expected {info.dtype}, got {input_tensor.dtype}. Casting...")
                 input_tensor = input_tensor.to(info.dtype)
            
            inputs[info.name] = input_tensor.contiguous() # Ensure contiguous

            # Set shape for this input tensor in the context
            self.context.set_input_shape(info.name, tuple(inputs[info.name].shape)) # type: ignore

        # Allocate output tensors based on current (potentially dynamic) shapes
        outputs_dict: Dict[str, torch.Tensor] = {}
        for info in self.output_info_list:
            # Output shape is determined by the engine after input shapes are set
            output_shape = tuple(self.context.get_tensor_shape(info.name)) # type: ignore
            outputs_dict[info.name] = torch.empty(output_shape, dtype=info.dtype, device=self.device)

        # Set tensor addresses for I/O
        for info in self.input_info_list:
            self.context.set_tensor_address(info.name, inputs[info.name].data_ptr()) # type: ignore
        for info in self.output_info_list:
            self.context.set_tensor_address(info.name, outputs_dict[info.name].data_ptr()) # type: ignore
        
        # Pick the CUDA stream (using torch's current stream for the device)
        stream = torch.cuda.current_stream(self.device)

        # Execute inference
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream): # type: ignore
            raise RuntimeError(f"TensorRT execute_async_v3() failed for engine {self.engine_path.name}")

        # Tell PyTorch that outputs are prepared on this stream
        for tensor in outputs_dict.values():
            tensor.record_stream(stream)
        
        # Stream synchronization will be handled by the caller if needed,
        # or can be added here if synchronous behavior per infer call is desired.
        # For pipelining, caller synchronization is often better.
        # stream.synchronize() # Uncomment for synchronous behavior

        return outputs_dict

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Inputs must be a dictionary matching input binding names
        if not isinstance(inputs, dict):
            raise TypeError(f"Input to {self.engine_path.name} TRTEngine must be a dictionary mapping "
                            "input names to torch.Tensors.")
        return self.infer(inputs)

    def get_input_details(self) -> List[TensorInfo]:
        return self.input_info_list

    def get_output_details(self) -> List[TensorInfo]:
        return self.output_info_list