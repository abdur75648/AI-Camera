# src/utils/image_processing.py

import cv2
import numpy as np
from typing import Tuple, List, Union

def letterbox(
    im: np.ndarray,
    new_shape: Union[Tuple[int, int], List[int]] = (640, 640),
    color: Union[Tuple[int, int, int], List[int]] = (114, 114, 114),
    auto: bool = True, # True for minimum rectangle, False for square
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    Returns resized image, ratio, and padding (dw, dh).

    Args:
        im (np.ndarray): Input image.
        new_shape (tuple): Target dimensions (height, width).
        color (tuple): Padding color.
        auto (bool): Adjust padding to new_shape. If False, minimum rectangle.
        scaleFill (bool): Stretch image to fill new_shape.
        scaleup (bool): Scale image up if smaller than new_shape.
        stride (int): Stride constraint.

    Returns:
        tuple: (Padded image, (ratio_height, ratio_width), (pad_height, pad_width))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r_h = new_shape[0] / shape[0]
    r_w = new_shape[1] / shape[1]
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r_h = min(r_h, 1.0)
        r_w = min(r_w, 1.0)

    # Use the minimum ratio to preserve aspect ratio
    r = min(r_h, r_w)

    # Compute padding
    new_unpad_shape = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    dw, dh = new_shape[1] - new_unpad_shape[1], new_shape[0] - new_unpad_shape[0]  # width, height padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad_shape = (new_shape[0], new_shape[1]) # H, W
        # ratio_h = new_shape[0] / shape[0]  # height gain
        # ratio_w = new_shape[1] / shape[1]  # width gain
        # Using the originally calculated r for ratio return consistency for coordinate scaling
        # but the image will be stretched to new_shape

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad_shape:  # resize
        im = cv2.resize(im, (new_unpad_shape[1], new_unpad_shape[0]), interpolation=cv2.INTER_LINEAR) # W, H for cv2.resize

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, (r, r), (dw, dh) # return same ratio for h & w for simplicity in unscaling


def preprocess_yolo_input(image_bgr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Prepares a BGR image for YOLO TensorRT inference.
    1. Letterboxes the image to target_shape (H, W).
    2. Converts BGR to RGB.
    3. Transposes from HWC to CHW.
    4. Normalizes to [0, 1].
    5. Adds a batch dimension.
    6. Converts to a contiguous float32 NumPy array.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        target_shape (Tuple[int, int]): Target shape (H, W) for the model.

    Returns:
        np.ndarray: Preprocessed image tensor.
        Tuple[float, float]: Ratios (height_ratio, width_ratio) for scaling back.
        Tuple[int, int]: Paddings (dw, dh) applied.
    """
    img_letterboxed, ratios, (pad_w, pad_h) = letterbox(image_bgr, new_shape=target_shape, auto=False, scaleup=False) # Use auto=False for standard letterboxing
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    
    # HWC to CHW, BGR to RGB
    img_chw = np.transpose(img_rgb, (2, 0, 1))  # CHW

    # Normalize to [0, 1] and add batch dimension
    img_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32) / 255.0
    
    # Ensure contiguous array
    return np.ascontiguousarray(img_tensor), ratios, (pad_w, pad_h)


def preprocess_reid_input(image_crop_bgr: np.ndarray, target_shape: Tuple[int, int] = (128, 64)) -> np.ndarray:
    """
    Prepares an image crop for ReID model inference.
    1. Resizes to target_shape (H, W).
    2. Converts BGR to RGB.
    3. Normalizes using ImageNet mean/std (common for ReID models).
    4. Transposes from HWC to CHW.
    5. Adds a batch dimension.
    6. Converts to a contiguous float32 NumPy array.

    Args:
        image_crop_bgr (np.ndarray): Input image crop in BGR format.
        target_shape (Tuple[int, int]): Target shape (H, W) for the ReID model.

    Returns:
        np.ndarray: Preprocessed image tensor.
    """
    # Resize
    resized_img = cv2.resize(image_crop_bgr, (target_shape[1], target_shape[0])) # W, H for cv2.resize

    # BGR to RGB
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Normalize (example with ImageNet mean/std, adjust if your ReID model used different ones)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_img = (img_rgb.astype(np.float32) / 255.0 - mean) / std

    # HWC to CHW
    img_chw = np.transpose(normalized_img, (2, 0, 1))

    # Add batch dimension and ensure contiguous
    img_tensor = np.expand_dims(img_chw, axis=0)
    return np.ascontiguousarray(img_tensor, dtype=np.float32)


def scale_bboxes(bboxes_letterboxed: np.ndarray,
                 original_shape: Tuple[int, int],
                 letterbox_shape: Tuple[int, int],
                 ratio: Tuple[float, float],
                 padding: Tuple[float, float]) -> np.ndarray:
    """
    Scales bounding boxes from letterboxed image coordinates back to original image coordinates.

    Args:
        bboxes_letterboxed (np.ndarray): Bounding boxes (x1, y1, x2, y2) in letterboxed image space.
        original_shape (Tuple[int, int]): Original image shape (H, W).
        letterbox_shape (Tuple[int, int]): Letterboxed image shape (H, W) used for inference.
        ratio (Tuple[float, float]): Scaling ratios (ratio_h, ratio_w) returned by letterbox.
        padding (Tuple[float, float]): Paddings (pad_w, pad_h) returned by letterbox.

    Returns:
        np.ndarray: Bounding boxes in original image space.
    """
    if bboxes_letterboxed.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    scaled_bboxes = bboxes_letterboxed.copy()
    pad_w, pad_h = padding
    ratio_h, ratio_w = ratio # Should be same if aspect ratio is preserved

    # Remove padding
    scaled_bboxes[:, 0] -= pad_w
    scaled_bboxes[:, 1] -= pad_h
    scaled_bboxes[:, 2] -= pad_w
    scaled_bboxes[:, 3] -= pad_h

    # Scale back to original size
    scaled_bboxes[:, 0] /= ratio_w
    scaled_bboxes[:, 1] /= ratio_h
    scaled_bboxes[:, 2] /= ratio_w
    scaled_bboxes[:, 3] /= ratio_h

    # Clip to original image dimensions
    original_h, original_w = original_shape
    scaled_bboxes[:, [0, 2]] = np.clip(scaled_bboxes[:, [0, 2]], 0, original_w)
    scaled_bboxes[:, [1, 3]] = np.clip(scaled_bboxes[:, [1, 3]], 0, original_h)

    return scaled_bboxes