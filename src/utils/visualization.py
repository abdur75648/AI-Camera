# src/utils/visualization.py

import cv2
import numpy as np
from . import image_processing # To import config if needed for colors/fonts
from .. import config # Import config directly from src
from typing import List

def draw_detections(
    frame: np.ndarray,
    bboxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: tuple
) -> np.ndarray:
    """
    Draws raw detection bounding boxes, scores, and class names on the frame.
    Mainly for debugging purposes.

    Args:
        frame (np.ndarray): The image frame to draw on.
        bboxes_xyxy (np.ndarray): Bounding boxes in (x1, y1, x2, y2) format.
        scores (np.ndarray): Detection confidence scores.
        class_ids (np.ndarray): Class IDs for each detection.
        class_names (tuple): Tuple of all possible class names.

    Returns:
        np.ndarray: Frame with detections drawn.
    """
    for i in range(len(bboxes_xyxy)):
        x1, y1, x2, y2 = map(int, bboxes_xyxy[i])
        score = scores[i]
        class_id = int(class_ids[i])
        
        if class_id < 0 or class_id >= len(class_names):
            label_name = "Unknown"
            color = (128, 128, 128) # Gray for unknown
        else:
            label_name = class_names[class_id]
            color = config.get_class_color(label_name)

        label = f"{label_name}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE_ID, config.FONT_THICKNESS
        )
        # Put text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            -1, # Filled
        )
        # Put text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline // 2), # Adjust for better vertical alignment
            config.FONT,
            config.FONT_SCALE_ID,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def draw_tracks(
    frame: np.ndarray,
    tracked_objects: list # List of tuples: (x1, y1, x2, y2, track_id, class_name, Optional[score])
) -> np.ndarray:
    """
    Draws tracked bounding boxes with track IDs and class names on the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        tracked_objects (list): A list of tuples, where each tuple contains
                                (x1, y1, x2, y2, track_id, class_name, Optional[score]).

    Returns:
        np.ndarray: Frame with tracks drawn.
    """
    for obj_data in tracked_objects:
        x1, y1, x2, y2 = map(int, obj_data[:4])
        track_id = obj_data[4]
        class_name = obj_data[5]
        
        color = config.get_track_color(class_name) # Use class-specific color for the track

        label = f"ID:{track_id} {class_name}"
        if len(obj_data) > 6: # If score is provided
            score = obj_data[6]
            label += f" {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE_ID, config.FONT_THICKNESS
        )
        # Put text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 2), # Add a small margin
            (x1 + text_width, y1),
            color,
            -1, # Filled
        )
        # Put text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline // 2 - 1), # Adjust for better vertical alignment
            config.FONT,
            config.FONT_SCALE_ID,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draws the current FPS on the top-left corner of the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        fps (float): The calculated frames per second.

    Returns:
        np.ndarray: Frame with FPS drawn.
    """
    fps_text = f"FPS: {fps:.2f}"
    
    # Calculate text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
    )
    
    # Position for the text (top-left corner)
    text_x = 10
    text_y = text_height + 10 + baseline // 2

    # Put text background (optional, for better visibility)
    cv2.rectangle(
        frame,
        (text_x - 5, text_y - text_height - baseline - 5),
        (text_x + text_width + 5, text_y + 5),
        (50, 50, 50), # Dark gray background
        -1,
    )

    cv2.putText(
        frame,
        fps_text,
        (text_x, text_y - baseline // 2),
        config.FONT,
        config.FONT_SCALE_INFO,
        (255, 255, 255), # White text
        config.FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame

def draw_info_panel(frame: np.ndarray, info_lines: List[str]) -> np.ndarray:
    """
    Draws multiple lines of informational text on the frame, typically at the top.
    Args:
        frame (np.ndarray): The image frame to draw on.
        info_lines (List[str]): A list of strings, each to be drawn on a new line.
    Returns:
        np.ndarray: Frame with info panel drawn.
    """
    start_x = 10
    start_y = 30 # Initial y position
    line_height_offset = 0

    max_text_width = 0

    # First pass to determine max width for background
    for line_index, text_line in enumerate(info_lines):
        (text_width, text_height), baseline = cv2.getTextSize(
            text_line, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
        )
        if text_width > max_text_width:
            max_text_width = text_width
        if line_index == 0: # Use first line's height for consistent spacing
            line_height_offset = text_height + baseline + 10 # 10px spacing

    # Draw background for the panel (optional)
    if info_lines:
        panel_height = len(info_lines) * line_height_offset
        cv2.rectangle(
            frame,
            (start_x - 5, start_y - line_height_offset + 15), # Adjust to cover first line properly
            (start_x + max_text_width + 5, start_y + panel_height - line_height_offset + 15),
            (50, 50, 50, 180), # Dark semi-transparent background (if alpha is supported by drawing context)
            -1,
        )


    current_y = start_y
    for text_line in info_lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            text_line, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
        )
        
        # text_y_position = current_y + text_height # Baseline is below text
        text_y_position = current_y + baseline + (text_height // 2) # Center text a bit better

        cv2.putText(
            frame,
            text_line,
            (start_x, text_y_position),
            config.FONT,
            config.FONT_SCALE_INFO,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
        current_y += line_height_offset # Move to next line position

    return frame