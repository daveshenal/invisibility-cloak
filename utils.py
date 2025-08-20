"""
Utility functions for the invisibility cloak project.

This module contains helper functions for camera operations,
image processing, and common utilities used across the project.
"""

import cv2
import numpy as np
from typing import Optional


def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize and return a camera capture object.
    
    Args:
        camera_index (int): Index of the camera to use (default: 0)
        
    Returns:
        cv2.VideoCapture: Camera capture object if successful, None otherwise
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open camera at index {camera_index}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap


def release_camera(cap: cv2.VideoCapture) -> None:
    """
    Safely release camera resources.
    
    Args:
        cap (cv2.VideoCapture): Camera capture object to release
    """
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def get_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Capture a frame from the camera.
    
    Args:
        cap (cv2.VideoCapture): Camera capture object
        
    Returns:
        np.ndarray: Captured frame if successful, None otherwise
    """
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Resize a frame to specified dimensions.
    
    Args:
        frame (np.ndarray): Input frame
        width (int): Target width
        height (int): Target height
        
    Returns:
        np.ndarray: Resized frame
    """
    return cv2.resize(frame, (width, height))


def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flip a frame horizontally (mirror effect).
    
    Args:
        frame (np.ndarray): Input frame
        flip_code (int): Flip direction (1 for horizontal, 0 for vertical)
        
    Returns:
        np.ndarray: Flipped frame
    """
    return cv2.flip(frame, flip_code)


def save_image(image: np.ndarray, filename: str) -> bool:
    """
    Save an image to disk.
    
    Args:
        image (np.ndarray): Image to save
        filename (str): Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cv2.imwrite(filename, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def load_image(filename: str) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        filename (str): Input filename
        
    Returns:
        np.ndarray: Loaded image if successful, None otherwise
    """
    try:
        image = cv2.imread(filename)
        if image is None:
            print(f"Error: Could not load image from {filename}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def create_trackbar_window(window_name: str, trackbar_name: str, 
                          initial_value: int, max_value: int, 
                          callback) -> None:
    """
    Create a window with a trackbar for HSV value adjustment.
    
    Args:
        window_name (str): Name of the window
        trackbar_name (str): Name of the trackbar
        initial_value (int): Initial value of the trackbar
        max_value (int): Maximum value of the trackbar
        callback: Function to call when trackbar value changes
    """
    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackbar_name, window_name, initial_value, max_value, callback)


def print_instructions() -> None:
    """
    Print usage instructions to the console.
    """
    print("\n" + "="*50)
    print("INVISIBILITY CLOAK PROJECT")
    print("="*50)
    print("Controls:")
    print("  'c' - Capture background")
    print("  'q' - Quit application")
    print("  's' - Save current frame")
    print("  'r' - Reset background")
    print("="*50 + "\n")


def validate_hsv_values(lower: np.ndarray, upper: np.ndarray) -> bool:
    """
    Validate HSV values to ensure they are within valid ranges.
    
    Args:
        lower (np.ndarray): Lower HSV bounds
        upper (np.ndarray): Upper HSV bounds
        
    Returns:
        bool: True if values are valid, False otherwise
    """
    # Check if arrays have correct shape
    if lower.shape != (3,) or upper.shape != (3,):
        return False
    
    # Check if values are within valid ranges
    if np.any(lower < 0) or np.any(upper > 255):
        return False
    
    # Check if lower bounds are less than upper bounds
    if np.any(lower > upper):
        return False
    
    return True
