"""
Cloak detection and masking module for the invisibility cloak project.

This module handles the detection of the cloak color, generation of masks,
and application of morphological operations to refine the detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CloakDetector:
    """
    A class to detect and create masks for objects of a specific color.
    
    This class uses HSV color space for more accurate color detection
    and applies morphological operations to create clean masks.
    """
    
    def __init__(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray, enable_trackbars: bool = True):
        """
        Initialize the cloak detector with HSV color bounds.
        
        Args:
            lower_hsv (np.ndarray): Lower HSV bounds [H, S, V]
            upper_hsv (np.ndarray): Upper HSV bounds [H, S, V]
            enable_trackbars (bool): Whether to create HSV adjustment trackbars
        """
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.enable_trackbars = enable_trackbars
        
        # Kernel for morphological operations
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Trackbar values for real-time adjustment
        self.trackbar_values = {
            'H_min': lower_hsv[0],
            'S_min': lower_hsv[1], 
            'V_min': lower_hsv[2],
            'H_max': upper_hsv[0],
            'S_max': upper_hsv[1],
            'V_max': upper_hsv[2]
        }
        
        if enable_trackbars:
            self._setup_trackbars()
    
    def _setup_trackbars(self) -> None:
        """Setup trackbars for real-time HSV value adjustment."""
        cv2.namedWindow('HSV Adjustments')
        
        # Create trackbars for each HSV component
        cv2.createTrackbar('H_min', 'HSV Adjustments', self.trackbar_values['H_min'], 179, self._on_h_min_change)
        cv2.createTrackbar('S_min', 'HSV Adjustments', self.trackbar_values['S_min'], 255, self._on_s_min_change)
        cv2.createTrackbar('V_min', 'HSV Adjustments', self.trackbar_values['V_min'], 255, self._on_v_min_change)
        cv2.createTrackbar('H_max', 'HSV Adjustments', self.trackbar_values['H_max'], 179, self._on_h_max_change)
        cv2.createTrackbar('S_max', 'HSV Adjustments', self.trackbar_values['S_max'], 255, self._on_s_max_change)
        cv2.createTrackbar('V_max', 'HSV Adjustments', self.trackbar_values['V_max'], 255, self._on_v_max_change)
    
    def _on_h_min_change(self, value: int) -> None:
        """Callback for H minimum trackbar change."""
        self.trackbar_values['H_min'] = value
        self.lower_hsv[0] = value
    
    def _on_s_min_change(self, value: int) -> None:
        """Callback for S minimum trackbar change."""
        self.trackbar_values['S_min'] = value
        self.lower_hsv[1] = value
    
    def _on_v_min_change(self, value: int) -> None:
        """Callback for V minimum trackbar change."""
        self.trackbar_values['V_min'] = value
        self.lower_hsv[2] = value
    
    def _on_h_max_change(self, value: int) -> None:
        """Callback for H maximum trackbar change."""
        self.trackbar_values['H_max'] = value
        self.upper_hsv[0] = value
    
    def _on_s_max_change(self, value: int) -> None:
        """Callback for S maximum trackbar change."""
        self.trackbar_values['S_max'] = value
        self.upper_hsv[1] = value
    
    def _on_v_max_change(self, value: int) -> None:
        """Callback for V maximum trackbar change."""
        self.trackbar_values['V_max'] = value
        self.upper_hsv[2] = value
    
    def detect_cloak(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the cloak in the given frame and return the mask.
        
        Args:
            frame (np.ndarray): Input BGR frame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (mask, refined_mask)
                - mask: Raw binary mask
                - refined_mask: Mask after morphological operations
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask based on HSV range
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations to refine the mask
        refined_mask = self._apply_morphology(mask)
        
        return mask, refined_mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to refine the mask.
        
        Args:
            mask (np.ndarray): Input binary mask
            
        Returns:
            np.ndarray: Refined mask
        """
        # Erosion to remove noise
        eroded = cv2.erode(mask, self.kernel, iterations=1)
        
        # Dilation to fill gaps
        dilated = cv2.dilate(eroded, self.kernel, iterations=1)
        
        # Additional opening operation to remove small objects
        opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, self.kernel)
        
        # Additional closing operation to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        
        return closed
    
    def create_invisibility_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask for the invisibility effect.
        
        Args:
            frame (np.ndarray): Input BGR frame
            
        Returns:
            np.ndarray: Binary mask where True indicates cloak pixels
        """
        _, refined_mask = self.detect_cloak(frame)
        
        # Invert the mask so that cloak pixels are True
        invisibility_mask = refined_mask > 0
        
        return invisibility_mask
    
    def get_hsv_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current HSV values.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (lower_hsv, upper_hsv)
        """
        return self.lower_hsv.copy(), self.upper_hsv.copy()
    
    def set_hsv_values(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> None:
        """
        Set new HSV values.
        
        Args:
            lower_hsv (np.ndarray): New lower HSV bounds
            upper_hsv (np.ndarray): New upper HSV bounds
        """
        self.lower_hsv = lower_hsv.copy()
        self.upper_hsv = upper_hsv.copy()
        
        # Update trackbar values
        self.trackbar_values['H_min'] = lower_hsv[0]
        self.trackbar_values['S_min'] = lower_hsv[1]
        self.trackbar_values['V_min'] = lower_hsv[2]
        self.trackbar_values['H_max'] = upper_hsv[0]
        self.trackbar_values['S_max'] = upper_hsv[1]
        self.trackbar_values['V_max'] = upper_hsv[2]
    
    def cleanup(self) -> None:
        """Clean up resources and close windows."""
        if self.enable_trackbars:
            cv2.destroyWindow('HSV Adjustments')


# Predefined color ranges for common cloak colors
class ColorRanges:
    """Predefined HSV color ranges for common cloak colors."""
    
    # Green cloak (most common)
    GREEN = (
        np.array([40, 40, 40]),    # Lower HSV
        np.array([80, 255, 255])   # Upper HSV
    )
    
    # Blue cloak
    BLUE = (
        np.array([100, 50, 50]),   # Lower HSV
        np.array([130, 255, 255])  # Upper HSV
    )
    
    # Red cloak (handles both ends of HSV spectrum)
    RED = (
        np.array([0, 50, 50]),     # Lower HSV
        np.array([10, 255, 255])   # Upper HSV
    )
    
    # Yellow cloak
    YELLOW = (
        np.array([20, 100, 100]),  # Lower HSV
        np.array([30, 255, 255])   # Upper HSV
    )
    
    # Purple cloak
    PURPLE = (
        np.array([130, 50, 50]),   # Lower HSV
        np.array([170, 255, 255])  # Upper HSV
    )


def create_detector_for_color(color_name: str, enable_trackbars: bool = True) -> CloakDetector:
    """
    Create a detector for a predefined color.
    
    Args:
        color_name (str): Name of the color ('green', 'blue', 'red', 'yellow', 'purple')
        enable_trackbars (bool): Whether to create HSV adjustment trackbars
        
    Returns:
        CloakDetector: Configured detector for the specified color
        
    Raises:
        ValueError: If color name is not recognized
    """
    color_name = color_name.lower()
    
    if color_name == 'green':
        lower, upper = ColorRanges.GREEN
    elif color_name == 'blue':
        lower, upper = ColorRanges.BLUE
    elif color_name == 'red':
        lower, upper = ColorRanges.RED
    elif color_name == 'yellow':
        lower, upper = ColorRanges.YELLOW
    elif color_name == 'purple':
        lower, upper = ColorRanges.PURPLE
    else:
        raise ValueError(f"Unknown color: {color_name}. Available colors: green, blue, red, yellow, purple")
    
    return CloakDetector(lower, upper, enable_trackbars)
