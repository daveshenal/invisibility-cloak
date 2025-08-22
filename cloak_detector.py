import cv2
import numpy as np
from typing import Tuple


class CloakDetector:
    """
    A class to detect and create masks for objects of a specific color.
    
    This class uses HSV color space for more accurate color detection
    and applies morphological operations to create clean masks.
    """
    
    def __init__(self, lower_hsv: np.ndarray, upper_hsv: np.ndarray, enable_trackbars: bool = True,
                 kernel_size: int = 5, erosion_iterations: int = 1, dilation_iterations: int = 1,
                 min_component_area: int = 0, smooth_kernel_size: int = 21):
        """
        Initialize the cloak detector with HSV color bounds.
        
        Args:
            lower_hsv (np.ndarray): Lower HSV bounds [H, S, V]
            upper_hsv (np.ndarray): Upper HSV bounds [H, S, V]
            enable_trackbars (bool): Whether to create HSV adjustment trackbars
            kernel_size (int): Morphological kernel size (odd number recommended)
            erosion_iterations (int): Number of erosion passes
            dilation_iterations (int): Number of dilation passes
            min_component_area (int): Remove components smaller than this area (0 disables)
            smooth_kernel_size (int): Gaussian blur kernel size for soft mask (odd number)
        """
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.enable_trackbars = enable_trackbars
        
        # Kernel for morphological operations
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        self.erosion_iterations = max(0, int(erosion_iterations))
        self.dilation_iterations = max(0, int(dilation_iterations))
        self.min_component_area = max(0, int(min_component_area))
        self.smooth_kernel_size = max(1, int(smooth_kernel_size))
        if self.smooth_kernel_size % 2 == 0:
            self.smooth_kernel_size += 1
        
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
        # Optional median blur to reduce salt-and-pepper noise
        filtered = cv2.medianBlur(mask, ksize=5)

        # Erosion to remove noise
        if self.erosion_iterations > 0:
            filtered = cv2.erode(filtered, self.kernel, iterations=self.erosion_iterations)
        
        # Dilation to fill gaps
        if self.dilation_iterations > 0:
            filtered = cv2.dilate(filtered, self.kernel, iterations=self.dilation_iterations)
        
        # Opening and closing to refine shapes
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, self.kernel)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, self.kernel)

        # Optionally remove tiny components by area
        if self.min_component_area > 0:
            contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            refined = np.zeros_like(filtered)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_component_area:
                    cv2.drawContours(refined, [cnt], -1, color=255, thickness=cv2.FILLED)
            filtered = refined

        return filtered
    
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

    def create_alpha_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a soft alpha mask (0..1) to enable feathered blending.
        
        Args:
            frame (np.ndarray): Input BGR frame
        
        Returns:
            np.ndarray: Float32 alpha mask in range [0.0, 1.0]
        """
        _, refined_mask = self.detect_cloak(frame)
        # Smooth edges to avoid blockiness using Gaussian blur
        blurred = cv2.GaussianBlur(refined_mask, (self.smooth_kernel_size, self.smooth_kernel_size), 0)
        alpha = blurred.astype(np.float32) / 255.0
        return alpha
    
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
    
    # Green cloak
    GREEN = (
        np.array([40, 40, 40]),
        np.array([80, 255, 255])
    )
    
    # Blue cloak
    BLUE = (
        np.array([100, 50, 50]),
        np.array([130, 255, 255])
    )
    
    # Red cloak
    RED = (
        np.array([0, 50, 50]),
        np.array([10, 255, 255])
    )
    
    # Yellow cloak
    YELLOW = (
        np.array([20, 100, 100]),
        np.array([30, 255, 255])
    )
    
    # Purple cloak
    PURPLE = (
        np.array([130, 50, 50]),
        np.array([170, 255, 255])
    )


def create_detector_for_color(color_name: str, enable_trackbars: bool = True,
                              kernel_size: int = 5, erosion_iterations: int = 1,
                              dilation_iterations: int = 1, min_component_area: int = 0,
                              smooth_kernel_size: int = 21) -> CloakDetector:
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
    
    return CloakDetector(lower, upper, enable_trackbars,
                        kernel_size=kernel_size,
                        erosion_iterations=erosion_iterations,
                        dilation_iterations=dilation_iterations,
                        min_component_area=min_component_area,
                        smooth_kernel_size=smooth_kernel_size)