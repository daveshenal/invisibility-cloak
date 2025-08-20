"""
Sample configuration file for the invisibility cloak project.

Copy this file to your project root and modify the values as needed.
You can import this file to customize the cloak detection parameters.
"""

import numpy as np

# HSV Color Ranges for different cloak colors
# These values can be fine-tuned for better detection

# Green Cloak (default)
GREEN_CLOAK = {
    'lower': np.array([40, 40, 40]),      # Lower HSV bounds
    'upper': np.array([80, 255, 255]),    # Upper HSV bounds
    'description': 'Standard green cloth detection'
}

# Blue Cloak
BLUE_CLOAK = {
    'lower': np.array([100, 50, 50]),     # Lower HSV bounds
    'upper': np.array([130, 255, 255]),   # Upper HSV bounds
    'description': 'Standard blue cloth detection'
}

# Red Cloak (handles both ends of HSV spectrum)
RED_CLOAK = {
    'lower': np.array([0, 50, 50]),       # Lower HSV bounds
    'upper': np.array([10, 255, 255]),    # Upper HSV bounds
    'description': 'Standard red cloth detection'
}

# Yellow Cloak
YELLOW_CLOAK = {
    'lower': np.array([20, 100, 100]),    # Lower HSV bounds
    'upper': np.array([30, 255, 255]),    # Upper HSV bounds
    'description': 'Standard yellow cloth detection'
}

# Purple Cloak
PURPLE_CLOAK = {
    'lower': np.array([130, 50, 50]),     # Lower HSV bounds
    'upper': np.array([170, 255, 255]),   # Upper HSV bounds
    'description': 'Standard purple cloth detection'
}

# Custom color ranges (add your own)
CUSTOM_CLOAK = {
    'lower': np.array([0, 0, 0]),         # Modify these values
    'upper': np.array([179, 255, 255]),   # for your specific color
    'description': 'Custom color detection - modify values as needed'
}

# Morphological operation settings
MORPHOLOGY_SETTINGS = {
    'kernel_size': 5,                      # Size of morphological kernel
    'erosion_iterations': 1,               # Number of erosion operations
    'dilation_iterations': 1,              # Number of dilation operations
    'opening_iterations': 1,               # Number of opening operations
    'closing_iterations': 1                # Number of closing operations
}

# Camera settings
CAMERA_SETTINGS = {
    'width': 640,                          # Camera frame width
    'height': 480,                         # Camera frame height
    'fps': 30,                            # Target frames per second
    'camera_index': 0                     # Camera device index
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'enable_fps_display': True,            # Show FPS counter
    'enable_overlay': True,                # Show information overlay
    'save_frames': False,                  # Save processed frames
    'frame_skip': 1                        # Process every Nth frame (1 = all frames)
}

# Background settings
BACKGROUND_SETTINGS = {
    'auto_resize': True,                   # Automatically resize background to match frame
    'background_dir': 'backgrounds',       # Directory to store background images
    'auto_load_latest': True,              # Automatically load most recent background
    'background_format': 'jpg'             # Format for saved background images
}

# Example usage:
# from assets.config_example import GREEN_CLOAK, MORPHOLOGY_SETTINGS
# 
# # Use in your code:
# lower_hsv = GREEN_CLOAK['lower']
# upper_hsv = GREEN_CLOAK['upper']
# kernel_size = MORPHOLOGY_SETTINGS['kernel_size']
