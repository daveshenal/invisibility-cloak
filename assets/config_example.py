"""
Sample configuration file for the invisibility cloak project.
"""

import numpy as np

# HSV Color Ranges for different cloak colors
# These values can be fine-tuned for better detection

# Green Cloak (default)
GREEN_CLOAK = {
    'lower': np.array([40, 40, 40]),
    'upper': np.array([80, 255, 255]),
    'description': 'Standard green cloth detection'
}

# Blue Cloak
BLUE_CLOAK = {
    'lower': np.array([100, 50, 50]),
    'upper': np.array([130, 255, 255]),
    'description': 'Standard blue cloth detection'
}

# Red Cloak
RED_CLOAK = {
    'lower': np.array([0, 50, 50]),
    'upper': np.array([10, 255, 255]),
    'description': 'Standard red cloth detection'
}

# Yellow Cloak
YELLOW_CLOAK = {
    'lower': np.array([20, 100, 100]),
    'upper': np.array([30, 255, 255]),
    'description': 'Standard yellow cloth detection'
}

# Purple Cloak
PURPLE_CLOAK = {
    'lower': np.array([130, 50, 50]),
    'upper': np.array([170, 255, 255]),
    'description': 'Standard purple cloth detection'
}

# Custom color ranges
CUSTOM_CLOAK = {
    'lower': np.array([0, 0, 0]),
    'upper': np.array([179, 255, 255]),
    'description': 'Custom color detection - modify values as needed'
}

# Morphological operation settings
MORPHOLOGY_SETTINGS = {
    'kernel_size': 5,                  # Size of morphological kernel
    'erosion_iterations': 1,           # Number of erosion operations
    'dilation_iterations': 1,          # Number of dilation operations
    'opening_iterations': 1,           # Number of opening operations
    'closing_iterations': 1            # Number of closing operations
}

# Camera settings
CAMERA_SETTINGS = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'camera_index': 0
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'enable_fps_display': True,        # Show FPS counter
    'enable_overlay': True,            # Show information overlay
    'save_frames': False,              # Save processed frames
    'frame_skip': 1                    # Process every Nth frame (1 = all frames)
}

# Background settings
BACKGROUND_SETTINGS = {
    'auto_resize': True,               # Automatically resize background to match frame
    'background_dir': 'backgrounds',   # Directory to store background images
    'auto_load_latest': True,          # Automatically load most recent background
    'background_format': 'jpg'         # Format for saved background images
}