# 🧙‍♂️ Invisibility Cloak Project 🪄

A classic invisibility cloak implementation using OpenCV and Python. This project captures a background image from your webcam and then makes objects of a specific color (like a green cloth) appear "invisible" by replacing them with the background in real-time.

## ✨ Features

- Real-time background capture and replacement
- Configurable cloak color detection using HSV color space
- Morphological operations for clean mask generation
- Interactive controls for easy operation
- Modular code structure

## 🛠️ How It Works - Dev Plan :D

1. **Background Capture**: Takes a snapshot of the background without any objects
2. **Color Detection**: Uses HSV color space to detect the specified cloak color
3. **Mask Generation**: Creates a binary mask of the detected cloak area
4. **Morphological Operations**: Refines the mask using erosion and dilation
5. **Background Replacement**: Replaces cloak pixels with corresponding background pixels
6. **Real-time Display**: Shows the result in a live video feed
