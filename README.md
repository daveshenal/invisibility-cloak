# 🧙‍♂️ Invisibility Cloak 🪄

![Status](https://img.shields.io/badge/status-Working-brightgreen)
![Next Version](https://img.shields.io/badge/Next%20Version-WIP-orange)

A real-time invisibility cloak implementation using OpenCV and Python. This project captures a background image from your webcam and then makes objects of a specific color (like a green cloth) appear "invisible" by replacing them with the background in real-time.

## ✨ Features

- Real-time background capture and replacement
- Configurable cloak color detection using HSV color space
- Morphological operations for clean mask generation
- Interactive controls for easy operation
- Professional modular code structure

## 🛠️ Requirements

- Python 3.10 or higher
- Webcam
- A cloth of the specified color (default: green)

## 📦 Installation

1. Clone or download this repository
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Step 1: Capture Background

First, capture a clean background image:

```bash
python background_capture.py
```

- Position yourself so the background is visible
- Press 'c' to capture the background
- Press 'q' to quit

### Step 2: Run the Invisibility Cloak

```bash
python invisibility_cloak.py
```

- The webcam will activate and show the real-time effect
- Hold a green cloth (or cloth of your specified color) in front of the camera
- The cloth should appear "invisible" showing the background instead
- Press 'q' to quit

## ⚙️ Configuration

You can change the cloak color by modifying the HSV values in `cloak_detector.py`:

```python
# Default green cloak values
LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([80, 255, 255])
```

## 📂 Project Structure

```
invisibility-cloak/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                  # MIT License
├── background_capture.py    # Background capture module
├── cloak_detector.py        # Cloak detection and masking
├── invisibility_cloak.py    # Main application
├── utils.py                 # Utility functions
└── assets/                  # Sample images/videos (optional)
```

## 🔮 How It Works

1. **Background Capture**: Takes a snapshot of the background without any objects
2. **Color Detection**: Uses HSV color space to detect the specified cloak color
3. **Mask Generation**: Creates a binary mask of the detected cloak area
4. **Morphological Operations**: Refines the mask using erosion and dilation
5. **Background Replacement**: Replaces cloak pixels with corresponding background pixels
6. **Real-time Display**: Shows the result in a live video feed

## 🐞 Troubleshooting

- **Camera not working**: Ensure your webcam is connected and not being used by another application
- **Poor detection**: Adjust the HSV values in `cloak_doctor.py` for better color matching
- **Performance issues**: Close other applications using the camera or reduce video resolution

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for the excellent computer vision library
- Python community for the robust ecosystem
