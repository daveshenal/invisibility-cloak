# 🧙‍♂️ Invisibility Cloak 🪄

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)

A real-time invisibility cloak implementation using OpenCV and Python. This project captures a background image from your webcam and then makes objects of a specific color appear "invisible" by replacing them with the background in real-time.

## ✨ Features

- Real-time background capture and replacement
- **Interactive color calibration** - Click on your cloak in real-time to automatically detect the perfect color range
- Configurable cloak color detection using HSV color space
- Interactive controls for easy operation
- Modular code structure
- Quality presets (low, medium, high, ultra) for different performance needs

## 🛠️ Requirements

- Python 3.10 or higher
- Webcam
- A cloth (an object) of any color (the system will automatically detect the color range)

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

**Main Workflow - Color Calibration:**

1. The webcam will activate and show the real-time feed
2. **Click on your cloak 3 times** in different areas to sample the color
3. The system automatically calculates the optimal HSV color range
4. A mask preview window shows what will be made invisible
5. Press Enter/Space to confirm, or 'r' to reset samples
6. Your cloak should now appear "invisible" showing the background instead

**Runtime Controls:**

- Press 'q' to quit
- Press 'b' to capture a new background
- Press 's' to save the current frame
- Press 'r' to reset to default HSV values

## ⚙️ Configuration

### Predefined Color Ranges

The system comes with optimized HSV ranges for common cloak colors:

- **Green** - Most common, works well in most lighting
- **Blue** - Good contrast, suitable for indoor/outdoor use
- **Red** - Handles hue wrap-around automatically
- **Yellow** - Bright and distinctive
- **Purple** - Unique color for special effects

### Custom Color Detection

Instead of predefined colors, the **main workflow** uses interactive calibration:

- Click on your cloak in real-time
- System automatically samples and calculates optimal HSV ranges
- Handles complex colors and lighting conditions
- Provides real-time mask preview during calibration

### Quality Settings

Choose performance vs. quality trade-offs:

- **Ultra**: Best quality, lower performance
- **High**: Balanced quality and performance (default)
- **Medium**: Good performance, acceptable quality
- **Low**: Maximum performance, basic quality

## 📂 Project Structure

```
invisibility-cloak/
├── README.md
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── background_capture.py    # Background capture module
├── cloak_detector.py        # Cloak detection and masking with predefined colors
├── invisibility_cloak.py    # Main application with interactive color calibration
├── utils.py                 # Utility functions
├── test_project.py          # Testing and validation scripts
└── assets/                  # Configuration examples and resources
```

## 🔮 How It Works

1. **Background Capture**: Takes a snapshot of the background without any objects
2. **Interactive Color Calibration**:
   - User clicks on cloak areas in real-time camera feed
   - System samples HSV values from clicked points
   - Automatically calculates optimal color range bounds
   - Provides real-time mask preview
3. **Mask Generation**: Creates a binary mask of the detected cloak area
4. **Morphological Operations**: Refines the mask using erosion and dilation
5. **Background Replacement**: Replaces cloak pixels with corresponding background pixels
6. **Real-time Display**: Shows the result in a live video feed

## 🎯 Key Advantages

- **No manual HSV tuning** - Click and go!
- **Adapts to any color** - Works with any cloak color, not just predefined ones
- **Real-time feedback** - See the mask preview while calibrating
- **Automatic optimization** - System finds the best color range for your specific setup

## 🐞 Troubleshooting

- **Camera not working**: Ensure your webcam is connected and not being used by another application
- **Poor detection**: Use the interactive calibration - click on different areas of your cloak
- **Performance issues**: Try a lower quality setting or close other applications using the camera
- **Color not detected**: Ensure good lighting and click on representative areas of your cloak

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for the excellent computer vision library
- Python community for the robust ecosystem
