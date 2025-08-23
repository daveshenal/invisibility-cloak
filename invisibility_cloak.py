"""
Main invisibility cloak application.
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional

from utils import initialize_camera, release_camera, get_frame, flip_frame, print_instructions
from cloak_detector import create_detector_for_color, ColorRanges
from background_capture import BackgroundCapture


class InvisibilityCloak:
    """
    Main class for the invisibility cloak application.
    
    This class orchestrates the background capture, cloak detection,
    and real-time video processing to create the invisibility effect.
    """
    
    def __init__(self, cloak_color: str = "green", camera_index: int = 0,
                 width: int = 640, height: int = 480, fps: int = 30,
                 quality: str = "high"):
        """
        Initialize the invisibility cloak application.
        
        Args:
            cloak_color (str): Color of the cloak to detect
            camera_index (int): Index of the camera to use
        """
        self.camera_index = camera_index
        self.cap = None
        self.background_capture = BackgroundCapture(width=width, height=height, fps=fps)
        self.cloak_detector = None
        self.background = None
        self.is_running = False
        self.width = width
        self.height = height
        self.fps_target = fps
        self.quality = quality
        
        # Initialize cloak detector
        try:
            # Choose smoothing parameters for high quality
            if quality == "ultra":
                kernel_size = 9
                erosion_iter = 1
                dilation_iter = 2
                min_area = 500
                smooth_kernel = 41
            elif quality == "high":
                kernel_size = 7
                erosion_iter = 1
                dilation_iter = 1
                min_area = 200
                smooth_kernel = 31
            elif quality == "medium":
                kernel_size = 5
                erosion_iter = 1
                dilation_iter = 1
                min_area = 100
                smooth_kernel = 21
            else:
                kernel_size = 5
                erosion_iter = 1
                dilation_iter = 1
                min_area = 0
                smooth_kernel = 15

            self.cloak_detector = create_detector_for_color(
                cloak_color,
                kernel_size=kernel_size,
                erosion_iterations=erosion_iter,
                dilation_iterations=dilation_iter,
                min_component_area=min_area,
                smooth_kernel_size=smooth_kernel,
            )
            print(f"Initialized cloak detector for {cloak_color} color")
        except ValueError as e:
            print(f"Error: {e}")
            print("Available colors: green, blue, red, yellow, purple")
            sys.exit(1)
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.last_time = cv2.getTickCount()
    
    def load_background(self, background_path: Optional[str] = None) -> bool:
        """
        Load a background image for the invisibility effect.
        
        Args:
            background_path (str, optional): Path to background image file.
                If None, will try to load the most recent background.
                
        Returns:
            bool: True if background was loaded successfully, False otherwise
        """
        if background_path:
            # Load specific background file
            if os.path.exists(background_path):
                return self.background_capture.load_background(background_path)
            else:
                print(f"Error: Background file not found: {background_path}")
                return False
        else:
            # Try to load the most recent background
            backgrounds = self.background_capture.list_backgrounds()
            if backgrounds:
                latest_background = backgrounds[-1]
                print(f"Loading most recent background: {latest_background}")
                return self.background_capture.load_background(latest_background)
            else:
                print("No background images found. Please capture a background first.")
                return False
    
    def capture_new_background(self) -> bool:
        """
        Capture a new background image.
        
        Returns:
            bool: True if background was captured successfully, False otherwise
        """
        print("Capturing new background...")
        return self.background_capture.capture_background(self.camera_index)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame to apply the invisibility effect.
        
        Args:
            frame (np.ndarray): Input BGR frame
            
        Returns:
            np.ndarray: Processed frame with invisibility effect
        """
        if self.background is None:
            return frame
        
        # Get the background image
        try:
            background = self.background_capture.get_background()
        except RuntimeError:
            return frame
        
        # Ensure background matches frame dimensions
        if background.shape[:2] != frame.shape[:2]:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        
        # Soft alpha mask for feathered blending
        alpha = self.cloak_detector.create_alpha_mask(frame)
        
        # Expand alpha to 3 channels
        alpha_3 = np.dstack([alpha, alpha, alpha])
        inv_alpha_3 = 1.0 - alpha_3
        
        # Blend: cloak areas replaced by background using soft edges
        result = (frame.astype(np.float32) * inv_alpha_3 + background.astype(np.float32) * alpha_3)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add informational overlay to the frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Frame with overlay information
        """
        overlay = frame.copy()
        
        # Add FPS counter
        current_time = cv2.getTickCount()
        self.frame_count += 1
        
        if current_time - self.last_time >= cv2.getTickFrequency():
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        # Draw FPS
        cv2.putText(overlay, f"FPS: {self.fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(overlay, "Press 'q' to quit, 'b' to capture background", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        # Add status information
        status = "Background: Loaded" if self.background is not None else "Background: Not loaded"
        cv2.putText(overlay, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0) if self.background is not None else (0, 0, 255), 2)
        
        return overlay
    
    def run(self) -> None:
        """
        Run the main invisibility cloak application.
        """
        try:
            # Initialize camera
            self.cap = initialize_camera(self.camera_index, width=self.width, height=self.height, fps=self.fps_target)
            
            # Try to load existing background
            if not self.load_background():
                print("No background loaded. Capturing new background...")
                if not self.capture_new_background():
                    print("Failed to capture background. Exiting.")
                    return
            
            # Get background reference
            try:
                self.background = self.background_capture.get_background()
            except RuntimeError:
                print("Error: No background available. Exiting.")
                return
            
            # Calibration step: ask user to click on the cloak 3 times to sample color
            if not self._calibrate_color_with_clicks(num_clicks=3):
                print("Calibration cancelled. Exiting.")
                return

            print("Invisibility cloak is running!")
            print("Controls:")
            print("  'q' - Quit application")
            print("  'b' - Capture new background")
            print("  's' - Save current frame")
            print("  'r' - Reset to default HSV values")
            
            self.is_running = True
            
            while self.is_running:
                # Capture frame
                frame = get_frame(self.cap)
                if frame is None:
                    print("Error: Could not capture frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = flip_frame(frame)
                
                # Process frame for invisibility effect
                processed_frame = self.process_frame(frame)
                
                # Add overlay information
                final_frame = self.add_overlay(processed_frame)
                
                # Display result
                cv2.imshow('Invisibility Cloak', final_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting application...")
                    break
                
                elif key == ord('b'):
                    print("Capturing new background...")
                    if self.capture_new_background():
                        try:
                            self.background = self.background_capture.get_background()
                            print("New background loaded successfully!")
                        except RuntimeError:
                            print("Failed to load new background.")
                    else:
                        print("Background capture failed.")
                
                elif key == ord('s'):
                    # Save current frame
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"invisibility_frame_{timestamp}.jpg"
                    if cv2.imwrite(filename, final_frame):
                        print(f"Frame saved: {filename}")
                    else:
                        print("Failed to save frame.")
                
                elif key == ord('r'):
                    # Reset HSV values to default
                    default_lower, default_upper = ColorRanges.GREEN
                    self.cloak_detector.set_hsv_values(default_lower, default_upper)
                    print("HSV values reset to default.")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user.")
        
        except Exception as e:
            print(f"Error during execution: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources and close windows."""
        self.is_running = False
        
        if self.cap is not None:
            release_camera(self.cap)
            self.cap = None
        
        if self.cloak_detector is not None:
            self.cloak_detector.cleanup()
        
        cv2.destroyAllWindows()
        print("Cleanup completed.")

    def _calibrate_color_with_clicks(self, num_clicks: int = 3) -> bool:
        """
        Interactive calibration: collect N clicks on the cloak to estimate HSV range.
        Shows live preview and mask while sampling, supports reset and cancel.
        
        Controls:
          - Left click: sample a point
          - 'r': reset samples
          - 'enter'/'space': confirm when enough samples
          - 'q': cancel
        """
        window_name = 'Calibration - Click cloak (0/{})'.format(num_clicks)
        mask_window = 'Mask Preview'

        samples_hsv: list[np.ndarray] = []
        click_points: list[tuple[int, int]] = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_points.append((x, y))

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)
        cv2.namedWindow(mask_window)

        try:
            while True:
                frame = get_frame(self.cap)
                if frame is None:
                    print("Error: Could not capture frame during calibration")
                    return False

                frame = flip_frame(frame)

                # Draw sampled points
                preview = frame.copy()
                for idx, (x, y) in enumerate(click_points):
                    cv2.circle(preview, (x, y), 6, (0, 255, 255), 2)
                    cv2.putText(preview, str(idx + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Guidance text
                cv2.putText(preview, 'Click {} cloak points. Enter to confirm, R to reset, Q to cancel'.format(num_clicks),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(preview, 'Samples: {}/{}'.format(len(click_points), num_clicks),
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow(window_name, preview)

                # If we have new clicks, sample HSV values
                if len(samples_hsv) < len(click_points):
                    # Sample a small 5x5 patch around each new click to average noise
                    x, y = click_points[-1]
                    h, w = frame.shape[:2]
                    x0, x1 = max(0, x - 2), min(w, x + 3)
                    y0, y1 = max(0, y - 2), min(h, y + 3)
                    patch_bgr = frame[y0:y1, x0:x1]
                    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                    median_hsv = np.median(patch_hsv.reshape(-1, 3), axis=0).astype(np.uint8)
                    samples_hsv.append(median_hsv)

                # Build a temporary mask preview if we have at least 1 sample
                if samples_hsv:
                    lower, upper, lower2, upper2 = self._estimate_hsv_ranges_from_samples(samples_hsv)
                    # Temporarily compute mask for preview
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    if lower2 is not None and upper2 is not None:
                        mask2 = cv2.inRange(hsv, lower2, upper2)
                        mask = cv2.bitwise_or(mask, mask2)
                    refined = cv2.GaussianBlur(mask, (15, 15), 0)
                    cv2.imshow(mask_window, refined)

                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32):  # Enter or Space
                    if len(samples_hsv) >= num_clicks:
                        lower, upper, lower2, upper2 = self._estimate_hsv_ranges_from_samples(samples_hsv)
                        # Apply to detector
                        self.cloak_detector.set_hsv_values(lower, upper, lower2, upper2)
                        break
                elif key == ord('r'):
                    samples_hsv.clear()
                    click_points.clear()
                    cv2.destroyWindow(mask_window)
                    cv2.namedWindow(mask_window)
                elif key == ord('q'):
                    return False

            cv2.destroyWindow(window_name)
            cv2.destroyWindow(mask_window)
            return True
        finally:
            try:
                cv2.setMouseCallback(window_name, lambda *args: None)
            except:
                pass

    def _estimate_hsv_ranges_from_samples(self, samples_hsv: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        From a list of sampled HSV points, compute lower/upper bounds.
        Handles hue wrap-around by returning an optional secondary range.
        """
        arr = np.stack(samples_hsv, axis=0).astype(np.int32)
        # Use median as robust center
        median = np.median(arr, axis=0)
        h_med, s_med, v_med = int(median[0]), int(median[1]), int(median[2])

        # Spread based on sample spread; add safe margins !!
        h_spread = int(np.maximum(5, np.std(arr[:, 0]) * 1.5 + 5))
        s_spread = int(np.maximum(30, np.std(arr[:, 1]) * 1.5 + 30))
        v_spread = int(np.maximum(30, np.std(arr[:, 2]) * 1.5 + 30))

        h_low = h_med - h_spread
        h_high = h_med + h_spread

        s_low = max(0, s_med - s_spread)
        s_high = min(255, s_med + s_spread)
        v_low = max(0, v_med - v_spread)
        v_high = min(255, v_med + v_spread)

        # Handle hue wrap-around by splitting into two ranges if needed
        lower2 = upper2 = None
        if h_low < 0:
            lower = np.array([0, s_low, v_low], dtype=np.uint8)
            upper = np.array([h_high % 180, s_high, v_high], dtype=np.uint8)
            lower2 = np.array([180 + h_low, s_low, v_low], dtype=np.uint8)
            upper2 = np.array([179, s_high, v_high], dtype=np.uint8)
        elif h_high > 179:
            lower = np.array([h_low % 180, s_low, v_low], dtype=np.uint8)
            upper = np.array([179, s_high, v_high], dtype=np.uint8)
            lower2 = np.array([0, s_low, v_low], dtype=np.uint8)
            upper2 = np.array([h_high - 180, s_high, v_high], dtype=np.uint8)
        else:
            lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        return lower, upper, lower2, upper2


def main():
    """Main function to run the invisibility cloak application."""
    print_instructions()
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Invisibility Cloak Application")
    parser.add_argument("--color", "-c", default="green", 
                       choices=["green", "blue", "red", "yellow", "purple"],
                       help="Color of the cloak to detect (default: green)")
    parser.add_argument("--camera", "-cam", type=int, default=0,
                       help="Camera index to use (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Capture height (default: 720)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Capture FPS target (default: 30)")
    parser.add_argument("--quality", choices=["ultra", "high", "medium", "low"], default="high",
                       help="Mask smoothing quality (default: high)")
    parser.add_argument("--background", "-b", type=str,
                       help="Path to background image file")
    
    args = parser.parse_args()
    
    # Create and run the invisibility cloak application
    try:
        app = InvisibilityCloak(
            cloak_color=args.color,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            quality=args.quality,
        )
        
        # Load background if specified
        if args.background:
            if not app.load_background(args.background):
                print(f"Failed to load background: {args.background}")
                return
        
        # Run the application
        app.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()