import cv2
import numpy as np
import os
from utils import initialize_camera, release_camera, get_frame, flip_frame, save_image, print_instructions


class BackgroundCapture:
    """
    A class to capture and manage background images for the invisibility effect.
    
    This class handles the capture of background images, saves them to disk,
    and provides methods to load and manage multiple background images.
    """
    
    def __init__(self, background_dir: str = "backgrounds"):
        """
        Initialize the background capture system.
        
        Args:
            background_dir (str): Directory to store background images
        """
        self.background_dir = background_dir
        self.current_background = None
        self.background_filename = None
        
        # Create backgrounds directory if it doesn't exist
        self._ensure_background_dir()
    
    def _ensure_background_dir(self) -> None:
        """Create the backgrounds directory if it doesn't exist."""
        if not os.path.exists(self.background_dir):
            os.makedirs(self.background_dir)
            print(f"Created backgrounds directory: {self.background_dir}")
    
    def capture_background(self, camera_index: int = 0) -> bool:
        """
        Capture a new background image from the camera.
        
        Args:
            camera_index (int): Index of the camera to use
            
        Returns:
            bool: True if background was captured successfully, False otherwise
        """
        cap = None
        try:
            # Initialize camera
            cap = initialize_camera(camera_index)
            
            print("Background capture mode:")
            print("1. Position yourself so the background is clearly visible")
            print("2. Make sure no objects are in the frame")
            print("3. Press 'c' to capture the background")
            print("4. Press 'q' to quit without capturing")
            
            while True:
                # Capture frame
                frame = get_frame(cap)
                if frame is None:
                    print("Error: Could not capture frame from camera")
                    return False
                
                # Flip frame horizontally for mirror effect
                frame = flip_frame(frame)
                
                # Display frame
                cv2.imshow('Background Capture', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Background capture cancelled.")
                    return False
                
                elif key == ord('c'):
                    # Capture background
                    self.current_background = frame.copy()
                    
                    # Generate filename with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.background_filename = f"background_{timestamp}.jpg"
                    filepath = os.path.join(self.background_dir, self.background_filename)
                    
                    # Save background image
                    if save_image(self.current_background, filepath):
                        print(f"Background captured and saved: {filepath}")
                        print(f"Background dimensions: {frame.shape[1]}x{frame.shape[0]}")
                        return True
                    else:
                        print("Error: Failed to save background image")
                        return False
                
                elif key == ord('s'):
                    # Show current frame info
                    print(f"Current frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
                    print(f"Frame data type: {frame.dtype}")
        
        except Exception as e:
            print(f"Error during background capture: {e}")
            return False
        
        finally:
            if cap is not None:
                release_camera(cap)
    
    def load_background(self, filename: str) -> bool:
        """
        Load a background image from file.
        
        Args:
            filename (str): Name of the background image file
            
        Returns:
            bool: True if background was loaded successfully, False otherwise
        """
        filepath = os.path.join(self.background_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: Background file not found: {filepath}")
            return False
        
        try:
            self.current_background = cv2.imread(filepath)
            if self.current_background is None:
                print(f"Error: Could not load background image: {filepath}")
                return False
            
            self.background_filename = filename
            print(f"Background loaded: {filepath}")
            print(f"Background dimensions: {self.current_background.shape[1]}x{self.current_background.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Error loading background: {e}")
            return False
    
    def get_background(self) -> np.ndarray:
        """
        Get the current background image.
        
        Returns:
            np.ndarray: Current background image
            
        Raises:
            RuntimeError: If no background is loaded
        """
        if self.current_background is None:
            raise RuntimeError("No background image loaded. Please capture or load a background first.")
        
        return self.current_background.copy()
    
    def list_backgrounds(self) -> list:
        """
        List all available background images.
        
        Returns:
            list: List of background image filenames
        """
        if not os.path.exists(self.background_dir):
            return []
        
        backgrounds = []
        for filename in os.listdir(self.background_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                backgrounds.append(filename)
        
        return sorted(backgrounds)
    
    def delete_background(self, filename: str) -> bool:
        """
        Delete a background image file.
        
        Args:
            filename (str): Name of the background image file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        filepath = os.path.join(self.background_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: Background file not found: {filepath}")
            return False
        
        try:
            os.remove(filepath)
            print(f"Background deleted: {filepath}")
            
            # If this was the current background, clear it
            if self.background_filename == filename:
                self.current_background = None
                self.background_filename = None
                print("Current background cleared.")
            
            return True
            
        except Exception as e:
            print(f"Error deleting background: {e}")
            return False
    
    def resize_background(self, width: int, height: int) -> bool:
        """
        Resize the current background to match video dimensions.
        
        Args:
            width (int): Target width
            height (int): Target height
            
        Returns:
            bool: True if resize was successful, False otherwise
        """
        if self.current_background is None:
            print("Error: No background image loaded.")
            return False
        
        try:
            self.current_background = cv2.resize(self.current_background, (width, height))
            print(f"Background resized to: {width}x{height}")
            return True
            
        except Exception as e:
            print(f"Error resizing background: {e}")
            return False


def main():
    """Main function to run the background capture application."""
    print_instructions()
    
    # Create background capture instance
    bg_capture = BackgroundCapture()
    
    # List existing backgrounds
    existing_backgrounds = bg_capture.list_backgrounds()
    if existing_backgrounds:
        print("Existing background images:")
        for i, bg in enumerate(existing_backgrounds, 1):
            print(f"  {i}. {bg}")
        print()
    
    # Capture new background
    print("Starting background capture...")
    if bg_capture.capture_background():
        print("Background capture completed successfully!")
        
        # Option to load existing background
        if existing_backgrounds:
            print("\nWould you like to load an existing background instead?")
            print("Enter the filename or press Enter to keep the new one:")
            
            user_input = input().strip()
            if user_input and user_input in existing_backgrounds:
                if bg_capture.load_background(user_input):
                    print(f"Loaded existing background: {user_input}")
                else:
                    print("Failed to load existing background, keeping new one.")
    else:
        print("Background capture failed or was cancelled.")


if __name__ == "__main__":
    main()