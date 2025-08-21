"""
Test script
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        from utils import initialize_camera, release_camera
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils module: {e}")
        return False
    
    try:
        from cloak_detector import CloakDetector, ColorRanges
        print("✓ Cloak detector module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cloak detector module: {e}")
        return False
    
    try:
        from background_capture import BackgroundCapture
        print("✓ Background capture module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import background capture module: {e}")
        return False
    
    try:
        from invisibility_cloak import InvisibilityCloak
        print("✓ Main invisibility cloak module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main module: {e}")
        return False
    
    return True


def test_cloak_detector():
    """Test basic cloak detector functionality."""
    print("\nTesting cloak detector functionality...")
    
    try:
        from cloak_detector import create_detector_for_color, ColorRanges
        import cv2
        
        # Test creating detectors for different colors (without trackbars for testing)
        colors = ['green', 'blue', 'red', 'yellow', 'purple']
        detectors = []
        for color in colors:
            detector = create_detector_for_color(color, enable_trackbars=False)
            detectors.append(detector)
            print(f"✓ Created detector for {color} color")
        
        # Test HSV values
        green_lower, green_upper = ColorRanges.GREEN
        print(f"✓ Green HSV range: {green_lower} to {green_upper}")
        
        # Clean up all detectors and windows
        for detector in detectors:
            detector.cleanup()
        
        # Ensure all windows are closed
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"✗ Cloak detector test failed: {e}")
        # Clean up on error too
        try:
            cv2.destroyAllWindows()
        except:
            pass
        return False


def test_background_capture():
    """Test background capture functionality."""
    print("\nTesting background capture functionality...")
    
    try:
        from background_capture import BackgroundCapture
        
        # Create background capture instance
        bg_capture = BackgroundCapture()
        print("✓ Background capture instance created")
        
        # Test listing backgrounds (should be empty initially)
        backgrounds = bg_capture.list_backgrounds()
        print(f"✓ Background listing works (found {len(backgrounds)} backgrounds)")
        
        return True
        
    except Exception as e:
        print(f"✗ Background capture test failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import validate_hsv_values
        import numpy as np
        
        # Test HSV validation
        valid_lower = np.array([40, 40, 40])
        valid_upper = np.array([80, 255, 255])
        
        if validate_hsv_values(valid_lower, valid_upper):
            print("✓ HSV validation works correctly")
        else:
            print("✗ HSV validation failed")
            return False
        
        # Test invalid HSV values
        invalid_lower = np.array([100, 100, 100])
        invalid_upper = np.array([50, 50, 50])
        
        if not validate_hsv_values(invalid_lower, invalid_upper):
            print("✓ HSV validation correctly rejects invalid values")
        else:
            print("✗ HSV validation should reject invalid values")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting project file structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'LICENSE',
        'utils.py',
        'cloak_detector.py',
        'background_capture.py',
        'invisibility_cloak.py'
    ]
    
    required_dirs = ['assets']
    
    all_files_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_files_exist = False
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}/ directory exists")
        else:
            print(f"✗ {directory}/ directory missing")
            all_files_exist = False
    
    return all_files_exist


def main():
    """Run all tests."""
    print("=" * 50)
    print("INVISIBILITY CLOAK PROJECT - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Cloak Detector", test_cloak_detector),
        ("Background Capture", test_background_capture),
        ("Utility Functions", test_utils)
    ]
    
    passed = 0
    total = len(tests)
    
    try:
        for test_name, test_func in tests:
            print(f"\n{test_name} Test:")
            print("-" * 30)
            
            try:
                if test_func():
                    passed += 1
                    print(f"✓ {test_name} test PASSED")
                else:
                    print(f"✗ {test_name} test FAILED")
            except Exception as e:
                print(f"✗ {test_name} test ERROR: {e}")
        
        print("\n" + "=" * 50)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed! The project is ready to use.")
            print("\nNext steps:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Capture background: python background_capture.py")
            print("3. Run invisibility cloak: python invisibility_cloak.py")
        else:
            print("Some tests failed. Please check the errors above.")
            return 1
        
        return 0
        
    finally:
        # Ensure all OpenCV windows are closed
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
