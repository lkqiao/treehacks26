#!/usr/bin/env python3
"""
Quick launcher for the Chinese Character Tutor
Run this to start the application with one command
"""

import sys
import os
import subprocess

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  中文 Chinese Character Tutor 🎨                          ║
║                                                          ║
║  Learn to write Chinese with real-time feedback          ║
║  Powered by MediaPipe hand detection & DTW matching      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Check if all required packages are installed."""
    required = ['cv2', 'mediapipe', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: uv sync")
        return False
    
    print("All dependencies installed")
    return True

def check_camera():
    """Check if camera is accessible."""
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        print("macOS: Settings → Privacy & Security → Camera → Grant access")
        print("Windows: Check camera in Device Manager")
        cap.release()
        return False
    
    cap.release()
    print("Camera accessible")
    return True

def check_characters_json():
    """Check if characters.json exists."""
    if not os.path.exists('characters.json'):
        print("❌ characters.json not found")
        return False
    
    print("Character database loaded")
    return True

def main():
    print_banner()
    
    print("\nPre-launch checks...")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Camera", check_camera),
        ("Character DB", check_characters_json)
    ]
    
    all_passed = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Some checks failed. Fix issues above and try again.")
        sys.exit(1)
    
    print("\nAll systems ready!")
    print("\nLaunching Chinese Character Tutor...\n")
    print("=" * 60)
    print("KEYBOARD SHORTCUTS:")
    print("  1 - Teaching Mode")
    print("  2 - Pinyin Recognition")
    print("  3 - English Translation")
    print()
    print("  SPACE - Submit/Next     C - Clear drawing")
    print("  M - Menu                Q - Quit")
    print("=" * 60)
    print()
    
    # Run the main app
    try:
        from main_app import TutorApp
        app = TutorApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
