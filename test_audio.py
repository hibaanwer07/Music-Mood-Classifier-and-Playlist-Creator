#!/usr/bin/env python3
"""
Test script to verify audio loading functionality
"""
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_utils import load_audio_file, validate_audio_file, setup_audio_environment
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_loading():
    """Test audio loading with a sample file"""
    print("Testing audio loading functionality...")

    # Set up audio environment
    setup_audio_environment()
    print("Audio environment setup completed")

    # Test with a sample audio file
    test_file = r"C:\Users\ASUS\Downloads\Music_Mood_Classifier_and_Playlist_Generator\Music_Mood_Classifier_and_Playlist_Generator\venv\archive (4)\Data\genres_original\jazz\jazz.00054.wav"

    if os.path.exists(test_file):
        print(f"Testing with file: {test_file}")

        try:
            # Validate the file first
            print("Validating audio file...")
            is_valid = validate_audio_file(test_file)
            print(f"File validation result: {is_valid}")

            if is_valid:
                # Try to load the file
                print("Loading audio file...")
                waveform, sample_rate = load_audio_file(test_file, target_sr=16000)
                print(f"Successfully loaded audio:")
                print(f"  - Shape: {waveform.shape}")
                print(f"  - Sample rate: {sample_rate}")
                print(f"  - Duration: {len(waveform)/sample_rate".2f"} seconds")
                print(f"  - Data type: {waveform.dtype}")
                return True
            else:
                print("File validation failed")
                return False

        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    else:
        print(f"Test file not found: {test_file}")
        return False

if __name__ == "__main__":
    print("Starting audio loading test...")
    success = test_audio_loading()

    if success:
        print("\n✅ Audio loading test PASSED!")
        print("The audio loading functionality is working correctly.")
    else:
        print("\n❌ Audio loading test FAILED!")
        print("There are still issues with audio loading.")
