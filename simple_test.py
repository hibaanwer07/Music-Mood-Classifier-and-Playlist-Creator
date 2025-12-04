import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_utils import load_audio_file, validate_audio_file, setup_audio_environment
    print("✅ Successfully imported audio_utils")

    # Set up audio environment
    setup_audio_environment()
    print("✅ Audio environment setup completed")

    # Test with the problematic file
    test_file = r"C:\Users\ASUS\Downloads\Music_Mood_Classifier_and_Playlist_Generator\Music_Mood_Classifier_and_Playlist_Generator\venv\archive (4)\Data\genres_original\jazz\jazz.00054.wav"

    if os.path.exists(test_file):
        print(f"✅ Test file exists: {test_file}")

        # Try to load the file
        try:
            waveform, sample_rate = load_audio_file(test_file, target_sr=16000)
            print("✅ Successfully loaded audio file!")
            print(f"   Shape: {waveform.shape}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Duration: {len(waveform)/sample_rate:.2f} seconds")
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")

except ImportError as e:
    print(f"❌ Failed to import audio_utils: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
