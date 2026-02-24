import librosa
import numpy as np
import warnings

def extract_features(file_path):
    """
    Extract audio features from a music file using librosa.

    Parameters:
    file_path (str): Path to the audio file

    Returns:
    dict: Dictionary containing extracted features
    """
    try:
        # Suppress warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load audio file with different backends
            try:
                y, sr = librosa.load(file_path, duration=30, sr=None)  # Load first 30 seconds with original sample rate
            except Exception as e:
                print(f"Primary load failed for {file_path}, trying alternative method: {e}")
                try:
                    # Try with explicit audio codec
                    y, sr = librosa.load(file_path, duration=30, sr=22050)
                except Exception as e2:
                    print(f"Alternative load also failed for {file_path}: {e2}")
                    return {}

        # Ensure we have valid audio data
        if len(y) == 0:
            print(f"Empty audio data for {file_path}")
            return {}

        # Extract features
        features = {}

        # Length
        features['length'] = len(y) / sr

        # Chroma features
        try:
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_stft_mean'] = float(np.mean(chroma_stft))
            features['chroma_stft_var'] = float(np.var(chroma_stft))
        except Exception as e:
            print(f"Error extracting chroma features: {e}")
            features['chroma_stft_mean'] = 0.0
            features['chroma_stft_var'] = 0.0

        # RMS (Root Mean Square) Energy
        try:
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_var'] = float(np.var(rms))
        except Exception as e:
            print(f"Error extracting RMS features: {e}")
            features['rms_mean'] = 0.0
            features['rms_var'] = 0.0

        # Spectral Centroid
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_var'] = float(np.var(spectral_centroid))
        except Exception as e:
            print(f"Error extracting spectral centroid: {e}")
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_var'] = 0.0

        # Spectral Bandwidth
        try:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_var'] = float(np.var(spectral_bandwidth))
        except Exception as e:
            print(f"Error extracting spectral bandwidth: {e}")
            features['spectral_bandwidth_mean'] = 0.0
            features['spectral_bandwidth_var'] = 0.0

        # Spectral Rolloff
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['rolloff_var'] = float(np.var(spectral_rolloff))
        except Exception as e:
            print(f"Error extracting spectral rolloff: {e}")
            features['rolloff_mean'] = 0.0
            features['rolloff_var'] = 0.0

        # Zero Crossing Rate
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))
            features['zero_crossing_rate_var'] = float(np.var(zero_crossing_rate))
        except Exception as e:
            print(f"Error extracting zero crossing rate: {e}")
            features['zero_crossing_rate_mean'] = 0.0
            features['zero_crossing_rate_var'] = 0.0

        # Harmony and Perceptr
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            harmony = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
            features['harmony_mean'] = float(np.mean(harmony))
            features['harmony_var'] = float(np.var(harmony))
            perceptr = librosa.feature.chroma_cqt(y=y_perc, sr=sr)
            features['perceptr_mean'] = float(np.mean(perceptr))
            features['perceptr_var'] = float(np.var(perceptr))
        except Exception as e:
            print(f"Error extracting harmony and perceptr: {e}")
            features['harmony_mean'] = 0.0
            features['harmony_var'] = 0.0
            features['perceptr_mean'] = 0.0
            features['perceptr_var'] = 0.0

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
        except Exception as e:
            print(f"Error extracting tempo: {e}")
            features['tempo'] = 0.0

        # MFCCs (Mel-frequency cepstral coefficients)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            for i in range(20):
                features[f'mfcc{i+1}_mean'] = 0.0
                features[f'mfcc{i+1}_var'] = 0.0

        return features

    except Exception as e:
        print(f"Unexpected error extracting features from {file_path}: {e}")
        return {}
