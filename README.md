# Music Mood Classifier & Playlist Creator

An intelligent system that listens to music like a vibe-detecting homie — classifies the mood of an audio track, then auto-creates a playlist that matches the emotion. Powered by deep learning, audio feature extraction, and a sprinkle of musical intuition.

# Overview

This project identifies the mood of a music file using audio features (MFCCs, chroma, tempo, etc.) and generates a playlist that resonates with that same energy. Whether it's chill, happy, sad, energetic, romantic, or lo-fi — the model gets the vibe and matches you with similar tracks.

# Features

 Mood Classification using deep learning

 Automatic Playlist Generator

 Extracts advanced audio features (MFCC, Spectral Contrast, Chroma)

 Works on MP3, WAV, and other common audio formats

 Trained on labeled emotion-based datasets

 Uses Librosa + TensorFlow/PyTorch pipeline

 Clean project structure for easy scaling

# Project Workflow

Upload an audio file

Model predicts the mood (Happy / Sad / Energetic / Calm / etc.)

System fetches recommendations from your dataset

Generates a ready-to-play playlist

# Tech Stack

Python 3.10+

Librosa (feature extraction)

TensorFlow / PyTorch (model training)

NumPy, Pandas, Scikit-learn

Matplotlib / Seaborn (visualization)


 # Installation & Setup
   
 # Install Dependencies
pip install -r requirements.txt

 # Run the App (if using Streamlit)
streamlit run app.py

# Training the Model
Extract Features
python src/feature_extraction.py

Train Mood Classifier
python src/model_training.py

Predict Mood
python src/predict_mood.py --file path/to/song.mp3

# Playlist Creation

After prediction:

python src/playlist_generator.py --mood "Happy"


This automatically fetches songs labeled with the same mood and generates a JSON/CSV playlist.

# Dataset

You can use:

GTZAN + mood-labeled variants

EMO-Music dataset

Custom curated datasets

Add more data → better accuracy → better vibes.

# Results

 High accuracy on multi-class mood detection

 Fast prediction (<1 sec per audio file)

 Smooth playlist generation



# License

MIT License • Free to use, remix, vibe with.

# Acknowledgments

Librosa team for amazing audio tools

TensorFlow/PyTorch communities

Open-source music datasets
