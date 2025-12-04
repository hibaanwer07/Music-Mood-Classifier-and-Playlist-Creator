# Music-Mood-Classifier-and-Playlist-Creator
Music Mood Classifier & Playlist Creator

An intelligent music companion that listens to the vibe, detects the mood, and builds the perfect playlist — all automatically.

Overview

The Music Mood Classifier & Playlist Creator is an AI-powered system that analyzes audio, predicts the mood of a song, and automatically generates playlists based on emotion categories. With real-time classification, smooth UI options, and smart playlist building, it brings together ML + music in one clean workflow.

Features

🎧 Mood Classification using machine learning models (Happy, Sad, Energetic, Calm, etc.)

🔍 Audio Feature Extraction (MFCCs, chroma, tempo, spectral features)

🎶 Automatic Playlist Generation based on predicted mood

📂 Upload & Analyze any audio file

⚡ Fast Prediction with optimized preprocessing

🧠 Customizable Model (SVM / Random Forest / Deep Learning)

📱 Clean & Responsive UI (if deployed with Flask/Streamlit)

🔄 Extendable Framework for adding new moods, features, or DSP modules

Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/music-mood-classifier.git
cd music-mood-classifier
2️⃣ Create Virtual Environment
python -m venv venv
source venv/Scripts/activate   # Windows
# OR
source venv/bin/activate       # Linux/Mac
3️⃣ Install Dependencies
pip install -r requirements.txt
Usage
Run the App
python app.py

Open:

http://localhost:5000

Upload an audio file

Get the predicted mood

Auto-generate a playlist matching the mood

Configuration

Modify the following based on your setup:

Mood Labels
MOOD_CLASSES = ["Happy", "Sad", "Energetic", "Calm"]
Audio Settings
SAMPLE_RATE = 22050
FEATURE_TYPE = "mfcc"
Playlist Settings
PLAYLIST_SIZE = 10
Model Training

Training notebooks available in notebooks/:

Data preprocessing

Feature extraction

Model comparison (SVM, RF, CNN, LSTM)

Final model saving

To train a new model:

python train.py
Contributing

Fork this repo

Create a new branch:

git checkout -b feature/new-feature

Add your improvements

Commit:

git commit -m "Add new feature"

Push and open a Pull Request

License

This project is licensed under the MIT License.

Acknowledgments

Librosa – Audio processing

Scikit-learn / TensorFlow – ML & DL models

Flask / Streamlit – Frontend UI

Everyone contributing to open-source music-tech ❤️
