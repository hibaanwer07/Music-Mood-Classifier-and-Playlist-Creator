import streamlit as st
import os
import shutil # For file operations
import spotipy # Spotify API client
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError # Spotify OAuth handling
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import time
from pathlib import Path # For .env path handling
import json # For playlist persistence
import joblib # For loading ML models
import pandas as pd # For data manipulation
import subprocess # For retraining model
from src.feature_extraction import extract_features
from src.spotify_integration import spotify_authenticate, get_playlist_id, create_playlist, refresh_spotify_token, get_spotify_recommendations, add_track_to_playlist

# ------------------------------------------------------------
# 🔧 Load environment variables from .env (secure and correct)
# ------------------------------------------------------------
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8502/callback').strip()

# ✅ Clean redirect URI
if SPOTIFY_REDIRECT_URI.startswith('"') and SPOTIFY_REDIRECT_URI.endswith('"'):
    SPOTIFY_REDIRECT_URI = SPOTIFY_REDIRECT_URI[1:-1].strip()
# ------------------------------------------------------------
# 🎧 Spotify OAuth Setup
# ------------------------------------------------------------
def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope='playlist-modify-public playlist-modify-private user-library-read'
    )

# ------------------------------------------------------------
# 📦 Load Default Moods (Adaptable Template)
# ------------------------------------------------------------
default_moods = ['sad', 'calm', 'happy', 'energetic']

# ------------------------------------------------------------
# 🍪 Playlist Persistence with File (since cookies not available)
# ------------------------------------------------------------
PLAYLISTS_FILE = 'playlists.json'

def load_playlists():
    try:
        if os.path.exists(PLAYLISTS_FILE):
            with open(PLAYLISTS_FILE, 'r') as f:
                return json.load(f)
        else:
            return {m: [] for m in default_moods}
    except:
        return {m: [] for m in default_moods}

def save_playlists(playlists):
    with open(PLAYLISTS_FILE, 'w') as f:
        json.dump(playlists, f)


# ------------------------------------------------------------
# 🎵 Simple Rule-Based Mood Prediction for Spotify Tracks
# ------------------------------------------------------------
def predict_spotify_mood(audio_features):
    valence = audio_features.get('valence', 0.5)
    energy = audio_features.get('energy', 0.5)
    danceability = audio_features.get('danceability', 0.5)
    tempo = audio_features.get('tempo', 120)

    if valence > 0.5 and energy > 0.6:
        return 'happy'
    elif valence < 0.4 and energy < 0.4:
        return 'sad'
    elif energy < 0.3 and tempo < 100:
        return 'calm'
    else:
        return 'energetic'



# Function for Upload and Predict interface
def upload_predict():
    st.header('📁 Upload & Predict Mood')

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a music file (WAV or MP3)", type=['wav', 'mp3'], help="Upload your song to analyze its mood.")
    with col2:
        if st.button('🔮 Predict', type="primary"):
            pass  # Placeholder for alignment

    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Save the uploaded file temporarily with original name
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        progress_bar.progress(20)
        status_text.text("Extracting audio features...")

        # Extract features
        try:
            features = extract_features(temp_path) # Extract features from the uploaded file
            features_df = pd.DataFrame([features])
            progress_bar.progress(50)
            status_text.text("Predicting mood...")

            # Predict mood
            X_scaled = st.session_state.scaler.transform(features_df) # Scale features
            predictions = st.session_state.model.predict(X_scaled)
            mood = st.session_state.label_encoder.inverse_transform(predictions)[0]
            progress_bar.progress(80)

            col1, col2 = st.columns(2) # Display results
            with col1:
                st.success(f"🎭 Predicted Mood: **{mood.capitalize()}**") 
            with col2:
                mood_emoji = {'happy': '😊', 'sad': '😢', 'energetic': '⚡', 'calm': '🧘'}.get(mood, '🎵') 
                st.markdown(f"### {mood_emoji}")

            # Automatically save to playlist
            # Create mood folder if not exists
            mood_dir = f"songs/{mood}"
            os.makedirs(mood_dir, exist_ok=True)
            file_path = f"{mood_dir}/{uploaded_file.name}"
            shutil.move(temp_path, file_path)
            # Add to session state
            if 'local_playlists' not in st.session_state:
                st.session_state.local_playlists = {m: [] for m in default_moods}
            song_entry = {'name': uploaded_file.name, 'path': file_path}
            if song_entry not in st.session_state.local_playlists[mood]:
                st.session_state.local_playlists[mood].append(song_entry)
            # Save playlists to file
            save_playlists(st.session_state.local_playlists)
            progress_bar.progress(90)

            # Song saved locally
            spotify_msg = st.empty()
            spotify_msg.success(f"✅ Song saved to {mood.capitalize()} playlist!")

            # Attempt to add to Spotify playlist if authenticated
            if st.session_state.get('spotify_authenticated', False):
                sp = spotify_authenticate()
                if sp:
                    results = sp.search(q=uploaded_file.name, type='track', limit=1)
                    if results['tracks']['items']:
                        uri = results['tracks']['items'][0]['uri']
                        add_track_to_playlist(mood, uri)
                        spotify_msg.success(f"✅ Song saved locally and added to Spotify {mood.capitalize()} playlist!")
                    else:
                        st.info("Song not found on Spotify.")
                else:
                    st.info("Spotify not authenticated.")

            progress_bar.progress(100)
            status_text.text("Complete!")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            progress_bar.empty()
            status_text.empty()

        # Clean up if temp file still exists (in case of error)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Retrain model
    if st.button('🔄 Retrain Model', help="Improve model with new data"):
        with st.spinner("Retraining model..."):
            import subprocess
            subprocess.run(['python', 'src/train_model.py']) # Retrain model script
        st.success("✅ Model retrained successfully!")

# Function for View Playlists interface
def view_playlists():
    st.header('🎧 View & Play Playlists')

    if 'local_playlists' not in st.session_state or not st.session_state.local_playlists:
        st.info("No playlists yet. Upload songs to create them!")
        return

    moods = list(st.session_state.local_playlists.keys())
    selected_mood = st.selectbox("Select Mood to View:", moods, help="Choose a mood to see its local songs and Spotify playlist.")

    if selected_mood:
        expander_local = st.expander(f"📱 Local Songs - {selected_mood.capitalize()}", expanded=True)
        with expander_local:
            songs = st.session_state.local_playlists[selected_mood]
            if songs:
                for i, song in enumerate(songs, 1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i}. **{song['name']}**")
                    with col2:
                        if st.button(f"▶️", key=f"play_{i}"):
                            st.audio(song['path'])
                    if os.path.exists(song['path']):
                        st.audio(song['path'], autoplay=False)
            else:
                st.info("No local songs yet. Upload some!")

        # Spotify Playlist
        if st.session_state.get('spotify_authenticated', False):
            with st.spinner("Loading Spotify playlist..."):
                try:
                    from src.spotify_integration import create_or_get_playlist, get_playlist_tracks
                    playlist_id = create_or_get_playlist(selected_mood)
                    if playlist_id:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.success("✅ Spotify Playlist Ready")
                        with col2:
                            st.markdown(f"### 🎼 [Open in Spotify](https://open.spotify.com/playlist/{playlist_id})")
                        embed_url = f"https://open.spotify.com/embed/playlist/{playlist_id}"
                        st.components.v1.iframe(embed_url, height=400, width=600)
                    else:
                        st.warning("Could not access Spotify playlist. Try adding tracks.")
                except Exception as e:
                    st.error(f"Error loading Spotify: {e}")
        else:
            st.warning("Authenticate Spotify in the Spotify tab to view playlists.")

# ------------------------------------------------------------
# 🧠 Spotify Integration (Enhanced with Library Analysis)
# ------------------------------------------------------------
def spotify_integration():
    st.header('Spotify Integration: Playlist Generator & Library Analysis')

    # 🚨 Validate credentials
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
        st.error(
            "⚠️ Missing Spotify credentials in `.env` file.\n\n"
            "Please add these lines to your `.env` file:\n\n"
            "SPOTIFY_CLIENT_ID=your_client_id_here\n"
            "SPOTIFY_CLIENT_SECRET=your_client_secret_here\n"
            "SPOTIFY_REDIRECT_URI=http://localhost:8502/callback\n\n"
            "Note: Make sure to run the app on port 8502 using: `streamlit run app.py --server.port 8502`\n\n"
            "**Setup Instructions:**\n"
            "1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).\n"
            "2. Create a new app.\n"
            "3. Set the Redirect URI to: `http://127.0.0.1:8502/callback`\n"
            "4. Copy the Client ID and Client Secret to your `.env` file."
        )
        return

    if 'spotify_mood_playlists' not in st.session_state:
        st.session_state.spotify_mood_playlists = {m: [] for m in default_moods}

    # Refresh Spotify token to ensure it's valid
    refresh_spotify_token()

    sp = spotify_authenticate()

    if sp:
        # Section 1: Generate Mood-Based Playlists from Search
        st.subheader("Generate Mood Playlists from Spotify Search")
        cols = st.columns(2)
        col_idx = 0
        for mood in default_moods:
            with cols[col_idx % 2]:
                if st.button(f"Create {mood.capitalize()} Playlist from Search"):
                    try:
                        user = sp.current_user()
                        playlist = sp.user_playlist_create(
                            user['id'],
                            f"{mood.capitalize()} Mood Playlist",
                            public=True,
                            description=f"Auto-generated playlist for {mood} mood"
                        )
                        results = sp.search(q=mood, type='track', limit=10)
                        track_uris = [track['uri'] for track in results['tracks']['items']]
                        sp.playlist_add_items(playlist['id'], track_uris)
                        st.session_state.spotify_mood_playlists[mood] = [{'name': track['name'], 'artist': track['artists'][0]['name'], 'uri': track['uri']} for track in results['tracks']['items']]
                        st.success(f"Created '{mood.capitalize()} Mood Playlist' with {len(track_uris)} tracks! Playlist ID: {playlist['id']}")
                    except Exception as e:
                        st.error(f"Error creating playlist: {e}")
            col_idx += 1

        # Display Generated Spotify Tracks
        if st.session_state.spotify_mood_playlists:
            for mood, tracks in st.session_state.spotify_mood_playlists.items():
                if tracks:
                    st.write(f"**{mood.capitalize()} Mood Tracks** ({len(tracks)} tracks)")
                    for track in tracks[:5]:
                        st.write(f"- {track['name']} by {track['artist']}")
                        st.markdown(f"[Play on Spotify]({track['uri']})")

                    if st.button(f"Create Spotify Playlist for {mood.capitalize()} from Generated Tracks"):
                        try:
                            user = sp.current_user()
                            if tracks:
                                playlist = sp.user_playlist_create(
                                    user['id'],
                                    f"Generated {mood.capitalize()} Mood",
                                    public=True,
                                    description=f"{mood} tracks from search"
                                )
                                track_uris = [t['uri'] for t in tracks]
                                sp.playlist_add_items(playlist['id'], track_uris)
                                st.success(f"Created generated '{mood.capitalize()} Mood' playlist with {len(track_uris)} tracks!")
                        except Exception as e:
                            st.error(f"Error creating playlist: {e}")



def visualizations_interface():
    st.header('📊 Mood Distribution Visualizations')

    if 'local_playlists' not in st.session_state or not st.session_state.local_playlists:
        st.info("No playlists yet. Upload songs to create them!")
        return

    moods = list(st.session_state.local_playlists.keys())
    counts = [len(st.session_state.local_playlists[mood]) for mood in moods]

    # Select chart type
    chart_type = st.selectbox("Select Chart Type:", ["Bar Chart", "Pie Chart"], help="Choose how to visualize the mood distribution.")

    fig, ax = plt.subplots()
    if chart_type == "Bar Chart":
        bars = ax.bar(moods, counts, color=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'])
        ax.set_xlabel('Mood')
        ax.set_ylabel('Number of Songs')
        ax.set_title('Mood Distribution')
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count), ha='center', va='bottom')
    elif chart_type == "Pie Chart":
        ax.pie(counts, labels=moods, autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'], startangle=90)
        ax.set_title('Mood Distribution')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

# ------------------------------------------------------------
# 🚀 Main Function
# ------------------------------------------------------------
def main():

    # Advanced Dark Theme CSS with Google Fonts, Gradients, Shadows

    # Advanced Minimal Dark Theme CSS with Google Fonts, Gradients, Shadows
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: normal;
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        font-weight: 500;
        color: #94a3b8;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        border: none;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    .stFileUploader > div > div > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        backdrop-filter: blur(10px);
    }
    .stAudio {
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    h1 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-size: 2em;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    div.stSuccess > div > div {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #10b981;
        color: #d1fae5;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    div.stError > div > div {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-radius: 12px;
        padding: 16px; 
        border-left: 4px solid #ef4444;
        color: #fecaca;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    div.stWarning > div > div {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #f59e0b;
        color: #fed7aa;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    div.stInfo > div > div {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        color: #dbeafe;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    p, div, span {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .stExpander > div > div > div > div {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stColumns > div {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 16px;
        margin: 4px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

    # Welcome Section
    st.title("🎵 Music Mood Classifier & Playlist Generator")

    # Load playlists from file
    if 'local_playlists' not in st.session_state:
        st.session_state.local_playlists = load_playlists()

    # Load model, scaler, label_encoder
    if 'model' not in st.session_state:
        try:
            st.session_state.model = joblib.load('best_mood_classifier.pkl')
            st.session_state.scaler = joblib.load('scaler.pkl')
            st.session_state.label_encoder = joblib.load('label_encoder.pkl')
        except Exception as e:
            st.error(f"Error loading models: {e}")

    # Initialize uploaded files and predictions
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}

    # Handle Spotify OAuth callback
    if 'code' in st.query_params:
        code = st.query_params['code']
        sp_oauth = get_spotify_oauth()
        try:
            token_info = sp_oauth.get_access_token(code, as_dict=True)
            st.session_state.spotify_token = token_info
            st.session_state.spotify_authenticated = True
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error during authentication: {e}")
            st.session_state.spotify_token = None
            st.session_state.spotify_authenticated = False
            st.query_params.clear()
            st.rerun()

    if 'spotify_mood_playlists' not in st.session_state:
        st.session_state.spotify_mood_playlists = {m: [] for m in default_moods}

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Local", "🎶 View Local Playlists", "🎧 Spotify (Link & Generate)", "📊 Visualization"])
    with tab1:
        upload_predict()
    with tab2:
        view_playlists()
    with tab3:
        spotify_integration()
    with tab4:
        visualizations_interface()

if __name__ == '__main__':
    main()
