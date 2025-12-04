
#integration

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import redis
import time
import mutagen
from pydub import AudioSegment
import os
import json

# Redis cache
r = redis.Redis(host='localhost', port=6379)

sp_oauth = SpotifyOAuth(client_id="4f39f7b71e3f4d978b2103dada7415ca", client_secret="020400323f4143e7bae06183868d8750", redirect_uri="http://127.0.0.1:8000/callback", scope="playlist-modify-public user-library-read")

def get_token():
    token = sp_oauth.get_access_token()
    return token['access_token']

sp = spotipy.Spotify(auth_manager=sp_oauth)

def fetch_features(track_id):
    cache_key = f"features:{track_id}"
    if r.exists(cache_key):
        return json.loads(r.get(cache_key))
    try:
        features = sp.audio_features(track_id)[0]  # valence, energy, etc.
        r.set(cache_key, json.dumps(features), ex=3600)  # Cache 1hr
        return features
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:  # Rate limit
            time.sleep(int(e.headers['Retry-After']) + 1)
            return fetch_features(track_id)  # Retry
        raise

def create_playlist(user_id, name, tracks):
    playlist = sp.user_playlist_create(user_id, name)
    sp.playlist_add_items(playlist['id'], tracks)
    return playlist['external_urls']['spotify']

# Local library
def scan_local(directory):
    tracks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                path = os.path.join(root, file)
                audio = AudioSegment.from_file(path)
                tags = mutagen.File(path, easy=True)
                tracks.append({"path": path, "metadata": tags})
    return tracks

# Unit tests (Pytest)
def test_fetch_features():
    assert 'valence' in fetch_features("spotify:track:4uLU6hMCjMI75M1A2tKUQC")