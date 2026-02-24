import joblib
import pandas as pd
from src.feature_extraction import extract_features

# Load models
model = joblib.load('best_mood_classifier.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Test with sample file
sample_file = '../temp_Aavanipponnoonjaal - Version, 01.mp3'
features = extract_features(sample_file)
if features:
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)[0]
    predicted_mood = label_encoder.inverse_transform([prediction])[0]
    print(f"Predicted Mood: {predicted_mood}")
else:
    print("Failed to extract features")
