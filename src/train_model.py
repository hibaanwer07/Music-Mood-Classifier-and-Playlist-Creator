import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import logging
# import random  # Removed as not used

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load data from CSV files
import os

def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df3_path = os.path.join(base_dir, "..", "Data", "features_3_sec.csv")
        df4_path = os.path.join(base_dir, "..", "Data", "features_30_sec.csv")
        df3 = pd.read_csv(df3_path)
        # df4 = pd.read_csv(df4_path)  # Not used
        # Combine the datasets if needed, or use one (e.g., df3 for 3-second features)
        data = df3.copy()  # Using df3 as primary dataset
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Prepare data: handle missing values, encode labels, scale features
def prepare_data(data):
    if data.empty:
        logger.error("Empty data received in prepare_data. Aborting preparation.")
        return None, None, None, None

    # Drop rows with missing values
    data = data.dropna()

    # Map genres to moods
    genre_to_mood = {
        'blues': 'sad',
        'classical': 'calm',
        'country': 'sad',
        'disco': 'happy',
        'hiphop': 'energetic',
        'jazz': 'calm',
        'metal': 'energetic',
        'pop': 'happy',
        'reggae': 'happy',
        'rock': 'energetic'
    }
    data.iloc[:, -1] = data.iloc[:, -1].map(genre_to_mood)
    # Remove rows where mapping failed (genre not in genre_to_mood)
    data = data.dropna(subset=[data.columns[-1]])

    # Select only numerical columns for features (exclude strings like filenames)
    X = data.select_dtypes(include=[np.number])
    y = data.iloc[:, -1]  # Now the last column is mood

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Data prepared. Features shape: {X_scaled.shape}, Labels shape: {y_encoded.shape}")
    return X_scaled, y_encoded, label_encoder, scaler

# Handle class imbalance using class weights
def compute_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    return class_weight_dict

# Train models with class weights
def train_models(X_train, X_test, y_train, y_test, class_weight_dict):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
        'SVM': SVC(kernel='rbf', random_state=42, class_weight=class_weight_dict),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        logger.info(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return results

# Save the best model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filename}")
def generate_playlist(data, mood_genre, num_tracks=10):
    # Map genres to moods (example mapping, adjust as needed)
    mood_map = {
        'happy': ['pop', 'disco', 'reggae'],
        'sad': ['blues', 'country'],
        'energetic': ['rock', 'metal', 'hiphop'],
        'calm': ['jazz', 'classical']
    }
    
    # Get genres for the specified mood
    target_genres = mood_map.get(mood_genre.lower(), [])
    if not target_genres:
        logger.warning(f"No genres found for mood: {mood_genre}")
        return []
    
    # Filter data for target genres
    filtered_data = data[data.iloc[:, -1].isin(target_genres)]
    if filtered_data.empty:
        logger.warning(f"No tracks found for mood: {mood_genre}")
        return []
    
    # Randomly select tracks
    playlist = filtered_data.sample(n=min(num_tracks, len(filtered_data)), random_state=42)
    logger.info(f"Generated playlist for mood '{mood_genre}' with {len(playlist)} tracks")
    return playlist
    logger.info(f"Generated playlist for mood '{mood_genre}' with {len(playlist)} tracks")
    return playlist

# Main function
def main():
    # Load and prepare data
    data = load_data()
    X, y, label_encoder, scaler = prepare_data(data)
    if X is None or y is None:
        logger.error("Data preparation failed. Exiting training.")
        return
    
    # Stratified train-test split to handle imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights
    class_weight_dict = compute_class_weights(y_train)
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test, class_weight_dict)
    
    # Save the best model (force RandomForest for better generalization)
    best_model_name = 'RandomForest'
    best_model = results[best_model_name]['model']
    save_model(best_model, 'best_mood_classifier.pkl')
    
    # Save scaler and label encoder
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info("Scaler and label encoder saved successfully.")

if __name__ == "__main__":
    main()

