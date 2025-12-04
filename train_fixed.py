import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTForAudioClassification, AutoFeatureExtractor, Trainer, TrainingArguments
import os
import optuna
from sklearn.metrics import f1_score, accuracy_score
import shap
import torchaudio
import librosa
import io
from fastapi import FastAPI, UploadFile
from audio_utils import load_audio_file, validate_audio_file
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Custom Dataset
# ---------------------------
class CustomAudioDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, transform=None):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(data_dir))
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):
                for file in os.listdir(class_folder):
                    if file.endswith(('.wav', '.mp3')):
                        self.audio_files.append(os.path.join(class_folder, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Use robust audio loading with fallback
        try:
            waveform, sample_rate = load_audio_file(audio_path, target_sr=16000)
            logger.info(f"Successfully loaded audio file: {audio_path}")
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            # Return a zero tensor as fallback (1 second of silence at 16kHz)
            waveform = np.zeros(16000)
            sample_rate = 16000

        waveform_np = waveform
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=1024
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

# ---------------------------
# Paths
# ---------------------------
print("Current working directory:", os.getcwd())

# Direct absolute path to dataset
data_dir_train = r"C:\Users\ASUS\Downloads\Music_Mood_Classifier_and_Playlist_Generator\Music_Mood_Classifier_and_Playlist_Generator\venv\archive (4)\Data\genres_original"

print("Train data dir exists:", os.path.exists(data_dir_train))
print("Train data dir is dir:", os.path.isdir(data_dir_train))

# ---------------------------
# Feature Extractor + Dataset
# ---------------------------
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
train_dataset = CustomAudioDataset(data_dir=data_dir_train, feature_extractor=feature_extractor)
test_dataset = CustomAudioDataset(data_dir=data_dir_train, feature_extractor=feature_extractor)

# ---------------------------
# Model
# ---------------------------
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(set(train_dataset.labels)),
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True
)

# ---------------------------
# Hyperparam tuning with Optuna
# ---------------------------
def objective(trial):
    args = TrainingArguments(
        output_dir="./results",
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {
            'f1': f1_score(p.label_ids, p.predictions.argmax(-1), average='macro'),
            'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))
        }
    )
    trainer.train()
    return trainer.evaluate()['eval_f1']

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)  # keep low for testing

# ---------------------------
# Optimization
# ---------------------------
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.onnx.export(model, torch.rand(1, 1024, 128), "model.onnx")

# ---------------------------
# Inference API
# ---------------------------
app = FastAPI()

@app.post("/infer")
async def infer(file: UploadFile):
    audio_bytes = await file.read()

    # Use robust audio loading for uploaded files too
    try:
        waveform, sample_rate = load_audio_file(io.BytesIO(audio_bytes), target_sr=16000)
        logger.info("Successfully loaded uploaded audio file")
    except Exception as e:
        logger.error(f"Failed to load uploaded audio: {e}")
        # Return error response
        return {"error": f"Failed to process audio file: {str(e)}"}

    waveform_np = waveform
    inputs = feature_extractor(
        waveform_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=1024
    )
    outputs = model(**inputs).logits
    preds = outputs.argmax(-1)
    return {"genre": preds.item()}
