import pickle
import joblib
import tensorflow as tf
from pathlib import Path

BASE_PATH = Path(__file__).parent / "models"

# 1. Risk Stratification
with open(f"{BASE_PATH}/risk_stratification/risk_stratification_model.pkl", "rb") as f:    risk_model = pickle.load(f)

# 2. Length of Stay Prediction
with open(f"{BASE_PATH}/length_of_stay/length_of_stay_prediction_model.pkl", "rb") as f:    los_model = pickle.load(f)

# 3. Patient Segmentation
with open(f"{BASE_PATH}/patient_segmentation/segmentation-kmeans_model.pkl", "rb") as f:    kmeans_model = pickle.load(f)
with open(f"{BASE_PATH}/patient_segmentation/segment_scaler.pkl", "rb") as f:    scaler = pickle.load(f)

# 4. Imaging Diagnosis
image_model = tf.keras.models.load_model(f"{BASE_PATH}/image_diagnosis/lung_cancer_cnn.keras")

# 5. Sequence Modeling
sequence_model = tf.keras.models.load_model(f"{BASE_PATH}/sequence_model/sequence_model.h5")
sequence_scaler = joblib.load(f"{BASE_PATH}/sequence_model/sequence_scaler.pkl")

# 6. Sentiment Analysis
sentiment_model = tf.keras.models.load_model(    f"{BASE_PATH}/sentiment_analysis/cnn_sentiment_model.keras")
with open(f"{BASE_PATH}/sentiment_analysis/cnn_tokenizer.pkl", "rb") as f:    sentiment_tokenizer = pickle.load(f)
