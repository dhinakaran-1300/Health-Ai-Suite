import numpy as np
from loaders import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. Risk Stratification
def predict_risk(features):
    return int(
        risk_model.predict([features])[0]
    )

# 2. Length of Stay Prediction
def predict_los(features):
    return float(
        los_model.predict([features])[0]
    )

# 3. Patient Segmentation
def predict_segment(features):
    features_scaled = scaler.transform([features])
    return int(
        kmeans_model.predict(features_scaled)[0]
    )

# 4. Imaging Diagnosis
def predict_image(img_array):
    img = img_array / 255.0
    img = np.expand_dims(img, axis=0)

    prob = float(
        image_model.predict(img)[0][0]
    )

    if prob >= 0.5:
        return {
            "Result": "Cancer",
            "probability": round(prob, 3)
        }
    else:
        return {
            "Result": "Normal"
        }

# 5. Sequence Modeling
def predict_sequence(sequence):
    seq = np.array(sequence, dtype=np.float32)
    t, f = seq.shape

    seq_scaled = (
        sequence_scaler
        .transform(seq.reshape(-1, f))
        .reshape(1, t, f)
    )

    prediction = sequence_model.predict(
        seq_scaled,
        verbose=0
    )

    return float(prediction[0][0])

# 6. Sentiment Analysis
def predict_sentiment(text):
    seq = sentiment_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)

    prob = sentiment_model.predict(padded)[0][0]

    return {
        "probability": float(prob),
        "label": "Positive" if prob > 0.5 else "Negative"
    }
