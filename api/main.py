from fastapi import FastAPI, UploadFile, File
from schemas import *
from inference import *
import cv2
import numpy as np


# Application Initialization
app = FastAPI(title="HealthAI Inference API")


# 1. Risk Stratification
@app.post("/risk")
def risk(data: TabularInput):
    return {"risk_class": predict_risk(data.features)}


# 2. Length of Stay Prediction
@app.post("/los")
def los(data: TabularInput):
    return {"length_of_stay": predict_los(data.features)}


# 3. Patient Segmentation
@app.post("/segment")
def segment(data: TabularInput):
    return {"cluster": predict_segment(data.features)}


# 4. Imaging Diagnosis
@app.post("/image")
async def image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return predict_image(img)


# 5. Sequence Modeling
@app.post("/sequence")
def sequence(data: SequenceInput):
    return {"prediction": predict_sequence(data.sequence)}


# 6. Sentiment Analysis
@app.post("/sentiment")
def sentiment(data: TextInput):
    return predict_sentiment(data.text)
