import streamlit as st
import whisper
import librosa
import numpy as np
import re
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model=load_model()

@st.cache_data
def get_models():
    df=pd.read_csv("features.csv")
    X=df[["speech_rate","num_pauses","pitch_variation","num_hesitations"]]
    scaler=StandardScaler().fit(X)
    iso=IsolationForest(contamination=0.2,random_state=42).fit(scaler.transform(X))
    return scaler,iso

scaler,iso=get_models()

def count_hesitations(text):
    return len(re.findall(r"\b(um+|uh+|uhm+|erm+|hmm+)\b",text.lower()))

def assess_cognitive_risk(audio_file_path):
    result=model.transcribe(audio_file_path)
    text=result["text"]

    y,sr=librosa.load(audio_file_path)
    duration=librosa.get_duration(y=y,sr=sr)
    pitches=librosa.yin(y,fmin=50,fmax=300)
    pitch_std=np.std(pitches)

    intervals=librosa.effects.split(y,top_db=25)
    pauses=0
    for i in range(1,len(intervals)):
        gap=(intervals[i][0]-intervals[i-1][1])/sr
        if gap>0.3: pauses+=1

    word_count=len(text.split())
    speech_rate=word_count/duration if duration>0 else 0
    hesitations=count_hesitations(text)

    features=np.array([[speech_rate,pauses,pitch_std,hesitations]])
    features_scaled=scaler.transform(features)
    risk=1 if iso.predict(features_scaled)[0]==-1 else 0

    return {
        "transcript":text,
        "speech_rate":round(speech_rate,2),
        "num_pauses":pauses,
        "num_hesitations":hesitations,
        "pitch_variation":round(pitch_std,2),
        "risk":risk
    }

st.title("MemoTag - Cognitive risk screener")
uploaded_file=st.file_uploader("Upload a .wav audio file",type=["wav"])

if uploaded_file is not None:
    with open("temp.wav","wb") as f:
        f.write(uploaded_file.read())

    result=assess_cognitive_risk("temp.wav")

    st.subheader("Transcript")
    st.text(result["transcript"])

    st.subheader("Extracted Features")
    st.write(f"Speech Rate: {result['speech_rate']} WPS")
    st.write(f"Pauses: {result['num_pauses']}")
    st.write(f"Hesitations: {result['num_hesitations']}")
    st.write(f"Pitch Variation: {result['pitch_variation']}")

    st.subheader("Prediction")
    if result["risk"]==1:
        st.write("Risk Detected")
    else:
        st.write("No Risk Detected")
# TO RUN THIS USE- streamlit run deploy.py
