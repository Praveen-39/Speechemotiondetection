import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import librosa
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import speech_recognition as sr
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit config
st.set_page_config(page_title="Speech Emotion & Sarcasm Detector", page_icon="🎭", layout="wide")

@st.cache_resource
def load_model_files():
    try:
        model = keras.models.load_model("models/emotion_recognition_model.h5")
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, label_encoder, scaler
    except Exception as e:
        st.error(f"Model file error: {e}")
        return None, None, None

def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=22050, duration=3)
        target_length = 22050 * 3
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1).tolist())
        features.extend(np.std(mfcc, axis=1).tolist())
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1).tolist())
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        features.extend(np.mean(mel, axis=1).tolist())
        features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
        features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
        features.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
        features.append(float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))))
        features.append(float(np.mean(librosa.feature.spectral_flatness(y=y))))
        features.append(float(np.mean(librosa.feature.zero_crossing_rate(y))))
        features.append(float(np.mean(librosa.feature.rms(y=y))))
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        features.append(float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0)
        features.append(float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.append(float(np.mean(y_harmonic)))
        features.append(float(np.mean(y_percussive)))
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1).tolist())
        features = np.array(features, dtype=np.float32)
        if len(features) < 73:
            features = np.pad(features, (0, 73 - len(features)), mode='constant')
        else:
            features = features[:73]
        return features
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return np.zeros(73, dtype=np.float32)

def predict_emotion(audio_file, model, label_encoder, scaler):
    features = extract_audio_features(audio_file)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_reshaped = features_scaled.reshape(1, 73, 1)
    preds = model.predict(features_reshaped, verbose=0)
    emotion_idx = np.argmax(preds[0])
    emotion = label_encoder.classes_[emotion_idx]
    confidence = preds[0][emotion_idx]
    return emotion, confidence, preds[0]

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except:
        return None

def analyze_sentiment(text):
    if not text:
        return 'neutral', 0.0, 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
    return sentiment, polarity, subjectivity

def detect_sarcasm(emotion, sentiment, confidence):
    positive_emotions = ['happy', 'surprised', 'calm']
    negative_emotions = ['angry', 'sad', 'fearful', 'disgust']
    scores = 0.0
    indicators = []
    if emotion in negative_emotions and sentiment == 'positive':
        scores += 0.6
        indicators.append("Negative emotion with positive words")
    if emotion in positive_emotions and sentiment == 'negative':
        scores += 0.5
        indicators.append("Positive emotion with negative words")
    if confidence > 0.8 and scores > 0:
        scores += 0.2
        indicators.append("High emotion confidence")
    return scores >= 0.5, scores, indicators

# --- Streamlit UI ---
st.markdown("## 🎭 Speech Emotion & Sarcasm Detector")
st.write("Upload a WAV audio file, and the system will predict the emotion, transcribe speech to text, analyze sentiment, and detect sarcasm.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    st.audio(uploaded_file, format="audio/wav")
    st.info("Analyzing audio ...")
    # Load models
    model, label_encoder, scaler = load_model_files()
    if model is not None:
        emotion, confidence, all_predictions = predict_emotion(tmp_path, model, label_encoder, scaler)
        text = speech_to_text(tmp_path)
        sentiment, polarity, subjectivity = analyze_sentiment(text)
        sarcasm_detected, sarcasm_score, indicators = detect_sarcasm(emotion, sentiment, confidence)
        st.success(f"Emotion: **{emotion.upper()}** ({confidence*100:.1f}%)")
        st.write(f"Transcribed Text: `{text if text else 'Could not transcribe'}`")
        st.write(f"Sentiment: **{sentiment.upper()}** (Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f})")
        if sarcasm_detected:
            st.error(f"⚠️ SARCASM DETECTED! Score: {sarcasm_score:.2f}")
            st.write(f"Indicators: {', '.join(indicators)}")
        else:
            st.success(f"No sarcasm detected. Score: {sarcasm_score:.2f}")
        # Plot emotion probabilities
        prob_df = pd.DataFrame({"Emotion": label_encoder.classes_, "Probability": all_predictions*100})
        fig = px.bar(prob_df, x="Emotion", y="Probability", title="Emotion Prediction Probabilities", color="Probability")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Model files not loaded. Make sure all *.h5 and *.pkl files are present in /models/ folder on GitHub and Streamlit Cloud.")

    os.unlink(tmp_path)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>Created for BSc Final Project | Speech Emotion & Sarcasm Detection | 2025</div>",
    unsafe_allow_html=True
)
