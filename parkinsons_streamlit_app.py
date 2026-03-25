# parkinsons_streamlit_app_safe.py
import os
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pandas as pd
import joblib
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -------------------------
# Config
# -------------------------
DATA_FILE = 'pd_voice_data.csv'
MODEL_FILE = 'best_gb_parkinsons_model.pkl'
SCALER_FILE = 'data_scaler.pkl'
LIVE_WAV_FILE = 'live_voice_sample.wav'
SAMPLE_RATE = 44100
RECORD_DURATION = 5  # seconds

# -------------------------
# Safe feature extraction
# -------------------------
def extract_acoustic_features(audio_file_path):
    try:
        sound = parselmouth.Sound(audio_file_path)
        pitch = call(sound, "To Pitch", 0.0, 50, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", 50, 600)

        def safe_call(obj, command, *args):
            try:
                return call(obj, command, *args)
            except:
                return 0.0

        features = {
            'MDVP:Fo(Hz)': safe_call(pitch, "Get mean", 0, 0, "Hertz"),
            'MDVP:Fhi(Hz)': safe_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Flo(Hz)': safe_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Jitter(%)': safe_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Jitter(Abs)': safe_call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) * 1000,
            'MDVP:RAP': safe_call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:PPQ': safe_call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'Jitter:DDP': safe_call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Shimmer': safe_call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:Shimmer(dB)': safe_call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ3': safe_call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ5': safe_call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:APQ': safe_call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:DDA': safe_call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'NHR': safe_call(sound, "Get noise to harmonics ratio", 0, 0),
            'HNR': safe_call(sound, "Get harmonicity (cc)", 0.01, 50, 0.1, 1.0),
            'RPDE': 0.0, 'DFA': 0.0, 'spread1': 0.0,
            'spread2': 0.0, 'D2': 0.0, 'PPE': 0.0
        }
        return np.array([list(features.values())], dtype=float).reshape(1, -1)
    except Exception as e:
        st.error(f"Failed to extract features: {e}")
        return np.zeros((1, 22))

# -------------------------
# Train + save model
# -------------------------
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found!")
        st.stop()

    data = pd.read_csv(DATA_FILE)
    if 'name' in data.columns:
        data = data.drop(columns=['name'])
    if 'status' not in data.columns:
        st.error("Dataset must have 'status' column")
        st.stop()

    X = data.drop(columns=['status'])
    y = data['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    gb = GradientBoostingClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}
    grid = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    with st.spinner("Training & hyperparameter tuning..."):
        grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Added metrics section
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    st.success(f" Model trained & saved! Test Accuracy: {acc*100:.2f}%")
    st.write("Best parameters:", grid.best_params_)

    # Show Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Show Precision, Recall, F1-score
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).T)

    return best_model, scaler

# -------------------------
# Predict
# -------------------------
def predict_from_audio(model, scaler, audio_path):
    features = extract_acoustic_features(audio_path)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]
    return int(pred), float(prob)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Parkinson's Voice Analyzer", page_icon="🎙", layout="centered")
st.title("🎙 Parkinson's Voice Analyzer")
st.write("Record a sustained vowel (e.g., 'Aaaah') and analyze voice features.")

# Load model
model, scaler = None, None
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        st.success("Loaded existing model and scaler")
    except:
        model, scaler = None, None

col1, col2 = st.columns(2)
with col1:
    if st.button("Retrain model"):
        model, scaler = train_and_save_model()

with col2:
    if st.button(" Record voice"):
        st.info(f"Recording for {RECORD_DURATION} sec... Speak 'Aaaah'")
        try:
            recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
            sd.wait()
            write(LIVE_WAV_FILE, SAMPLE_RATE, recording)
            st.success(f"Recording saved to {LIVE_WAV_FILE}")
        except Exception as e:
            st.error(f"Recording failed: {e}")

if os.path.exists(LIVE_WAV_FILE):
    if st.button("Analyze recording"):
        if model is None or scaler is None:
            st.warning("Model not loaded. Retrain first.")
        else:
            with st.spinner("Extracting features and predicting..."):
                pred, prob = predict_from_audio(model, scaler, LIVE_WAV_FILE)
            if pred == 1:
                st.error(f"Parkinson's Detected — Confidence: {prob*100:.2f}%")
            else:
                st.success(f" Healthy Voice — Confidence: {prob*100:.2f}%")
            if st.button("Save prediction log"):
                with open("prediction_log.txt", "a") as f:
                    f.write(f"Prediction: {'Parkinson' if pred==1 else 'Healthy'} | Confidence: {prob*100:.2f}%\n")
                st.success("Saved to prediction_log.txt")

st.markdown("---")
st.write("### Tips:")
st.write("- Record a loud, steady 'Aaaah' for 4–5 seconds")
st.write("- Reduce background noise")
st.write("- If feature extraction fails, try again")
