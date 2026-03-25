import pandas as pd
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import joblib
import parselmouth
from parselmouth.praat import call
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os
import sys

# --- CONFIGURATION ---
DATA_FILE = 'pd_voice_data.csv'      # Must be in the same folder
MODEL_FILE = 'best_gb_parkinsons_model.pkl'
SCALER_FILE = 'data_scaler.pkl'
LIVE_WAV_FILE = 'live_voice_sample.wav'
RECORD_DURATION = 3                  # seconds
SAMPLE_RATE = 44100


# =========================================================================
# 1. FEATURE EXTRACTION FUNCTION
# =========================================================================

def extract_acoustic_features(audio_file_path):
    """Extracts the 22 features required by the trained model (UCI format)."""
    try:
        sound = parselmouth.Sound(audio_file_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        
        # Features must match the 22 columns of the UCI dataset (excluding 'name' and 'status')
        features = {
            'MDVP:Fo(Hz)': call(pitch, "Get mean", 0, 0, "Hertz"),
            'MDVP:Fhi(Hz)': call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Flo(Hz)': call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Jitter(%)': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Jitter(Abs)': call(point_process, "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3) * 1000,
            'MDVP:RAP': call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:PPQ': call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'Jitter:DDP': call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Shimmer': call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:Shimmer(dB)': call([sound, point_process], "Get shimmer (dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ3': call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ5': call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:APQ': call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:DDA': call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'NHR': call(sound, "Get noise to harmonics ratio", 0, 0, 0.0001, 0.02, 1.3),
            'HNR': call(sound, "Get harmonicity (cc)", 0.01, 75, 0.1, 1.0),
            # Complex features usually set to 0.0 or mean if not extractable from simple Praat script:
            'RPDE': 0.0, 'DFA': 0.0, 'spread1': 0.0, 'spread2': 0.0, 'D2': 0.0, 'PPE': 0.0,
        }
        
        # Reshape to a 1x22 array for the model
        return np.array([value for value in features.values()]).reshape(1, -1)

    except Exception as e:
        print(f"❌ CRITICAL ERROR during feature extraction: {e}")
        print("💡 HINT: The input audio may be silence or unusable noise. Try speaking louder or checking your mic.")
        sys.exit(1)


# =========================================================================
# 2. TRAINING AND SAVING FUNCTION
# =========================================================================

def train_and_tune_model():
    """Loads data, trains Gradient Boosting with Grid Search, saves model/scaler."""
    print("1. Loading and preparing data...")
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"FATAL: Training data file '{DATA_FILE}' not found. Please provide a CSV with voice features.")
        sys.exit(1)
    except PermissionError as e:
        print(f"FATAL: Permission denied while reading '{DATA_FILE}'. Is the file open in Excel/Notepad? Restart your PC if error persists.")
        sys.exit(1)
        
    data.drop('name', axis=1, inplace=True) 
    X = data.drop('status', axis=1) # Features
    y = data['status']              # Target

    # Split, Scale, and Resample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    print("2. Starting Hyperparameter Tuning (This may take a minute)...")
    gb_clf_base = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=gb_clf_base, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_res, y_train_res)
    best_gb_clf = grid_search.best_estimator_
    
    # Evaluation
    y_pred_gb_tuned = best_gb_clf.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, y_pred_gb_tuned)
    
    print("\n--- Training Complete ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")

    # Save the model and scaler
    joblib.dump(best_gb_clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"✅ SUCCESS! Final Model saved to {MODEL_FILE}")
    print(f"✅ Scaler saved to {SCALER_FILE}")
    
    return best_gb_clf, scaler


# =========================================================================
# 3. LIVE PREDICTION FUNCTION
# =========================================================================

def predict_live(model, scaler):
    """Records voice, extracts features, scales, and makes a prediction."""
    print("\n--- Starting Live Voice Recording ---")
    print(f"--- Recording for {RECORD_DURATION} seconds ---")
    print("📢 Please say a sustained vowel sound (e.g., 'Aaaah') now...")
    
    try:
        # Record Audio
        recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
        sd.wait() 
        write(LIVE_WAV_FILE, SAMPLE_RATE, recording)
        print(f"✅ Recording saved to {LIVE_WAV_FILE}")
        
    except Exception as e:
        print(f"❌ Recording Error: {e}")
        print("💡 HINT: Check your microphone connection and system audio settings.")
        return # Exit prediction function gracefully

    # Extract, Scale, and Predict
    print("🔬 Analyzing voice features...")
    new_patient_features = extract_acoustic_features(LIVE_WAV_FILE) 
    
    new_patient_features_scaled = scaler.transform(new_patient_features)
    predicted_status = model.predict(new_patient_features_scaled)[0]
    probability = model.predict_proba(new_patient_features_scaled)[0][predicted_status]
    
    # Display Results
    status_text = "Parkinson's Disease" if predicted_status == 1 else "Healthy"
    
    print("\n==============================================")
    print("📊 LIVE PARKINSON'S PREDICTION RESULT")
    print("==============================================")
    print(f"Prediction: {status_text}")
    print(f"Confidence: {probability*100:.2f}%")
    print("==============================================")


# =========================================================================
# 4. MAIN EXECUTION FLOW
# =========================================================================

if __name__ == '__main__':
    
    # Check if model files exist
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("Model files not found. Starting training and tuning...")
        # Run the full training, tuning, and saving process
        model, scaler = train_and_tune_model()
    else:
        print("✅ Loading existing Optimized Model and Scaler...")
        try:
            # Load saved model and scaler
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
        except Exception as e:
            print(f"FATAL: Could not load model files. Error: {e}")
            sys.exit(1)
    
    # After the model is trained OR loaded, proceed to live prediction
    predict_live(model, scaler)