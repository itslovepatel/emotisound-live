# app.py
import os
import sys

# --- CONFIGURATION: Add FFmpeg to System Path ---
# This allows the server to find the audio tools downloaded by build.sh
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_path

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Try importing your audio processor
try:
    from audio_processor import extract_features
except ImportError:
    extract_features = None

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model
# The model is created by build.sh running train_model.py before this app starts
print("⏳ Loading AI Brain...")
try:
    with open('emotisound_brain.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    print("🧠 AI Brain Loaded!")
except FileNotFoundError:
    print("❌ ERROR: 'emotisound_brain.pkl' missing. Did build.sh run train_model.py?")
    model = None

EMOTIONS = {0: "Melancholy", 1: "Serenity", 2: "Turbulence", 3: "Euphoria"}

def get_personality_match(features):
    dance, energy, valence, acoustic, tempo, loudness = features
    traits = []
    if acoustic > 0.6: traits.append("High Openness")
    if energy > 0.7: traits.append("High Extraversion")
    if valence > 0.7: traits.append("High Agreeableness")
    if valence < 0.3 and energy > 0.6: traits.append("High Neuroticism")
    return traits if traits else ["Balanced Personality"]

def get_synesthesia_color(valence, energy):
    hue = int((1 - energy) * 240)
    saturation = int(valence * 100)
    return f"hsl({hue}, {saturation}%, 50%)"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    # Save file temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # 1. Extract Features
        if extract_features is None:
            return jsonify({"error": "Audio Processor not loaded"})
            
        features_dict = extract_features(filepath)
        if not features_dict:
            return jsonify({"error": "Error reading audio file (Check FFmpeg/WAV)"})

        # 2. Prepare for Model
        if model is None:
            return jsonify({"error": "Model not loaded"})

        feature_vector = np.array([
            features_dict['danceability'], features_dict['energy'], features_dict['valence'],
            features_dict['acousticness'], features_dict['tempo'], features_dict['loudness']
        ]).reshape(1, -1)
        
        scaled_features = scaler.transform(feature_vector)
        
        # 3. Predict
        probs = model.predict_proba(scaled_features)[0]
        prediction = np.argmax(probs)
        
        # 4. Result
        result = {
            "main_emotion": EMOTIONS[prediction],
            "dna_sequence": [float(round(p, 2)) for p in probs.tolist()],
            "synesthesia_color": get_synesthesia_color(features_dict['valence'], features_dict['energy']),
            "personality_match": get_personality_match(feature_vector[0]),
            "extracted_features": features_dict
        }
        
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass 
            
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Crash Error: {str(e)}")
        return jsonify({"error": f"Server Crash: {str(e)}"})

if __name__ == '__main__':
    # Production Port Configuration
    print("🚀 STARTING SERVER...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)