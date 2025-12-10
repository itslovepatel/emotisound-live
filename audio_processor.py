# audio_processor.py
import librosa
import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        danceability = np.mean(onset_env) 
        danceability = min(1.0, danceability / 10.0)
        
        energy = np.mean(rms) * 5 
        energy = min(1.0, max(0.0, energy))
        
        valence = np.mean(cent) 
        valence = min(1.0, (valence - 500) / 4000)
        valence = max(0.0, valence)
        
        acousticness = 1.0 - (energy + (np.mean(zcr)*2)) / 2
        acousticness = max(0.0, acousticness)
        
        loudness = librosa.amplitude_to_db(rms).mean()
        
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        
        return {
            "danceability": round(float(danceability), 2),
            "energy":       round(float(energy), 2),
            "valence":      round(float(valence), 2),
            "acousticness": round(float(acousticness), 2),
            "tempo":        round(float(tempo), 0),
            "loudness":     round(float(loudness), 1)
        }
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None