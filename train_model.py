# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("🎵 Loading Spotify Dataset...")
try:
    # LIMIT TO 10,000 ROWS for Free Tier Server Stability
    df = pd.read_csv('spotify_tracks.csv', nrows=10000)
    
    df.columns = df.columns.str.lower()
    required_features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']
    
    if not all(col in df.columns for col in required_features):
        print("❌ Error: Missing columns")
        exit()

    df = df.dropna(subset=required_features)
    X = df[required_features].values
    print(f"✅ Data Loaded! Training on {len(X)} songs.")

except FileNotFoundError:
    print("❌ Error: 'spotify_tracks.csv' not found.")
    exit()

# Engineer Labels (Russell's Model)
y = []
for row in X:
    valence, energy = row[2], row[1]
    if valence < 0.5 and energy < 0.5: y.append(0) # Sad
    elif valence >= 0.5 and energy < 0.5: y.append(1) # Calm
    elif valence < 0.5 and energy >= 0.5: y.append(2) # Angry
    else: y.append(3) # Happy
y = np.array(y)

# Training
print("🧠 Training The Ensemble Brain...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf1 = RandomForestClassifier(n_estimators=50, random_state=42) # Reduced estimators for speed
clf2 = GradientBoostingClassifier(random_state=42)
clf3 = SVC(probability=True, kernel='rbf')
clf4 = KNeighborsClassifier(n_neighbors=7)
clf5 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', clf1), ('gb', clf2), ('svm', clf3), ('knn', clf4), ('nn', clf5)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🚀 Model Accuracy: {acc*100:.2f}%")

with open('emotisound_brain.pkl', 'wb') as f:
    pickle.dump((ensemble, scaler), f)
print("💾 Brain saved successfully as 'emotisound_brain.pkl'")