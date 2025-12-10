
# 🎵 EmotiSound - Music Emotion DNA Analyzer

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/lovepatel/emotisound-live)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**EmotiSound** is an AI-powered audio analysis tool that "listens" to raw music files, extracts complex acoustic features, and predicts the emotional DNA of the song. Unlike basic mood classifiers, it generates a multi-dimensional emotional fingerprint and even maps music to synesthesia-inspired colors.

### 🔴 **Live Demo:** [Try EmotiSound Now on Hugging Face Spaces](https://huggingface.co/spaces/lovepatel/emotisound-live)

---

## 🚀 Features

- 🎧 **Advanced Audio Feature Extraction** using Librosa (Danceability, Energy, Valence, Tempo, Brightness, etc.)
- 🧠 **Ensemble AI Model** – Voting Classifier with 5 powerful algorithms (Random Forest, Gradient Boosting, SVM, KNN, MLP)
- 🧬 **Emotion DNA Radar Chart** – Visualizes the unique emotional profile of any song
- 🎨 **Synesthesia Color Engine** – Converts sound frequencies and energy into vivid colors
- ☁️ **Fully Containerized** – Runs seamlessly with Docker and Hugging Face Spaces

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, NumPy, Pandas
- **Audio Processing:** Librosa, SoundFile, pydub + FFmpeg
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Deployment:** Docker, Gunicorn, Hugging Face Spaces

---

## ⚙️ How It Works

1. User uploads an `.mp3` or `.wav` file
2. System extracts low-level and high-level acoustic features using **Librosa**
3. Features are normalized and passed to the pre-trained **Ensemble Voting Classifier**
4. Model outputs emotion probabilities across 4 core dimensions:
   - Euphoria ⚡
   - Melancholy 🌧️
   - Serenity ☮️
   - Turbulence 🌪️
5. Results displayed as interactive radar chart + synesthesia color palette

---

## 📦 Local Installation

### Prerequisites
- Python 3.9 or higher
- FFmpeg installed and added to system PATH ([download here](https://ffmpeg.org/download.html))

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/itslovepatel/emotisound-live.git
   cd emotisound-live
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain the model**
   ```bash
   python train_model.py
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

   Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📂 Project Structure

```
emotisound-live/
├── templates/
│   └── index.html              # Frontend with Chart.js radar visualization
├── static/
│   └── (CSS, JS, assets)
├── app.py                      # Main Flask application
├── audio_processor.py          # Feature extraction logic
├── train_model.py              # Model training script
├── emotisound_brain.pkl        # Trained ensemble model
├── requirements.txt            # Python dependencies
├── Dockerfile                  # For Docker deployment
├── spotify_tracks.csv          # Training dataset
└── README.md                   # You are here :)
```

---

## 🤝 Contributing

Contributions are very welcome! Ideas for future features:
- Real-time microphone input
- Spotify/YouTube URL analysis
- Mood-based playlist generator
- Export emotion DNA as NFT metadata

Feel free to fork and open a Pull Request!

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

**Built with 🎵 passion and ☕ caffeine by [Love Patel](https://github.com/itslovepatel)**
```

