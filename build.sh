#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Install Python Dependencies
pip install -r requirements.txt

# 2. Download and Extract FFmpeg (Static Build)
if [ ! -d "ffmpeg" ]; then
  echo "Downloading FFmpeg..."
  curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o ffmpeg.tar.xz
  tar -xf ffmpeg.tar.xz
  mv ffmpeg-master-latest-linux64-gpl ffmpeg
  rm ffmpeg.tar.xz
fi

# 3. Train the Brain (Create the .pkl file on the server)
echo "🧠 Training AI Brain on the server..."
python train_model.py