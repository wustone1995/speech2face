#!/bin/bash

mkdir -p speech2face/preprocess/data/data
mkdir -p speech2face/preprocess/data/data/speaker_video_embeddings

mkdir -p speech2face/preprocess/data/data/audios
mkdir -p speech2face/preprocess/data/data/videos
mkdir -p speech2face/preprocess/data/data/audio_spectrograms
mkdir -p speech2face/preprocess/data/data/frames
mkdir -p speech2face/preprocess/data/cropped_frames
mkdir -p speech2face/preprocess/data/data/pretrained_model
mkdir -p speech2face/preprocess/data/data/cropped_models

echo "Done setting up directories set"

# pip install -r ../requirements.txt

# git clone https://github.com/davidsandberg/facenet.git facenet/
# pip install face_recognition
# sudo apt-get --assume-yes install ffmpeg
# sudo apt-get --assume-yes --fix-missing install youtube-dl
pip install face_recognition
pip install ffmpeg
pip install youtube-dl
pip install keras_vggface

echo "Done setting up directories and environment, now proceed to download dataset...."
