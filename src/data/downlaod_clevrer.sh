#!/bin/bash

# ==========================================
# CLEVRER Dataset Download Script
# ==========================================

# 1. Define the base directory on the cluster
BASE_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER"
TRAIN_DIR="$BASE_DIR/train_video"

echo "🚀 Starting dataset download setup for the new project..."

# 2. Create the necessary directories
echo "📁 Creating directory structure at $BASE_DIR..."
mkdir -p "$TRAIN_DIR"

# 3. Download the training videos from the official MIT server
# Note: Using -c allows resuming if the cluster connection drops
echo "⏳ Downloading training videos (video_train.zip)..."
wget -c --show-progress -P "$TRAIN_DIR" http://data.csail.mit.edu/clevrer/videos/train/video_train.zip

# 4. Unzip the downloaded file
echo "📦 Extracting video files. This may take a few minutes..."
unzip -q "$TRAIN_DIR/video_train.zip" -d "$TRAIN_DIR"

# 5. Clean up the zip file to save storage space
echo "🧹 Cleaning up the zip archive..."
rm "$TRAIN_DIR/video_train.zip"

echo "✅ Download and extraction complete! Your training videos are locked and loaded in $TRAIN_DIR"