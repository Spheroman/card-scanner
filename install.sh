#!/bin/bash

# Debian-based system installation script for Card Scanner
set -e

echo "Starting Card Scanner installation..."

# 1. Update and install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git libgl1-mesa-glx libglib2.0-0

# 2. Setup Virtual Environment
echo "Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure systemd service
echo "Configuring systemd service..."
WORKING_DIR=$(pwd)
CURRENT_USER=$(whoami)

# Create a local copy of the service file with correct paths
sed -e "s|%WORKING_DIR%|${WORKING_DIR}|g" \
    -e "s|%USER%|${CURRENT_USER}|g" \
    card-scanner.service > card-scanner.service.tmp

sudo mv card-scanner.service.tmp /etc/systemd/system/card-scanner.service

# 5. Start the service
echo "Starting Card Scanner service..."
sudo systemctl daemon-reload
sudo systemctl enable card-scanner
sudo systemctl start card-scanner

echo "Installation complete!"
echo "You can check the service status with: sudo systemctl status card-scanner"
echo "API is running at http://localhost:8000"
