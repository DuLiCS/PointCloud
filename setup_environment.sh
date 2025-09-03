#!/bin/bash

# Setup script for Point Cloud Completion Python Environment

echo "Setting up Python environment for Point Cloud Completion..."
echo "=========================================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv pointcloud_env

# Activate virtual environment
echo "Activating virtual environment..."
source pointcloud_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.4.0
pip install shapely>=1.7.0

echo "=========================================================="
echo "Setup completed!"
echo ""
echo "To use the environment:"
echo "1. Activate it: source pointcloud_env/bin/activate"
echo "2. Run tests: python test_conversion.py"
echo "3. Run main: python main.py"
echo "4. Deactivate when done: deactivate"
echo "=========================================================="
