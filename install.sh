#!/bin/bash

# NeuroInsight BCI Project Installation Script
# This script installs all required dependencies and sets up the project

echo "🧠 NeuroInsight BCI Project - Installation Script"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 detected"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install required packages
echo "📦 Installing required Python packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "❌ Failed to install some packages. Please check the error messages above."
    exit 1
fi

# Install the package in development mode
echo "🔧 Installing NeuroInsight package in development mode..."
pip3 install -e .

if [ $? -eq 0 ]; then
    echo "✅ NeuroInsight package installed successfully"
else
    echo "❌ Failed to install NeuroInsight package. Please check the error messages above."
    exit 1
fi

# Test the installation
echo "🧪 Testing installation..."
python3 -c "from neuroinsight.eeg_simulator import EEGSimulator; print('✅ EEGSimulator imported successfully')"

if [ $? -eq 0 ]; then
    echo "✅ Installation test passed"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "   1. Run the demo: python3 demo_eeg_simulation.py"
echo "   2. Or use the command: neuroinsight-demo"
echo "   3. Check the README.md for more usage examples"
echo ""
echo "📚 Documentation: README.md"
echo "🔧 Source code: neuroinsight/"
echo "" 