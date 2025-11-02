#!/bin/bash
# ========================================================================
# Lightning AI Installation Script for IndexTTS2 Amharic Fine-tuning
# ========================================================================

set -e  # Exit on error

echo "========================================="
echo "IndexTTS2 Amharic - Lightning AI Setup"
echo "========================================="

# Check Python version
python_version=\
echo "Python version: \"

if [[ \ == "False" ]]; then
    echo "ERROR: Python 3.10+ required. Found: \"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Install PyTorch with CUDA support (Lightning AI has NVIDIA GPUs)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# Install core requirements
echo "Installing core requirements..."
pip install -r requirements.txt --no-cache-dir

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python3 -c "import gradio; print(f'Gradio: {gradio.__version__}')"

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data: python scripts/prepare_amharic_data.py"
echo "2. Train vocabulary: python scripts/train_amharic_vocabulary.py"
echo "3. Start training: python train_amharic_full.py --data_dir <path> --vocab <vocab_file>"
echo ""
