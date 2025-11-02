# 📦 Installation Guide - IndexTTS2 Amharic Fine-tuning

## 🌍 Universal Installation (Works Everywhere!)

This guide covers installation for:
- ✅ **Lightning AI** (Cloud GPU platform)
- ✅ **Windows** (Local PC with/without GPU)
- ✅ **Linux** (Ubuntu, Debian, CentOS, etc.)
- ✅ **macOS** (Apple Silicon & Intel)
- ✅ **Google Colab** (Free/Pro)
- ✅ **Any PC** with Python 3.10+

---

## 🚀 Quick Start (One Command)

### **Lightning AI / Linux / macOS:**
\\\ash
chmod +x install_lightning.sh && ./install_lightning.sh
\\\

### **Windows:**
\\\powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
\\\

### **Google Colab:**
\\\python
!git clone https://github.com/Diakonrobel/Indextts_V2Am.git
%cd Indextts_V2Am
!pip install -r requirements.txt -q --no-cache-dir
\\\

---

## 📋 Detailed Installation Steps

### **Prerequisites:**

1. **Python 3.10 - 3.12** (required)
2. **CUDA 12.1+** (optional, for GPU acceleration)
3. **16GB+ RAM** (recommended)
4. **50GB+ free disk space**

### **Step 1: Install Python**

#### **Windows:**
`powershell
# Download from python.org or use winget
winget install Python.Python.3.11
`

#### **Ubuntu/Debian:**
`ash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
`

#### **macOS:**
`ash
brew install python@3.11
`

#### **Lightning AI:**
Python is pre-installed ✅

---

### **Step 2: Install System Dependencies**

#### **Windows:**
`powershell
# Install ffmpeg (for audio processing)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
`

#### **Ubuntu/Debian:**
`ash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 git git-lfs
`

#### **macOS:**
`ash
brew install ffmpeg libsndfile git-lfs
`

#### **Lightning AI / Google Colab:**
Pre-installed ✅

---

### **Step 3: Clone Repository**

`ash
git clone https://github.com/Diakonrobel/Indextts_V2Am.git
cd Indextts_V2Am

# Install Git LFS for large files
git lfs install
git lfs pull
`

---

### **Step 4: Install Python Dependencies**

#### **Method 1: Using requirements.txt (Recommended)**

**Lightning AI / Linux / macOS:**
`ash
pip install -r requirements.txt --no-cache-dir
`

**Windows:**
`powershell
pip install -r requirements.txt
`

**Google Colab:**
`python
!pip install -r requirements.txt -q --no-cache-dir
`

#### **Method 2: Minimal Installation (Core only)**

`ash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate librosa sentencepiece
pip install gradio tensorboard omegaconf pyyaml tqdm
`

#### **Method 3: With Optional Features**

`ash
# With DeepSpeed (for distributed training)
pip install -r requirements.txt
pip install deepspeed

# With Web UI
pip install -r requirements.txt
pip install gradio fastapi uvicorn

# Development mode
pip install -r requirements.txt
pip install pytest black flake8 mypy
`

---

### **Step 5: Verify Installation**

`ash
# Check Python
python --version
# Should show: Python 3.10.x or 3.11.x or 3.12.x

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check other libraries
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import transformers; print('Transformers OK')"
python -c "import librosa; print('Librosa OK')"
python -c "import gradio; print('Gradio OK')"
`

**Expected Output:**
`
PyTorch: 2.0.0+cu121 (or higher)
CUDA: True (if GPU available)
TorchAudio: 2.0.0+cu121 (or higher)
Transformers OK
Librosa OK
Gradio OK
`

---

## 🔧 Platform-Specific Instructions

### **🌩️ Lightning AI**

Lightning AI provides pre-configured GPU instances. Just install dependencies:

`ash
# Automatic installation
bash install_lightning.sh

# Or manual
pip install -r requirements.txt --no-cache-dir
`

**Tips:**
- Lightning AI has CUDA pre-installed ✅
- Use --no-cache-dir to save disk space
- GPU (T4/V100/A100) auto-detected

---

### **🪟 Windows (Local PC)**

#### **With GPU (NVIDIA):**

1. **Install CUDA Toolkit 12.1+**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Or use: choco install cuda

2. **Install cuDNN** (optional but recommended)
   - Download: https://developer.nvidia.com/cudnn

3. **Run installation script:**
   `powershell
   powershell -ExecutionPolicy Bypass -File install_windows.ps1
   `

#### **CPU Only:**

`powershell
# Install PyTorch CPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
`

**Common Windows Issues:**

1. **Visual Studio Build Tools Required:**
   `powershell
   # Install Visual Studio Build Tools
   choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
   `

2. **Permissions Error:**
   `powershell
   # Run PowerShell as Administrator
   # Or use: --user flag
   pip install -r requirements.txt --user
   `

---

### **🐧 Linux (Ubuntu/Debian/CentOS)**

#### **With GPU:**

`ash
# Install NVIDIA drivers (if not already)
sudo apt install nvidia-driver-535

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# Run installation script
bash install_lightning.sh
`

#### **CPU Only:**

`ash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
`

---

### **🍎 macOS (Apple Silicon / Intel)**

**Note:** macOS doesn't support CUDA. Use CPU or MPS (Apple Silicon):

`ash
# Install dependencies
brew install ffmpeg libsndfile portaudio

# Install Python packages
pip install -r requirements.txt

# Verify MPS (Apple Silicon only)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
`

---

### **📓 Google Colab**

**Free Tier (T4 GPU):**

`python
# Clone repository
!git clone https://github.com/Diakonrobel/Indextts_V2Am.git
%cd Indextts_V2Am

# Install dependencies
!pip install -r requirements.txt -q --no-cache-dir

# Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
`

**Pro Tier (V100/A100):**
Same as above, but with more VRAM and faster training!

---

## ⚡ Performance Optimization

### **For T4 GPU (16GB VRAM):**

`ash
# Use Full Layer Training config (already optimized)
python train_amharic_full.py --data_dir <path> --vocab <vocab>
`

Config automatically uses:
- ✅ Batch size: 1
- ✅ Gradient accumulation: 16
- ✅ Mixed precision (FP16)
- ✅ Gradient checkpointing
- ✅ CPU offload

### **For V100/A100 (32GB+ VRAM):**

Edit configs/amharic_config.yaml:
`yaml
training:
    batch_size: 2  # or 4 for A100
    gradient_accumulation_steps: 8  # adjust accordingly
`

---

## 🐛 Troubleshooting

### **1. "CUDA not found" Error**

`ash
# Reinstall PyTorch with CUDA
pip uninstall torch torchaudio torchvision
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
`

### **2. "ImportError: libsndfile" (Linux)**

`ash
sudo apt install libsndfile1 libsndfile1-dev
`

### **3. "No module named 'sentencepiece'" (Windows)**

`powershell
pip install sentencepiece --prefer-binary
`

### **4. "Out of Memory" Error**

`ash
# Verify config uses:
# - batch_size: 1
# - mixed_precision: true
# - gradient_checkpointing: true

# Or reduce max_text_tokens
# Edit configs/amharic_config.yaml:
# max_text_tokens: 400  # from 600
`

### **5. "Microsoft Visual C++ Required" (Windows)**

`powershell
# Install Visual Studio Build Tools
choco install visualstudio2022buildtools

# Or download from:
# https://visualstudio.microsoft.com/downloads/
`

### **6. "ffmpeg not found"**

**Windows:**
`powershell
choco install ffmpeg
# Or download: https://ffmpeg.org/download.html
`

**Linux:**
`ash
sudo apt install ffmpeg
`

**macOS:**
`ash
brew install ffmpeg
`

---

## ✅ Verification Checklist

After installation, run these checks:

`ash
# 1. Python version
python --version  # Should be 3.10+

# 2. PyTorch installation
python -c "import torch; print(torch.__version__)"

# 3. CUDA availability (if GPU)
python -c "import torch; print(torch.cuda.is_available())"

# 4. Audio processing
python -c "import librosa, soundfile; print('Audio OK')"

# 5. Transformers
python -c "import transformers; print('Transformers OK')"

# 6. Gradio (Web UI)
python -c "import gradio; print('Gradio OK')"

# 7. Run GPU check (if available)
python tools/gpu_check.py
`

---

## 📚 Next Steps

Once installed:

1. **Prepare Data:**
   `ash
   python scripts/prepare_amharic_data.py \\
       --audio_dir ./data/audio \\
       --text_dir ./data/text \\
       --output_dir ./data/prepared
   `

2. **Train Vocabulary:**
   `ash
   python scripts/train_amharic_vocabulary.py \\
       --text_files ./data/text/*.txt \\
       --output_dir ./models \\
       --vocab_size 8000
   `

3. **Start Training:**
   `ash
   python train_amharic_full.py \\
       --data_dir ./data/prepared \\
       --vocab ./models/amharic_bpe.model
   `

---

## 💡 Installation Variants

### **Minimal (Core only):**
`ash
pip install torch torchaudio librosa sentencepiece transformers
`
**Size:** ~3GB | **Time:** ~5 min

### **Standard (Full features):**
`ash
pip install -r requirements.txt
`
**Size:** ~8GB | **Time:** ~15 min

### **Complete (With all extras):**
`ash
pip install -r requirements.txt
pip install deepspeed flash-attn  # Linux only
`
**Size:** ~12GB | **Time:** ~30 min

---

## 🆘 Get Help

If you encounter issues:

1. **Check logs:** Look for error messages
2. **Verify Python version:** Must be 3.10-3.12
3. **Update pip:** pip install --upgrade pip
4. **Clear cache:** pip cache purge
5. **Reinstall:** pip install -r requirements.txt --force-reinstall

**Still having issues?**
- Check FULL_LAYER_TRAINING_UPDATE.md for troubleshooting
- See PROJECT_ANALYSIS.md for technical details
- Review GitHub Issues

---

## 📝 Summary

| Environment | Installation Command | Time | Notes |
|-------------|---------------------|------|-------|
| **Lightning AI** | ash install_lightning.sh | 10-15 min | GPU auto-detected |
| **Windows GPU** | install_windows.ps1 | 15-20 min | Requires CUDA |
| **Linux GPU** | ash install_lightning.sh | 10-15 min | Requires CUDA |
| **macOS** | pip install -r requirements.txt | 15-20 min | CPU/MPS only |
| **Google Colab** | !pip install -r requirements.txt | 5-10 min | GPU included |

---

**Ready to train?** All dependencies installed! 🚀

Next: [FULL_LAYER_TRAINING_UPDATE.md](FULL_LAYER_TRAINING_UPDATE.md)
