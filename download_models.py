#!/usr/bin/env python3
"""
IndexTTS2 Model Downloader
Automatically downloads required pretrained model files from Hugging Face
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
    from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "IndexTeam/IndexTTS-2"
MODEL_DIR = "./checkpoints"

# Required files and their approximate sizes
REQUIRED_FILES = {
    "bpe.model": "~5 MB",
    "gpt.pth": "~200 MB",
    "config.yaml": "~2 KB",
    "s2mel.pth": "~50 MB",
    "wav2vec2bert_stats.pt": "~1 MB"
}

def download_models():
    """
    Download all required model files from Hugging Face
    """
    print("="*60)
    print("IndexTTS2 Model Downloader")
    print("="*60)
    print(f"\nDownloading from: {REPO_ID}")
    print(f"Saving to: {os.path.abspath(MODEL_DIR)}")
    print(f"Total files: {len(REQUIRED_FILES)}")
    print(f"Approximate total size: ~256 MB\n")
    
    # Create checkpoint directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    downloaded = []
    skipped = []
    failed = []
    
    for idx, (filename, size) in enumerate(REQUIRED_FILES.items(), 1):
        local_path = os.path.join(MODEL_DIR, filename)
        
        # Check if file already exists
        if os.path.exists(local_path):
            print(f"[{idx}/{len(REQUIRED_FILES)}] ✓ {filename} - Already exists (skipped)")
            skipped.append(filename)
            continue
        
        print(f"[{idx}/{len(REQUIRED_FILES)}] ⬇ {filename} ({size}) - Downloading...")
        
        try:
            # Download from Hugging Face
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False  # Copy files instead of symlinks on Windows
            )
            print(f"[{idx}/{len(REQUIRED_FILES)}] ✓ {filename} - Download complete")
            downloaded.append(filename)
        except Exception as e:
            print(f"[{idx}/{len(REQUIRED_FILES)}] ✗ {filename} - Download failed: {e}")
            failed.append(filename)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"✓ Downloaded: {len(downloaded)} files")
    if downloaded:
        for f in downloaded:
            print(f"  - {f}")
    
    print(f"\n⊘ Skipped (already exist): {len(skipped)} files")
    if skipped:
        for f in skipped:
            print(f"  - {f}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)} files")
        for f in failed:
            print(f"  - {f}")
        print("\n⚠️  Some files failed to download. Please check your internet connection.")
        return False
    else:
        print("\n✅ All required model files are ready!")
        print(f"\nYou can now run: python webui.py")
        return True

if __name__ == "__main__":
    try:
        success = download_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
