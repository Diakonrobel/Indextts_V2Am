#!/usr/bin/env python3
"""
Professional Amharic IndexTTS2 Web Interface Launcher
Complete training and inference platform with modern UI
"""
import os
import sys
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'gradio', 'torch', 'torchaudio', 'numpy', 'pyyaml', 
        'psutil', 'tqdm', 'sentencepiece', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_gpu_availability():
    """Check GPU availability and display information"""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU Available: {gpu_count} CUDA device(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   Device {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Recommend T4 for this project
        if any('T4' in torch.cuda.get_device_properties(i).name for i in range(gpu_count)):
            print("✅ T4 GPU detected - Optimized for this project!")
        else:
            print("💡 Recommended GPU: NVIDIA T4 (16GB) for optimal performance")
    else:
        print("⚠️  No GPU detected - Training will be very slow")
        print("💡 Consider using a machine with NVIDIA GPU for training")
    
    return torch.cuda.is_available()

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        "checkpoints",
        "datasets", 
        "logs/gradio",
        "logs/training",
        "outputs/audio",
        "outputs/models",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def launch_gradio_app(port=7860, share=False, in_browser=True):
    """Launch the Gradio application"""
    
    # Check if the app file exists
    app_path = Path("amharic_gradio_app.py")
    if not app_path.exists():
        print("❌ Gradio app file not found: amharic_gradio_app.py")
        print("💡 Make sure you're in the correct directory")
        return False
    
    print(f"🚀 Launching Amharic IndexTTS2 Web Interface...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📱 Mode: {'Public' if share else 'Local'}")
    
    # Additional launch options for production
    launch_args = {
        'server_name': "0.0.0.0",
        'server_port': port,
        'share': share,
        'inbrowser': in_browser,
        'show_error': True,
        'quiet': False,
        'show_tips': True,
        'height': 800,
        'title': "Amharic IndexTTS2 - Professional TTS Platform"
    }
    
    try:
        # Import and run the app
        from amharic_gradio_app import main as gradio_main
        gradio_main()
        return True
    except ImportError as e:
        print(f"❌ Error importing Gradio app: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error launching Gradio app: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Amharic IndexTTS2 Web Interface Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Port to launch the interface on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit")
    parser.add_argument("--setup-only", action="store_true", help="Only setup directories and exit")
    
    args = parser.parse_args()
    
    print("🎙️  Amharic IndexTTS2 - Professional TTS Platform")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking Dependencies...")
    if not check_dependencies():
        return 1
    
    # Check GPU
    print("\n🖥️  Checking System Resources...")
    has_gpu = check_gpu_availability()
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Handle check-only mode
    if args.check_only:
        print("\n✅ All checks passed! Ready to launch.")
        return 0
    
    # Handle setup-only mode  
    if args.setup_only:
        print("\n✅ Setup complete! Ready to launch.")
        return 0
    
    # Launch the application
    print("\n" + "=" * 60)
    success = launch_gradio_app(
        port=args.port, 
        share=args.share,
        in_browser=not args.no_browser
    )
    
    if not success:
        print("\n❌ Failed to launch application")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)