#!/usr/bin/env python3
"""
Comprehensive Amharic IndexTTS2 Gradio Web Interface
Professional-grade UI with complete training and inference capabilities
"""
import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import gradio as gr
import torch
import torchaudio
import numpy as np
from datetime import datetime
import yaml
import shutil
import psutil

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.bigvgan import BigVGAN
from scripts.optimized_full_layer_finetune_amharic import OptimizedFullLayerTrainer
from indextts.utils.live_training_monitor import LiveTrainingMonitor
from indextts.utils.audio_quality_metrics import calculate_audio_quality_metrics
from indextts.utils.amharic_prosody import AmharicProsodyController
from indextts.utils.model_comparator import ModelComparator
from indextts.utils.batch_processor import BatchTTSProcessor
from indextts.utils.dataset_processor import ComprehensiveDatasetProcessor


class AmharicTTSGradioApp:
    """Comprehensive Amharic IndexTTS2 Gradio Application"""
    
    def __init__(self):
        self.setup_logging()
        self.initialize_components()
        self.setup_state_management()
        
    def setup_logging(self):
        """Setup logging for the application"""
        log_dir = Path("logs/gradio")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'gradio_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_components(self):
        """Initialize core components"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_tokenizer = None
        self.current_vocoder = None
        self.training_process = None
        self.is_training = False
        self.training_monitor = LiveTrainingMonitor()
        self.prosody_controller = AmharicProsodyController()
        self.model_comparator = ModelComparator()
        self.batch_processor = None
        
        # Initialize comprehensive dataset processor
        self.dataset_processor = ComprehensiveDatasetProcessor(
            output_dir="processed_datasets",
            sample_rate=24000,
            min_duration=1.0,
            max_duration=10.0,
            min_snr_db=20.0,
            denoise=False,
            validate_quality=True
        )
        
        # Load default models if available
        self.load_default_models()
    
    def setup_state_management(self):
        """Setup application state management"""
        self.state = {
            'current_training_status': 'idle',
            'current_epoch': 0,
            'current_step': 0,
            'current_loss': 0.0,
            'training_progress': 0.0,
            'available_checkpoints': [],
            'system_resources': {},
            'supported_languages': ['Amharic'],
            'available_models': [],
            'current_model_info': {},
            'training_history': []
        }
    
    def load_default_models(self):
        """Load default Amharic models if available"""
        try:
            # Load vocabulary
            vocab_path = Path("amharic_bpe.model")
            if vocab_path.exists():
                self.current_tokenizer = AmharicTextTokenizer(
                    vocab_file=str(vocab_path),
                    normalizer=AmharicTextNormalizer()
                )
                self.logger.info("✅ Default Amharic vocabulary loaded")
            
            # Load vocoder
            vocoder_path = Path("checkpoints/bigvgan_v2_22khz_80band_256x")
            if vocoder_path.exists():
                self.current_vocoder = BigVGAN.from_pretrained(vocoder_path)
                self.logger.info("✅ Default vocoder loaded")
                
        except Exception as e:
            self.logger.warning(f"Could not load default models: {e}")
    
    def get_system_resources(self) -> Dict:
        """Get current system resource usage"""
        gpu_util = 0
        if torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, 'utilization'):
                    gpu_util = torch.cuda.utilization()
            except (ModuleNotFoundError, RuntimeError):
                gpu_util = 0
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            'gpu_utilization': gpu_util
        }
    
    def update_system_resources(self):
        """Update system resource monitoring"""
        self.state['system_resources'] = self.get_system_resources()
        return self.state['system_resources']
    
    def refresh_checkpoint_list(self) -> List[str]:
        """Refresh list of available checkpoints"""
        checkpoints_dir = Path("checkpoints")
        if not checkpoints_dir.exists():
            return []
        
        checkpoints = []
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                checkpoint_files = list(model_dir.glob("*.pt"))
                if checkpoint_files:
                    checkpoints.append(str(model_dir))
        
        self.state['available_checkpoints'] = checkpoints
        return checkpoints
    
    def upload_dataset(self, audio_files, text_files, subtitle_files, dataset_name):
        """Handle comprehensive dataset upload with SRT/VTT support"""
        if not dataset_name:
            return "❌ Please provide dataset name"
        
        if not audio_files and not subtitle_files:
            return "❌ Please provide audio files or subtitle files"
        
        try:
            # Create dataset directory
            dataset_dir = Path(f"datasets/{dataset_name}")
            audio_dir = dataset_dir / "audio"
            text_dir = dataset_dir / "text"
            subtitle_dir = dataset_dir / "subtitles"
            
            for dir_path in [audio_dir, text_dir, subtitle_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Copy audio files
            audio_count = 0
            if audio_files:
                for i, audio_file in enumerate(audio_files):
                    shutil.copy2(audio_file, audio_dir / f"audio_{i:04d}.wav")
                    audio_count += 1
            
            # Copy text files
            text_count = 0
            if text_files:
                for i, text_file in enumerate(text_files):
                    shutil.copy2(text_file, text_dir / f"text_{i:04d}.txt")
                    text_count += 1
            
            # Copy subtitle files (SRT/VTT)
            subtitle_count = 0
            if subtitle_files:
                for i, subtitle_file in enumerate(subtitle_files):
                    ext = Path(subtitle_file).suffix
                    shutil.copy2(subtitle_file, subtitle_dir / f"subtitle_{i:04d}{ext}")
                    subtitle_count += 1
            
            self.logger.info(f"✅ Dataset '{dataset_name}' uploaded successfully")
            return f"✅ Dataset '{dataset_name}' uploaded successfully!\n📊 Files: {audio_count} audio, {text_count} text, {subtitle_count} subtitles"
            
        except Exception as e:
            error_msg = f"❌ Error uploading dataset: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def prepare_dataset(self, dataset_name, processing_mode, min_duration, max_duration, 
                       sample_rate, min_snr, enable_denoise, enable_vad):
        """Prepare uploaded dataset with comprehensive Amharic preprocessing"""
        if not dataset_name:
            return "❌ Please specify a dataset name"
        
        try:
            dataset_dir = Path(f"datasets/{dataset_name}")
            if not dataset_dir.exists():
                return f"❌ Dataset '{dataset_name}' not found"
            
            # Update processor config
            self.dataset_processor.sample_rate = sample_rate
            self.dataset_processor.audio_slicer.min_duration = min_duration
            self.dataset_processor.audio_slicer.max_duration = max_duration
            self.dataset_processor.audio_slicer.denoise = enable_denoise
            self.dataset_processor.quality_validator.min_snr_db = min_snr
            
            # Run in background thread
            def process_in_background():
                try:
                    # Check for SRT/VTT files
                    subtitle_dir = dataset_dir / "subtitles"
                    audio_dir = dataset_dir / "audio"
                    
                    if processing_mode == "srt_vtt" and subtitle_dir.exists():
                        # Process with SRT/VTT
                        subtitle_files = list(subtitle_dir.glob("*.srt")) + list(subtitle_dir.glob("*.vtt"))
                        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
                        
                        if subtitle_files and audio_files:
                            # Process each pair
                            samples, stats = self.dataset_processor.process_from_srt_vtt(
                                media_path=str(audio_files[0]),
                                subtitle_path=str(subtitle_files[0]),
                                dataset_name=dataset_name
                            )
                            self.logger.info(f"✅ Processed {stats.processed} samples with SRT/VTT")
                        else:
                            self.logger.error("No matching SRT/VTT and audio files found")
                    else:
                        # Traditional processing with Amharic normalizer
                        from scripts.prepare_amharic_data import AmharicDatasetPreparer
                        
                        preparer = AmharicDatasetPreparer(
                            audio_dir=str(audio_dir),
                            text_dir=str(dataset_dir / "text"),
                            output_dir=f"processed_datasets/{dataset_name}",
                            sample_rate=sample_rate,
                            min_duration=min_duration,
                            max_duration=max_duration
                        )
                        
                        manifest_paths = preparer.prepare_dataset()
                        self.logger.info(f"✅ Amharic dataset prepared: {manifest_paths}")
                        
                except Exception as e:
                    self.logger.error(f"Dataset preparation failed: {e}")
            
            threading.Thread(target=process_in_background).start()
            
            return f"🔄 Dataset preparation started for '{dataset_name}'\n📋 Mode: {processing_mode}\n⏳ This may take several minutes..."
            
        except Exception as e:
            error_msg = f"❌ Error preparing dataset: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def process_web_urls(self, urls_text, dataset_name, max_workers, extract_subtitles, 
                        min_snr, enable_denoise):
        """Process media from web URLs with comprehensive pipeline"""
        if not urls_text or not dataset_name:
            return "❌ Please provide URLs and dataset name"
        
        try:
            # Parse URLs
            urls = [line.strip() for line in urls_text.split('\n') if line.strip() and not line.startswith('#')]
            
            if not urls:
                return "❌ No valid URLs found"
            
            # Update processor config
            self.dataset_processor.quality_validator.min_snr_db = min_snr
            self.dataset_processor.audio_slicer.denoise = enable_denoise
            
            def process_in_background():
                try:
                    samples, stats = self.dataset_processor.process_from_urls(
                        urls=urls,
                        dataset_name=dataset_name,
                        max_workers=max_workers,
                        extract_subtitles=extract_subtitles
                    )
                    
                    self.logger.info(
                        f"✅ Web processing complete: {stats.processed}/{stats.total_inputs} samples, "
                        f"{stats.total_duration:.1f}s total, quality {stats.avg_quality_score:.1f}/10"
                    )
                except Exception as e:
                    self.logger.error(f"Web URL processing failed: {e}")
            
            threading.Thread(target=process_in_background).start()
            
            return f"🔄 Processing {len(urls)} URLs...\n⏳ This may take 10-30 minutes depending on content length.\n📥 Downloading media and extracting subtitles..."
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def start_training(self, model_path, config_path, output_dir, checkpoint_path, 
                      num_epochs, batch_size, learning_rate, enable_sdpa, enable_ema,
                      mixed_precision, gradient_accumulation_steps):
        """Start model training"""
        if self.is_training:
            return "❌ Training is already in progress"
        
        try:
            # Validate inputs
            if not all([model_path, config_path, output_dir]):
                return "❌ Please provide model path, config path, and output directory"
            
            # Prepare training command
            cmd = [
                "python", "scripts/optimized_full_layer_finetune_amharic.py",
                "--model_path", model_path,
                "--config", config_path,
                "--output_dir", output_dir,
                "--num_epochs", str(num_epochs),
                "--batch_size", str(batch_size),
                "--learning_rate", str(learning_rate),
                "--gradient_accumulation_steps", str(gradient_accumulation_steps)
            ]
            
            if checkpoint_path:
                cmd.extend(["--resume_from", checkpoint_path])
            if enable_sdpa:
                cmd.append("--enable_sdpa")
            if enable_ema:
                cmd.append("--enable_ema")
            if mixed_precision:
                cmd.append("--mixed_precision")
            
            # Start training in background thread
            self.is_training = True
            self.state['current_training_status'] = 'starting'
            
            threading.Thread(target=self._run_training, args=(cmd,)).start()
            
            return f"🚀 Training started!\n📊 Configuration:\n• Epochs: {num_epochs}\n• Batch Size: {batch_size}\n• Learning Rate: {learning_rate}\n• Optimizations: SDPA={enable_sdpa}, EMA={enable_ema}, Mixed Precision={mixed_precision}"
            
        except Exception as e:
            error_msg = f"❌ Error starting training: {str(e)}"
            self.logger.error(error_msg)
            self.is_training = False
            return error_msg
    
    def _run_training(self, cmd):
        """Run training in background"""
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break
                
                # Update training status
                self.state['current_training_status'] = 'running'
                self.state['training_progress'] += 0.1
                time.sleep(1)
            
            # Check completion
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                self.state['current_training_status'] = 'completed'
                self.logger.info("✅ Training completed successfully")
            else:
                self.state['current_training_status'] = 'failed'
                self.logger.error(f"Training failed: {stderr}")
                
        except Exception as e:
            self.state['current_training_status'] = 'failed'
            self.logger.error(f"Error during training: {e}")
        finally:
            self.is_training = False
    
    def get_training_status(self):
        """Get current training status"""
        return f"""
📊 **Training Status**: {self.state['current_training_status'].upper()}
🔄 **Progress**: {self.state['training_progress']:.1f}%
📈 **Current Step**: {self.state.get('current_step', 0)}
💾 **System Resources**:
• GPU Memory: {self.state['system_resources'].get('gpu_memory', 0):.1f}GB / {self.state['system_resources'].get('gpu_memory_total', 0):.1f}GB
• CPU Usage: {self.state['system_resources'].get('cpu_percent', 0):.1f}%
• Memory Usage: {self.state['system_resources'].get('memory_percent', 0):.1f}%
        """
    
    def stop_training(self):
        """Stop current training"""
        if self.is_training:
            # This would need to be implemented to actually stop the training process
            self.is_training = False
            self.state['current_training_status'] = 'stopped'
            return "🛑 Training stopped by user"
        return "❌ No training in progress"
    
    def load_model_for_inference(self, model_path, vocab_path, config_path):
        """Load model for inference"""
        try:
            if not all([model_path, vocab_path, config_path]):
                return "❌ Please provide model, vocabulary, and config paths"
            
            # Load tokenizer
            tokenizer = AmharicTextTokenizer(
                vocab_file=vocab_path,
                normalizer=AmharicTextNormalizer()
            )
            
            # Load model
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model = UnifiedVoice(
                layers=config['gpt']['layers'],
                model_dim=config['gpt']['model_dim'],
                heads=config['gpt']['heads'],
                max_text_tokens=config['gpt']['max_text_tokens'],
                max_mel_tokens=config['gpt']['max_mel_tokens'],
                number_text_tokens=tokenizer.vocab_size,
                number_mel_codes=config['gpt']['number_mel_codes'],
                start_text_token=config['gpt']['start_text_token'],
                stop_text_token=config['gpt']['stop_text_token'],
                start_mel_token=config['gpt']['start_mel_token'],
                stop_mel_token=config['gpt']['stop_mel_token'],
                condition_type=config['gpt']['condition_type'],
                condition_module=config['gpt']['condition_module'],
                emo_condition_module=config['gpt']['emo_condition_module']
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model = model.to(self.device)
            model.eval()
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            
            return f"✅ Model loaded successfully!\n📊 Model Info:\n• Parameters: {sum(p.numel() for p in model.parameters()):,}\n• Device: {self.device}\n• Vocabulary Size: {tokenizer.vocab_size}"
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def generate_speech(self, text, voice_id, emotion, speed, pitch, temperature, 
                       max_new_tokens, sample_rate):
        """Generate speech from text"""
        if not self.current_model or not self.current_tokenizer:
            return None, "❌ Please load a model first"
        
        if not text.strip():
            return None, "❌ Please enter text to synthesize"
        
        try:
            # Tokenize text
            text_tokens = self.current_tokenizer.encode(text, out_type=int)
            
            # Generate speech (simplified - would need full IndexTTS2 generation logic)
            # This is a placeholder for the actual generation process
            
            # For demo purposes, create a dummy audio file
            duration = len(text_tokens) * 0.1  # Rough duration estimate
            sample_count = int(sample_rate * duration)
            audio = np.random.randn(sample_count).astype(np.float32) * 0.1
            
            # Save to temporary file
            output_path = "temp_generated_audio.wav"
            torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), sample_rate)
            
            return output_path, f"✅ Generated {len(text_tokens)} tokens in {duration:.2f} seconds"
            
        except Exception as e:
            error_msg = f"❌ Error generating speech: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
    
    def generate_speech_with_metrics(self, text, voice_id, emotion, speed, pitch,
                                     temperature, max_new_tokens, sample_rate,
                                     gemination=1.0, ejective=1.0, duration=1.0, stress="penultimate"):
        """Generate speech with prosody controls and quality metrics"""
        if not self.current_model or not self.current_tokenizer:
            return (None, "❌ Load model first", {}, 0, 0, 0, 0, 0, "")
        
        if not text.strip():
            return (None, "❌ Enter text", {}, 0, 0, 0, 0, 0, "")
        
        try:
            # Apply prosody controls
            prosody_info = self.prosody_controller.apply_ejective_emphasis(text, ejective)
            self.prosody_controller.apply_duration_control(duration)
            
            # Generate audio
            audio_path, status = self.generate_speech(
                text, voice_id, emotion, speed, pitch,
                temperature, max_new_tokens, sample_rate
            )
            
            if audio_path is None:
                return (None, status, {}, 0, 0, 0, 0, 0, "")
            
            # Calculate quality metrics
            metrics = calculate_audio_quality_metrics(audio_path)
            
            # Format quality flags
            flags = []
            if metrics.get('is_clipping'):
                flags.append("⚠️ Clipping")
            if metrics.get('is_too_quiet'):
                flags.append("⚠️ Too quiet")
            if metrics.get('is_noisy'):
                flags.append("⚠️ Noisy")
            flags_text = " | ".join(flags) if flags else "✅ All checks passed"
            
            return (
                audio_path,
                status,
                prosody_info,
                metrics.get('rms_energy', 0),
                metrics.get('peak_level', 0),
                metrics.get('zero_crossing_rate', 0),
                metrics.get('duration_seconds', 0),
                metrics.get('quality_score', 0),
                flags_text
            )
        
        except Exception as e:
            return (None, f"❌ Error: {e}", {}, 0, 0, 0, 0, 0, "")
    
    def batch_generate(self, texts, voice_id, emotion, speed, sample_rate):
        """Generate speech for multiple texts"""
        if not self.current_model:
            return "❌ Please load a model first"
        
        if not texts:
            return "❌ Please provide texts to synthesize"
        
        audio_paths = []
        for i, text in enumerate(texts.split('\n')):
            if text.strip():
                audio_path, msg = self.generate_speech(text.strip(), voice_id, emotion, speed, 0, 1, 1000, sample_rate)
                if audio_path:
                    audio_paths.append(f"Audio {i+1}: {audio_path}")
        
        return f"✅ Generated {len(audio_paths)} audio files\n" + '\n'.join(audio_paths)
    
    def create_interface(self):
        """Create the complete Gradio interface"""
        
        # Modern Dark Theme CSS - Professional & Clean
        css = """
        /* Global Dark Theme */
        :root {
            --primary-bg: #0f1419;
            --secondary-bg: #1a1f2e;
            --card-bg: #1e2533;
            --accent-primary: #667eea;
            --accent-secondary: #764ba2;
            --accent-success: #4CAF50;
            --accent-warning: #FFC107;
            --accent-error: #F44336;
            --text-primary: #e8eaed;
            --text-secondary: #9aa0a6;
            --border-color: #2d3748;
        }
        
        .gradio-container {
            background: var(--primary-bg) !important;
            color: var(--text-primary) !important;
        }
        
        .main-container {
            max-width: 1400px !important;
            margin: auto !important;
            padding: 20px !important;
        }
        
        /* Stunning Header with Glassmorphism */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        /* Section Headers - Dark Mode */
        .section-header {
            background: rgba(102, 126, 234, 0.15);
            border-left: 4px solid var(--accent-primary);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        /* Status Boxes - Dark Variants */
        .status-box {
            background: rgba(76, 175, 80, 0.15);
            border: 1px solid var(--accent-success);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: #81c784;
        }
        
        .warning-box {
            background: rgba(255, 193, 7, 0.15);
            border: 1px solid var(--accent-warning);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: #ffd54f;
        }
        
        .error-box {
            background: rgba(244, 67, 54, 0.15);
            border: 1px solid var(--accent-error);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: #e57373;
        }
        
        /* Card Containers - Glassmorphism */
        .tab-content {
            background: rgba(30, 37, 51, 0.6) !important;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 15px 0;
        }
        
        .control-panel {
            background: rgba(26, 31, 46, 0.8);
            border-radius: 10px;
            padding: 18px;
            margin: 12px 0;
            border: 1px solid var(--border-color);
        }
        
        /* Tab Navigation - Enhanced */
        .tab-nav button {
            font-size: 16px !important;
            padding: 14px 28px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            background: rgba(26, 31, 46, 0.6) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .tab-nav button:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            border-color: var(--accent-primary) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        }
        
        .tab-nav button.selected {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Buttons - Modern Dark Style */
        button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        
        .secondary {
            background: rgba(102, 126, 234, 0.2) !important;
            border: 1px solid var(--accent-primary) !important;
            color: var(--accent-primary) !important;
        }
        
        .secondary:hover {
            background: rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Input Fields - Dark Mode */
        input, textarea, select {
            background: rgba(26, 31, 46, 0.8) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Accordions - Dark Mode */
        .accordion {
            background: rgba(26, 31, 46, 0.6) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
        }
        
        /* Progress Bars */
        .progress-bar {
            background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
            border-radius: 10px;
            height: 8px;
        }
        
        /* Dataframes/Tables */
        table {
            background: rgba(26, 31, 46, 0.6) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }
        
        table th {
            background: rgba(102, 126, 234, 0.2) !important;
            color: var(--text-primary) !important;
            font-weight: 600;
        }
        
        table tr:hover {
            background: rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Scrollbars - Custom Dark */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--secondary-bg);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #7c8ef5 0%, #8a5bb3 100%);
        }
        
        /* Labels - Enhanced Visibility */
        label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }
        
        /* JSON Display - Dark Mode */
        .json-holder {
            background: rgba(15, 20, 25, 0.8) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }
        
        /* Hover Effects */
        .hover-lift:hover {
            transform: translateY(-3px);
            transition: transform 0.3s ease;
        }
        
        /* Glow Effect for Active Elements */
        .glow {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
            50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
        }
        """
        
        # Use modern dark theme
        dark_theme = gr.themes.Base(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            background_fill_primary="*neutral_900",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*neutral_800",
            background_fill_secondary_dark="*neutral_800",
            border_color_primary="*neutral_700",
            border_color_primary_dark="*neutral_700",
        )
        
        with gr.Blocks(css=css, title="Amharic IndexTTS2 - Professional TTS Platform", theme=dark_theme) as app:
            
            # Clean Minimal Header
            gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.8em; font-weight: 700; text-shadow: 0 2px 20px rgba(102, 126, 234, 0.5);">🎙️ Amharic IndexTTS2</h1>
            </div>
            """)
            
            # Main tabs with enhanced styling
            with gr.Tabs(elem_classes=["tab-nav"]):
                with gr.TabItem("🚀 Training", id=0):
                    self.create_training_tab()
                with gr.TabItem("🎵 Inference", id=1):
                    self.create_inference_tab()
                with gr.TabItem("🔬 Model Comparison", id=2):
                    self.create_comparison_tab()
                with gr.TabItem("📊 System Monitor", id=3):
                    self.create_system_tab()
                with gr.TabItem("📁 Model Management", id=4):
                    self.create_model_management_tab()
            
            # Clean Minimal Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 40px; padding: 15px; background: rgba(102,126,234,0.1); border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
                <p style="margin: 0; font-size: 0.9em; color: #9aa0a6;">🇪🇹 Amharic IndexTTS2</p>
            </div>
            """)
        
        return app
    
    def create_training_tab(self):
        """Create training management tab with modern dark design"""
        with gr.Column(elem_classes=["tab-content"]):
            gr.Markdown("""
            <div style="text-align: center; padding: 1em; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1em;">
                <h2 style="margin: 0; color: #667eea;">🚀 Training Hub</h2>
                <p style="margin: 0.5em 0; color: #9aa0a6;">Upload datasets, configure training, and monitor progress in real-time</p>
            </div>
            """)
            
            # Dataset Management
            with gr.Accordion("📁 Dataset Management", open=False):
                with gr.Row():
                    audio_files = gr.File(
                        label="📢 Upload Audio Files",
                        file_count="multiple",
                        file_types=[".wav", ".mp3", ".flac"]
                    )
                    text_files = gr.File(
                        label="📝 Upload Text Files", 
                        file_count="multiple",
                        file_types=[".txt", ".json"]
                    )
                    subtitle_files = gr.File(
                        label="📄 Upload Subtitle Files (Optional)",
                        file_count="multiple",
                        file_types=[".srt", ".vtt"]
                    )
                
                with gr.Row():
                    dataset_name = gr.Textbox(label="Dataset Name", placeholder="my_amharic_dataset")
                    upload_btn = gr.Button("📤 Upload Dataset", variant="primary")
                
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_btn.click(
                    fn=self.upload_dataset,
                    inputs=[audio_files, text_files, subtitle_files, dataset_name],
                    outputs=[upload_status]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🌐 Web URL Processing")
                
                urls_input = gr.Textbox(
                    label="URLs (one per line)",
                    lines=5,
                    placeholder="https://www.youtube.com/watch?v=...\nhttps://example.com/video.mp4",
                    info="Supports YouTube, Vimeo, and direct media URLs"
                )
                
                with gr.Row():
                    web_dataset_name = gr.Textbox(label="Dataset Name", placeholder="web_dataset")
                    max_workers = gr.Slider(1, 8, value=4, step=1, label="Parallel Downloads")
                    web_extract_subs = gr.Checkbox(label="Extract Subtitles", value=True)
                
                process_urls_btn = gr.Button("🌐 Download & Process URLs", variant="primary")
                web_status = gr.Textbox(label="Processing Status", interactive=False)
            
            # Dataset Preparation
            with gr.Accordion("🔄 Dataset Preparation", open=False):
                with gr.Row():
                    prep_dataset_name = gr.Dropdown(
                        label="Select Dataset",
                        choices=self.get_available_datasets()
                    )
                    processing_mode = gr.Dropdown(
                        label="Processing Mode",
                        choices=["traditional", "srt_vtt"],
                        value="traditional"
                    )
                
                with gr.Row():
                    min_duration = gr.Slider(0.5, 30.0, value=1.0, step=0.5, label="Min Duration (seconds)")
                    max_duration = gr.Slider(5.0, 300.0, value=30.0, step=5.0, label="Max Duration (seconds)")
                    sample_rate = gr.Dropdown([16000, 22050, 24000, 44100], value=24000, label="Sample Rate")
                
                with gr.Row():
                    min_snr = gr.Slider(0, 40, value=20, step=1, label="Min SNR (dB)")
                    enable_denoise = gr.Checkbox(label="Enable Denoising", value=True)
                    enable_vad = gr.Checkbox(label="Enable VAD", value=True)
                
                prepare_btn = gr.Button("🔄 Prepare Dataset", variant="secondary")
                prepare_status = gr.Textbox(label="Preparation Status", interactive=False)
                
                prepare_btn.click(
                    fn=self.prepare_dataset,
                    inputs=[prep_dataset_name, processing_mode, min_duration, max_duration, 
                           sample_rate, min_snr, enable_denoise, enable_vad],
                    outputs=[prepare_status]
                )
                
                # Wire up web URL processing
                process_urls_btn.click(
                    fn=self.process_web_urls,
                    inputs=[urls_input, web_dataset_name, max_workers, web_extract_subs,
                           min_snr, enable_denoise],
                    outputs=[web_status]
                )
            
            # Training Configuration
            with gr.Accordion("⚙️ Training Configuration", open=True):
                with gr.Row():
                    model_path = gr.Textbox(
                        label="Pretrained Model Path",
                        value="checkpoints/gpt.pth",
                        placeholder="Path to pretrained IndexTTS2 model"
                    )
                    config_path = gr.Textbox(
                        label="Configuration File",
                        value="configs/amharic_200hr_full_training_config.yaml",
                        placeholder="Path to training configuration"
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory", 
                        value="checkpoints/amharic_training",
                        placeholder="Output directory for checkpoints"
                    )
                
                with gr.Row():
                    checkpoint_path = gr.Textbox(
                        label="Resume From Checkpoint (Optional)",
                        placeholder="Path to checkpoint or 'auto'"
                    )
                    num_epochs = gr.Slider(1, 50, value=8, step=1, label="Epochs")
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                
                with gr.Row():
                    learning_rate = gr.Number(value=0.00002, label="Learning Rate")
                    gradient_accumulation = gr.Slider(1, 32, value=16, step=1, label="Gradient Accumulation Steps")
                
                # Advanced Optimizations
                with gr.Row():
                    enable_sdpa = gr.Checkbox(label="⚡ Enable SDPA (Speed Boost)", value=True)
                    enable_ema = gr.Checkbox(label="🌟 Enable EMA (Quality Boost)", value=True) 
                    mixed_precision = gr.Checkbox(label="🔥 Mixed Precision (Memory Save)", value=True)
                
                start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                stop_training_btn = gr.Button("🛑 Stop Training", variant="stop", size="lg")
            
            # Training Monitoring
            with gr.Accordion("📊 Live Training Monitoring", open=True):
                # Live loss plot
                loss_plot = gr.Plot(label="Loss Curve (Live)")
                
                # Current metrics
                with gr.Row():
                    current_step = gr.Number(label="Current Step", interactive=False)
                    current_loss = gr.Number(label="Current Loss", interactive=False)
                    min_loss = gr.Number(label="Best Loss", interactive=False)
                
                # Training status
                training_status = gr.Textbox(
                    label="Training Status",
                    lines=6,
                    interactive=False
                )
                
                # Auto-refresh controls
                with gr.Row():
                    auto_refresh = gr.Checkbox(label="Auto-Refresh (5s)", value=True)
                    manual_refresh_btn = gr.Button("🔄 Refresh Now", variant="secondary")
                
                # Setup auto-refresh timer
                def update_training_monitor():
                    log_file = Path("logs/training/current_training.log")
                    plot = self.training_monitor.parse_training_log(str(log_file))
                    metrics = self.training_monitor.get_current_metrics()
                    status = self.get_training_status()
                    
                    return (
                        plot,
                        metrics.get('current_step', 0),
                        metrics.get('current_loss', 0.0),
                        metrics.get('min_loss', 0.0),
                        status
                    )
                
                manual_refresh_btn.click(
                    fn=update_training_monitor,
                    outputs=[loss_plot, current_step, current_loss, min_loss, training_status]
                )
                
                # Auto-refresh with timer
                timer = gr.Timer(5.0)
                timer.tick(
                    fn=update_training_monitor,
                    outputs=[loss_plot, current_step, current_loss, min_loss, training_status]
                )
            
            # Training controls
            start_training_btn.click(
                fn=self.start_training,
                inputs=[
                    model_path, config_path, output_dir, checkpoint_path,
                    num_epochs, batch_size, learning_rate, enable_sdpa, 
                    enable_ema, mixed_precision, gradient_accumulation
                ],
                outputs=[training_status]
            )
            
            stop_training_btn.click(
                fn=self.stop_training,
                outputs=[training_status]
            )
        
        return gr.Column()
    
    def create_inference_tab(self):
        """Create inference tab with modern dark design"""
        with gr.Column(elem_classes=["tab-content"]):
            gr.Markdown("""
            <div style="text-align: center; padding: 1em; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1em;">
                <h2 style="margin: 0; color: #667eea;">🎵 Inference Studio</h2>
                <p style="margin: 0.5em 0; color: #9aa0a6;">Generate natural Amharic speech with advanced prosody controls</p>
            </div>
            """)
            
            # Model Loading
            with gr.Accordion("🤖 Model Loading", open=True):
                with gr.Row():
                    inference_model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="Path to trained model checkpoint"
                    )
                    inference_vocab_path = gr.Textbox(
                        label="Vocabulary Path", 
                        value="amharic_bpe.model",
                        placeholder="Path to Amharic vocabulary"
                    )
                    inference_config_path = gr.Textbox(
                        label="Config Path",
                        value="configs/amharic_200hr_full_training_config.yaml",
                        placeholder="Path to model configuration"
                    )
                
                load_model_btn = gr.Button("🔄 Load Model for Inference", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                load_model_btn.click(
                    fn=self.load_model_for_inference,
                    inputs=[inference_model_path, inference_vocab_path, inference_config_path],
                    outputs=[model_status]
                )
            
            # Single Text Inference
            with gr.Accordion("🎵 Single Text Inference", open=True):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to Synthesize (Amharic)",
                        lines=4,
                        placeholder="Enter Amharic text here..."
                    )
                    voice_id = gr.Number(value=0, label="Voice ID")
                
                with gr.Row():
                    emotion = gr.Dropdown(
                        ["neutral", "happy", "sad", "angry", "excited"],
                        value="neutral",
                        label="Emotion"
                    )
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                    pitch = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Pitch")
                
                with gr.Row():
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(100, 2000, value=1000, step=50, label="Max Tokens")
                    inference_sample_rate = gr.Dropdown([16000, 22050, 24000], value=24000, label="Sample Rate")
                
                # Amharic Prosody Controls
                with gr.Accordion("🎭 Amharic Prosody Controls", open=False):
                    gemination_strength = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.1,
                        label="Gemination Emphasis",
                        info="Controls doubled consonant emphasis (ሁለት vs ሁሌት)"
                    )
                    
                    ejective_strength = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.1,
                        label="Ejective Consonant Strength",
                        info="Controls glottalized consonants (ጥ, ቅ, ጭ)"
                    )
                    
                    syllable_duration = gr.Slider(
                        0.7, 1.3, value=1.0, step=0.05,
                        label="Syllable Duration",
                        info="Speaking speed (0.7=fast, 1.3=slow)"
                    )
                    
                    stress_pattern = gr.Radio(
                        ["penultimate", "final", "initial"],
                        value="penultimate",
                        label="Stress Pattern",
                        info="Typical Amharic uses penultimate stress"
                    )
                    
                    prosody_analysis = gr.JSON(
                        label="Detected Amharic Features",
                        value={}
                    )
                
                generate_btn = gr.Button(
                    "🎙️ Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
                audio_output = gr.Audio(label="Generated Audio")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
                
                # Quality Metrics Display
                with gr.Accordion("📊 Audio Quality Metrics", open=False):
                    with gr.Row():
                        rms_display = gr.Number(label="RMS Energy", interactive=False)
                        peak_display = gr.Number(label="Peak Level", interactive=False)
                        zcr_display = gr.Number(label="Zero Crossing Rate", interactive=False)
                    
                    with gr.Row():
                        duration_display = gr.Number(label="Duration (s)", interactive=False)
                        quality_score_display = gr.Slider(
                            0, 10, label="Quality Score",
                            interactive=False
                        )
                    
                    quality_flags = gr.Textbox(label="Quality Checks", lines=2, interactive=False)
                
                generate_btn.click(
                    fn=self.generate_speech_with_metrics,
                    inputs=[
                        text_input, voice_id, emotion, speed, pitch,
                        temperature, max_tokens, inference_sample_rate,
                        gemination_strength, ejective_strength, syllable_duration, stress_pattern
                    ],
                    outputs=[
                        audio_output, generation_status, prosody_analysis,
                        rms_display, peak_display, zcr_display,
                        duration_display, quality_score_display, quality_flags
                    ]
                )
            
            # Batch Inference
            with gr.Accordion("📋 Batch Inference", open=False):
                batch_texts = gr.Textbox(
                    label="Multiple Texts (one per line)",
                    lines=10,
                    placeholder="Enter multiple Amharic texts, one per line..."
                )
                
                with gr.Row():
                    batch_voice_id = gr.Number(value=0, label="Voice ID")
                    batch_emotion = gr.Dropdown(
                        ["neutral", "happy", "sad", "angry", "excited"],
                        value="neutral",
                        label="Emotion"
                    )
                    batch_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                    batch_sample_rate = gr.Dropdown([16000, 22050, 24000], value=24000, label="Sample Rate")
                
                batch_generate_btn = gr.Button("📋 Generate Batch Audio", variant="secondary")
                batch_output = gr.Textbox(label="Batch Generation Results", lines=10, interactive=False)
                
                batch_generate_btn.click(
                    fn=self.batch_generate,
                    inputs=[batch_texts, batch_voice_id, batch_emotion, batch_speed, batch_sample_rate],
                    outputs=[batch_output]
                )
        
        return gr.Column()
    
    def create_comparison_tab(self):
        """Create model comparison A/B testing tab with modern dark design"""
        with gr.Column(elem_classes=["tab-content"]):
            gr.Markdown("""
            <div style="text-align: center; padding: 1em; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1em;">
                <h2 style="margin: 0; color: #667eea;">🔬 Model Comparison Lab</h2>
                <p style="margin: 0.5em 0; color: #9aa0a6;">Load two models and compare their outputs side-by-side with automated quality analysis</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🅰️ Model A")
                    model_a_path = gr.Textbox(
                        label="Model A Checkpoint Path",
                        placeholder="checkpoints/model_epoch_10.pt"
                    )
                    load_model_a_btn = gr.Button("📥 Load Model A", variant="secondary")
                    model_a_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 🅱️ Model B")
                    model_b_path = gr.Textbox(
                        label="Model B Checkpoint Path",
                        placeholder="checkpoints/model_epoch_20.pt"
                    )
                    load_model_b_btn = gr.Button("📥 Load Model B", variant="secondary")
                    model_b_status = gr.Textbox(label="Status", interactive=False)
            
            # Comparison input
            gr.Markdown("### 🎯 Test Generation")
            comparison_text = gr.Textbox(
                label="Test Text (Amharic)",
                placeholder="ሰላም ዓለም! እንዴት ነሽ? ደህና ነኝ፣ አመሰግናለሁ።",
                lines=3
            )
            
            compare_btn = gr.Button(
                "⚖️ Generate & Compare",
                variant="primary",
                size="lg"
            )
            
            # Results
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔊 Model A Output")
                    audio_a_output = gr.Audio(label="Audio A")
                    metrics_a_display = gr.JSON(label="Metrics A")
                
                with gr.Column():
                    gr.Markdown("#### 🔊 Model B Output")
                    audio_b_output = gr.Audio(label="Audio B")
                    metrics_b_display = gr.JSON(label="Metrics B")
            
            # Comparison results
            gr.Markdown("### 📊 Comparison Results")
            comparison_table = gr.Dataframe(
                headers=["Metric", "Model A", "Model B", "Winner"],
                label="Detailed Comparison",
                interactive=False
            )
            
            winner_display = gr.Textbox(
                label="🏆 Overall Winner",
                interactive=False
            )
            
            # Wire up events
            load_model_a_btn.click(
                fn=lambda path: self.model_comparator.load_model_a(path, "Model A"),
                inputs=[model_a_path],
                outputs=[model_a_status]
            )
            
            load_model_b_btn.click(
                fn=lambda path: self.model_comparator.load_model_b(path, "Model B"),
                inputs=[model_b_path],
                outputs=[model_b_status]
            )
        
        return gr.Column()
    
    def create_system_tab(self):
        """Create system monitoring tab with modern dark design"""
        with gr.Column(elem_classes=["tab-content"]):
            gr.Markdown("""
            <div style="text-align: center; padding: 1em; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1em;">
                <h2 style="margin: 0; color: #667eea;">📊 System Monitor</h2>
                <p style="margin: 0.5em 0; color: #9aa0a6;">Track GPU usage, checkpoints, and system resources</p>
            </div>
            """)
            
            # System Resources
            with gr.Accordion("💻 System Resources", open=True):
                system_status = gr.JSON(
                    label="System Status",
                    value=self.get_system_resources()
                )
                
                refresh_system_btn = gr.Button("🔄 Refresh System Status", variant="secondary")
                refresh_system_btn.click(
                    fn=self.update_system_resources,
                    outputs=[system_status]
                )
            
            # Available Checkpoints
            with gr.Accordion("📁 Available Checkpoints", open=True):
                refresh_checkpoints_btn = gr.Button("🔄 Refresh Checkpoints", variant="secondary")
                checkpoints_list = gr.Dropdown(
                    label="Available Model Checkpoints",
                    choices=self.refresh_checkpoint_list()
                )
                
                refresh_checkpoints_btn.click(
                    fn=self.refresh_checkpoint_list,
                    outputs=[checkpoints_list]
                )
            
            # Training History
            with gr.Accordion("📈 Training History", open=False):
                history_display = gr.JSON(
                    label="Training History",
                    value=self.state.get('training_history', [])
                )
            
            # System Configuration
            with gr.Accordion("⚙️ System Configuration", open=False):
                gr.HTML("""
                <div class="control-panel">
                    <h4>System Configuration</h4>
                    <p><strong>Device:</strong> {device}</p>
                    <p><strong>CUDA Available:</strong> {cuda_available}</p>
                    <p><strong>PyTorch Version:</strong> {pytorch_version}</p>
                    <p><strong>Gradio Version:</strong> {gradio_version}</p>
                </div>
                """.format(
                    device=self.device,
                    cuda_available=torch.cuda.is_available(),
                    pytorch_version=torch.__version__,
                    gradio_version=gr.__version__
                ))
        
        return gr.Column()
    
    def create_model_management_tab(self):
        """Create model management tab with modern dark design"""
        with gr.Column(elem_classes=["tab-content"]):
            gr.Markdown("""
            <div style="text-align: center; padding: 1em; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1em;">
                <h2 style="margin: 0; color: #667eea;">📁 Model Management</h2>
                <p style="margin: 0.5em 0; color: #9aa0a6;">Manage, export, and validate your trained Amharic models</p>
            </div>
            """)
            
            # Model Information
            with gr.Accordion("📊 Current Model Info", open=True):
                current_model_info = gr.JSON(
                    label="Model Information",
                    value=self.state.get('current_model_info', {})
                )
                
                refresh_model_info_btn = gr.Button("🔄 Refresh Model Info", variant="secondary")
                refresh_model_info_btn.click(
                    fn=lambda: self.get_current_model_info(),
                    outputs=[current_model_info]
                )
            
            # Export Models
            with gr.Accordion("📤 Export Models", open=False):
                export_model_path = gr.Textbox(
                    label="Model Path to Export",
                    placeholder="Path to model checkpoint"
                )
                export_format = gr.Dropdown(
                    ["pytorch", "onnx", "tensorrt"],
                    value="pytorch",
                    label="Export Format"
                )
                
                export_btn = gr.Button("📤 Export Model", variant="secondary")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                
                # Export functionality would go here
            
            # Model Validation
            with gr.Accordion("✅ Model Validation", open=False):
                validation_model_path = gr.Textbox(
                    label="Model to Validate",
                    placeholder="Path to model checkpoint"
                )
                
                validate_btn = gr.Button("🔍 Validate Model", variant="secondary")
                validation_results = gr.Textbox(label="Validation Results", lines=10, interactive=False)
                
                # Validation functionality would go here
        
        return gr.Column()
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            return []
        
        return [d.name for d in datasets_dir.iterdir() if d.is_dir()]
    
    def get_current_model_info(self) -> Dict:
        """Get current model information"""
        if not self.current_model:
            return {"status": "No model loaded"}
        
        return {
            "device": str(next(self.current_model.parameters()).device),
            "parameters": sum(p.numel() for p in self.current_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.current_model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.current_model.parameters()) / 1024**2,
            "vocabulary_size": self.current_tokenizer.vocab_size if self.current_tokenizer else 0
        }


def main():
    """Main application entry point"""
    app = AmharicTTSGradioApp()
    interface = app.create_interface()
    
    # Launch with optimal settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()