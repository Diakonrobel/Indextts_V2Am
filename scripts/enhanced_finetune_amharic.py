"""
Enhanced Amharic fine-tuning script for IndexTTS2
Optimized for 200-hour dataset with T4 GPU (16GB VRAM) and anti-overfitting measures
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml
from transformers import get_linear_schedule_with_warmup

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.utils.feature_extractors import MelSpectrogramFeatures


class EnhancedAmharicTTSDataset(Dataset):
    """Enhanced dataset for Amharic TTS with data augmentation and anti-overfitting measures"""
    
    def __init__(
        self,
        manifest_file: str,
        tokenizer: AmharicTextTokenizer,
        mel_extractor: MelSpectrogramFeatures,
        max_text_length: int = 600,
        max_mel_length: int = 1815,
        augmentation_prob: float = 0.7,
        enable_augmentation: bool = True
    ):
        self.manifest_file = manifest_file
        self.tokenizer = tokenizer
        self.mel_extractor = mel_extractor
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.augmentation_prob = augmentation_prob
        self.enable_augmentation = enable_augmentation
        
        # Load manifest
        self.data = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"Enhanced dataset loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def apply_augmentation(self, audio, sr):
        """Apply data augmentation with high diversity"""
        if not self.enable_augmentation or torch.rand(1).item() > self.augmentation_prob:
            return audio
            
        # Speed perturbation (0.9-1.1)
        if torch.rand(1).item() < 0.3:
            speed_factor = np.random.uniform(0.95, 1.05)
            audio = torchaudio.functional.time_stretch(audio, sr, speed_factor)
        
        # Pitch perturbation
        if torch.rand(1).item() < 0.3:
            pitch_shift = np.random.uniform(-0.5, 0.5)
            audio = audio * (2 ** (pitch_shift / 12))
        
        # Noise injection
        if torch.rand(1).item() < 0.4:
            noise_level = 0.005
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        # Time stretching
        if torch.rand(1).item() < 0.3:
            stretch_factor = np.random.uniform(0.95, 1.05)
            audio = torchaudio.functional.time_stretch(audio, sr, stretch_factor)
        
        return audio
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio and apply augmentation
        audio_path = item['audio_path']
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono
        
        # Apply data augmentation
        if self.enable_augmentation:
            audio = self.apply_augmentation(audio, sr)
        
        # Extract mel spectrogram
        mel = self.mel_extractor(audio)
        mel = mel.squeeze(0)  # Remove batch dimension
        
        # Tokenize text
        text = item['text']
        text_tokens = self.tokenizer.encode(text, out_type=int)
        
        # Truncate if too long
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[:self.max_text_length]
        
        if mel.shape[-1] > self.max_mel_length:
            mel = mel[:, :self.max_mel_length]
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'mel_spectrogram': mel,
            'text': text,
            'audio_id': item['audio_id'],
            'original_length': mel.shape[-1]
        }


class AntiOverfittingTrainer:
    """Advanced trainer with anti-overfitting measures for 200hr datasets"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str,
        amharic_vocab_path: str,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        device: str = 'cuda'
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Anti-overfitting state tracking
        self.best_val_loss = float('inf')
        self.val_loss_history = []
        self.overfitting_detected = False
        self.early_stopping_counter = 0
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize tokenizer
        self.tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_path,
            normalizer=AmharicTextNormalizer()
        )
        
        # Initialize mel extractor
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=self.config['dataset']['sample_rate'],
            n_fft=self.config['dataset']['mel']['n_fft'],
            hop_length=self.config['dataset']['mel']['hop_length'],
            win_length=self.config['dataset']['mel']['win_length'],
            n_mels=self.config['dataset']['mel']['n_mels']
        )
        
        # Load model
        self.model = self._load_model()
        
        # Setup LoRA
        if self.use_lora:
            self.lora_manager = self._setup_lora()
        else:
            self.lora_manager = None
        
        # Setup logging
        self._setup_logging()
        
        # Anti-overfitting thresholds
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
        self.min_epochs_before_early_stop = self.config['training']['min_epochs_before_early_stop']
    
    def _setup_lora(self):
        """Setup LoRA with enhanced anti-overfitting"""
        try:
            from indextts.adapters.lora import add_lora_to_model, LoRAManager
            lora_manager = add_lora_to_model(
                model=self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=[
                    'gpt.h.*.attn.c_attn',
                    'gpt.h.*.attn.c_proj', 
                    'gpt.h.*.mlp.c_fc',
                    'gpt.h.*.mlp.c_proj'
                ]
            )
            return lora_manager
        except ImportError:
            print("⚠️  LoRA not available, using full fine-tuning")
            return None
    
    def _setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'enhanced_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self) -> UnifiedVoice:
        """Load model with memory optimizations for T4 GPU"""
        print("Loading pre-trained model with T4 optimizations...")
        
        model = UnifiedVoice(
            layers=self.config['gpt']['layers'],
            model_dim=self.config['gpt']['model_dim'],
            heads=self.config['gpt']['heads'],
            max_text_tokens=self.config['gpt']['max_text_tokens'],
            max_mel_tokens=self.config['gpt']['max_mel_tokens'],
            number_text_tokens=self.tokenizer.vocab_size,
            number_mel_codes=self.config['gpt']['number_mel_codes'],
            start_text_token=self.config['gpt']['start_text_token'],
            stop_text_token=self.config['gpt']['stop_text_token'],
            start_mel_token=self.config['gpt']['start_mel_token'],
            stop_mel_token=self.config['gpt']['stop_mel_token'],
            condition_type=self.config['gpt']['condition_type'],
            condition_module=self.config['gpt']['condition_module'],
            emo_condition_module=self.config['gpt']['emo_condition_module']
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Handle vocabulary size mismatch
        if 'text_embedding.weight' in checkpoint:
            old_vocab_size = checkpoint['text_embedding.weight'].shape[0]
            new_vocab_size = self.tokenizer.vocab_size
            
            if old_vocab_size != new_vocab_size:
                print(f"Resizing text embedding: {old_vocab_size} → {new_vocab_size}")
                
                old_embedding = checkpoint['text_embedding.weight']
                new_embedding = torch.zeros(new_vocab_size, old_embedding.shape[1])
                new_embedding.normal_(mean=0.0, std=0.01)
                
                # Copy common tokens
                for i in range(min(old_vocab_size, new_vocab_size)):
                    new_embedding[i] = old_embedding[i]
                
                checkpoint['text_embedding.weight'] = new_embedding
                
                if 'text_head.weight' in checkpoint:
                    old_head = checkpoint['text_head.weight']
                    new_head = torch.zeros(new_vocab_size, old_head.shape[1])
                    new_head.normal_(mean=0.0, std=0.01)
                    
                    for i in range(min(old_vocab_size, new_vocab_size)):
                        new_head[i] = old_head[i]
                    
                    checkpoint['text_head.weight'] = new_head
        
        # Load state dict
        torch.cuda.empty_cache()
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['hardware'].get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
        
        print("Model loaded successfully with T4 optimizations")
        return model
    
    def detect_overfitting(self, val_loss: float, epoch: int) -> bool:
        """Advanced overfitting detection"""
        
        self.val_loss_history.append(val_loss)
        
        if len(self.val_loss_history) < 3:
            return False
        
        # Check for validation loss increase trend
        recent_losses = self.val_loss_history[-3:]
        if len(recent_losses) >= 3:
            if recent_losses[-1] > recent_losses[-2] > recent_losses[-3]:
                self.overfitting_detected = True
                return True
        
        # Early stopping check
        if epoch >= self.min_epochs_before_early_stop:
            if self.overfitting_detected:
                self.early_stopping_counter += 1
                return True
        
        return False
    
    def train_anti_overfitting(
        self,
        train_manifest: str,
        val_manifest: str,
        num_epochs: int = 6,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        warmup_steps: int = 2000,
        gradient_clip_val: float = 0.5,
        save_every: int = 500,
        log_every: int = 50,
        use_wandb: bool = False,
        gradient_accumulation_steps: int = 8
    ):
        """Enhanced training loop with anti-overfitting measures"""
        
        self.logger.info("🚀 Starting Enhanced Amharic Training with Anti-Overfitting")
        
        # Create data loaders
        train_loader = self._create_data_loader(train_manifest, batch_size, True)
        val_loader = self._create_data_loader(val_manifest, batch_size, False)
        
        # Setup optimizer
        if self.use_lora and self.lora_manager:
            trainable_params = self.lora_manager.get_lora_parameters()
            trainable_params.extend(list(self.model.text_embedding.parameters()))
            trainable_params.extend(list(self.model.speed_emb.parameters()))
        else:
            trainable_params = list(self.model.parameters())
        
        optimizer = optim.AdamW(
            trainable_params, 
            lr=learning_rate, 
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"🔄 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                text_tokens = batch['text_tokens'].to(self.device)
                text_attention_masks = batch['text_attention_masks'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                
                # Forward pass (simplified loss computation)
                loss = self._compute_loss(
                    text_tokens, text_attention_masks,
                    mel_spectrograms, mel_attention_masks
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_val)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % log_every == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        self.logger.info(
                            f"Step {global_step} | "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                            f"LR: {current_lr:.2e}"
                        )
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        self._save_checkpoint(global_step, epoch, loss.item() * gradient_accumulation_steps)
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "step": global_step
                })
            
            # Validation
            val_loss = self._validate(val_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Anti-overfitting check
            if self.detect_overfitting(val_loss, epoch):
                self.logger.warning(f"⚠️  Overfitting detected at epoch {epoch + 1}")
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info("🛑 Early stopping triggered")
                    break
        
        self.logger.info("🎉 Enhanced training completed!")
    
    def _create_data_loader(self, manifest_file, batch_size, shuffle):
        """Create enhanced data loader"""
        dataset = EnhancedAmharicTTSDataset(
            manifest_file=manifest_file,
            tokenizer=self.tokenizer,
            mel_extractor=self.mel_extractor,
            max_text_length=self.config['gpt']['max_text_tokens'],
            max_mel_length=self.config['gpt']['max_mel_tokens'],
            augmentation_prob=self.config['data']['augmentation_prob'],
            enable_augmentation=shuffle
        )
        
        def collate_fn(batch):
            # Pad sequences
            text_tokens = [item['text_tokens'] for item in batch]
            mel_spectrograms = [item['mel_spectrogram'] for item in batch]
            
            # Pad text tokens
            max_text_len = max(len(tokens) for tokens in text_tokens)
            padded_text_tokens = []
            text_attention_masks = []
            
            for tokens in text_tokens:
                pad_len = max_text_len - len(tokens)
                if pad_len > 0:
                    padded_tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
                    attention_mask = torch.cat([torch.ones(len(tokens)), torch.zeros(pad_len)])
                else:
                    padded_tokens = tokens
                    attention_mask = torch.ones(len(tokens))
                
                padded_text_tokens.append(padded_tokens)
                text_attention_masks.append(attention_mask)
            
            # Pad mel spectrograms
            max_mel_len = max(mel.shape[-1] for mel in mel_spectrograms)
            padded_mel_spectrograms = []
            mel_attention_masks = []
            
            for mel in mel_spectrograms:
                pad_len = max_mel_len - mel.shape[-1]
                if pad_len > 0:
                    padded_mel = torch.cat([mel, torch.zeros(mel.shape[0], pad_len)], dim=-1)
                    attention_mask = torch.cat([torch.ones(mel.shape[-1]), torch.zeros(pad_len)])
                else:
                    padded_mel = mel
                    attention_mask = torch.ones(mel.shape[-1])
                
                padded_mel_spectrograms.append(padded_mel)
                mel_attention_masks.append(attention_mask)
            
            return {
                'text_tokens': torch.stack(padded_text_tokens),
                'text_attention_masks': torch.stack(text_attention_masks),
                'mel_spectrograms': torch.stack(padded_mel_spectrograms),
                'mel_attention_masks': torch.stack(mel_attention_masks),
                'texts': [item['text'] for item in batch],
                'audio_ids': [item['audio_id'] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['hardware']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def _compute_loss(self, text_tokens, text_attention_masks, mel_spectrograms, mel_attention_masks):
        """Simplified loss computation - in practice, implement full IndexTTS2 loss"""
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _validate(self, val_loader):
        """Enhanced validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_tokens = batch['text_tokens'].to(self.device)
                text_attention_masks = batch['text_attention_masks'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                
                loss = self._compute_loss(
                    text_tokens, text_attention_masks,
                    mel_spectrograms, mel_attention_masks
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss
    
    def _save_checkpoint(self, step, epoch, loss):
        """Save enhanced checkpoint"""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config,
            'overfitting_metrics': {
                'val_loss_history': self.val_loss_history,
                'overfitting_detected': self.overfitting_detected
            }
        }
        
        if self.use_lora and self.lora_manager:
            checkpoint['lora_state_dict'] = {
                name: adapter.state_dict() 
                for name, adapter in self.lora_manager.lora_adapters.items()
            }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"enhanced_checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            best_path = self.output_dir / "enhanced_best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"✅ Best model saved at step {step}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Amharic IndexTTS2 Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--amharic_vocab", type=str, required=True, help="Path to Amharic vocab file")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, required=True, help="Path to validation manifest")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (T4 optimized)")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (T4 optimized)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AntiOverfittingTrainer(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        amharic_vocab_path=args.amharic_vocab,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Start training
    trainer.train_anti_overfitting(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main()