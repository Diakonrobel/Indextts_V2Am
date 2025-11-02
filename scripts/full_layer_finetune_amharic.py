"""
Full Layer Training Script for IndexTTS2 Amharic
Optimized for 200-hour dataset with T4 GPU (16GB VRAM) - ALL LAYERS TRAINING
NO LoRA - Based on community evidence that LoRA fails for new languages
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


class FullLayerAmharicTTSDataset(Dataset):
    """Dataset for full layer training with enhanced augmentation"""
    
    def __init__(
        self,
        manifest_file: str,
        tokenizer: AmharicTextTokenizer,
        mel_extractor: MelSpectrogramFeatures,
        max_text_length: int = 600,
        max_mel_length: int = 1815,
        augmentation_prob: float = 0.8,  # Higher for full training
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
        
        print(f"Full layer dataset loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def apply_enhanced_augmentation(self, audio, sr):
        """Enhanced augmentation for full layer training"""
        if not self.enable_augmentation or torch.rand(1).item() > self.augmentation_prob:
            return audio
            
        # Speed perturbation (wider range)
        if torch.rand(1).item() < 0.4:
            speed_factor = np.random.uniform(0.9, 1.1)
            audio = torchaudio.functional.time_stretch(audio, sr, speed_factor)
        
        # Pitch perturbation (wider range)
        if torch.rand(1).item() < 0.4:
            pitch_shift = np.random.uniform(-1.0, 1.0)  # Wider pitch range
            audio = audio * (2 ** (pitch_shift / 12))
        
        # Noise injection
        if torch.rand(1).item() < 0.5:
            noise_level = 0.01
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        # Time stretching
        if torch.rand(1).item() < 0.4:
            stretch_factor = np.random.uniform(0.9, 1.1)
            audio = torchaudio.functional.time_stretch(audio, sr, stretch_factor)
        
        return audio
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio and apply enhanced augmentation
        audio_path = item['audio_path']
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono
        
        # Apply data augmentation
        if self.enable_augmentation:
            audio = self.apply_enhanced_augmentation(audio, sr)
        
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


class FullLayerTrainer:
    """Full layer trainer for IndexTTS2 - NO LoRA, ALL Layers"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str,
        amharic_vocab_path: str,
        device: str = 'cuda',
        mixed_precision: bool = True  # CRITICAL for full training
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Full training state tracking
        self.best_val_loss = float('inf')
        self.val_loss_history = []
        self.overfitting_detected = False
        self.early_stopping_counter = 0
        self.training_stats = {}
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize tokenizer
        self.tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_path,
            normalizer=AmharicTextNormalizer()
        )
        
        # Initialize mel extractor with optimization
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=self.config['dataset']['sample_rate'],
            n_fft=self.config['dataset']['mel']['n_fft'],
            hop_length=self.config['dataset']['mel']['hop_length'],
            win_length=self.config['dataset']['mel']['win_length'],
            n_mels=self.config['dataset']['mel']['n_mels']
        )
        
        # Load model with full training setup
        self.model = self._load_model_full_training()
        
        # Setup enhanced logging
        self._setup_full_training_logging()
        
        # Anti-overfitting thresholds for full training
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
        self.min_epochs_before_early_stop = self.config['training']['min_epochs_before_early_stop']
        
        print("🚀 FULL LAYER TRAINING ENABLED - NO LoRA")
        print("   Community Evidence: LoRA fails for new language adaptation")
        print(f"   All {self.config['gpt']['layers']} layers will be trained")
        print("   Optimized for Amharic script (231+ characters)")
    
    def _setup_full_training_logging(self):
        """Setup logging for full layer training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'full_layer_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model_full_training(self) -> UnifiedVoice:
        """Load model with full layer training setup"""
        print("🔄 Loading pre-trained model for FULL LAYER training...")
        print("   💡 NO LoRA adapters will be used")
        print("   🎯 ALL model parameters will be trained")
        
        # Create model with enhanced parameters
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
        print("📥 Loading pre-trained weights...")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Handle vocabulary size mismatch with full training strategy
        if 'text_embedding.weight' in checkpoint:
            old_vocab_size = checkpoint['text_embedding.weight'].shape[0]
            new_vocab_size = self.tokenizer.vocab_size
            
            if old_vocab_size != new_vocab_size:
                print(f"🔧 Resizing text embedding: {old_vocab_size} → {new_vocab_size}")
                
                old_embedding = checkpoint['text_embedding.weight']
                new_embedding = torch.zeros(new_vocab_size, old_embedding.shape[1])
                # Use Xavier initialization for new embeddings
                nn.init.xavier_uniform_(new_embedding)
                
                # Copy common tokens
                min_size = min(old_vocab_size, new_vocab_size)
                new_embedding[:min_size] = old_embedding[:min_size]
                
                checkpoint['text_embedding.weight'] = new_embedding
                
                # Handle text head
                if 'text_head.weight' in checkpoint:
                    old_head = checkpoint['text_head.weight']
                    new_head = torch.zeros(new_vocab_size, old_head.shape[1])
                    nn.init.xavier_uniform_(new_head)
                    
                    min_size = min(old_vocab_size, new_vocab_size)
                    new_head[:min_size] = old_head[:min_size]
                    
                    checkpoint['text_head.weight'] = new_head
        
        # Load state dict
        torch.cuda.empty_cache()
        model.load_state_dict(checkpoint, strict=False)
        
        # Move to device with mixed precision
        if self.mixed_precision:
            model = model.half()  # Use FP16 for memory efficiency
        
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['hardware'].get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model loaded successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Memory optimization: {'Mixed Precision' if self.mixed_precision else 'Full Precision'}")
        
        return model
    
    def _get_trainable_parameters(self):
        """Get all trainable parameters for full training"""
        # ALL parameters are trainable in full training
        trainable_params = []
        
        # Core model parameters
        trainable_params.extend(list(self.model.parameters()))
        
        # Ensure all parameters require gradients
        for param in trainable_params:
            param.requires_grad = True
        
        print(f"🔢 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        return trainable_params
    
    def detect_overfitting_full_training(self, val_loss: float, epoch: int) -> bool:
        """Enhanced overfitting detection for full layer training"""
        
        self.val_loss_history.append(val_loss)
        
        # Don't check for overfitting too early
        if len(self.val_loss_history) < 5:  # Wait longer for full training
            return False
        
        # Check for validation loss increase trend
        recent_losses = self.val_loss_history[-5:]  # Look at last 5 losses
        if len(recent_losses) >= 5:
            # Check if last 3 losses are all increasing
            if recent_losses[-1] > recent_losses[-2] > recent_losses[-3]:
                self.overfitting_detected = True
                return True
        
        # Early stopping check (with more patience for full training)
        if epoch >= self.min_epochs_before_early_stop:
            if self.overfitting_detected:
                self.early_stopping_counter += 1
                return True
        
        return False
    
    def train_full_layers(
        self,
        train_manifest: str,
        val_manifest: str,
        num_epochs: int = 8,  # Increased for full training
        batch_size: int = 1,  # Reduced for memory
        learning_rate: float = 2e-5,  # Lower for stability
        warmup_steps: int = 3000,  # Increased for full training
        gradient_clip_val: float = 0.3,  # Tighter clipping
        save_every: int = 250,
        log_every: int = 25,
        use_wandb: bool = True,
        gradient_accumulation_steps: int = 16  # Increased for full training
    ):
        """Full layer training loop with enhanced monitoring"""
        
        self.logger.info("🚀 Starting FULL LAYER TRAINING - NO LoRA")
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"   Epochs: {num_epochs}")
        self.logger.info(f"   Batch Size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
        self.logger.info(f"   Learning Rate: {learning_rate}")
        self.logger.info(f"   Gradient Accumulation: {gradient_accumulation_steps}")
        
        # Create data loaders
        train_loader = self._create_full_training_data_loader(train_manifest, batch_size, True)
        val_loader = self._create_full_training_data_loader(val_manifest, batch_size, False)
        
        # Get ALL trainable parameters
        trainable_params = self._get_trainable_parameters()
        
        # Setup optimizer with enhanced settings for full training
        optimizer = optim.AdamW(
            trainable_params, 
            lr=learning_rate, 
            weight_decay=self.config['training']['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)  # Standard AdamW betas
        )
        
        # Enhanced scheduler for full training
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            last_epoch=-1
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        epoch_train_losses = []
        
        self.logger.info(f"🎯 Starting training with {len(train_loader)} batches per epoch")
        
        for epoch in range(num_epochs):
            self.logger.info(f"🔄 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            optimizer.zero_grad()
            
            # Enhanced progress bar
            progress_bar = tqdm(
                train_loader, 
                desc=f"Full Training Epoch {epoch + 1}", 
                leave=False,
                dynamic_ncols=True
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
                text_attention_masks = batch['text_attention_masks'].to(self.device, non_blocking=True)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device, non_blocking=True)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
                    loss = self._compute_full_training_loss(
                        text_tokens, text_attention_masks,
                        mel_spectrograms, mel_attention_masks
                    )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Gradient accumulation with enhanced clipping
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Enhanced gradient clipping for full training
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, 
                        gradient_clip_val,
                        norm_type=2.0,
                        error_if_nonfinite=True
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Enhanced logging
                    if global_step % log_every == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_val)
                        
                        self.logger.info(
                            f"Step {global_step} | "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Grad Norm: {grad_norm:.4f}"
                        )
                        
                        if use_wandb:
                            wandb.log({
                                "train_loss": loss.item() * gradient_accumulation_steps,
                                "learning_rate": current_lr,
                                "gradient_norm": grad_norm.item(),
                                "step": global_step,
                                "epoch": epoch,
                                "batch_idx": batch_idx
                            })
                    
                    # Save checkpoint
                    if global_step % save_every == 0:
                        self._save_full_training_checkpoint(
                            global_step, epoch, 
                            loss.item() * gradient_accumulation_steps
                        )
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "step": global_step
                })
                
                # Memory management
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Store epoch training loss
            avg_train_loss = epoch_loss / len(train_loader)
            epoch_train_losses.append(avg_train_loss)
            
            # Enhanced validation
            val_loss = self._validate_full_training(val_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1} Summary | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            
            # Anti-overfitting check (enhanced for full training)
            if self.detect_overfitting_full_training(val_loss, epoch):
                self.logger.warning(f"⚠️  Overfitting detected at epoch {epoch + 1}")
                self.logger.warning(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info("🛑 Early stopping triggered")
                    break
            
            # Wandb logging
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "overfitting_detected": self.overfitting_detected
                })
        
        # Final save
        self._save_full_training_checkpoint(
            global_step, epoch, val_loss, is_final=True
        )
        
        self.logger.info("🎉 FULL LAYER training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Final statistics
        self._log_training_summary(epoch_train_losses)
    
    def _create_full_training_data_loader(self, manifest_file, batch_size, shuffle):
        """Create data loader optimized for full training"""
        dataset = FullLayerAmharicTTSDataset(
            manifest_file=manifest_file,
            tokenizer=self.tokenizer,
            mel_extractor=self.mel_extractor,
            max_text_length=self.config['gpt']['max_text_tokens'],
            max_mel_length=self.config['gpt']['max_mel_tokens'],
            augmentation_prob=self.config['data']['augmentation_prob'],
            enable_augmentation=shuffle
        )
        
        def full_training_collate_fn(batch):
            """Collate function optimized for memory"""
            text_tokens = [item['text_tokens'] for item in batch]
            mel_spectrograms = [item['mel_spectrogram'] for item in batch]
            
            # Pad text tokens efficiently
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
            collate_fn=full_training_collate_fn,
            pin_memory=True,
            persistent_workers=False  # Save memory
        )
    
    def _compute_full_training_loss(self, text_tokens, text_attention_masks, mel_spectrograms, mel_attention_masks):
        """Simplified loss computation - implement full IndexTTS2 loss in practice"""
        # This would implement the full IndexTTS2 loss function
        # including text-to-mel generation loss, emotion loss, etc.
        
        # For now, return a placeholder loss that requires gradients
        # In practice, this would be the complete IndexTTS2 loss computation
        base_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Add regularization loss
        reg_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                reg_loss += torch.norm(param, p=2) * 0.01
        
        total_loss = base_loss + reg_loss
        return total_loss
    
    def _validate_full_training(self, val_loader):
        """Enhanced validation for full training"""
        self.model.eval()
        total_loss = 0
        validation_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_tokens = batch['text_tokens'].to(self.device)
                text_attention_masks = batch['text_attention_masks'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                
                loss = self._compute_full_training_loss(
                    text_tokens, text_attention_masks,
                    mel_spectrograms, mel_attention_masks
                )
                
                total_loss += loss.item()
                
                # Store metrics for overfitting detection
                validation_metrics.append({
                    'loss': loss.item(),
                    'text_length': text_tokens.shape[1],
                    'mel_length': mel_spectrograms.shape[-1]
                })
        
        avg_loss = total_loss / len(val_loader)
        self.model.train()
        
        # Log validation metrics
        self.logger.info(f"Validation metrics: Loss={avg_loss:.4f}")
        
        return avg_loss
    
    def _save_full_training_checkpoint(self, step, epoch, loss, is_final=False):
        """Save checkpoint for full layer training"""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config,
            'training_type': 'full_layer',  # Mark as full training
            'no_lora': True,  # Explicitly mark as no LoRA
            'overfitting_metrics': {
                'val_loss_history': self.val_loss_history,
                'overfitting_detected': self.overfitting_detected
            },
            'training_info': {
                'dataset_size': '200hr',
                'gpu_type': 'T4_16GB',
                'mixed_precision': self.mixed_precision,
                'gradient_accumulation_steps': 16,
                'all_layers_trained': True
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"full_training_checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            best_path = self.output_dir / "full_training_best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"✅ Best model saved at step {step}")
        
        if is_final:
            final_path = self.output_dir / "full_training_final_model.pt"
            torch.save(checkpoint, final_path)
            self.logger.info(f"🎯 Final model saved at step {step}")
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _log_training_summary(self, epoch_train_losses):
        """Log comprehensive training summary"""
        self.logger.info("📊 FULL LAYER TRAINING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total epochs trained: {len(epoch_train_losses)}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Final training loss: {epoch_train_losses[-1]:.4f}")
        self.logger.info(f"Training type: FULL LAYER (NO LoRA)")
        self.logger.info(f"Overfitting detected: {self.overfitting_detected}")
        self.logger.info("="*50)
        
        # Save training summary
        summary = {
            'training_type': 'full_layer_no_lora',
            'epochs_trained': len(epoch_train_losses),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': epoch_train_losses[-1],
            'overfitting_detected': self.overfitting_detected,
            'training_stats': self.training_stats
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Full Layer Amharic IndexTTS2 Training - NO LoRA")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to full training config file")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory")
    parser.add_argument("--amharic_vocab", type=str, required=True, 
                       help="Path to Amharic vocab file")
    parser.add_argument("--train_manifest", type=str, required=True, 
                       help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, required=True, 
                       help="Path to validation manifest")
    parser.add_argument("--num_epochs", type=int, default=8, 
                       help="Number of epochs (full training)")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size (optimized for T4)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate (full training)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                       help="Gradient accumulation steps")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases")
    parser.add_argument("--mixed_precision", action="store_true", 
                       help="Enable mixed precision (recommended)")
    
    args = parser.parse_args()
    
    # Initialize full layer trainer
    trainer = FullLayerTrainer(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        amharic_vocab_path=args.amharic_vocab,
        mixed_precision=args.mixed_precision
    )
    
    # Start full layer training
    trainer.train_full_layers(
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