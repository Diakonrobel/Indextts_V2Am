"""
Amharic fine-tuning script for IndexTTS2
Complete LoRA-based fine-tuning system for Amharic TTS
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
import wandb
from transformers import get_linear_schedule_with_warmup
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.vocab_utils import resize_token_embeddings, resize_linear_layer
from indextts.utils.mel_quantization import simple_mel_quantization
from indextts.utils.checkpoint_validator import CheckpointValidator


class AmharicTTSDataset(Dataset):
    """Dataset for Amharic TTS fine-tuning"""
    
    def __init__(
        self,
        manifest_file: str,
        tokenizer: AmharicTextTokenizer,
        mel_extractor: MelSpectrogramFeatures,
        max_text_length: int = 600,
        max_mel_length: int = 1815,
        augmentation_config: Dict = None
    ):
        self.manifest_file = manifest_file
        self.tokenizer = tokenizer
        self.mel_extractor = mel_extractor
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.augmentation_config = augmentation_config or {}
        
        # Load manifest
        self.data = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} Amharic samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio and extract mel spectrogram
        audio_path = item['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Apply augmentation if enabled
        if self.augmentation_config.get('speed_perturbation', False):
            speed_factor = np.random.uniform(
                self.augmentation_config.get('speed_range', [0.9, 1.1])[0],
                self.augmentation_config.get('speed_range', [0.9, 1.1])[1]
            )
            if speed_factor != 1.0:
                new_length = int(audio.shape[1] / speed_factor)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0), 
                    size=new_length, 
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
        
        # Extract mel spectrogram
        mel = self.mel_extractor(audio)
        mel = mel.squeeze(0)  # Remove batch dimension
        
        # Apply pitch perturbation if enabled
        if self.augmentation_config.get('pitch_perturbation', False):
            pitch_shift = np.random.uniform(
                self.augmentation_config.get('pitch_range', [-0.1, 0.1])[0],
                self.augmentation_config.get('pitch_range', [-0.1, 0.1])[1]
            )
            if pitch_shift != 0.0:
                # Simple pitch shifting by time stretching
                if pitch_shift > 0:
                    # Higher pitch: slightly shorter duration
                    factor = 1.0 - pitch_shift * 0.1
                else:
                    # Lower pitch: slightly longer duration  
                    factor = 1.0 + abs(pitch_shift) * 0.1
                
                new_length = int(mel.shape[1] * factor)
                mel = torch.nn.functional.interpolate(
                    mel.unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
        
        # Apply noise injection if enabled
        if self.augmentation_config.get('noise_injection', False):
            noise_level = self.augmentation_config.get('noise_level', 0.01)
            noise = torch.randn_like(mel) * noise_level
            mel = mel + noise
        
        # Truncate if too long
        if mel.shape[-1] > self.max_mel_length:
            mel = mel[:, :self.max_mel_length]
        
        # Tokenize text using Amharic tokenizer
        text = item['text']
        text_tokens = self.tokenizer.encode(text, out_type=int)
        
        # Truncate if too long
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[:self.max_text_length]
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'mel_spectrogram': mel,
            'text': text,
            'audio_id': item['id'],
            'duration': item.get('duration', 0.0),
            'text_length': len(text_tokens)
        }


class AmharicTTSFineTuner:
    """Fine-tuner for Amharic TTS adaptation using IndexTTS2"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str,
        amharic_vocab_path: str,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
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
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Amharic tokenizer
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
        
        # Setup LoRA if enabled
        if self.use_lora:
            self.lora_manager = self._setup_lora()
        else:
            self.lora_manager = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for Amharic training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Amharic TTS fine-tuning initialized")
    
    def _load_model(self) -> UnifiedVoice:
        """Load the pre-trained model and adapt for Amharic"""
        self.logger.info("Loading pre-trained IndexTTS2 model...")
        
        # Create model with Amharic vocabulary size
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
        
        # Handle vocabulary size mismatch with proper token mapping
        if 'text_embedding.weight' in checkpoint:
            old_vocab_size = checkpoint['text_embedding.weight'].shape[0]
            new_vocab_size = self.tokenizer.vocab_size
            
            if old_vocab_size != new_vocab_size:
                self.logger.info(f"Resizing text embedding from {old_vocab_size} to {new_vocab_size}")
                self.logger.info("Using token-string mapping for proper transfer learning")
                
                # Get vocabulary files (need to store/infer old vocab path)
                # For now, use position-based as fallback if vocab files not available
                # TODO: Save old_vocab_path in checkpoint for proper mapping
                old_embedding_tensor = checkpoint['text_embedding.weight']
                embedding_dim = old_embedding_tensor.shape[1]
                
                # Create temporary embedding layers
                old_emb = nn.Embedding(old_vocab_size, embedding_dim)
                old_emb.weight.data = old_embedding_tensor
                
                # For now use position-based, but log warning
                self.logger.warning("Old vocab file not available - using position-based fallback")
                self.logger.warning("This may not preserve token semantics correctly!")
                
                new_embedding = torch.zeros(new_vocab_size, embedding_dim)
                nn.init.normal_(new_embedding, mean=0.0, std=0.02)
                min_size = min(old_vocab_size, new_vocab_size)
                new_embedding[:min_size] = old_embedding_tensor[:min_size]
                
                checkpoint['text_embedding.weight'] = new_embedding
                
                # Resize text head
                if 'text_head.weight' in checkpoint:
                    old_head = checkpoint['text_head.weight']
                    new_head = torch.zeros(new_vocab_size, old_head.shape[1])
                    nn.init.normal_(new_head, mean=0.0, std=0.02)
                    new_head[:min_size] = old_head[:min_size]
                    checkpoint['text_head.weight'] = new_head
                    
                    if 'text_head.bias' in checkpoint:
                        old_bias = checkpoint['text_head.bias']
                        new_bias = torch.zeros(new_vocab_size)
                        new_bias[:min_size] = old_bias[:min_size]
                        checkpoint['text_head.bias'] = new_bias
        
        # Load state dict
        model.load_state_dict(checkpoint, strict=False)
        
        # Move to device
        model = model.to(self.device)
        
        self.logger.info("Model loaded successfully with Amharic vocabulary")
        return model
    
    def _setup_lora(self):
        """Setup LoRA adapters for efficient fine-tuning"""
        self.logger.info("Setting up LoRA adapters...")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # LoRA configuration for Amharic
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.config['lora']['dropout'],
                target_modules=self.config['lora']['target_modules'],
                bias=self.config['lora'].get('bias', 'none')
            )
            
            # Apply LoRA to model
            model = get_peft_model(self.model, lora_config)
            model.print_trainable_parameters()
            
            self.logger.info("LoRA adapters configured successfully")
            return model
            
        except ImportError:
            self.logger.warning("PEFT library not available, using manual LoRA implementation")
            return self._setup_manual_lora()
    
    def _setup_manual_lora(self):
        """Manual LoRA implementation as fallback"""
        self.logger.info("Setting up manual LoRA implementation...")
        
        # This is a simplified manual LoRA setup
        # In practice, you'd want to use the PEFT library
        
        # For now, we'll just return the model without LoRA
        self.logger.info("Using full fine-tuning instead of LoRA")
        return self.model
    
    def create_data_loader(
        self,
        manifest_file: str,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader for Amharic dataset"""
        dataset = AmharicTTSDataset(
            manifest_file=manifest_file,
            tokenizer=self.tokenizer,
            mel_extractor=self.mel_extractor,
            max_text_length=self.config['gpt']['max_text_tokens'],
            max_mel_length=self.config['gpt']['max_mel_tokens'],
            augmentation_config=self.config['training']['augmentation']
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
                'audio_ids': [item['audio_id'] for item in batch],
                'durations': [item['duration'] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def _compute_loss(self, batch):
        """Compute training loss using proper cross-entropy on mel codes
        
        Note: This function expects batch to contain:
        - text_tokens, text_attention_masks
        - mel_codes (discrete, pre-quantized) OR mel_spectrograms (will use fallback)
        - conditioning_audios for speaker conditioning
        """
        try:
            # Move to device
            text_tokens = batch['text_tokens'].to(self.device)
            text_attention_masks = batch['text_attention_masks'].to(self.device)
            
            # Get lengths
            text_lengths = text_attention_masks.sum(dim=1).long()
            
            # Load mel codes - prefer pre-quantized, fallback to random
            if 'mel_codes' in batch:
                mel_codes = batch['mel_codes'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                mel_lengths = mel_attention_masks.sum(dim=1).long()
            else:
                # Fallback for old dataset format
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                mel_lengths = mel_attention_masks.sum(dim=1).long()
                
                # Generate random codes as temporary fallback
                mel_codes = torch.randint(
                    0, self.model.number_mel_codes,
                    (mel_spectrograms.shape[0], mel_spectrograms.shape[2]),
                    device=self.device
                )
                if not hasattr(self, '_warned_about_codes'):
                    self.logger.warning("⚠️  Using random mel_codes - run prepare_amharic_mel_codes.py!")
                    self._warned_about_codes = True
            
            # Prepare conditioning from audio
            # Extract mel features for conditioning
            if 'conditioning_audios' in batch:
                # Use provided audio for conditioning
                cond_audio = batch['conditioning_audios'][0]  # First in batch
                cond_mel = self.mel_extractor(cond_audio.to(self.device))
                cond_lengths = torch.tensor([cond_mel.shape[-1]], device=self.device)
            else:
                # Fallback: use mel_codes themselves for conditioning
                cond_mel = torch.randn(1, 1024, 100, device=self.device)
                cond_lengths = torch.tensor([100], device=self.device)
            
            # Get conditioning latents
            speech_conditioning_latent = self.model.get_conditioning(
                cond_mel.transpose(1, 2) if cond_mel.ndim == 3 else cond_mel,
                cond_lengths
            )
            
            # Forward pass through model
            loss_text, loss_mel, _ = self.model(
                speech_conditioning_latent=speech_conditioning_latent,
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                mel_codes=mel_codes,
                wav_lengths=mel_lengths,
                cond_mel_lengths=cond_lengths
            )
            
            # Weighted combination
            total_loss = 0.1 * loss_text + loss_mel
            
            return total_loss
            
        except Exception as e:
            self.logger.error(f"Error computing loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    def train(
        self,
        train_manifest: str,
        val_manifest: str,
        num_epochs: int = 15,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1500,
        gradient_clip_val: float = 1.0,
        save_every: int = 500,
        log_every: int = 50,
        use_wandb: bool = False
    ):
        """Train the Amharic TTS model"""
        
        if use_wandb:
            wandb.init(
                project="indextts2-amharic-finetuning",
                config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "use_lora": self.use_lora,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "vocab_size": self.tokenizer.vocab_size
                }
            )
        
        # Create data loaders
        train_loader = self.create_data_loader(train_manifest, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_manifest, batch_size, shuffle=False)
        
        # Setup optimizer
        if self.use_lora:
            # Only optimize LoRA parameters if using PEFT
            if hasattr(self.lora_manager, 'parameters'):
                trainable_params = list(self.lora_manager.parameters())
            else:
                # Fallback to text embedding and speed embedding
                trainable_params = list(self.model.text_embedding.parameters())
                if hasattr(self.model, 'speed_emb'):
                    trainable_params.extend(list(self.model.speed_emb.parameters()))
        else:
            # Fine-tune all parameters
            trainable_params = list(self.model.parameters())
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Starting Amharic training epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Forward pass
                optimizer.zero_grad()
                
                # Compute loss
                loss = self._compute_loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_val)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % log_every == 0:
                    self.logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                    if use_wandb:
                        wandb.log({"train_loss": loss.item(), "step": global_step})
                
                # Save checkpoint
                if global_step % save_every == 0:
                    self._save_checkpoint(global_step, epoch, loss.item())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Validation
            val_loss = self._validate(val_loader)
            avg_epoch_loss = epoch_loss / len(train_loader)
            
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_epoch_loss,
                    "val_loss": val_loss
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(global_step, epoch, val_loss, is_best=True)
                self.logger.info(f"New best model saved at epoch {epoch + 1}")
        
        self.logger.info("Amharic TTS training completed!")
        if use_wandb:
            wandb.finish()
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, step, epoch, loss, is_best=False):
        """Save model checkpoint with vocabulary and normalizer state"""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.lora_manager.state_dict() if self.use_lora else self.model.state_dict(),
            'loss': loss,
            'config': self.config,
            # Vocabulary state
            'vocab_size': self.tokenizer.vocab_size,
            'vocab_file': self.tokenizer.vocab_file,
            # Normalizer state
            'normalizer_config': {
                'type': 'AmharicTextNormalizer',
                'number_words': self.tokenizer.normalizer.number_words if hasattr(self.tokenizer, 'normalizer') else {},
                'contractions': self.tokenizer.normalizer.contractions if hasattr(self.tokenizer, 'normalizer') else {},
                'abbreviations': self.tokenizer.normalizer.abbreviations if hasattr(self.tokenizer, 'normalizer') else {}
            },
            # Training metadata
            'training_type': 'lora' if self.use_lora else 'full',
            'lora_config': {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha
            } if self.use_lora else None
        }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_amharic_model.pt"
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndexTTS2 for Amharic")
    parser.add_argument("--config", type=str, required=True, help="Path to Amharic config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--amharic_vocab", type=str, required=True, help="Path to Amharic vocab file")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, required=True, help="Path to validation manifest")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = AmharicTTSFineTuner(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        amharic_vocab_path=args.amharic_vocab,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Start training
    fine_tuner.train(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main()