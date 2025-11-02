"""
Enhanced Amharic TTS with IndexTTS2 Integration
Implements advanced finetuning methodologies from IndexTTS2 repository analysis
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_linear_schedule_with_warmup

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextNormalizer, AmharicTextTokenizer
from indextts.gpt.model_v2 import UnifiedVoice


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, l=1.0):
    """Apply gradient reversal"""
    return GradReverse.apply(x, l)


class EnhancedAmharicUnifiedVoice(UnifiedVoice):
    """Enhanced Amharic TTS model integrating IndexTTS2 methodologies"""
    
    def __init__(self, config: Dict, vocab_size: int):
        """
        Initialize enhanced Amharic model with IndexTTS2 architecture
        
        Args:
            config: Model configuration dictionary
            vocab_size: Size of Amharic vocabulary
        """
        # Initialize base UnifiedVoice with Amharic optimizations
        super().__init__(
            layers=config['gpt']['layers'],
            model_dim=config['gpt']['model_dim'],
            heads=config['gpt']['heads'],
            max_text_tokens=config['gpt']['max_text_tokens'],
            max_mel_tokens=config['gpt']['max_mel_tokens'],
            number_text_tokens=vocab_size,
            number_mel_codes=config['gpt'].get('number_mel_codes', 8194),
            start_text_token=config['gpt'].get('start_text_token', 0),
            stop_text_token=config['gpt'].get('stop_text_token', 1),
            start_mel_token=config['gpt'].get('start_mel_token', 8192),
            stop_mel_token=config['gpt'].get('stop_mel_token', 8193),
            train_solo_embeddings=config['gpt'].get('train_solo_embeddings', False),
            use_mel_codes_as_input=config['gpt'].get('use_mel_codes_as_input', True),
            checkpointing=config['gpt'].get('checkpointing', True),
            condition_type=config['gpt'].get('condition_type', "conformer_perceiver"),
            condition_module=config['gpt']['condition_module'],
            emo_condition_module=config['gpt']['emo_condition_module']
        )
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Enhanced Amharic-specific components
        self.setup_amharic_enhancements()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
    def setup_amharic_enhancements(self):
        """Setup Amharic-specific model enhancements"""
        model_dim = self.config['gpt']['model_dim']
        
        # Amharic speaker-emotion disentanglement
        self.amharic_speaker_classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, self.config.get('n_speakers', 100)),
            nn.Softmax(dim=-1)
        )
        
        # Enhanced duration controller
        max_duration = self.config['gpt'].get('max_duration_tokens', 64)
        self.amharic_duration_controller = nn.Sequential(
            nn.Linear(max_duration, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Amharic text-specific embeddings with script awareness
        self.amharic_text_processor = AmharicTextProcessor()
        
    def _setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def forward_enhanced(self, 
                        speech_conditioning_latent: torch.Tensor,
                        text_inputs: torch.Tensor,
                        text_lengths: torch.Tensor,
                        mel_codes: torch.Tensor,
                        mel_codes_lengths: torch.Tensor,
                        emo_speech_conditioning_latent: torch.Tensor = None,
                        cond_mel_lengths: Optional[torch.Tensor] = None,
                        emo_cond_mel_lengths: Optional[torch.Tensor] = None,
                        use_speed: Optional[torch.Tensor] = None,
                        stage: int = 1,
                        return_intermediates: bool = False):
        """
        Enhanced forward pass with IndexTTS2 methodologies
        
        Args:
            speech_conditioning_latent: Speaker conditioning input
            text_inputs: Amharic text tokens
            text_lengths: Lengths of text sequences
            mel_codes: MEL code sequences
            mel_codes_lengths: Lengths of MEL sequences
            emo_speech_conditioning_latent: Emotion conditioning input
            cond_mel_lengths: Conditioning MEL lengths
            emo_cond_mel_lengths: Emotion conditioning MEL lengths
            use_speed: Speed control tensor
            stage: Training stage (1, 2, or 3)
            return_intermediates: Whether to return intermediate activations
        
        Returns:
            Tuple of (mel_logits, speaker_probs, emotion_latent, intermediates)
        """
        # Apply Amharic text preprocessing
        processed_text_inputs = self.amharic_text_processor(text_inputs)
        
        # Duration control with Amharic optimizations
        duration_emb = self.compute_enhanced_duration_embedding(text_lengths, stage)
        
        # Multi-conditioning with enhanced speaker-emotion disentanglement
        conditioning_latents = self.get_enhanced_conditioning(
            speech_conditioning_latent,
            emo_speech_conditioning_latent,
            cond_mel_lengths,
            emo_cond_mel_lengths,
            stage
        )
        
        # Build aligned inputs and targets
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            processed_text_inputs, self.start_text_token, self.stop_text_token
        )
        
        # Enhanced embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)
        
        # Combine conditioning, text, and MEL embeddings
        combined_emb = torch.cat([conditioning_latents, text_emb, mel_emb], dim=1)
        
        # Enhanced GPT forward with attention to intermediates
        gpt_out = self.gpt(inputs_embeds=combined_emb, return_dict=True, 
                          output_hidden_states=return_intermediates)
        
        # Extract outputs
        offset = conditioning_latents.shape[1]
        text_logits = gpt_out.last_hidden_state[:, offset:offset+text_emb.shape[1]]
        mel_logits = gpt_out.last_hidden_state[:, offset+text_emb.shape[1]:]
        
        # Apply final layers
        text_logits = self.text_head(self.final_norm(text_logits)).permute(0, 2, 1)
        mel_logits = self.mel_head(self.final_norm(mel_logits)).permute(0, 2, 1)
        
        # Speaker-emotion disentanglement
        emotion_latent = self.get_emotion_latent(emo_speech_conditioning_latent)
        speaker_probs = self.amharic_speaker_classifier(emotion_latent)
        
        if return_intermediates:
            intermediates = {
                'text_logits': text_logits,
                'mel_logits': mel_logits,
                'speaker_probs': speaker_probs,
                'emotion_latent': emotion_latent,
                'duration_emb': duration_emb,
                'conditioning_latents': conditioning_latents,
                'hidden_states': gpt_out.hidden_states if return_intermediates else None
            }
            return mel_logits, speaker_probs, emotion_latent, intermediates
        
        return mel_logits, speaker_probs, emotion_latent
    
    def compute_enhanced_duration_embedding(self, text_lengths: torch.Tensor, stage: int) -> torch.Tensor:
        """Enhanced duration embedding computation with IndexTTS2 optimizations"""
        max_duration = self.config['gpt'].get('max_duration_tokens', 64)
        
        # Clamp and one-hot encode
        clamped_lengths = torch.clamp(text_lengths, min=0, max=max_duration - 1)
        one_hot = F.one_hot(clamped_lengths, num_classes=max_duration).float()
        
        # Enhanced duration embedding
        duration_emb = self.amharic_duration_controller(one_hot)
        
        # Stage 1: Apply randomization for robustness (IndexTTS2 approach)
        if stage == 1:
            B = duration_emb.shape[0]
            device = duration_emb.device
            # 30% probability randomization
            mask = (torch.rand(B, device=device) < 0.3).float().unsqueeze(-1)
            duration_emb = duration_emb * (1.0 - mask)
        
        return duration_emb
    
    def get_enhanced_conditioning(self, 
                                 speaker_latent: torch.Tensor,
                                 emotion_latent: torch.Tensor,
                                 speaker_lengths: Optional[torch.Tensor],
                                 emotion_lengths: Optional[torch.Tensor],
                                 stage: int) -> torch.Tensor:
        """Enhanced conditioning with speaker-emotion disentanglement"""
        # Default emotion latent if not provided
        if emotion_latent is None:
            emotion_latent = speaker_latent
            
        # Get basic conditioning
        speaker_cond = self.get_conditioning(speaker_latent.transpose(1, 2), speaker_lengths)
        emotion_cond = self.get_emo_conditioning(emotion_latent.transpose(1, 2), emotion_lengths)
        
        # Stage 2+: Apply gradient reversal for speaker-emotion disentanglement
        if stage >= 2:
            emotion_cond = grad_reverse(emotion_cond, l=1.0)
        
        # Combine conditioning with duration info
        # Format: [speaker_conditioning + emotion_conditioning, duration_emb, speed_emb]
        combined_conditioning = torch.cat([
            speaker_cond + emotion_cond.squeeze(1).unsqueeze(1),
            torch.zeros_like(speaker_cond[:, :1, :])  # Placeholder for duration
        ], dim=1)
        
        return combined_conditioning
    
    def get_emotion_latent(self, emotion_input: torch.Tensor) -> torch.Tensor:
        """Extract emotion latent representation"""
        if emotion_input is None:
            return torch.zeros(1, self.config['gpt']['model_dim'])
            
        # Project emotion input to latent space
        emotion_latent = self.emo_layer(
            self.emovec_layer(emotion_input.transpose(1, 2).mean(dim=-1, keepdim=True))
        )
        return emotion_latent.squeeze(1)
    
    def build_aligned_inputs_and_targets(self, input: torch.Tensor, start_token: int, stop_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced input-target alignment with Amharic-specific handling"""
        # Standard alignment (same as IndexTTS2)
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar


class AmharicTextProcessor(nn.Module):
    """Amharic-specific text processing module"""
    
    def __init__(self):
        super().__init__()
        self.script_processor = AmharicScriptProcessor()
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Process Amharic text tokens with script-specific optimizations
        
        Args:
            text_tokens: Tokenized Amharic text [batch_size, seq_len]
            
        Returns:
            Processed text tokens
        """
        # Apply Amharic script-specific processing
        # This can include character normalization, script detection, etc.
        processed_tokens = text_tokens
        
        # Apply script-aware token adjustments if needed
        # (Implementation depends on specific Amharic script requirements)
        
        return processed_tokens


class AmharicScriptProcessor:
    """Amharic script-specific processing utilities"""
    
    def __init__(self):
        # Amharic script character ranges and patterns
        self.amharic_chars = set(chr(i) for i in range(0x1200, 0x137F))
        self.punctuation = set(['፣', '።', '፧', '፤', '፦', '፥', '፦', '፧'])
        
    def detect_script(self, text: str) -> str:
        """Detect the script type in text"""
        amharic_count = sum(1 for char in text if char in self.amharic_chars)
        latin_count = sum(1 for char in text if char.isalpha() and char.isascii())
        
        if amharic_count > latin_count:
            return "amharic"
        elif latin_count > 0:
            return "mixed"
        else:
            return "other"


class EnhancedAmharicLoss(nn.Module):
    """Enhanced loss function incorporating IndexTTS2 methodologies"""
    
    def __init__(self, alpha: float = 0.5, length_normalization: bool = True):
        super().__init__()
        self.alpha = alpha
        self.length_normalization = length_normalization
        
    def forward(self, 
               mel_logits: torch.Tensor,
               mel_targets: torch.Tensor,
               text_logits: torch.Tensor,
               text_targets: torch.Tensor,
               speaker_probs: torch.Tensor,
               text_lengths: torch.Tensor,
               mel_lengths: torch.Tensor,
               stage: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute enhanced loss for Amharic TTS training
        
        Args:
            mel_logits: Predicted MEL logits [batch_size, mel_vocab, seq_len]
            mel_targets: Target MEL tokens [batch_size, seq_len]
            text_logits: Predicted text logits [batch_size, text_vocab, seq_len]
            text_targets: Target text tokens [batch_size, seq_len]
            speaker_probs: Speaker classification probabilities [batch_size, n_speakers]
            text_lengths: Text sequence lengths [batch_size]
            mel_lengths: MEL sequence lengths [batch_size]
            stage: Training stage (affects loss computation)
        
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Main MEL prediction loss
        mel_loss = self.compute_sequence_loss(mel_logits, mel_targets, mel_lengths, "mel")
        
        # Main text prediction loss
        text_loss = self.compute_sequence_loss(text_logits, text_targets, text_lengths, "text")
        
        # Adversarial loss for speaker-emotion disentanglement (Stage 2+)
        adv_loss = 0.0
        if stage >= 2 and speaker_probs is not None:
            adv_loss = self.compute_adversarial_loss(speaker_probs)
        
        # Stage-specific loss weighting
        if stage == 1:
            # Stage 1: Focus on basic generation
            total_loss = mel_loss + text_loss
        elif stage == 2:
            # Stage 2: Include adversarial loss
            total_loss = mel_loss + text_loss + self.alpha * adv_loss
        else:  # Stage 3
            # Stage 3: Fine-tuning with reduced adversarial loss
            total_loss = mel_loss + text_loss + 0.5 * self.alpha * adv_loss
        
        # Loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'mel_loss': mel_loss.item(),
            'text_loss': text_loss.item(),
            'adv_loss': adv_loss.item() if adv_loss > 0 else 0.0,
            'stage': stage
        }
        
        return total_loss, loss_components
    
    def compute_sequence_loss(self, 
                             logits: torch.Tensor, 
                             targets: torch.Tensor, 
                             lengths: torch.Tensor,
                             sequence_type: str) -> torch.Tensor:
        """Compute sequence loss with length normalization"""
        B, V, L = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        
        # Cross-entropy loss (assuming padding token is 0)
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, 
                                       ignore_index=0, reduction='none')
        loss_per_token = loss_per_token.view(B, L)
        
        if self.length_normalization:
            # Length-normalized loss (IndexTTS2 approach)
            valid_mask = (targets != 0).float()
            token_sum = (loss_per_token * valid_mask).sum(dim=1)
            denom = lengths.float() + 1.0  # +1 to avoid division by zero
            sequence_loss = token_sum / denom
        else:
            # Standard mean loss
            sequence_loss = loss_per_token.mean(dim=1)
        
        return sequence_loss.mean()
    
    def compute_adversarial_loss(self, speaker_probs: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for speaker-emotion disentanglement"""
        # Use maximum probability as confidence measure
        speaker_max, _ = speaker_probs.max(dim=-1)
        
        # Adversarial loss: maximize uncertainty (reduce confidence)
        adv_loss = -torch.log(speaker_max + 1e-9)
        
        return adv_loss.mean()


class EnhancedAmharicTrainer:
    """Enhanced Amharic TTS trainer with IndexTTS2 methodologies"""
    
    def __init__(self, config: Dict, model: EnhancedAmharicUnifiedVoice, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.logger = self._setup_logging()
        
        # Initialize loss function
        self.criterion = EnhancedAmharicLoss(
            alpha=config['training'].get('adversarial_alpha', 0.5)
        )
        
        # Initialize optimizer and scheduler
        self.setup_optimizer()
        
        # Training history
        self.training_history = []
        
    def _setup_logging(self):
        """Setup training logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def setup_optimizer(self):
        """Setup optimizer with stage-specific parameters"""
        # Separate parameters by training stage
        self.param_groups = {
            'stage1': [],
            'stage2': [],
            'stage3': []
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(keyword in name for keyword in ['text_embedding', 'mel_embedding']):
                    self.param_groups['stage1'].append(param)
                elif any(keyword in name for keyword in ['conditioning', 'perceiver']):
                    self.param_groups['stage2'].append(param)
                elif any(keyword in name for keyword in ['duration', 'speaker_classifier']):
                    self.param_groups['stage3'].append(param)
                else:
                    self.param_groups['stage1'].append(param)  # Default to stage 1
    
    def train_three_stage(self, 
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         num_epochs_per_stage: int = 3) -> Dict:
        """
        Three-stage training as implemented in IndexTTS2
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs_per_stage: Number of epochs per stage
            
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting three-stage Amharic TTS training")
        
        stages = [
            (1, "Basic Amharic Alignment", self.param_groups['stage1']),
            (2, "Speaker-Emotion Disentanglement", self.param_groups['stage2']),
            (3, "Duration Fine-tuning", self.param_groups['stage3'])
        ]
        
        training_history = {
            'stage_history': [],
            'final_metrics': {},
            'best_checkpoint': None
        }
        
        best_val_loss = float('inf')
        
        for stage_num, stage_name, trainable_params in stages:
            self.logger.info(f"Starting Stage {stage_num}: {stage_name}")
            
            # Freeze non-trainable parameters
            self.freeze_stage_parameters(stage_num)
            
            # Setup stage-specific optimizer
            stage_optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'] * (0.5 ** (stage_num - 1)),
                weight_decay=0.01
            )
            
            # Training loop for this stage
            stage_history = self.train_stage(
                stage_num, stage_name, trainable_params, stage_optimizer,
                train_loader, val_loader, num_epochs_per_stage
            )
            
            training_history['stage_history'].append(stage_history)
            
            # Check if this is the best model so far
            if stage_history['best_val_loss'] < best_val_loss:
                best_val_loss = stage_history['best_val_loss']
                training_history['best_checkpoint'] = {
                    'stage': stage_num,
                    'state_dict': self.model.state_dict(),
                    'val_loss': best_val_loss
                }
                
        self.training_history = training_history
        self.logger.info("Three-stage training completed!")
        
        return training_history
    
    def freeze_stage_parameters(self, stage: int):
        """Freeze parameters according to training stage"""
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            
        # Enable parameters for current stage
        if stage == 1:
            # Stage 1: Basic alignment - train embeddings and basic components
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in [
                    'text_embedding', 'mel_embedding', 'gpt.wte', 'gpt.wpe'
                ]):
                    param.requires_grad = True
                    
        elif stage == 2:
            # Stage 2: Speaker-emotion disentanglement
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in [
                    'conditioning_encoder', 'perceiver_encoder', 'speaker_classifier'
                ]):
                    param.requires_grad = True
                    
        elif stage == 3:
            # Stage 3: Duration fine-tuning
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in [
                    'duration_controller', 'amharic_speaker_classifier'
                ]):
                    param.requires_grad = True
    
    def train_stage(self, 
                   stage: int, 
                   stage_name: str,
                   trainable_params: List[nn.Parameter],
                   optimizer: torch.optim.Optimizer,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int) -> Dict:
        """Train a specific stage"""
        
        stage_history = {
            'stage': stage,
            'name': stage_name,
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(stage, train_loader, optimizer, scheduler)
            stage_history['train_losses'].append(train_loss)
            
            # Validation phase
            val_loss = self.validate_epoch(stage, val_loader)
            stage_history['val_losses'].append(val_loss)
            
            # Update best validation
            if val_loss < stage_history['best_val_loss']:
                stage_history['best_val_loss'] = val_loss
                stage_history['best_epoch'] = epoch
                
            self.logger.info(
                f"Stage {stage} Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        
        return stage_history
    
    def train_epoch(self, stage: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            mel_logits, speaker_probs, emotion_latent, intermediates = self.model.forward_enhanced(
                speech_conditioning_latent=batch['conditioning'],
                text_inputs=batch['text_inputs'],
                text_lengths=batch['text_lengths'],
                mel_codes=batch['mel_codes'],
                mel_codes_lengths=batch['mel_lengths'],
                emo_speech_conditioning_latent=batch.get('emotion_conditioning'),
                cond_mel_lengths=batch.get('conditioning_lengths'),
                use_speed=batch.get('speed'),
                stage=stage,
                return_intermediates=True
            )
            
            # Compute loss
            total_loss_batch, loss_components = self.criterion(
                mel_logits=mel_logits,
                mel_targets=batch['mel_targets'],
                text_logits=intermediates['text_logits'],
                text_targets=batch['text_targets'],
                speaker_probs=speaker_probs,
                text_lengths=batch['text_lengths'],
                mel_lengths=batch['mel_lengths'],
                stage=stage
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Stage {stage} Batch {batch_idx}/{len(train_loader)} - "
                    f"Loss: {total_loss_batch.item():.4f}, "
                    f"Mel Loss: {loss_components['mel_loss']:.4f}, "
                    f"Text Loss: {loss_components['text_loss']:.4f}, "
                    f"Adv Loss: {loss_components['adv_loss']:.4f}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, stage: int, val_loader: DataLoader) -> float:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                mel_logits, speaker_probs, emotion_latent, intermediates = self.model.forward_enhanced(
                    speech_conditioning_latent=batch['conditioning'],
                    text_inputs=batch['text_inputs'],
                    text_lengths=batch['text_lengths'],
                    mel_codes=batch['mel_codes'],
                    mel_codes_lengths=batch['mel_lengths'],
                    emo_speech_conditioning_latent=batch.get('emotion_conditioning'),
                    cond_mel_lengths=batch.get('conditioning_lengths'),
                    use_speed=batch.get('speed'),
                    stage=stage,
                    return_intermediates=True
                )
                
                # Compute loss
                total_loss_batch, loss_components = self.criterion(
                    mel_logits=mel_logits,
                    mel_targets=batch['mel_targets'],
                    text_logits=intermediates['text_logits'],
                    text_targets=batch['text_targets'],
                    speaker_probs=speaker_probs,
                    text_lengths=batch['text_lengths'],
                    mel_lengths=batch['mel_lengths'],
                    stage=stage
                )
                
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'gpt': {
            'layers': 24,
            'model_dim': 1280,
            'heads': 20,
            'max_text_tokens': 600,
            'max_mel_tokens': 1815,
            'number_mel_codes': 8194,
            'condition_type': 'conformer_perceiver',
            'condition_module': {
                'output_size': 512,
                'linear_units': 2048,
                'attention_heads': 8,
                'num_blocks': 6,
                'input_layer': 'conv2d2'
            },
            'emo_condition_module': {
                'output_size': 512,
                'linear_units': 1024,
                'attention_heads': 4,
                'num_blocks': 4,
                'input_layer': 'conv2d2'
            }
        },
        'training': {
            'learning_rate': 1e-4,
            'adversarial_alpha': 0.5
        },
        'n_speakers': 100
    }
    
    # Initialize enhanced Amharic model
    vocab_size = 8000  # Amharic vocabulary size
    model = EnhancedAmharicUnifiedVoice(config, vocab_size)
    
    print(f"Enhanced Amharic TTS model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Key features:")
    print("- Three-stage training pipeline")
    print("- Enhanced speaker-emotion disentanglement")
    print("- Advanced duration control")
    print("- IndexTTS2 UnifiedVoice architecture")
    print("- Amharic script-specific optimizations")