"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
        Returns metadata for the PreprocessedWindowDataset.
        """
        metadata_file = os.path.join(self.save_dir, "window_metadata.pt")
        
        # Check if already preprocessed
        if not force_reprocess and os.path.exists(metadata_file):
            print("📁 Found existing preprocessed windows, loading metadata...")
            try:
                metadata = torch.load(metadata_file)
                print(f"✅ Loaded {len(metadata)} preprocessed windows from disk")
                return metadata
            except:
                print("⚠️ Failed to load metadata, will reprocess...")
        
        print("🔄 Pre-processing ALL windows (this may take a while)...")
        print("💡 This is done ONCE - subsequent training will be very fast!")
        
        metadata = []
        window_count = 0
        
        start_time = time.time()
        
        for sample_idx, item in enumerate(self.data):
            if sample_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Processing sample {sample_idx+1}/{len(self.data)} | "
                      f"Windows so far: {window_count:,} | Time: {elapsed:.1f}s")
            
            try:
                # Load and resample audio once per file
                audio = item['audio']['array']
                original_sr = item['audio']['sampling_rate']
                
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
                # Get boundary positions
                boundary_positions = []
                if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                    starts = item['phonetic_detail']['start']
                    stops = item['phonetic_detail']['stop']
                    
                    for start, stop in zip(starts, stops):
                        start_sample = int(start * self.sample_rate)
                        stop_sample = int(stop * self.sample_rate)
                        boundary_positions.extend([start_sample, stop_sample])
                
                boundary_positions = sorted(list(set(boundary_positions)))
                boundary_positions = [pos for pos in boundary_positions if 0 <= pos < len(audio)]
                
                # DEBUG: Show boundary info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: {len(boundary_positions)} boundaries, "
                          f"audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                
                # Generate and save positive windows
                pos_count = self._save_positive_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += pos_count
                
                # Generate and save negative windows
                neg_count = self._save_negative_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += neg_count
                
                # DEBUG: Show window generation info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: Generated {pos_count} positive, {neg_count} negative windows")
                
            except Exception as e:
                print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
                # Add more detailed debugging
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                continue
        
        # Save metadata
        torch.save(metadata, metadata_file)
        
        preprocessing_time = time.time() - start_time
        positive_count = sum(1 for w in metadata if w['label'] == 1)
        negative_count = len(metadata) - positive_count
        
        print(f"✅ Pre-processing complete!")
        print(f"📊 Statistics:")
        print(f"   Total windows: {len(metadata):,}")
        
        if len(metadata) > 0:
            print(f"   Positive windows: {positive_count:,} ({positive_count/len(metadata)*100:.1f}%)")
            print(f"   Negative windows: {negative_count:,} ({negative_count/len(metadata)*100:.1f}%)")
            if positive_count > 0:
                print(f"   Class balance ratio: {negative_count/positive_count:.1f}:1")
                print(f"   Class balance quality: {'✅ GOOD' if 1 <= negative_count/positive_count <= 10 else '⚠️ IMBALANCED'}")
            else:
                print(f"   Class balance ratio: N/A (no positive windows)")
            print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
            print(f"   Disk usage: ~{len(metadata) * 48 / 1024:.1f} MB")  # Increased for 1.5s windows
            print(f"   Average windows per audio file: {len(metadata)/len(self.data):.1f}")
        else:
            print(f"   ⚠️ WARNING: No windows were generated!")
            print(f"   Positive windows: {positive_count:,}")
            print(f"   Negative windows: {negative_count:,}")
            print(f"   This indicates a problem with the audio processing or boundary detection.")
        
        return metadata
    
    def _save_positive_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save exactly ONE positive window per boundary for proper balance."""
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # FIXED: Generate exactly ONE window per boundary at exact position
            window_end = boundary_pos
            start_pos = window_end - self.window_samples
            
            if start_pos < 0 or window_end >= len(audio):
                continue
            
            # Extract and validate window
            window_audio = audio[start_pos:window_end]
            
            if len(window_audio) != self.window_samples:
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (padding, 0), mode='constant')
                else:
                    window_audio = window_audio[:self.window_samples]
            
            if np.all(window_audio == 0):
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Save processed window
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'window_start': start_pos
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata
                metadata.append({
                    'window_file': window_filename,
                    'label': 1,
                    'file_id': file_id,
                    'boundary_pos': boundary_pos
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save negative windows with reasonable limits for balance."""
        saved_count = 0
        
        # Create exclusion zones around boundaries (for negative sampling only!)
        exclusion_zones = set()
        for boundary_pos in boundary_positions:
            for offset in range(-self.exclusion_zone_samples, self.exclusion_zone_samples + 1):
                if 0 <= boundary_pos + offset < len(audio):
                    exclusion_zones.add(boundary_pos + offset)
        
        # Find valid positions for negative windows
        valid_end_positions = []
        for end_pos in range(self.window_samples, len(audio)):
            if end_pos not in exclusion_zones:
                valid_end_positions.append(end_pos)
        
        # DEBUG: Show exclusion zone statistics for first few samples
        if sample_idx < 5:
            total_possible = len(audio) - self.window_samples
            excluded_count = len(exclusion_zones)
            valid_count = len(valid_end_positions)
            print(f"   DEBUG Negative sampling: {excluded_count:,} excluded, {valid_count:,}/{total_possible:,} valid positions")
        
        # Sample negative positions - aim for 1:1 or 1:2 ratio with positive windows
        # Since we now have exactly len(boundary_positions) positive windows, balance accordingly
        target_negative = len(boundary_positions) * 2  # 2:1 negative:positive ratio for better discrimination
        num_negative = min(
            int(len(valid_end_positions) * self.negative_sampling_ratio),
            target_negative,  # Target balanced ratio
            self.max_negative_per_file  # Still enforce upper limit
        )
        
        if num_negative > 0 and len(valid_end_positions) > 0:
            sampled_positions = random.sample(valid_end_positions, 
                                            min(num_negative, len(valid_end_positions)))
            
            for end_pos in sampled_positions:
                # Stop when we reach the maximum negative windows per file
                if saved_count >= self.max_negative_per_file:
                    break
                    
                start_pos = end_pos - self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                if len(window_audio) != self.window_samples or np.all(window_audio == 0):
                    continue
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Save processed window
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'metadata': {
                        'boundary_pos': None,
                        'window_start': start_pos
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0,
                        'file_id': file_id,
                        'boundary_pos': None
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        else:
            if sample_idx < 5:
                print(f"   DEBUG: No negative windows generated - {num_negative} calculated from {len(valid_end_positions)} valid positions")
        
        return saved_count


class PreprocessedWindowDataset(Dataset):
    """
    STAGE 2: Ultra-fast dataset that loads pre-processed windows.
    No redundant processing - just loads tensors from disk.
    """
    
    def __init__(self, metadata, save_dir="./preprocessed_windows"):
        """
        Initialize dataset with pre-processed window metadata.
        
        Args:
            metadata: List of window metadata from preprocessing
            save_dir: Directory containing preprocessed windows
        """
        self.metadata = metadata
        self.save_dir = save_dir
        
        print(f"📁 PreprocessedWindowDataset loaded:")
        print(f"   Total windows: {len(metadata):,}")
        print(f"   Source directory: {save_dir}")
        
        # Verify a few files exist
        for i in range(min(5, len(metadata))):
            window_path = os.path.join(save_dir, metadata[i]['window_file'])
            if not os.path.exists(window_path):
                raise FileNotFoundError(f"Preprocessed window not found: {window_path}")
        
        print(f"✅ Verified preprocessed windows are available")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """ULTRA-FAST: Just load pre-processed tensor from disk."""
        try:
            meta = self.metadata[idx]
            window_path = os.path.join(self.save_dir, meta['window_file'])
            
            # Load pre-processed window (very fast!)
            window_data = torch.load(window_path, map_location='cpu')
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return dummy sample as fallback
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                    'file_id': f'dummy_{idx}',
                'metadata': {'boundary_pos': None, 'window_start': 0}
                }

class StableLocalBoundaryClassifier(nn.Module):
    """
    STABLE 128-DIM CNN classifier with modern techniques for gradient stability.
    
    Uses:
    - Residual connections to prevent gradient explosion
    - Layer normalization for gradient stability
    - Intelligent weight initialization
    - Progressive channel reduction
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=128, dropout_rate=0.3, use_residual=True, use_layer_norm=True):
        """
        Initialize the stable local boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head (128 for stable performance)
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model with explicit settings for memory efficiency
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,  # Explicit dtype
                low_cpu_mem_usage=True      # Memory efficient loading
            )
            print(f"      ✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"      ❌ Error loading Wav2Vec2: {e}")
            raise e
        
        # Freeze Wav2Vec2 parameters if specified
        if freeze_wav2vec2:
            print(f"      🔒 Freezing Wav2Vec2 parameters...")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print(f"      ✅ Wav2Vec2 parameters frozen")
        
        # Get wav2vec2 hidden size
        wav2vec2_hidden_size = self.wav2vec2.config.hidden_size
        print(f"      📏 Wav2Vec2 hidden size: {wav2vec2_hidden_size}")
        
        print(f"      🏗️ Building STABLE 128-dim classification head with modern techniques...")
        
        # STABLE PROGRESSIVE ARCHITECTURE: 768 -> 256 -> 128 -> 64
        
        # Layer 1: 768 -> 256 with residual prep
        self.conv1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(256) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128 with residual
        self.conv2 = nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64 (final features)
        self.conv3 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(64) if use_layer_norm else nn.Identity()
        
        # RESIDUAL PROJECTIONS - Match dimensions for skip connections
        if use_residual:
            self.residual_proj1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=1)  # 768->256
            self.residual_proj2 = nn.Conv1d(256, hidden_dim, kernel_size=1)            # 256->128
        
        # Position-aware final layer - focuses on END of window
        self.boundary_detector = nn.Sequential(
            nn.Linear(64, 32),  # Per-timestep features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)    # Per-timestep boundary scores
        )
        
        # INTELLIGENT WEIGHT INITIALIZATION - Modern best practices
        self._intelligent_weight_init()
        
        print(f"      ✅ STABLE 128-dim architecture built successfully")
        print(f"      🔧 Features: Residual={use_residual}, LayerNorm={use_layer_norm}")
        print(f"      🛡️ Gradient stability techniques applied")
        
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices for stability."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks - PROVEN STABLE
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier for linear layers with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu') * 0.5)  # Conservative
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Intelligent weight initialization complete")
        
    def forward(self, input_values):
        """
        Forward pass with residual connections and layer normalization.
        STABLE: Multiple techniques prevent gradient explosion.
        
        Args:
            input_values: Audio input [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Boundary logits [batch_size]
        """
        # FIXED: Since Wav2Vec2 is frozen, ALWAYS use no_grad for stability
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Detach to prevent gradient flow through frozen Wav2Vec2
        hidden_states = hidden_states.detach()
        
        # Transpose for temporal convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === LAYER 1: 768 -> 256 with optional residual ===
        if self.use_residual:
            identity1 = self.residual_proj1(x)  # Project input to 256 dims
        
        x = self.conv1(x)  # [batch, 256, seq_len]
        
        # Apply layer normalization (transpose for LayerNorm)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 256] for LayerNorm
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, 256, seq_len] back to conv format
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION - prevents gradient explosion
        if self.use_residual:
            x = x + identity1
        
        x = self.dropout1(x)
        
        # === LAYER 2: 256 -> 128 with optional residual ===
        if self.use_residual:
            identity2 = self.residual_proj2(x)  # Project to 128 dims
        
        x = self.conv2(x)  # [batch, 128, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 128]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, 128, seq_len]
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION
        if self.use_residual:
            x = x + identity2
        
        x = self.dropout2(x)
        
        # === LAYER 3: 128 -> 64 (no residual needed) ===
        x = self.conv3(x)  # [batch, 64, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 64]
            x = self.norm3(x)
            x = x.transpose(1, 2)  # [batch, 64, seq_len]
        
        x = F.relu(x)
        
        # === BOUNDARY DETECTION ===
        # Get per-timestep boundary scores
        x = x.transpose(1, 2)  # [batch, seq_len, 64] for linear layers
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # FOCUS ON THE END: Take last 20% of window (where boundary should be)
        seq_len = timestep_scores.size(1)
        end_portion = max(1, seq_len // 5)  # Last 20% of sequence
        end_scores = timestep_scores[:, -end_portion:]  # [batch, end_portion]
        
        # Max pooling over the end portion - find strongest boundary signal
        boundary_logit = torch.max(end_scores, dim=1)[0]  # [batch]
        
        return boundary_logit


# Keep the old class as LocalBoundaryClassifier for compatibility
LocalBoundaryClassifier = StableLocalBoundaryClassifier

class SafeBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE with Logits Loss with label smoothing for better calibration.
    Combines sigmoid and BCE into one operation for better numerical stability.
    """
    
    def __init__(self, pos_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Calculate BCE loss from logits with label smoothing and numerical stability.
        
        Args:
            logits: Model logits [batch_size] (NOT probabilities)
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: BCE loss with label smoothing
        """
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Label smoothing: 0 -> eps/2, 1 -> 1-eps/2
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Use BCEWithLogitsLoss for numerical stability (safe with mixed precision)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    
    Args:
        model: StableLocalBoundaryClassifier instance
        train_dataloader: Training data loader
"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
        Returns metadata for the PreprocessedWindowDataset.
        """
        metadata_file = os.path.join(self.save_dir, "window_metadata.pt")
        
        # Check if already preprocessed
        if not force_reprocess and os.path.exists(metadata_file):
            print("📁 Found existing preprocessed windows, loading metadata...")
            try:
                metadata = torch.load(metadata_file)
                print(f"✅ Loaded {len(metadata)} preprocessed windows from disk")
                return metadata
            except:
                print("⚠️ Failed to load metadata, will reprocess...")
        
        print("🔄 Pre-processing ALL windows (this may take a while)...")
        print("💡 This is done ONCE - subsequent training will be very fast!")
        
        metadata = []
        window_count = 0
        
        start_time = time.time()
        
        for sample_idx, item in enumerate(self.data):
            if sample_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Processing sample {sample_idx+1}/{len(self.data)} | "
                      f"Windows so far: {window_count:,} | Time: {elapsed:.1f}s")
            
            try:
                # Load and resample audio once per file
                audio = item['audio']['array']
                original_sr = item['audio']['sampling_rate']
                
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
                # Get boundary positions
                boundary_positions = []
                if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                    starts = item['phonetic_detail']['start']
                    stops = item['phonetic_detail']['stop']
                    
                    for start, stop in zip(starts, stops):
                        start_sample = int(start * self.sample_rate)
                        stop_sample = int(stop * self.sample_rate)
                        boundary_positions.extend([start_sample, stop_sample])
                
                boundary_positions = sorted(list(set(boundary_positions)))
                boundary_positions = [pos for pos in boundary_positions if 0 <= pos < len(audio)]
                
                # DEBUG: Show boundary info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: {len(boundary_positions)} boundaries, "
                          f"audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                
                # Generate and save positive windows
                pos_count = self._save_positive_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += pos_count
                
                # Generate and save negative windows
                neg_count = self._save_negative_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += neg_count
                
                # DEBUG: Show window generation info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: Generated {pos_count} positive, {neg_count} negative windows")
                
            except Exception as e:
                print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
                # Add more detailed debugging
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                continue
        
        # Save metadata
        torch.save(metadata, metadata_file)
        
        preprocessing_time = time.time() - start_time
        positive_count = sum(1 for w in metadata if w['label'] == 1)
        negative_count = len(metadata) - positive_count
        
        print(f"✅ Pre-processing complete!")
        print(f"📊 Statistics:")
        print(f"   Total windows: {len(metadata):,}")
        
        if len(metadata) > 0:
            print(f"   Positive windows: {positive_count:,} ({positive_count/len(metadata)*100:.1f}%)")
            print(f"   Negative windows: {negative_count:,} ({negative_count/len(metadata)*100:.1f}%)")
            if positive_count > 0:
                print(f"   Class balance ratio: {negative_count/positive_count:.1f}:1")
                print(f"   Class balance quality: {'✅ GOOD' if 1 <= negative_count/positive_count <= 10 else '⚠️ IMBALANCED'}")
            else:
                print(f"   Class balance ratio: N/A (no positive windows)")
            print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
            print(f"   Disk usage: ~{len(metadata) * 48 / 1024:.1f} MB")  # Increased for 1.5s windows
            print(f"   Average windows per audio file: {len(metadata)/len(self.data):.1f}")
        else:
            print(f"   ⚠️ WARNING: No windows were generated!")
            print(f"   Positive windows: {positive_count:,}")
            print(f"   Negative windows: {negative_count:,}")
            print(f"   This indicates a problem with the audio processing or boundary detection.")
        
        return metadata
    
    def _save_positive_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save exactly ONE positive window per boundary for proper balance."""
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # FIXED: Generate exactly ONE window per boundary at exact position
            window_end = boundary_pos
            start_pos = window_end - self.window_samples
            
            if start_pos < 0 or window_end >= len(audio):
                continue
            
            # Extract and validate window
            window_audio = audio[start_pos:window_end]
            
            if len(window_audio) != self.window_samples:
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (padding, 0), mode='constant')
                else:
                    window_audio = window_audio[:self.window_samples]
            
            if np.all(window_audio == 0):
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Save processed window
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'window_start': start_pos
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata
                metadata.append({
                    'window_file': window_filename,
                    'label': 1,
                    'file_id': file_id,
                    'boundary_pos': boundary_pos
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save negative windows with reasonable limits for balance."""
        saved_count = 0
        
        # Create exclusion zones around boundaries (for negative sampling only!)
        exclusion_zones = set()
        for boundary_pos in boundary_positions:
            for offset in range(-self.exclusion_zone_samples, self.exclusion_zone_samples + 1):
                if 0 <= boundary_pos + offset < len(audio):
                    exclusion_zones.add(boundary_pos + offset)
        
        # Find valid positions for negative windows
        valid_end_positions = []
        for end_pos in range(self.window_samples, len(audio)):
            if end_pos not in exclusion_zones:
                valid_end_positions.append(end_pos)
        
        # DEBUG: Show exclusion zone statistics for first few samples
        if sample_idx < 5:
            total_possible = len(audio) - self.window_samples
            excluded_count = len(exclusion_zones)
            valid_count = len(valid_end_positions)
            print(f"   DEBUG Negative sampling: {excluded_count:,} excluded, {valid_count:,}/{total_possible:,} valid positions")
        
        # Sample negative positions - aim for 1:1 or 1:2 ratio with positive windows
        # Since we now have exactly len(boundary_positions) positive windows, balance accordingly
        target_negative = len(boundary_positions) * 2  # 2:1 negative:positive ratio for better discrimination
        num_negative = min(
            int(len(valid_end_positions) * self.negative_sampling_ratio),
            target_negative,  # Target balanced ratio
            self.max_negative_per_file  # Still enforce upper limit
        )
        
        if num_negative > 0 and len(valid_end_positions) > 0:
            sampled_positions = random.sample(valid_end_positions, 
                                            min(num_negative, len(valid_end_positions)))
            
            for end_pos in sampled_positions:
                # Stop when we reach the maximum negative windows per file
                if saved_count >= self.max_negative_per_file:
                    break
                    
                start_pos = end_pos - self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                if len(window_audio) != self.window_samples or np.all(window_audio == 0):
                    continue
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Save processed window
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'metadata': {
                        'boundary_pos': None,
                        'window_start': start_pos
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0,
                        'file_id': file_id,
                        'boundary_pos': None
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        else:
            if sample_idx < 5:
                print(f"   DEBUG: No negative windows generated - {num_negative} calculated from {len(valid_end_positions)} valid positions")
        
        return saved_count


class PreprocessedWindowDataset(Dataset):
    """
    STAGE 2: Ultra-fast dataset that loads pre-processed windows.
    No redundant processing - just loads tensors from disk.
    """
    
    def __init__(self, metadata, save_dir="./preprocessed_windows"):
        """
        Initialize dataset with pre-processed window metadata.
        
        Args:
            metadata: List of window metadata from preprocessing
            save_dir: Directory containing preprocessed windows
        """
        self.metadata = metadata
        self.save_dir = save_dir
        
        print(f"📁 PreprocessedWindowDataset loaded:")
        print(f"   Total windows: {len(metadata):,}")
        print(f"   Source directory: {save_dir}")
        
        # Verify a few files exist
        for i in range(min(5, len(metadata))):
            window_path = os.path.join(save_dir, metadata[i]['window_file'])
            if not os.path.exists(window_path):
                raise FileNotFoundError(f"Preprocessed window not found: {window_path}")
        
        print(f"✅ Verified preprocessed windows are available")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """ULTRA-FAST: Just load pre-processed tensor from disk."""
        try:
            meta = self.metadata[idx]
            window_path = os.path.join(self.save_dir, meta['window_file'])
            
            # Load pre-processed window (very fast!)
            window_data = torch.load(window_path, map_location='cpu')
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return dummy sample as fallback
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                    'file_id': f'dummy_{idx}',
                'metadata': {'boundary_pos': None, 'window_start': 0}
                }

class StableLocalBoundaryClassifier(nn.Module):
    """
    STABLE 128-DIM CNN classifier with modern techniques for gradient stability.
    
    Uses:
    - Residual connections to prevent gradient explosion
    - Layer normalization for gradient stability
    - Intelligent weight initialization
    - Progressive channel reduction
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=128, dropout_rate=0.3, use_residual=True, use_layer_norm=True):
        """
        Initialize the stable local boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head (128 for stable performance)
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model with explicit settings for memory efficiency
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,  # Explicit dtype
                low_cpu_mem_usage=True      # Memory efficient loading
            )
            print(f"      ✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"      ❌ Error loading Wav2Vec2: {e}")
            raise e
        
        # Freeze Wav2Vec2 parameters if specified
        if freeze_wav2vec2:
            print(f"      🔒 Freezing Wav2Vec2 parameters...")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print(f"      ✅ Wav2Vec2 parameters frozen")
        
        # Get wav2vec2 hidden size
        wav2vec2_hidden_size = self.wav2vec2.config.hidden_size
        print(f"      📏 Wav2Vec2 hidden size: {wav2vec2_hidden_size}")
        
        print(f"      🏗️ Building STABLE 128-dim classification head with modern techniques...")
        
        # STABLE PROGRESSIVE ARCHITECTURE: 768 -> 256 -> 128 -> 64
        
        # Layer 1: 768 -> 256 with residual prep
        self.conv1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(256) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128 with residual
        self.conv2 = nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64 (final features)
        self.conv3 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(64) if use_layer_norm else nn.Identity()
        
        # RESIDUAL PROJECTIONS - Match dimensions for skip connections
        if use_residual:
            self.residual_proj1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=1)  # 768->256
            self.residual_proj2 = nn.Conv1d(256, hidden_dim, kernel_size=1)            # 256->128
        
        # Position-aware final layer - focuses on END of window
        self.boundary_detector = nn.Sequential(
            nn.Linear(64, 32),  # Per-timestep features
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 1)    # Per-timestep boundary scores
        )
        
        # INTELLIGENT WEIGHT INITIALIZATION - Modern best practices
        self._intelligent_weight_init()
        
        print(f"      ✅ STABLE 128-dim architecture built successfully")
        print(f"      🔧 Features: Residual={use_residual}, LayerNorm={use_layer_norm}")
        print(f"      🛡️ Gradient stability techniques applied")
        
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices for stability."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks - PROVEN STABLE
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier for linear layers with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu') * 0.5)  # Conservative
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Intelligent weight initialization complete")
        
    def forward(self, input_values):
        """
        Forward pass with residual connections and layer normalization.
        STABLE: Multiple techniques prevent gradient explosion.
        
        Args:
            input_values: Audio input [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Boundary logits [batch_size]
        """
        # FIXED: Since Wav2Vec2 is frozen, ALWAYS use no_grad for stability
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Detach to prevent gradient flow through frozen Wav2Vec2
        hidden_states = hidden_states.detach()
        
        # Transpose for temporal convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === LAYER 1: 768 -> 256 with optional residual ===
        if self.use_residual:
            identity1 = self.residual_proj1(x)  # Project input to 256 dims
        
        x = self.conv1(x)  # [batch, 256, seq_len]
        
        # Apply layer normalization (transpose for LayerNorm)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 256] for LayerNorm
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, 256, seq_len] back to conv format
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION - prevents gradient explosion
        if self.use_residual:
            x = x + identity1
        
        x = self.dropout1(x)
        
        # === LAYER 2: 256 -> 128 with optional residual ===
        if self.use_residual:
            identity2 = self.residual_proj2(x)  # Project to 128 dims
        
        x = self.conv2(x)  # [batch, 128, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 128]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, 128, seq_len]
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION
        if self.use_residual:
            x = x + identity2
        
        x = self.dropout2(x)
        
        # === LAYER 3: 128 -> 64 (no residual needed) ===
        x = self.conv3(x)  # [batch, 64, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 64]
            x = self.norm3(x)
            x = x.transpose(1, 2)  # [batch, 64, seq_len]
        
        x = F.relu(x)
        
        # === BOUNDARY DETECTION ===
        # Get per-timestep boundary scores
        x = x.transpose(1, 2)  # [batch, seq_len, 64] for linear layers
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # FOCUS ON THE END: Take last 20% of window (where boundary should be)
        seq_len = timestep_scores.size(1)
        end_portion = max(1, seq_len // 5)  # Last 20% of sequence
        end_scores = timestep_scores[:, -end_portion:]  # [batch, end_portion]
        
        # Max pooling over the end portion - find strongest boundary signal
        boundary_logit = torch.max(end_scores, dim=1)[0]  # [batch]
        
        return boundary_logit


# Keep the old class as LocalBoundaryClassifier for compatibility
LocalBoundaryClassifier = StableLocalBoundaryClassifier

class SafeBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE with Logits Loss with label smoothing for better calibration.
    Combines sigmoid and BCE into one operation for better numerical stability.
    """
    
    def __init__(self, pos_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Calculate BCE loss from logits with label smoothing and numerical stability.
        
        Args:
            logits: Model logits [batch_size] (NOT probabilities)
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: BCE loss with label smoothing
        """
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Label smoothing: 0 -> eps/2, 1 -> 1-eps/2
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Use BCEWithLogitsLoss for numerical stability (safe with mixed precision)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    
    Args:
        model: StableLocalBoundaryClassifier instance
        train_dataloader: Training data loader
"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
        Returns metadata for the PreprocessedWindowDataset.
        """
        metadata_file = os.path.join(self.save_dir, "window_metadata.pt")
        
        # Check if already preprocessed
        if not force_reprocess and os.path.exists(metadata_file):
            print("📁 Found existing preprocessed windows, loading metadata...")
            try:
                metadata = torch.load(metadata_file)
                print(f"✅ Loaded {len(metadata)} preprocessed windows from disk")
                return metadata
            except:
                print("⚠️ Failed to load metadata, will reprocess...")
        
        print("🔄 Pre-processing ALL windows (this may take a while)...")
        print("💡 This is done ONCE - subsequent training will be very fast!")
        
        metadata = []
        window_count = 0
        
        start_time = time.time()
        
        for sample_idx, item in enumerate(self.data):
            if sample_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Processing sample {sample_idx+1}/{len(self.data)} | "
                      f"Windows so far: {window_count:,} | Time: {elapsed:.1f}s")
            
            try:
                # Load and resample audio once per file
                audio = item['audio']['array']
                original_sr = item['audio']['sampling_rate']
                
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
                # Get boundary positions
                boundary_positions = []
                if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                    starts = item['phonetic_detail']['start']
                    stops = item['phonetic_detail']['stop']
                    
                    for start, stop in zip(starts, stops):
                        start_sample = int(start * self.sample_rate)
                        stop_sample = int(stop * self.sample_rate)
                        boundary_positions.extend([start_sample, stop_sample])
                
                boundary_positions = sorted(list(set(boundary_positions)))
                boundary_positions = [pos for pos in boundary_positions if 0 <= pos < len(audio)]
                
                # DEBUG: Show boundary info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: {len(boundary_positions)} boundaries, "
                          f"audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                
                # Generate and save positive windows
                pos_count = self._save_positive_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += pos_count
                
                # Generate and save negative windows
                neg_count = self._save_negative_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += neg_count
                
                # DEBUG: Show window generation info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: Generated {pos_count} positive, {neg_count} negative windows")
                
            except Exception as e:
                print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
                # Add more detailed debugging
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                continue
        
        # Save metadata
        torch.save(metadata, metadata_file)
        
        preprocessing_time = time.time() - start_time
        positive_count = sum(1 for w in metadata if w['label'] == 1)
        negative_count = len(metadata) - positive_count
        
        print(f"✅ Pre-processing complete!")
        print(f"📊 Statistics:")
        print(f"   Total windows: {len(metadata):,}")
        
        if len(metadata) > 0:
            print(f"   Positive windows: {positive_count:,} ({positive_count/len(metadata)*100:.1f}%)")
            print(f"   Negative windows: {negative_count:,} ({negative_count/len(metadata)*100:.1f}%)")
            if positive_count > 0:
                print(f"   Class balance ratio: {negative_count/positive_count:.1f}:1")
                print(f"   Class balance quality: {'✅ GOOD' if 1 <= negative_count/positive_count <= 10 else '⚠️ IMBALANCED'}")
            else:
                print(f"   Class balance ratio: N/A (no positive windows)")
            print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
            print(f"   Disk usage: ~{len(metadata) * 48 / 1024:.1f} MB")  # Increased for 1.5s windows
            print(f"   Average windows per audio file: {len(metadata)/len(self.data):.1f}")
        else:
            print(f"   ⚠️ WARNING: No windows were generated!")
            print(f"   Positive windows: {positive_count:,}")
            print(f"   Negative windows: {negative_count:,}")
            print(f"   This indicates a problem with the audio processing or boundary detection.")
        
        return metadata
    
    def _save_positive_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save exactly ONE positive window per boundary for proper balance."""
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # FIXED: Generate exactly ONE window per boundary at exact position
            window_end = boundary_pos
            start_pos = window_end - self.window_samples
            
            if start_pos < 0 or window_end >= len(audio):
                continue
            
            # Extract and validate window
            window_audio = audio[start_pos:window_end]
            
            if len(window_audio) != self.window_samples:
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (padding, 0), mode='constant')
                else:
                    window_audio = window_audio[:self.window_samples]
            
            if np.all(window_audio == 0):
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Save processed window
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'window_start': start_pos
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata
                metadata.append({
                    'window_file': window_filename,
                    'label': 1,
                    'file_id': file_id,
                    'boundary_pos': boundary_pos
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save negative windows with reasonable limits for balance."""
        saved_count = 0
        
        # Create exclusion zones around boundaries (for negative sampling only!)
        exclusion_zones = set()
        for boundary_pos in boundary_positions:
            for offset in range(-self.exclusion_zone_samples, self.exclusion_zone_samples + 1):
                if 0 <= boundary_pos + offset < len(audio):
                    exclusion_zones.add(boundary_pos + offset)
        
        # Find valid positions for negative windows
        valid_end_positions = []
        for end_pos in range(self.window_samples, len(audio)):
            if end_pos not in exclusion_zones:
                valid_end_positions.append(end_pos)
        
        # DEBUG: Show exclusion zone statistics for first few samples
        if sample_idx < 5:
            total_possible = len(audio) - self.window_samples
            excluded_count = len(exclusion_zones)
            valid_count = len(valid_end_positions)
            print(f"   DEBUG Negative sampling: {excluded_count:,} excluded, {valid_count:,}/{total_possible:,} valid positions")
        
        # Sample negative positions - aim for 1:1 or 1:2 ratio with positive windows
        # Since we now have exactly len(boundary_positions) positive windows, balance accordingly
        target_negative = len(boundary_positions) * 2  # 2:1 negative:positive ratio for better discrimination
        num_negative = min(
            int(len(valid_end_positions) * self.negative_sampling_ratio),
            target_negative,  # Target balanced ratio
            self.max_negative_per_file  # Still enforce upper limit
        )
        
        if num_negative > 0 and len(valid_end_positions) > 0:
            sampled_positions = random.sample(valid_end_positions, 
                                            min(num_negative, len(valid_end_positions)))
            
            for end_pos in sampled_positions:
                # Stop when we reach the maximum negative windows per file
                if saved_count >= self.max_negative_per_file:
                    break
                    
                start_pos = end_pos - self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                if len(window_audio) != self.window_samples or np.all(window_audio == 0):
                    continue
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Save processed window
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'metadata': {
                        'boundary_pos': None,
                        'window_start': start_pos
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0,
                        'file_id': file_id,
                        'boundary_pos': None
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        else:
            if sample_idx < 5:
                print(f"   DEBUG: No negative windows generated - {num_negative} calculated from {len(valid_end_positions)} valid positions")
        
        return saved_count


class PreprocessedWindowDataset(Dataset):
    """
    STAGE 2: Ultra-fast dataset that loads pre-processed windows.
    No redundant processing - just loads tensors from disk.
    """
    
    def __init__(self, metadata, save_dir="./preprocessed_windows"):
        """
        Initialize dataset with pre-processed window metadata.
        
        Args:
            metadata: List of window metadata from preprocessing
            save_dir: Directory containing preprocessed windows
        """
        self.metadata = metadata
        self.save_dir = save_dir
        
        print(f"📁 PreprocessedWindowDataset loaded:")
        print(f"   Total windows: {len(metadata):,}")
        print(f"   Source directory: {save_dir}")
        
        # Verify a few files exist
        for i in range(min(5, len(metadata))):
            window_path = os.path.join(save_dir, metadata[i]['window_file'])
            if not os.path.exists(window_path):
                raise FileNotFoundError(f"Preprocessed window not found: {window_path}")
        
        print(f"✅ Verified preprocessed windows are available")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """ULTRA-FAST: Just load pre-processed tensor from disk."""
        try:
            meta = self.metadata[idx]
            window_path = os.path.join(self.save_dir, meta['window_file'])
            
            # Load pre-processed window (very fast!)
            window_data = torch.load(window_path, map_location='cpu')
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return dummy sample as fallback
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                    'file_id': f'dummy_{idx}',
                'metadata': {'boundary_pos': None, 'window_start': 0}
                }

class StableLocalBoundaryClassifier(nn.Module):
    """
    STABLE 128-DIM CNN classifier with modern techniques for gradient stability.
    
    Uses:
    - Residual connections to prevent gradient explosion
    - Layer normalization for gradient stability
    - Intelligent weight initialization
    - Progressive channel reduction
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=128, dropout_rate=0.3, use_residual=True, use_layer_norm=True):
        """
        Initialize the stable local boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head (128 for stable performance)
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model with explicit settings for memory efficiency
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,  # Explicit dtype
                low_cpu_mem_usage=True      # Memory efficient loading
            )
            print(f"      ✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"      ❌ Error loading Wav2Vec2: {e}")
            raise e
        
        # Freeze Wav2Vec2 parameters if specified
        if freeze_wav2vec2:
            print(f"      🔒 Freezing Wav2Vec2 parameters...")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print(f"      ✅ Wav2Vec2 parameters frozen")
        
        # Get wav2vec2 hidden size
        wav2vec2_hidden_size = self.wav2vec2.config.hidden_size
        print(f"      📏 Wav2Vec2 hidden size: {wav2vec2_hidden_size}")
        
        print(f"      🏗️ Building STABLE 128-dim classification head with modern techniques...")
        
        # STABLE PROGRESSIVE ARCHITECTURE: 768 -> 256 -> 128 -> 64
        
        # Layer 1: 768 -> 256 with residual prep
        self.conv1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(256) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128 with residual
        self.conv2 = nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64 (final features)
        self.conv3 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(64) if use_layer_norm else nn.Identity()
        
        # RESIDUAL PROJECTIONS - Match dimensions for skip connections
        if use_residual:
            self.residual_proj1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=1)  # 768->256
            self.residual_proj2 = nn.Conv1d(256, hidden_dim, kernel_size=1)            # 256->128
        
        # Position-aware final layer - focuses on END of window
        self.boundary_detector = nn.Sequential(
            nn.Linear(64, 32),  # Per-timestep features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)    # Per-timestep boundary scores
        )
        
        # INTELLIGENT WEIGHT INITIALIZATION - Modern best practices
        self._intelligent_weight_init()
        
        print(f"      ✅ STABLE 128-dim architecture built successfully")
        print(f"      🔧 Features: Residual={use_residual}, LayerNorm={use_layer_norm}")
        print(f"      🛡️ Gradient stability techniques applied")
        
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices for stability."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks - PROVEN STABLE
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier for linear layers with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu') * 0.5)  # Conservative
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Intelligent weight initialization complete")
        
    def forward(self, input_values):
        """
        Forward pass with residual connections and layer normalization.
        STABLE: Multiple techniques prevent gradient explosion.
        
        Args:
            input_values: Audio input [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Boundary logits [batch_size]
        """
        # FIXED: Since Wav2Vec2 is frozen, ALWAYS use no_grad for stability
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Detach to prevent gradient flow through frozen Wav2Vec2
        hidden_states = hidden_states.detach()
        
        # Transpose for temporal convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === LAYER 1: 768 -> 256 with optional residual ===
        if self.use_residual:
            identity1 = self.residual_proj1(x)  # Project input to 256 dims
        
        x = self.conv1(x)  # [batch, 256, seq_len]
        
        # Apply layer normalization (transpose for LayerNorm)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 256] for LayerNorm
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, 256, seq_len] back to conv format
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION - prevents gradient explosion
        if self.use_residual:
            x = x + identity1
        
        x = self.dropout1(x)
        
        # === LAYER 2: 256 -> 128 with optional residual ===
        if self.use_residual:
            identity2 = self.residual_proj2(x)  # Project to 128 dims
        
        x = self.conv2(x)  # [batch, 128, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 128]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, 128, seq_len]
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION
        if self.use_residual:
            x = x + identity2
        
        x = self.dropout2(x)
        
        # === LAYER 3: 128 -> 64 (no residual needed) ===
        x = self.conv3(x)  # [batch, 64, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 64]
            x = self.norm3(x)
            x = x.transpose(1, 2)  # [batch, 64, seq_len]
        
        x = F.relu(x)
        
        # === BOUNDARY DETECTION ===
        # Get per-timestep boundary scores
        x = x.transpose(1, 2)  # [batch, seq_len, 64] for linear layers
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # FOCUS ON THE END: Take last 20% of window (where boundary should be)
        seq_len = timestep_scores.size(1)
        end_portion = max(1, seq_len // 5)  # Last 20% of sequence
        end_scores = timestep_scores[:, -end_portion:]  # [batch, end_portion]
        
        # Max pooling over the end portion - find strongest boundary signal
        boundary_logit = torch.max(end_scores, dim=1)[0]  # [batch]
        
        return boundary_logit


# Keep the old class as LocalBoundaryClassifier for compatibility
LocalBoundaryClassifier = StableLocalBoundaryClassifier

class SafeBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE with Logits Loss with label smoothing for better calibration.
    Combines sigmoid and BCE into one operation for better numerical stability.
    """
    
    def __init__(self, pos_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Calculate BCE loss from logits with label smoothing and numerical stability.
        
        Args:
            logits: Model logits [batch_size] (NOT probabilities)
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: BCE loss with label smoothing
        """
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Label smoothing: 0 -> eps/2, 1 -> 1-eps/2
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Use BCEWithLogitsLoss for numerical stability (safe with mixed precision)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    
    Args:
        model: StableLocalBoundaryClassifier instance
        train_dataloader: Training data loader
"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
        Returns metadata for the PreprocessedWindowDataset.
        """
        metadata_file = os.path.join(self.save_dir, "window_metadata.pt")
        
        # Check if already preprocessed
        if not force_reprocess and os.path.exists(metadata_file):
            print("📁 Found existing preprocessed windows, loading metadata...")
            try:
                metadata = torch.load(metadata_file)
                print(f"✅ Loaded {len(metadata)} preprocessed windows from disk")
                return metadata
            except:
                print("⚠️ Failed to load metadata, will reprocess...")
        
        print("🔄 Pre-processing ALL windows (this may take a while)...")
        print("💡 This is done ONCE - subsequent training will be very fast!")
        
        metadata = []
        window_count = 0
        
        start_time = time.time()
        
        for sample_idx, item in enumerate(self.data):
            if sample_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Processing sample {sample_idx+1}/{len(self.data)} | "
                      f"Windows so far: {window_count:,} | Time: {elapsed:.1f}s")
            
            try:
                # Load and resample audio once per file
                audio = item['audio']['array']
                original_sr = item['audio']['sampling_rate']
                
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
                # Get boundary positions
                boundary_positions = []
                if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                    starts = item['phonetic_detail']['start']
                    stops = item['phonetic_detail']['stop']
                    
                    for start, stop in zip(starts, stops):
                        start_sample = int(start * self.sample_rate)
                        stop_sample = int(stop * self.sample_rate)
                        boundary_positions.extend([start_sample, stop_sample])
                
                boundary_positions = sorted(list(set(boundary_positions)))
                boundary_positions = [pos for pos in boundary_positions if 0 <= pos < len(audio)]
                
                # DEBUG: Show boundary info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: {len(boundary_positions)} boundaries, "
                          f"audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                
                # Generate and save positive windows
                pos_count = self._save_positive_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += pos_count
                
                # Generate and save negative windows
                neg_count = self._save_negative_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += neg_count
                
                # DEBUG: Show window generation info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: Generated {pos_count} positive, {neg_count} negative windows")
                
            except Exception as e:
                print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
                # Add more detailed debugging
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                continue
        
        # Save metadata
        torch.save(metadata, metadata_file)
        
        preprocessing_time = time.time() - start_time
        positive_count = sum(1 for w in metadata if w['label'] == 1)
        negative_count = len(metadata) - positive_count
        
        print(f"✅ Pre-processing complete!")
        print(f"📊 Statistics:")
        print(f"   Total windows: {len(metadata):,}")
        
        if len(metadata) > 0:
            print(f"   Positive windows: {positive_count:,} ({positive_count/len(metadata)*100:.1f}%)")
            print(f"   Negative windows: {negative_count:,} ({negative_count/len(metadata)*100:.1f}%)")
            if positive_count > 0:
                print(f"   Class balance ratio: {negative_count/positive_count:.1f}:1")
                print(f"   Class balance quality: {'✅ GOOD' if 1 <= negative_count/positive_count <= 10 else '⚠️ IMBALANCED'}")
            else:
                print(f"   Class balance ratio: N/A (no positive windows)")
            print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
            print(f"   Disk usage: ~{len(metadata) * 48 / 1024:.1f} MB")  # Increased for 1.5s windows
            print(f"   Average windows per audio file: {len(metadata)/len(self.data):.1f}")
        else:
            print(f"   ⚠️ WARNING: No windows were generated!")
            print(f"   Positive windows: {positive_count:,}")
            print(f"   Negative windows: {negative_count:,}")
            print(f"   This indicates a problem with the audio processing or boundary detection.")
        
        return metadata
    
    def _save_positive_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save exactly ONE positive window per boundary for proper balance."""
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # FIXED: Generate exactly ONE window per boundary at exact position
            window_end = boundary_pos
            start_pos = window_end - self.window_samples
            
            if start_pos < 0 or window_end >= len(audio):
                continue
            
            # Extract and validate window
            window_audio = audio[start_pos:window_end]
            
            if len(window_audio) != self.window_samples:
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (padding, 0), mode='constant')
                else:
                    window_audio = window_audio[:self.window_samples]
            
            if np.all(window_audio == 0):
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Save processed window
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'window_start': start_pos
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata
                metadata.append({
                    'window_file': window_filename,
                    'label': 1,
                    'file_id': file_id,
                    'boundary_pos': boundary_pos
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save negative windows with reasonable limits for balance."""
        saved_count = 0
        
        # Create exclusion zones around boundaries (for negative sampling only!)
        exclusion_zones = set()
        for boundary_pos in boundary_positions:
            for offset in range(-self.exclusion_zone_samples, self.exclusion_zone_samples + 1):
                if 0 <= boundary_pos + offset < len(audio):
                    exclusion_zones.add(boundary_pos + offset)
        
        # Find valid positions for negative windows
        valid_end_positions = []
        for end_pos in range(self.window_samples, len(audio)):
            if end_pos not in exclusion_zones:
                valid_end_positions.append(end_pos)
        
        # DEBUG: Show exclusion zone statistics for first few samples
        if sample_idx < 5:
            total_possible = len(audio) - self.window_samples
            excluded_count = len(exclusion_zones)
            valid_count = len(valid_end_positions)
            print(f"   DEBUG Negative sampling: {excluded_count:,} excluded, {valid_count:,}/{total_possible:,} valid positions")
        
        # Sample negative positions - aim for 1:1 or 1:2 ratio with positive windows
        # Since we now have exactly len(boundary_positions) positive windows, balance accordingly
        target_negative = len(boundary_positions) * 2  # 2:1 negative:positive ratio for better discrimination
        num_negative = min(
            int(len(valid_end_positions) * self.negative_sampling_ratio),
            target_negative,  # Target balanced ratio
            self.max_negative_per_file  # Still enforce upper limit
        )
        
        if num_negative > 0 and len(valid_end_positions) > 0:
            sampled_positions = random.sample(valid_end_positions, 
                                            min(num_negative, len(valid_end_positions)))
            
            for end_pos in sampled_positions:
                # Stop when we reach the maximum negative windows per file
                if saved_count >= self.max_negative_per_file:
                    break
                    
                start_pos = end_pos - self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                if len(window_audio) != self.window_samples or np.all(window_audio == 0):
                    continue
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Save processed window
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'metadata': {
                        'boundary_pos': None,
                        'window_start': start_pos
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0,
                        'file_id': file_id,
                        'boundary_pos': None
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        else:
            if sample_idx < 5:
                print(f"   DEBUG: No negative windows generated - {num_negative} calculated from {len(valid_end_positions)} valid positions")
        
        return saved_count


class PreprocessedWindowDataset(Dataset):
    """
    STAGE 2: Ultra-fast dataset that loads pre-processed windows.
    No redundant processing - just loads tensors from disk.
    """
    
    def __init__(self, metadata, save_dir="./preprocessed_windows"):
        """
        Initialize dataset with pre-processed window metadata.
        
        Args:
            metadata: List of window metadata from preprocessing
            save_dir: Directory containing preprocessed windows
        """
        self.metadata = metadata
        self.save_dir = save_dir
        
        print(f"📁 PreprocessedWindowDataset loaded:")
        print(f"   Total windows: {len(metadata):,}")
        print(f"   Source directory: {save_dir}")
        
        # Verify a few files exist
        for i in range(min(5, len(metadata))):
            window_path = os.path.join(save_dir, metadata[i]['window_file'])
            if not os.path.exists(window_path):
                raise FileNotFoundError(f"Preprocessed window not found: {window_path}")
        
        print(f"✅ Verified preprocessed windows are available")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """ULTRA-FAST: Just load pre-processed tensor from disk."""
        try:
            meta = self.metadata[idx]
            window_path = os.path.join(self.save_dir, meta['window_file'])
            
            # Load pre-processed window (very fast!)
            window_data = torch.load(window_path, map_location='cpu')
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return dummy sample as fallback
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                    'file_id': f'dummy_{idx}',
                'metadata': {'boundary_pos': None, 'window_start': 0}
                }

class StableLocalBoundaryClassifier(nn.Module):
    """
    STABLE 128-DIM CNN classifier with modern techniques for gradient stability.
    
    Uses:
    - Residual connections to prevent gradient explosion
    - Layer normalization for gradient stability
    - Intelligent weight initialization
    - Progressive channel reduction
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=128, dropout_rate=0.3, use_residual=True, use_layer_norm=True):
        """
        Initialize the stable local boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head (128 for stable performance)
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model with explicit settings for memory efficiency
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,  # Explicit dtype
                low_cpu_mem_usage=True      # Memory efficient loading
            )
            print(f"      ✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"      ❌ Error loading Wav2Vec2: {e}")
            raise e
        
        # Freeze Wav2Vec2 parameters if specified
        if freeze_wav2vec2:
            print(f"      🔒 Freezing Wav2Vec2 parameters...")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print(f"      ✅ Wav2Vec2 parameters frozen")
        
        # Get wav2vec2 hidden size
        wav2vec2_hidden_size = self.wav2vec2.config.hidden_size
        print(f"      📏 Wav2Vec2 hidden size: {wav2vec2_hidden_size}")
        
        print(f"      🏗️ Building STABLE 128-dim classification head with modern techniques...")
        
        # STABLE PROGRESSIVE ARCHITECTURE: 768 -> 256 -> 128 -> 64
        
        # Layer 1: 768 -> 256 with residual prep
        self.conv1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(256) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128 with residual
        self.conv2 = nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64 (final features)
        self.conv3 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(64) if use_layer_norm else nn.Identity()
        
        # RESIDUAL PROJECTIONS - Match dimensions for skip connections
        if use_residual:
            self.residual_proj1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=1)  # 768->256
            self.residual_proj2 = nn.Conv1d(256, hidden_dim, kernel_size=1)            # 256->128
        
        # Position-aware final layer - focuses on END of window
        self.boundary_detector = nn.Sequential(
            nn.Linear(64, 32),  # Per-timestep features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)    # Per-timestep boundary scores
        )
        
        # INTELLIGENT WEIGHT INITIALIZATION - Modern best practices
        self._intelligent_weight_init()
        
        print(f"      ✅ STABLE 128-dim architecture built successfully")
        print(f"      🔧 Features: Residual={use_residual}, LayerNorm={use_layer_norm}")
        print(f"      🛡️ Gradient stability techniques applied")
        
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices for stability."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks - PROVEN STABLE
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier for linear layers with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu') * 0.5)  # Conservative
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Intelligent weight initialization complete")
        
    def forward(self, input_values):
        """
        Forward pass with residual connections and layer normalization.
        STABLE: Multiple techniques prevent gradient explosion.
        
        Args:
            input_values: Audio input [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Boundary logits [batch_size]
        """
        # FIXED: Since Wav2Vec2 is frozen, ALWAYS use no_grad for stability
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Detach to prevent gradient flow through frozen Wav2Vec2
        hidden_states = hidden_states.detach()
        
        # Transpose for temporal convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === LAYER 1: 768 -> 256 with optional residual ===
        if self.use_residual:
            identity1 = self.residual_proj1(x)  # Project input to 256 dims
        
        x = self.conv1(x)  # [batch, 256, seq_len]
        
        # Apply layer normalization (transpose for LayerNorm)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 256] for LayerNorm
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, 256, seq_len] back to conv format
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION - prevents gradient explosion
        if self.use_residual:
            x = x + identity1
        
        x = self.dropout1(x)
        
        # === LAYER 2: 256 -> 128 with optional residual ===
        if self.use_residual:
            identity2 = self.residual_proj2(x)  # Project to 128 dims
        
        x = self.conv2(x)  # [batch, 128, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 128]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, 128, seq_len]
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION
        if self.use_residual:
            x = x + identity2
        
        x = self.dropout2(x)
        
        # === LAYER 3: 128 -> 64 (no residual needed) ===
        x = self.conv3(x)  # [batch, 64, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 64]
            x = self.norm3(x)
            x = x.transpose(1, 2)  # [batch, 64, seq_len]
        
        x = F.relu(x)
        
        # === BOUNDARY DETECTION ===
        # Get per-timestep boundary scores
        x = x.transpose(1, 2)  # [batch, seq_len, 64] for linear layers
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # FOCUS ON THE END: Take last 20% of window (where boundary should be)
        seq_len = timestep_scores.size(1)
        end_portion = max(1, seq_len // 5)  # Last 20% of sequence
        end_scores = timestep_scores[:, -end_portion:]  # [batch, end_portion]
        
        # Max pooling over the end portion - find strongest boundary signal
        boundary_logit = torch.max(end_scores, dim=1)[0]  # [batch]
        
        return boundary_logit


# Keep the old class as LocalBoundaryClassifier for compatibility
LocalBoundaryClassifier = StableLocalBoundaryClassifier

class SafeBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE with Logits Loss with label smoothing for better calibration.
    Combines sigmoid and BCE into one operation for better numerical stability.
    """
    
    def __init__(self, pos_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Calculate BCE loss from logits with label smoothing and numerical stability.
        
        Args:
            logits: Model logits [batch_size] (NOT probabilities)
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: BCE loss with label smoothing
        """
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Label smoothing: 0 -> eps/2, 1 -> 1-eps/2
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Use BCEWithLogitsLoss for numerical stability (safe with mixed precision)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    
    Args:
        model: StableLocalBoundaryClassifier instance
        train_dataloader: Training data loader
"""
LOCAL WINDOW BOUNDARY DETECTION - A Smarter Approach
=====================================================

🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification

Instead of the problematic sequence-to-sequence approach, this implements:
1. 0.5s sliding windows with binary classification
2. Smart sampling: positive windows end at boundaries, negative windows avoid boundaries
3. Simple CNN classifier instead of complex sequence model
4. Direct binary predictions with post-processing grouping

This approach should be much more effective because:
- Clear binary task instead of sparse sequence labeling
- Balanced training data instead of 1% positive class
- Strong local patterns instead of weak global signals
- No complex peak detection needed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from typing import List, Tuple, Dict, Any
import warnings
import time
from datetime import datetime
import random
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WindowPreprocessor:
    """
    STAGE 1: Pre-process all windows once and save to disk.
    This eliminates redundant processing across epochs.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.08, negative_exclusion_zone=0.05, 
                 negative_sampling_ratio=0.3, save_dir="./preprocessed_windows",
                 max_windows_per_file=20, max_positive_per_file=None, max_negative_per_file=50):
        """
        Initialize the window preprocessor.
        
        Args:
            save_dir: Directory to save preprocessed windows
            max_windows_per_file: Maximum total windows per audio file (None = no limit)
            max_positive_per_file: Maximum positive windows per audio file (None = ALL boundaries)
            max_negative_per_file: Maximum negative windows per audio file
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)
        self.exclusion_zone_samples = int(negative_exclusion_zone * sample_rate)
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        
        # FIXED: Handle None values properly
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file  # None = ALL boundaries
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🔧 WindowPreprocessor Configuration:")
        print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"   Boundary tolerance: {boundary_tolerance}s ({self.boundary_tolerance_samples} samples)")
        print(f"   Negative exclusion zone: {negative_exclusion_zone}s ({self.exclusion_zone_samples} samples)")
        print(f"   Negative sampling ratio: {negative_sampling_ratio}")
        
        # IMPROVED: Show proper limits
        pos_limit = "ALL boundaries" if max_positive_per_file is None else str(max_positive_per_file)
        total_limit = "No limit" if max_windows_per_file is None else str(max_windows_per_file)
        print(f"   Max positive per file: {pos_limit}")
        print(f"   Max negative per file: {max_negative_per_file}")
        print(f"   Max total per file: {total_limit}")
        print(f"   Save directory: {save_dir}")
    
    def preprocess_all_windows(self, force_reprocess=False):
        """
        Pre-process ALL windows and save to disk.
        Returns metadata for the PreprocessedWindowDataset.
        """
        metadata_file = os.path.join(self.save_dir, "window_metadata.pt")
        
        # Check if already preprocessed
        if not force_reprocess and os.path.exists(metadata_file):
            print("📁 Found existing preprocessed windows, loading metadata...")
            try:
                metadata = torch.load(metadata_file)
                print(f"✅ Loaded {len(metadata)} preprocessed windows from disk")
                return metadata
            except:
                print("⚠️ Failed to load metadata, will reprocess...")
        
        print("🔄 Pre-processing ALL windows (this may take a while)...")
        print("💡 This is done ONCE - subsequent training will be very fast!")
        
        metadata = []
        window_count = 0
        
        start_time = time.time()
        
        for sample_idx, item in enumerate(self.data):
            if sample_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Processing sample {sample_idx+1}/{len(self.data)} | "
                      f"Windows so far: {window_count:,} | Time: {elapsed:.1f}s")
            
            try:
                # Load and resample audio once per file
                audio = item['audio']['array']
                original_sr = item['audio']['sampling_rate']
                
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
                # Get boundary positions
                boundary_positions = []
                if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                    starts = item['phonetic_detail']['start']
                    stops = item['phonetic_detail']['stop']
                    
                    for start, stop in zip(starts, stops):
                        start_sample = int(start * self.sample_rate)
                        stop_sample = int(stop * self.sample_rate)
                        boundary_positions.extend([start_sample, stop_sample])
                
                boundary_positions = sorted(list(set(boundary_positions)))
                boundary_positions = [pos for pos in boundary_positions if 0 <= pos < len(audio)]
                
                # DEBUG: Show boundary info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: {len(boundary_positions)} boundaries, "
                          f"audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                
                # Generate and save positive windows
                pos_count = self._save_positive_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += pos_count
                
                # Generate and save negative windows
                neg_count = self._save_negative_windows(
                    audio, boundary_positions, item.get('id', f'sample_{sample_idx}'), 
                    sample_idx, metadata
                )
                window_count += neg_count
                
                # DEBUG: Show window generation info for first few samples
                if sample_idx < 5:
                    print(f"   DEBUG Sample {sample_idx}: Generated {pos_count} positive, {neg_count} negative windows")
                
            except Exception as e:
                print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
                # Add more detailed debugging
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                continue
        
        # Save metadata
        torch.save(metadata, metadata_file)
        
        preprocessing_time = time.time() - start_time
        positive_count = sum(1 for w in metadata if w['label'] == 1)
        negative_count = len(metadata) - positive_count
        
        print(f"✅ Pre-processing complete!")
        print(f"📊 Statistics:")
        print(f"   Total windows: {len(metadata):,}")
        
        if len(metadata) > 0:
            print(f"   Positive windows: {positive_count:,} ({positive_count/len(metadata)*100:.1f}%)")
            print(f"   Negative windows: {negative_count:,} ({negative_count/len(metadata)*100:.1f}%)")
            if positive_count > 0:
                print(f"   Class balance ratio: {negative_count/positive_count:.1f}:1")
                print(f"   Class balance quality: {'✅ GOOD' if 1 <= negative_count/positive_count <= 10 else '⚠️ IMBALANCED'}")
            else:
                print(f"   Class balance ratio: N/A (no positive windows)")
            print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
            print(f"   Disk usage: ~{len(metadata) * 48 / 1024:.1f} MB")  # Increased for 1.5s windows
            print(f"   Average windows per audio file: {len(metadata)/len(self.data):.1f}")
        else:
            print(f"   ⚠️ WARNING: No windows were generated!")
            print(f"   Positive windows: {positive_count:,}")
            print(f"   Negative windows: {negative_count:,}")
            print(f"   This indicates a problem with the audio processing or boundary detection.")
        
        return metadata
    
    def _save_positive_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save exactly ONE positive window per boundary for proper balance."""
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # FIXED: Generate exactly ONE window per boundary at exact position
            window_end = boundary_pos
            start_pos = window_end - self.window_samples
            
            if start_pos < 0 or window_end >= len(audio):
                continue
            
            # Extract and validate window
            window_audio = audio[start_pos:window_end]
            
            if len(window_audio) != self.window_samples:
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (padding, 0), mode='constant')
                else:
                    window_audio = window_audio[:self.window_samples]
            
            if np.all(window_audio == 0):
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Save processed window
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'window_start': start_pos
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata
                metadata.append({
                    'window_file': window_filename,
                    'label': 1,
                    'file_id': file_id,
                    'boundary_pos': boundary_pos
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """Generate and save negative windows with reasonable limits for balance."""
        saved_count = 0
        
        # Create exclusion zones around boundaries (for negative sampling only!)
        exclusion_zones = set()
        for boundary_pos in boundary_positions:
            for offset in range(-self.exclusion_zone_samples, self.exclusion_zone_samples + 1):
                if 0 <= boundary_pos + offset < len(audio):
                    exclusion_zones.add(boundary_pos + offset)
        
        # Find valid positions for negative windows
        valid_end_positions = []
        for end_pos in range(self.window_samples, len(audio)):
            if end_pos not in exclusion_zones:
                valid_end_positions.append(end_pos)
        
        # DEBUG: Show exclusion zone statistics for first few samples
        if sample_idx < 5:
            total_possible = len(audio) - self.window_samples
            excluded_count = len(exclusion_zones)
            valid_count = len(valid_end_positions)
            print(f"   DEBUG Negative sampling: {excluded_count:,} excluded, {valid_count:,}/{total_possible:,} valid positions")
        
        # Sample negative positions - aim for 1:1 or 1:2 ratio with positive windows
        # Since we now have exactly len(boundary_positions) positive windows, balance accordingly
        target_negative = len(boundary_positions) * 2  # 2:1 negative:positive ratio for better discrimination
        num_negative = min(
            int(len(valid_end_positions) * self.negative_sampling_ratio),
            target_negative,  # Target balanced ratio
            self.max_negative_per_file  # Still enforce upper limit
        )
        
        if num_negative > 0 and len(valid_end_positions) > 0:
            sampled_positions = random.sample(valid_end_positions, 
                                            min(num_negative, len(valid_end_positions)))
            
            for end_pos in sampled_positions:
                # Stop when we reach the maximum negative windows per file
                if saved_count >= self.max_negative_per_file:
                    break
                    
                start_pos = end_pos - self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                if len(window_audio) != self.window_samples or np.all(window_audio == 0):
                    continue
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Save processed window
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'metadata': {
                        'boundary_pos': None,
                        'window_start': start_pos
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0,
                        'file_id': file_id,
                        'boundary_pos': None
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        else:
            if sample_idx < 5:
                print(f"   DEBUG: No negative windows generated - {num_negative} calculated from {len(valid_end_positions)} valid positions")
        
        return saved_count


class PreprocessedWindowDataset(Dataset):
    """
    STAGE 2: Ultra-fast dataset that loads pre-processed windows.
    No redundant processing - just loads tensors from disk.
    """
    
    def __init__(self, metadata, save_dir="./preprocessed_windows"):
        """
        Initialize dataset with pre-processed window metadata.
        
        Args:
            metadata: List of window metadata from preprocessing
            save_dir: Directory containing preprocessed windows
        """
        self.metadata = metadata
        self.save_dir = save_dir
        
        print(f"📁 PreprocessedWindowDataset loaded:")
        print(f"   Total windows: {len(metadata):,}")
        print(f"   Source directory: {save_dir}")
        
        # Verify a few files exist
        for i in range(min(5, len(metadata))):
            window_path = os.path.join(save_dir, metadata[i]['window_file'])
            if not os.path.exists(window_path):
                raise FileNotFoundError(f"Preprocessed window not found: {window_path}")
        
        print(f"✅ Verified preprocessed windows are available")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """ULTRA-FAST: Just load pre-processed tensor from disk."""
        try:
            meta = self.metadata[idx]
            window_path = os.path.join(self.save_dir, meta['window_file'])
            
            # Load pre-processed window (very fast!)
            window_data = torch.load(window_path, map_location='cpu')
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return dummy sample as fallback
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                    'file_id': f'dummy_{idx}',
                'metadata': {'boundary_pos': None, 'window_start': 0}
                }

class StableLocalBoundaryClassifier(nn.Module):
    """
    STABLE 128-DIM CNN classifier with modern techniques for gradient stability.
    
    Uses:
    - Residual connections to prevent gradient explosion
    - Layer normalization for gradient stability
    - Intelligent weight initialization
    - Progressive channel reduction
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=128, dropout_rate=0.3, use_residual=True, use_layer_norm=True):
        """
        Initialize the stable local boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head (128 for stable performance)
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model with explicit settings for memory efficiency
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,  # Explicit dtype
                low_cpu_mem_usage=True      # Memory efficient loading
            )
            print(f"      ✅ Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"      ❌ Error loading Wav2Vec2: {e}")
            raise e
        
        # Freeze Wav2Vec2 parameters if specified
        if freeze_wav2vec2:
            print(f"      🔒 Freezing Wav2Vec2 parameters...")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print(f"      ✅ Wav2Vec2 parameters frozen")
        
        # Get wav2vec2 hidden size
        wav2vec2_hidden_size = self.wav2vec2.config.hidden_size
        print(f"      📏 Wav2Vec2 hidden size: {wav2vec2_hidden_size}")
        
        print(f"      🏗️ Building STABLE 128-dim classification head with modern techniques...")
        
        # STABLE PROGRESSIVE ARCHITECTURE: 768 -> 256 -> 128 -> 64
        
        # Layer 1: 768 -> 256 with residual prep
        self.conv1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(256) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128 with residual
        self.conv2 = nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64 (final features)
        self.conv3 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(64) if use_layer_norm else nn.Identity()
        
        # RESIDUAL PROJECTIONS - Match dimensions for skip connections
        if use_residual:
            self.residual_proj1 = nn.Conv1d(wav2vec2_hidden_size, 256, kernel_size=1)  # 768->256
            self.residual_proj2 = nn.Conv1d(256, hidden_dim, kernel_size=1)            # 256->128
        
        # Position-aware final layer - focuses on END of window
        self.boundary_detector = nn.Sequential(
            nn.Linear(64, 32),  # Per-timestep features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)    # Per-timestep boundary scores
        )
        
        # INTELLIGENT WEIGHT INITIALIZATION - Modern best practices
        self._intelligent_weight_init()
        
        print(f"      ✅ STABLE 128-dim architecture built successfully")
        print(f"      🔧 Features: Residual={use_residual}, LayerNorm={use_layer_norm}")
        print(f"      🛡️ Gradient stability techniques applied")
        
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices for stability."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks - PROVEN STABLE
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier for linear layers with conservative gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu') * 0.5)  # Conservative
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Intelligent weight initialization complete")
        
    def forward(self, input_values):
        """
        Forward pass with residual connections and layer normalization.
        STABLE: Multiple techniques prevent gradient explosion.
        
        Args:
            input_values: Audio input [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Boundary logits [batch_size]
        """
        # FIXED: Since Wav2Vec2 is frozen, ALWAYS use no_grad for stability
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Detach to prevent gradient flow through frozen Wav2Vec2
        hidden_states = hidden_states.detach()
        
        # Transpose for temporal convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === LAYER 1: 768 -> 256 with optional residual ===
        if self.use_residual:
            identity1 = self.residual_proj1(x)  # Project input to 256 dims
        
        x = self.conv1(x)  # [batch, 256, seq_len]
        
        # Apply layer normalization (transpose for LayerNorm)
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 256] for LayerNorm
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, 256, seq_len] back to conv format
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION - prevents gradient explosion
        if self.use_residual:
            x = x + identity1
        
        x = self.dropout1(x)
        
        # === LAYER 2: 256 -> 128 with optional residual ===
        if self.use_residual:
            identity2 = self.residual_proj2(x)  # Project to 128 dims
        
        x = self.conv2(x)  # [batch, 128, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 128]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, 128, seq_len]
        
        x = F.relu(x)
        
        # RESIDUAL CONNECTION
        if self.use_residual:
            x = x + identity2
        
        x = self.dropout2(x)
        
        # === LAYER 3: 128 -> 64 (no residual needed) ===
        x = self.conv3(x)  # [batch, 64, seq_len]
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = x.transpose(1, 2)  # [batch, seq_len, 64]
            x = self.norm3(x)
            x = x.transpose(1, 2)  # [batch, 64, seq_len]
        
        x = F.relu(x)
        
        # === BOUNDARY DETECTION ===
        # Get per-timestep boundary scores
        x = x.transpose(1, 2)  # [batch, seq_len, 64] for linear layers
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # FOCUS ON THE END: Take last 20% of window (where boundary should be)
        seq_len = timestep_scores.size(1)
        end_portion = max(1, seq_len // 5)  # Last 20% of sequence
        end_scores = timestep_scores[:, -end_portion:]  # [batch, end_portion]
        
        # Max pooling over the end portion - find strongest boundary signal
        boundary_logit = torch.max(end_scores, dim=1)[0]  # [batch]
        
        return boundary_logit


# Keep the old class as LocalBoundaryClassifier for compatibility
LocalBoundaryClassifier = StableLocalBoundaryClassifier

class SafeBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable BCE with Logits Loss with label smoothing for better calibration.
    Combines sigmoid and BCE into one operation for better numerical stability.
    """
    
    def __init__(self, pos_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Calculate BCE loss from logits with label smoothing and numerical stability.
        
        Args:
            logits: Model logits [batch_size] (NOT probabilities)
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: BCE loss with label smoothing
        """
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Label smoothing: 0 -> eps/2, 1 -> 1-eps/2
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Use BCEWithLogitsLoss for numerical stability (safe with mixed precision)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    
    Args:
        model: LocalBoundaryClassifier instance
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        pos_weight: Weight for positive class in loss function
        use_mixed_precision: Whether to use automatic mixed precision
        early_stopping_patience: Number of epochs to wait for improvement
        min_improvement: Minimum improvement required to reset patience
        
    Returns:
        dict: Training history
    """
    print("🚀 Starting OPTIMIZED Local Window Boundary Detection Training!")
    print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Device: {device}")
    print(f"📊 Training batches: {len(train_dataloader)}")
    print(f"📊 Validation batches: {len(val_dataloader)}")
    print(f"🎯 Epochs: {num_epochs}, Learning rate: {learning_rate}")
    print(f"⚖️ Positive class weight: {pos_weight}")
    print(f"🎯 Mixed precision: {use_mixed_precision}")
    print(f"⏰ Early stopping patience: {early_stopping_patience}")
    print(f"📈 Min improvement threshold: {min_improvement}")
    print("=" * 80)
    
    # Setup training components with STRONG regularization
    criterion = SafeBCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # BETTER: Reduce LR only when validation F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Monitor F1 score (higher is better)
        factor=0.5,          # Reduce LR by half when plateauing
        patience=3,          # Wait 3 epochs before reducing
        verbose=True,        # Print when LR changes
        min_lr=1e-6         # Don't go below this LR
    )
    
    # Setup mixed precision if enabled
    scaler = None
    if use_mixed_precision and device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print(f"✅ Automatic Mixed Precision enabled")
    
    # Training history and early stopping
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rates': []
    }
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n🔄 EPOCH {epoch+1}/{num_epochs}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📚 Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"🎯 Patience: {patience_counter}/{early_stopping_patience}")
        
        if device == 'cuda':
            print(f"📊 {get_gpu_memory_info()}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # DEBUG: Check for corrupted input data (only print if there's a problem)
            if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                print(f"🚨 CORRUPTED INPUT DATA in batch {batch_idx}!")
                print(f"   Input shape: {input_values.shape}")
                print(f"   Input range: [{input_values.min():.6f}, {input_values.max():.6f}]")
                print(f"   NaN count: {torch.sum(torch.isnan(input_values))}")
                print(f"   Inf count: {torch.sum(torch.isinf(input_values))}")
                continue
            
            # Forward pass with optional mixed precision
            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(input_values)  # Get logits (not probabilities)
                    
                    loss = criterion(logits, labels)  # Safe BCEWithLogitsLoss
                    
                    # DEBUG: Check for NaN/Inf (only print if there's a problem)
                    if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                        print(f"🚨 MODEL OUTPUT IS NaN/Inf in batch {batch_idx}!")
                        print(f"   Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                        print(f"   NaN count: {torch.sum(torch.isnan(logits))}")
                        print(f"   Inf count: {torch.sum(torch.isinf(logits))}")
                        
                        # Check if model weights have become NaN
                        nan_params = []
                        for name, param in model.named_parameters():
                            if param.requires_grad and torch.any(torch.isnan(param)):
                                nan_params.append(name)
                        
                        if nan_params:
                            print(f"   🔥 MODEL WEIGHTS ARE NaN: {nan_params[:3]}...")
                            print(f"   💀 TRAINING CORRUPTED - STOPPING IMMEDIATELY!")
                            return history  # Stop training immediately
                        
                        continue
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"🚨 LOSS IS NaN/Inf in batch {batch_idx}!")
                        print(f"   Loss: {loss.item():.6f}")
                        print(f"   Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                        continue
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Check for inf/nan gradients before stepping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('gradient_clip_norm', 1.0))
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                logits = model(input_values)  # Get logits (not probabilities)
                
                loss = criterion(logits, labels)  # Safe BCEWithLogitsLoss
                
                # DEBUG: Check for NaN/Inf (only print if there's a problem)
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    print(f"🚨 MODEL OUTPUT IS NaN/Inf in batch {batch_idx}!")
                    print(f"   Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                    print(f"   NaN count: {torch.sum(torch.isnan(logits))}")
                    print(f"   Inf count: {torch.sum(torch.isinf(logits))}")
                    continue
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"🚨 LOSS IS NaN/Inf in batch {batch_idx}!")
                    print(f"   Loss: {loss.item():.6f}")
                    print(f"   Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Smart gradient clipping for stable architecture
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('gradient_clip_norm', 1.0))
                
                optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Less frequent progress logging for speed
            if batch_idx % 1000 == 0:  # Reduced from 100 to 200
                progress = (batch_idx + 1) / len(train_dataloader) * 100
                mem_info = f" | {get_gpu_memory_info()}" if device == 'cuda' else ""
                print(f"   📦 Batch {batch_idx+1:4d}/{len(train_dataloader)} ({progress:5.1f}%) | Loss: {loss.item():.4f}{mem_info}")
                
                # Clear cache less frequently for speed
                if device == 'cuda' and batch_idx % 1000 == 0:
                    clear_gpu_memory()
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        print("🔍 Starting validation...")
        val_loss, val_metrics = evaluate_local_classifier(model, val_dataloader, device, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping logic - FOCUS ON VALIDATION LOSS, NOT JUST F1
        val_loss_improved = val_loss < best_val_loss if 'best_val_loss' in locals() else True
        f1_improved = val_metrics['f1'] - best_val_f1 > min_improvement
        
        # Save model if BOTH validation loss improves AND F1 doesn't degrade significantly
        if val_loss_improved and val_metrics['f1'] >= best_val_f1 - 0.05:  # Allow 5% F1 degradation for better loss
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_local_model.pth')
            print(f"💾 New best model saved! F1: {best_val_f1:.4f}, Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
            if val_loss > best_val_loss * 1.5:  # Warn if val loss is diverging badly
                print(f"⚠️ WARNING: Validation loss diverging! Current: {val_loss:.4f}, Best: {best_val_loss:.4f}")
        
        # Check early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered! Best F1: {best_val_f1:.4f}")
            print(f"🎯 Training stopped early at epoch {epoch+1}/{num_epochs}")
            break
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start_time
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f} (Best: {best_val_f1:.4f})")
        print(f"   Val Precision: {val_metrics['precision']:.4f}")
        print(f"   Val Recall: {val_metrics['recall']:.4f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   ⏱️ Epoch Time: {epoch_time:.1f}s")
        print("=" * 80)
    
    final_epochs = len(history['train_loss'])
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation F1: {best_val_f1:.4f}")
    print(f"📊 Completed {final_epochs}/{num_epochs} epochs")
    if final_epochs < num_epochs:
        print(f"⚡ Saved {num_epochs - final_epochs} epochs with early stopping!")
    
    return history

def evaluate_local_classifier(model, dataloader, device, criterion=None, threshold=0.5):
    """
    Evaluate the local boundary classifier.
    
    Args:
        model: Trained LocalBoundaryClassifier
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)
        threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        tuple: (loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass - get logits
            logits = model(input_values)
            
            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()
            
            # Convert logits to probabilities, then to binary predictions
            probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
            # Use fixed 0.5 threshold for proper probability calibration
            binary_predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(binary_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    avg_loss = total_loss / num_batches if criterion else 0.0
    return avg_loss, metrics

def predict_boundaries_on_full_audio(model, audio, processor, device, 
                                   window_duration=1.5, stride=0.02, 
                                   sample_rate=16000, threshold=0.75, 
                                   grouping_distance=0.05):
    """
    Apply local boundary classifier to full audio with sliding windows.
    
    Args:
        model: Trained LocalBoundaryClassifier
        audio: Audio array
        processor: Wav2Vec2 processor
        device: Device for inference
        window_duration: Window duration in seconds
        stride: Stride between windows in seconds
        sample_rate: Sample rate
        threshold: Threshold for binary classification (default: 0.75)
        grouping_distance: Distance for grouping adjacent predictions
        
    Returns:
        list: Predicted boundary positions in samples
    """
    model.eval()
    
    window_samples = int(window_duration * sample_rate)
    stride_samples = int(stride * sample_rate)
    grouping_samples = int(grouping_distance * sample_rate)
    
    # Collect all window predictions
    window_predictions = []
    
    with torch.no_grad():
        for start_pos in range(0, len(audio) - window_samples + 1, stride_samples):
            end_pos = start_pos + window_samples
            window_audio = audio[start_pos:end_pos]
            
            # Process window
            inputs = processor(window_audio, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            
            # Get prediction - FIXED: Convert logits to probabilities
            logits = model(input_values)
            prediction = torch.sigmoid(logits).item()  # Convert to probability [0,1]
            
            if prediction > threshold:
                window_predictions.append({
                    'position': end_pos,  # Boundary at end of window
                    'confidence': prediction
                })
    
    # Group nearby predictions
    if not window_predictions:
        return []
    
    # Sort by position
    window_predictions.sort(key=lambda x: x['position'])
    
    # Group predictions that are close together
    grouped_boundaries = []
    current_group = [window_predictions[0]]
    
    for pred in window_predictions[1:]:
        if pred['position'] - current_group[-1]['position'] <= grouping_samples:
            current_group.append(pred)
        else:
            # Finalize current group - take position with highest confidence
            best_pred = max(current_group, key=lambda x: x['confidence'])
            grouped_boundaries.append(best_pred['position'])
            current_group = [pred]
    
    # Don't forget the last group
    if current_group:
        best_pred = max(current_group, key=lambda x: x['confidence'])
        grouped_boundaries.append(best_pred['position'])
    
    return grouped_boundaries

def load_timit_data_for_local_windows(split='train', max_samples=None):
    """Load TIMIT data for local window approach."""
    try:
        from wav2seg import load_data  # Try to import from original file
        data = load_data(split, max_samples)
        
        # DEBUG: Show the structure of first item
        if len(data) > 0:
            first_item = data[0]
            print(f"   DEBUG: First item keys: {list(first_item.keys())}")
            if 'audio' in first_item:
                print(f"   DEBUG: Audio keys: {list(first_item['audio'].keys())}")
            if 'phonetic_detail' in first_item:
                print(f"   DEBUG: Phonetic detail keys: {list(first_item['phonetic_detail'].keys())}")
                if 'start' in first_item['phonetic_detail']:
                    starts = first_item['phonetic_detail']['start']
                    print(f"   DEBUG: Found {len(starts)} phonetic segments")
        
        return data
    
    except ImportError:
        print("   ⚠️ Could not import load_data from wav2seg - returning empty data")
        return []

def evaluate_on_full_audio(model, test_data, processor, device, config):
    """
    Comprehensive evaluation on full audio files using sliding windows.
    Similar to the sequence model evaluation but for local windows.
    
    Args:
        model: Trained LocalBoundaryClassifier
        test_data: Test dataset (original TIMIT data)
        processor: Wav2Vec2 processor
        device: Device for inference
        config: Configuration dictionary
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print("🔍 Starting comprehensive full-audio evaluation...")
    print(f"📅 Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Test samples: {len(test_data)}")
    print("=" * 80)
    
    model.eval()
    results = {
        'file_results': [],
        'overall_metrics': {},
        'worst_cases': [],
        'best_cases': []
    }
    
    eval_start_time = time.time()
    tolerance_frames = int(config['boundary_tolerance'] * config.get('sample_rate', 16000))
    
    for sample_idx, item in enumerate(test_data):
        if sample_idx % 50 == 0:
            progress = sample_idx / len(test_data) * 100
            print(f"   📊 Sample {sample_idx+1:4d}/{len(test_data)} ({progress:5.1f}%)")
        
        try:
            # Extract audio and resample if necessary
            audio = item['audio']['array']
            original_sr = item['audio']['sampling_rate']
            sample_rate = config.get('sample_rate', 16000)
            
            if original_sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
            
            # Get true boundaries
            true_boundaries = []
            if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                starts = item['phonetic_detail']['start']
                stops = item['phonetic_detail']['stop']
                
                for start, stop in zip(starts, stops):
                    start_sample = int(start * sample_rate)
                    stop_sample = int(stop * sample_rate)
                    true_boundaries.extend([start_sample, stop_sample])
            
            true_boundaries = sorted(list(set(true_boundaries)))
            true_boundaries = [pos for pos in true_boundaries if 0 <= pos < len(audio)]
            
            # Predict boundaries using sliding windows
            pred_boundaries = predict_boundaries_on_full_audio(
                model, audio, processor, device,
                window_duration=config['window_duration'],  # 1.5s windows
                stride=0.02,  # 20ms stride
                sample_rate=sample_rate,
                threshold=config['threshold'],  # Use fixed 0.5 threshold
                grouping_distance=0.05  # Reduced grouping distance
            )
            
            # Calculate metrics
            mae, precision, recall, f1 = calculate_boundary_metrics_local(
                true_boundaries, pred_boundaries, tolerance_frames
            )
            
            file_result = {
                'file_id': item.get('id', f'sample_{sample_idx}'),
                'true_boundaries': true_boundaries,
                'pred_boundaries': pred_boundaries,
                'mae': mae,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_true_boundaries': len(true_boundaries),
                'num_pred_boundaries': len(pred_boundaries),
                'audio_length_frames': len(audio)
            }
            
            results['file_results'].append(file_result)
            
        except Exception as e:
            print(f"   ⚠️ Error processing sample {sample_idx}: {e}")
            continue
    
    eval_time = time.time() - eval_start_time
    
    # Calculate overall metrics
    all_maes = [r['mae'] for r in results['file_results']]
    all_precisions = [r['precision'] for r in results['file_results']]
    all_recalls = [r['recall'] for r in results['file_results']]
    all_f1s = [r['f1'] for r in results['file_results']]
    all_true_counts = [r['num_true_boundaries'] for r in results['file_results']]
    all_pred_counts = [r['num_pred_boundaries'] for r in results['file_results']]
    
    # Filter out infinite MAE values
    finite_maes = [mae for mae in all_maes if not np.isinf(mae)]
    
    results['overall_metrics'] = {
        'mean_mae': np.mean(finite_maes) if finite_maes else float('inf'),
        'std_mae': np.std(finite_maes) if finite_maes else 0.0,
        'median_mae': np.median(finite_maes) if finite_maes else float('inf'),
        'mean_precision': np.mean(all_precisions),
        'std_precision': np.std(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'std_recall': np.std(all_recalls),
        'mean_f1': np.mean(all_f1s),
        'std_f1': np.std(all_f1s),
        'median_f1': np.median(all_f1s),
        'total_samples': len(results['file_results']),
        'samples_with_infinite_mae': len(all_maes) - len(finite_maes),
        'avg_true_boundaries_per_sample': np.mean(all_true_counts),
        'avg_pred_boundaries_per_sample': np.mean(all_pred_counts),
        'evaluation_time_seconds': eval_time
    }
    
    # Find best and worst cases
    results['worst_cases'] = sorted(
        results['file_results'], 
        key=lambda x: x['f1'] if not np.isinf(x['mae']) else -1,
        reverse=False
    )[:10]
    
    results['best_cases'] = sorted(
        results['file_results'], 
        key=lambda x: x['f1'],
        reverse=True
    )[:10]
    
    print(f"\n📈 Evaluation Phase Complete:")
    print(f"   Total samples processed: {len(results['file_results'])}")
    print(f"   Total evaluation time: {eval_time:.1f}s")
    
    # Print metrics
    metrics = results['overall_metrics']
    print(f"\n📊 LOCAL WINDOW EVALUATION RESULTS:")
    print(f"   📈 MAE: {metrics['mean_mae']:.2f} ± {metrics['std_mae']:.2f} frames")
    print(f"   📈 F1: {metrics['mean_f1']:.3f} ± {metrics['std_f1']:.3f}")
    print(f"   📈 Precision: {metrics['mean_precision']:.3f} ± {metrics['std_precision']:.3f}")
    print(f"   📈 Recall: {metrics['mean_recall']:.3f} ± {metrics['std_recall']:.3f}")
    print(f"   📊 Avg true boundaries per sample: {metrics['avg_true_boundaries_per_sample']:.1f}")
    print(f"   📊 Avg pred boundaries per sample: {metrics['avg_pred_boundaries_per_sample']:.1f}")
    
    return results

def calculate_boundary_metrics_local(true_boundaries, pred_boundaries, tolerance):
    """Calculate boundary detection metrics for local window approach."""
    if len(true_boundaries) == 0 and len(pred_boundaries) == 0:
        return 0.0, 1.0, 1.0, 1.0
    
    if len(true_boundaries) == 0:
        return float('inf'), 0.0, 1.0, 0.0
    
    if len(pred_boundaries) == 0:
        return float('inf'), 1.0, 0.0, 0.0
    
    # FIXED: Convert to numpy arrays to avoid list-int subtraction errors
    true_boundaries = np.array(true_boundaries)
    pred_boundaries = np.array(pred_boundaries)
    
    # Calculate MAE
    mae_values = []
    for true_bound in true_boundaries:
        distances = np.abs(pred_boundaries - true_bound)
        min_distance = np.min(distances)
        mae_values.append(min_distance)
    
    mae = np.mean(mae_values)
    
    # Calculate precision, recall, F1
    true_positives = 0
    
    # For each predicted boundary, check if there's a true boundary within tolerance
    for pred_bound in pred_boundaries:
        distances = np.abs(true_boundaries - pred_bound)
        if np.min(distances) <= tolerance:
            true_positives += 1
    
    precision = true_positives / len(pred_boundaries) if len(pred_boundaries) > 0 else 0.0
    recall = true_positives / len(true_boundaries) if len(true_boundaries) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return mae, precision, recall, f1

def plot_local_window_results(results, history, save_path='local_window_evaluation.png'):
    """
    Create comprehensive visualization plots for local window results.
    
    Args:
        results: Evaluation results dictionary
        history: Training history
        save_path: Path to save the plots
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training History - Loss and Metrics
    plt.subplot(3, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('🚀 Local Window Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    plt.plot(history['val_f1'], label='Val F1', color='green', linewidth=2)
    plt.plot(history['val_precision'], label='Val Precision', color='orange', linewidth=1, alpha=0.7)
    plt.plot(history['val_recall'], label='Val Recall', color='red', linewidth=1, alpha=0.7)
    plt.title('🎯 Local Window Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 3)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='purple', linewidth=2)
    plt.title('📊 Local Window Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 4)
    plt.plot(history['learning_rates'], label='Learning Rate', color='orange', linewidth=2)
    plt.title('📈 Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Evaluation Results
    # MAE distribution
    maes = [r['mae'] for r in results['file_results']]
    finite_maes = [mae for mae in maes if not np.isinf(mae)]
    
    plt.subplot(3, 4, 5)
    if finite_maes:
        plt.hist(finite_maes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(finite_maes), color='red', linestyle='--', label=f'Mean: {np.mean(finite_maes):.1f}')
        plt.title(f'MAE Distribution ({len(finite_maes)}/{len(maes)} finite)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'All MAE values infinite', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('MAE Distribution (All infinite)')
    plt.xlabel('MAE (frames)')
    plt.ylabel('Frequency')
    
    # F1 score distribution
    plt.subplot(3, 4, 6)
    f1s = [r['f1'] for r in results['file_results']]
    plt.hist(f1s, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(np.mean(f1s), color='red', linestyle='--', label=f'Mean: {np.mean(f1s):.3f}')
    plt.title('F1 Score Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Precision vs Recall scatter
    plt.subplot(3, 4, 7)
    precisions = [r['precision'] for r in results['file_results']]
    recalls = [r['recall'] for r in results['file_results']]
    plt.scatter(recalls, precisions, alpha=0.6, color='green')
    plt.title('Precision vs Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    
    # Boundary count comparison
    plt.subplot(3, 4, 8)
    true_counts = [r['num_true_boundaries'] for r in results['file_results']]
    pred_counts = [r['num_pred_boundaries'] for r in results['file_results']]
    
    # FIXED: Handle empty results gracefully
    if true_counts and pred_counts:
        plt.scatter(true_counts, pred_counts, alpha=0.6, color='purple')
        max_count = max(max(true_counts), max(pred_counts))
        plt.plot([0, max_count], [0, max_count], 'r--', alpha=0.7, label='Perfect Prediction')
        plt.title('Predicted vs True Boundary Counts')
        plt.xlabel('True Boundaries')
        plt.ylabel('Predicted Boundaries')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No evaluation results available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Predicted vs True Boundary Counts (No Data)')
    
    # 3. Sample Cases
    # Best case visualization
    plt.subplot(3, 4, 9)
    if results['best_cases']:
        best_case = results['best_cases'][0]
        true_bounds = np.array(best_case['true_boundaries'])
        pred_bounds = np.array(best_case['pred_boundaries'])
        
        if len(true_bounds) > 0:
            plt.scatter(true_bounds, [1] * len(true_bounds), color='blue', label='True', s=50, marker='|')
        if len(pred_bounds) > 0:
            plt.scatter(pred_bounds, [0.5] * len(pred_bounds), color='red', label='Predicted', s=50, marker='|')
        plt.title(f'Best Case: {best_case["file_id"][:15]}\nF1: {best_case["f1"]:.3f}')
        plt.xlabel('Frame Position')
        plt.ylabel('Boundary Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No best case available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Best Case (No Data)')
    
    # Worst case visualization
    plt.subplot(3, 4, 10)
    if results['worst_cases']:
        worst_case = results['worst_cases'][0]
        true_bounds = np.array(worst_case['true_boundaries'])
        pred_bounds = np.array(worst_case['pred_boundaries'])
        
        if len(true_bounds) > 0:
            plt.scatter(true_bounds, [1] * len(true_bounds), color='blue', label='True', s=50, marker='|')
        if len(pred_bounds) > 0:
            plt.scatter(pred_bounds, [0.5] * len(pred_bounds), color='red', label='Predicted', s=50, marker='|')
        plt.title(f'Worst Case: {worst_case["file_id"][:15]}\nF1: {worst_case["f1"]:.3f}')
        plt.xlabel('Frame Position')
        plt.ylabel('Boundary Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No worst case available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Worst Case (No Data)')
    
    # Performance comparison (if we had sequence results)
    plt.subplot(3, 4, 11)
    # Placeholder for comparison with sequence approach
    categories = ['Local Window', 'Sequence (Ref)']
    f1_scores = [results['overall_metrics']['mean_f1'], 0.141]  # Reference sequence F1
    colors = ['green', 'red']
    plt.bar(categories, f1_scores, color=colors, alpha=0.7)
    plt.title('🏆 Local Window vs Sequence Approach')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    summary_text = f"""
🎯 LOCAL WINDOW RESULTS SUMMARY

📊 Performance:
   F1 Score: {results['overall_metrics']['mean_f1']:.3f} ± {results['overall_metrics']['std_f1']:.3f}
   Precision: {results['overall_metrics']['mean_precision']:.3f}
   Recall: {results['overall_metrics']['mean_recall']:.3f}
   
📈 Boundary Prediction:
   Avg True: {results['overall_metrics']['avg_true_boundaries_per_sample']:.1f}
   Avg Pred: {results['overall_metrics']['avg_pred_boundaries_per_sample']:.1f}
   Ratio: {results['overall_metrics']['avg_pred_boundaries_per_sample']/results['overall_metrics']['avg_true_boundaries_per_sample']:.2f}
   
⏱️ Performance:
   Evaluation Time: {results['overall_metrics']['evaluation_time_seconds']:.1f}s
   Samples: {results['overall_metrics']['total_samples']}
    """
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Local window plots saved to {save_path}")

def print_local_window_analysis(results, history, top_k=5):
    """
    Print detailed analysis of local window results.
    
    Args:
        results: Evaluation results
        history: Training history  
        top_k: Number of cases to show
    """
    print("\n" + "="*80)
    print("🧠 LOCAL WINDOW BOUNDARY DETECTION - DETAILED ANALYSIS")
    print("="*80)
    
    # Training summary
    print(f"🎯 TRAINING SUMMARY:")
    print(f"   Best Validation F1: {max(history['val_f1']):.4f}")
    print(f"   Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
    print(f"   Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Training Epochs: {len(history['train_loss'])}")
    
    # Evaluation summary
    metrics = results['overall_metrics']
    print(f"\n📊 FULL AUDIO EVALUATION SUMMARY:")
    print(f"   Mean F1 Score: {metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f}")
    print(f"   Median F1 Score: {metrics['median_f1']:.4f}")
    print(f"   Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"   Mean Recall: {metrics['mean_recall']:.4f}")
    if metrics['mean_mae'] != float('inf'):
        print(f"   Mean MAE: {metrics['mean_mae']:.2f} ± {metrics['std_mae']:.2f} frames")
    print(f"   Samples with predictions: {metrics['total_samples'] - metrics['samples_with_infinite_mae']}/{metrics['total_samples']}")
    
    # Boundary analysis
    print(f"\n📈 BOUNDARY PREDICTION ANALYSIS:")
    print(f"   Avg true boundaries per sample: {metrics['avg_true_boundaries_per_sample']:.1f}")
    print(f"   Avg predicted boundaries per sample: {metrics['avg_pred_boundaries_per_sample']:.1f}")
    ratio = metrics['avg_pred_boundaries_per_sample'] / metrics['avg_true_boundaries_per_sample']
    print(f"   Prediction ratio (pred/true): {ratio:.2f}")
    if ratio < 0.5:
        print(f"   ⚠️ Severe underprediction detected!")
    elif ratio > 2.0:
        print(f"   ⚠️ Overprediction detected!")
    else:
        print(f"   ✅ Reasonable prediction ratio!")
    
    # Best cases
    print(f"\n🏆 TOP {top_k} BEST CASES:")
    for i, case in enumerate(results['best_cases'][:top_k], 1):
        print(f"   {i}. {case['file_id']}: F1={case['f1']:.3f}, P={case['precision']:.3f}, R={case['recall']:.3f}")
        print(f"      True: {case['num_true_boundaries']}, Pred: {case['num_pred_boundaries']}")
    
    # Worst cases
    print(f"\n📉 TOP {top_k} WORST CASES:")
    for i, case in enumerate(results['worst_cases'][:top_k], 1):
        print(f"   {i}. {case['file_id']}: F1={case['f1']:.3f}, P={case['precision']:.3f}, R={case['recall']:.3f}")
        print(f"      True: {case['num_true_boundaries']}, Pred: {case['num_pred_boundaries']}")
        if case['num_pred_boundaries'] == 0:
            print(f"      ⚠️ No predictions made!")
    
    print("="*80)

def debug_window_predictions(model, dataloader, device, num_samples=10, threshold=0.5):
    """
    Debug function to analyze individual window predictions.
    Shows what the model is actually learning on individual windows.
    
    Args:
        model: The trained model
        dataloader: DataLoader instance
        device: Device to run on
        num_samples: Number of samples to analyze
        threshold: Threshold for binary classification
    """
    print("\n🔍 DEBUGGING WINDOW PREDICTIONS")
    print("=" * 50)
    
    model.eval()
    positive_windows = []
    negative_windows = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(positive_windows) >= num_samples and len(negative_windows) >= num_samples:
                break
                
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            file_ids = batch['file_id']
            
            # Get predictions - FIXED: Convert logits to probabilities
            logits = model(input_values)
            predictions = torch.sigmoid(logits)  # Convert to probabilities [0,1]
            
            # Separate positive and negative windows
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predictions[i].item()  # Now this is a probability 0-1
                file_id = file_ids[i]
                
                sample_info = {
                    'file_id': file_id,
                    'true_label': label,
                    'prediction': pred,
                    'binary_pred': 1 if pred > threshold else 0,  # Use configured threshold
                    'confidence': abs(pred - threshold)  # Distance from threshold
                }
                
                if label == 1 and len(positive_windows) < num_samples:
                    positive_windows.append(sample_info)
                elif label == 0 and len(negative_windows) < num_samples:
                    negative_windows.append(sample_info)
    
    print(f"📊 POSITIVE WINDOWS (Boundary endings):")
    for i, window in enumerate(positive_windows, 1):
        correct = "✅" if window['binary_pred'] == 1 else "❌"
        print(f"   {i:2d}. {window['file_id'][:20]:20s} | Pred: {window['prediction']:.3f} | "
              f"Binary: {window['binary_pred']} | Conf: {window['confidence']:.3f} {correct}")
    
    print(f"\n📊 NEGATIVE WINDOWS (Non-boundary endings):")
    for i, window in enumerate(negative_windows, 1):
        correct = "✅" if window['binary_pred'] == 0 else "❌"
        print(f"   {i:2d}. {window['file_id'][:20]:20s} | Pred: {window['prediction']:.3f} | "
              f"Binary: {window['binary_pred']} | Conf: {window['confidence']:.3f} {correct}")
    
    # Calculate accuracy on this subset
    all_windows = positive_windows + negative_windows
    correct_predictions = sum(1 for w in all_windows if w['binary_pred'] == w['true_label'])
    accuracy = correct_predictions / len(all_windows) if all_windows else 0
    
    print(f"\n📈 Debug Sample Statistics:")
    print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{len(all_windows)})")
    
    # Analyze prediction distributions
    pos_preds = [w['prediction'] for w in positive_windows]
    neg_preds = [w['prediction'] for w in negative_windows]
    
    if pos_preds:
        print(f"   Positive windows - Mean pred: {np.mean(pos_preds):.3f}, "
              f"Min: {np.min(pos_preds):.3f}, Max: {np.max(pos_preds):.3f}")
    if neg_preds:
        print(f"   Negative windows - Mean pred: {np.mean(neg_preds):.3f}, "
              f"Min: {np.min(neg_preds):.3f}, Max: {np.max(neg_preds):.3f}")
    
    print("=" * 50)

def analyze_window_confusion_matrix(model, dataloader, device):
    """
    Analyze the confusion matrix for window-level predictions.
    """
    print("\n📊 WINDOW-LEVEL CONFUSION MATRIX ANALYSIS")
    print("=" * 50)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Convert logits to probabilities with fixed 0.5 threshold
            logits = model(input_values)
            predictions = torch.sigmoid(logits)  # Convert to probabilities [0,1]
            binary_predictions = (predictions > 0.5).float()  # Fixed 0.5 threshold
            
            all_predictions.extend(binary_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("📈 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              0      1")
    print(f"   Actual 0  {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"          1  {cm[1,0]:6d} {cm[1,1]:6d}")
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n📊 Detailed Metrics:")
    print(f"   True Positives:  {tp:6d}")
    print(f"   True Negatives:  {tn:6d}")
    print(f"   False Positives: {fp:6d}")
    print(f"   False Negatives: {fn:6d}")
    print(f"   Precision:       {precision:.4f}")
    print(f"   Recall:          {recall:.4f}")
    print(f"   Specificity:     {specificity:.4f}")
    print(f"   F1 Score:        {f1:.4f}")
    print(f"   Accuracy:        {accuracy:.4f}")
    
    # Class distribution
    pos_samples = np.sum(all_labels)
    total_samples = len(all_labels)
    class_balance = pos_samples / total_samples
    
    print(f"\n📈 Class Distribution:")
    print(f"   Positive samples: {int(pos_samples):6d} ({class_balance:.3f})")
    print(f"   Negative samples: {int(total_samples - pos_samples):6d} ({1-class_balance:.3f})")
    print(f"   Total samples:    {total_samples:6d}")
    
    print("=" * 50)

def custom_collate_fn(batch):
    """
    Custom collate function that filters out None values and handles errors gracefully.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        dict: Properly collated batch
    """
    # Filter out None values
    valid_samples = [sample for sample in batch if sample is not None]
    
    if len(valid_samples) == 0:
        # If all samples are None, create a dummy batch
        print("⚠️ Warning: Empty batch detected, creating dummy batch")
        dummy_sample = {
            'input_values': torch.zeros(8000),
            'label': torch.tensor(0.0),
            'file_id': 'dummy_batch',
            'metadata': {'boundary_pos': None, 'window_start': 0}
        }
        valid_samples = [dummy_sample]
    
    if len(valid_samples) < len(batch):
        print(f"⚠️ Warning: Filtered {len(batch) - len(valid_samples)} invalid samples from batch")
    
    try:
        # Collate valid samples
        collated = {
            'input_values': torch.stack([sample['input_values'] for sample in valid_samples]),
            'label': torch.stack([sample['label'] for sample in valid_samples]),
            'file_id': [sample['file_id'] for sample in valid_samples],
            'metadata': [sample['metadata'] for sample in valid_samples]
        }
        return collated
    except Exception as e:
        print(f"⚠️ Error in collation: {e}")
        # Create a completely safe dummy batch
        batch_size = len(valid_samples)
        return {
            'input_values': torch.zeros(batch_size, 8000),
            'label': torch.zeros(batch_size),
            'file_id': [f'error_dummy_{i}' for i in range(batch_size)],
            'metadata': [{'boundary_pos': None, 'window_start': 0} for _ in range(batch_size)]
        }

def validate_dataset_samples(dataset, num_samples=10):
    """
    Validate a few samples from the dataset to check for issues.
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of samples to check
    """
    print(f"🔍 Validating {num_samples} dataset samples...")
    
    valid_count = 0
    error_count = 0
    
    # Check a few samples from different parts of the dataset
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        try:
            sample = dataset[idx]
            
            # Validate sample structure
            required_keys = ['input_values', 'label', 'file_id', 'metadata']
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing key: {key}")
            
            # Validate tensor shapes and types
            if not isinstance(sample['input_values'], torch.Tensor):
                raise ValueError(f"input_values is not a tensor: {type(sample['input_values'])}")
            
            if not isinstance(sample['label'], torch.Tensor):
                raise ValueError(f"label is not a tensor: {type(sample['label'])}")
            
            # Check for NaN or infinite values
            if torch.any(torch.isnan(sample['input_values'])) or torch.any(torch.isinf(sample['input_values'])):
                raise ValueError("input_values contains NaN or infinite values")
            
            if torch.isnan(sample['label']) or torch.isinf(sample['label']):
                raise ValueError("label contains NaN or infinite values")
            
            valid_count += 1
            
            if i == 0:  # Print details for first sample
                print(f"   ✅ Sample {idx}: {sample['file_id']}")
                print(f"      Input shape: {sample['input_values'].shape}")
                print(f"      Label: {sample['label'].item()}")
                print(f"      Label type: {sample['label'].dtype}")
                
        except Exception as e:
            error_count += 1
            print(f"   ❌ Sample {idx} failed validation: {e}")
    
    print(f"📊 Validation Results: {valid_count}/{num_samples} samples valid")
    if error_count > 0:
        print(f"⚠️ Found {error_count} problematic samples - these may cause training issues")
    else:
        print("✅ All sampled data looks good!")
    
    return valid_count == num_samples

def clear_gpu_memory():
    """Clear GPU memory and print current usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"   🔧 GPU Memory: {allocated:.1f} MB allocated, {cached:.1f} MB cached")

def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return "CPU mode"
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    free = total - allocated
    
    return f"GPU: {allocated:.1f}/{total:.1f} MB ({allocated/total*100:.1f}%), Free: {free:.1f} MB"

def find_optimal_threshold(model, val_dataloader, device, thresholds=None):
    """
    Find optimal threshold using validation data based on F1 score.
    
    Args:
        model: Trained model
        val_dataloader: Validation data loader
        device: Device for inference
        thresholds: List of thresholds to test (default: 0.05 to 0.95 in steps of 0.05)
        
    Returns:
        tuple: (optimal_threshold, best_f1, threshold_results)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)  # 0.05, 0.10, 0.15, ..., 0.95
    
    print(f"\n🔍 FINDING OPTIMAL THRESHOLD")
    print(f"📊 Testing {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print("=" * 60)
    
    model.eval()
    
    # Collect all predictions and labels first
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_values)
            probabilities = torch.sigmoid(logits)  # Convert to probabilities [0,1]
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    print(f"📈 Probability Distribution Analysis:")
    print(f"   Total samples: {len(all_labels):,}")
    print(f"   Positive samples: {np.sum(all_labels):,} ({np.mean(all_labels):.1%})")
    print(f"   Negative samples: {len(all_labels) - np.sum(all_labels):,} ({1-np.mean(all_labels):.1%})")
    print(f"   Probability range: {np.min(all_probabilities):.3f} to {np.max(all_probabilities):.3f}")
    print(f"   Mean probability: {np.mean(all_probabilities):.3f}")
    
    # Analyze by class
    pos_probs = all_probabilities[all_labels == 1]
    neg_probs = all_probabilities[all_labels == 0]
    
    if len(pos_probs) > 0:
        print(f"   Positive class - Mean: {np.mean(pos_probs):.3f}, Std: {np.std(pos_probs):.3f}")
    if len(neg_probs) > 0:
        print(f"   Negative class - Mean: {np.mean(neg_probs):.3f}, Std: {np.std(neg_probs):.3f}")
    
    # Test different thresholds
    threshold_results = []
    best_f1 = 0.0
    optimal_threshold = 0.5
    
    print(f"\n📋 Threshold Performance:")
    print(f"{'Threshold':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Accuracy':>9}")
    print("-" * 50)
    
    for threshold in thresholds:
        # Apply threshold
        predictions = (all_probabilities > threshold).astype(float)
        
        # Calculate metrics
        if len(np.unique(predictions)) == 1:
            # All predictions are the same class
            if predictions[0] == 1:
                precision = np.mean(all_labels)  # If predicting all positive
                recall = 1.0 if np.sum(all_labels) > 0 else 0.0
            else:
                precision = 0.0  # If predicting all negative
                recall = 0.0
        else:
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, predictions, average='binary', zero_division=0
            )
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        accuracy = np.mean(predictions == all_labels)
        
        threshold_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        # Update best threshold
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
        
        print(f"{threshold:9.2f} {precision:9.3f} {recall:9.3f} {f1:9.3f} {accuracy:9.3f}")
    
    print("=" * 50)
    print(f"🏆 OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
    print(f"🎯 Best F1 Score: {best_f1:.3f}")
    
    # Additional analysis
    optimal_result = next(r for r in threshold_results if r['threshold'] == optimal_threshold)
    print(f"📊 At optimal threshold:")
    print(f"   Precision: {optimal_result['precision']:.3f}")
    print(f"   Recall: {optimal_result['recall']:.3f}")
    print(f"   Accuracy: {optimal_result['accuracy']:.3f}")
    
    return optimal_threshold, best_f1, threshold_results

def plot_threshold_analysis(threshold_results, save_path='threshold_analysis.png'):
    """
    Plot threshold analysis results.
    
    Args:
        threshold_results: Results from find_optimal_threshold
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1s = [r['f1'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    
    plt.figure(figsize=(12, 8))
    
    # Main metrics plot
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 score detail
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, f1s, 'g-', linewidth=2)
    best_idx = np.argmax(f1s)
    plt.scatter(thresholds[best_idx], f1s[best_idx], color='red', s=100, zorder=5)
    plt.annotate(f'Best: {thresholds[best_idx]:.2f}', 
                 xy=(thresholds[best_idx], f1s[best_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold (Detail)')
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    plt.subplot(2, 2, 3)
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.scatter(recalls[best_idx], precisions[best_idx], color='red', s=100, zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(thresholds, accuracies, 'purple', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Threshold analysis plot saved to {save_path}")

def main():
    """
    Main function for ULTRA-OPTIMIZED local window boundary detection.
    
    🚀 TWO-STAGE APPROACH FOR MAXIMUM EFFICIENCY:
    
    📦 STAGE 1: Pre-process ALL windows ONCE and save to disk
    ✅ Process audio files with Wav2Vec2 once per window
    ✅ Save processed tensors to disk for reuse
    ✅ Generate metadata for ultra-fast access
    ✅ Takes ~5-15 minutes but only done ONCE
    
    ⚡ STAGE 2: Lightning-fast training with preprocessed windows
    ✅ Load pre-processed tensors from disk (no redundant processing)
    ✅ 100x faster than on-the-fly generation per epoch
    ✅ Memory efficient - no huge datasets in RAM
    ✅ Subsequent runs skip Stage 1 entirely
    
    🎯 OTHER OPTIMIZATIONS:
    ✅ Mixed precision training (1.5-2x speedup)
    ✅ 4x larger batch sizes (4x fewer iterations)
    ✅ Early stopping (saves unnecessary epochs)
    ✅ Improved boundary tolerance (better F1 scores)
    ✅ DATA-DRIVEN threshold selection (no more arbitrary thresholds!)
    
    Expected: First run 20-30 min, subsequent runs 5-10 min!
    """
    print("🎤 LOCAL WINDOW BOUNDARY DETECTION - TWO-STAGE OPTIMIZED")
    print("🚀 PARADIGM SHIFT: From Sequence Labeling to Local Binary Classification")
    print("⚡ WITH REVOLUTIONARY TWO-STAGE PREPROCESSING FOR MAXIMUM SPEED")
    print("=" * 80)
    print(f"🚀 Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("💡 TWO-STAGE APPROACH:")
    print("   📦 Stage 1: Pre-process windows ONCE → Save to disk")
    print("   ⚡ Stage 2: Lightning-fast training from preprocessed tensors")
    print("   🔄 Subsequent runs skip Stage 1 for maximum speed!")
    print("   📊 NEW: Data-driven threshold selection!")
    
    # BULLETPROOF 128-DIM CONFIGURATION with modern stability techniques
    config = {
        # INTELLIGENT 128-DIM TRAINING
        'batch_size': 64,                    # LARGER: Residual connections allow bigger batches
        'num_epochs': 15,                    # More epochs with better architecture
        'learning_rate': 3e-5,               # OPTIMAL: Scaled for 128-dim model with modern techniques
        'weight_decay': 5e-5,                # MODERATE: Layer norm reduces need for heavy regularization
        
        # STABLE MODEL ARCHITECTURE
        'hidden_dim': 128,                   # TARGET: 128 dimensions with stability techniques
        'dropout_rate': 0.3,                 # MODERATE: Residual connections reduce overfitting
        'label_smoothing': 0.05,             # LIGHT: Better calibration without over-smoothing
        'use_residual': True,                # CRITICAL: Residual connections for gradient stability
        'use_layer_norm': True,              # CRITICAL: Layer normalization for gradient scaling
        
        # PROPER WINDOW GENERATION
        'max_windows_per_file': None,        # No artificial limits
        'max_positive_per_file': None,       # Generate ONE window per boundary
        'max_negative_per_file': 20,         # Balanced ratio
        
        # IMPROVED NEGATIVE SAMPLING
        'window_duration': 1.5,              # 1.5s windows for context
        'boundary_tolerance': 0.06,          # 60ms tolerance for positive windows
        'negative_exclusion_zone': 0.05,     # 50ms exclusion zone
        'negative_sampling_ratio': 0.3,      # 30% of valid positions
        
        # SMART REGULARIZATION FOR 128-DIM
        'pos_weight': 1.8,                   # MODERATE: Slight positive bias with stable architecture
        'gradient_clip_norm': 1.0,           # HIGHER: Residual connections allow higher clipping
        
        # HARDWARE OPTIMIZATIONS
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': False,            # DISABLED: Ensure stability
        'use_gradient_checkpointing': True,  # Memory optimization
        
        # EARLY STOPPING AND EFFICIENCY
        'early_stopping_patience': 5,       # Less patience - model should learn faster
        'min_improvement': 0.001,           # Smaller improvement threshold
        
        # FIXED THRESHOLD APPROACH
        'threshold': 0.5,                    # Standard 0.5 threshold
        
        # TWO-STAGE PREPROCESSING - New directory for stable approach
        'train_preprocessed_dir': './preprocessed_windows_train_stable',  # NEW DIR for stable approach
        'test_preprocessed_dir': './preprocessed_windows_test_stable',    # NEW DIR for stable approach
        'force_reprocess': True,             # FORCE reprocessing with stable settings
        'sample_rate': 16000,
    }
    
    # Start timing
    pipeline_start_time = time.time()
    
    print(f"⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print(f"🔧 Using device: {config['device']}")
    
    # GPU Information
    if config['device'] == 'cuda':
        print(f"\n🎮 GPU Information:")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    print("-" * 80)
    
    # Initialize processor
    print("🔄 Initializing Wav2Vec2 processor...")
    processor_start = time.time()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    processor_time = time.time() - processor_start
    print(f"✅ Processor initialized in {processor_time:.2f}s")
    
    # Load datasets - Use ALL available files for maximum diversity
    print("\n📂 Loading TIMIT datasets...")
    data_start = time.time()
    train_data = load_timit_data_for_local_windows('train', None)  # Use ALL train files
    test_data = load_timit_data_for_local_windows('test', None)    # Use ALL test files
    data_time = time.time() - data_start
    
    print(f"📊 Dataset sizes: Train={len(train_data)}, Test={len(test_data)}")
    print(f"✅ Data loading completed in {data_time:.2f}s")
    
    # ===============================
    # STAGE 1: PREPROCESSING
    # ===============================
    print("\n" + "="*80)
    print("🚀 STAGE 1: PREPROCESSING ALL WINDOWS")
    print("💡 This is done ONCE - subsequent runs will be much faster!")
    print("="*80)
    
    # Create preprocessors for train and test data
    print("\n🔄 Creating window preprocessors...")
    train_preprocessor = WindowPreprocessor(
        train_data, processor, 
        window_duration=config['window_duration'],
        boundary_tolerance=config['boundary_tolerance'],
        negative_exclusion_zone=config['negative_exclusion_zone'],
        negative_sampling_ratio=config['negative_sampling_ratio'],
        save_dir=config['train_preprocessed_dir'],
        max_windows_per_file=config['max_windows_per_file'],
        max_positive_per_file=config['max_positive_per_file'],
        max_negative_per_file=config['max_negative_per_file']
    )
    
    test_preprocessor = WindowPreprocessor(
        test_data, processor,
        window_duration=config['window_duration'],
        boundary_tolerance=config['boundary_tolerance'],
        negative_exclusion_zone=config['negative_exclusion_zone'],
        negative_sampling_ratio=config['negative_sampling_ratio'],
        save_dir=config['test_preprocessed_dir'],
        max_windows_per_file=config['max_windows_per_file'],
        max_positive_per_file=config['max_positive_per_file'],
        max_negative_per_file=config['max_negative_per_file']
    )
    
    # Preprocess training data
    print(f"\n📦 Preprocessing TRAINING data...")
    preprocessing_start = time.time()
    train_metadata = train_preprocessor.preprocess_all_windows(force_reprocess=config['force_reprocess'])
    train_preprocessing_time = time.time() - preprocessing_start
    
    # Preprocess test data
    print(f"\n📦 Preprocessing TEST data...")
    test_preprocessing_start = time.time()
    test_metadata = test_preprocessor.preprocess_all_windows(force_reprocess=config['force_reprocess'])
    test_preprocessing_time = time.time() - test_preprocessing_start
    
    total_preprocessing_time = train_preprocessing_time + test_preprocessing_time
    print(f"\n✅ PREPROCESSING COMPLETE!")
    print(f"   Train preprocessing: {train_preprocessing_time/60:.1f} minutes")
    print(f"   Test preprocessing: {test_preprocessing_time/60:.1f} minutes")
    print(f"   Total preprocessing: {total_preprocessing_time/60:.1f} minutes")
    print(f"   💾 All windows saved to disk for lightning-fast training!")
    
    # ===============================
    # STAGE 2: FAST TRAINING
    # ===============================
    print("\n" + "="*80)
    print("⚡ STAGE 2: ULTRA-FAST TRAINING WITH PREPROCESSED WINDOWS")
    print("🚀 No redundant processing - just loading tensors from disk!")
    print("="*80)
    
    # Create ultra-fast datasets from preprocessed data
    print("\n📁 Creating ultra-fast datasets from preprocessed windows...")
    dataset_start = time.time()
    
    train_dataset = PreprocessedWindowDataset(
        train_metadata, 
        save_dir=config['train_preprocessed_dir']
    )
    
    test_dataset = PreprocessedWindowDataset(
        test_metadata,
        save_dir=config['test_preprocessed_dir']
    )
    
    dataset_time = time.time() - dataset_start
    print(f"✅ Ultra-fast datasets created in {dataset_time:.2f}s")
    
    # Validate datasets before proceeding
    print("\n🔍 Validating dataset integrity...")
    train_valid = validate_dataset_samples(train_dataset, num_samples=20)
    test_valid = validate_dataset_samples(test_dataset, num_samples=10)
    
    if not train_valid or not test_valid:
        print("⚠️ Dataset validation found issues - proceeding with caution")
    else:
        print("✅ Dataset validation passed!")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    
    # Split train for validation
    val_size = len(train_dataset) // 5
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Use custom collate function to handle any remaining issues
    train_dataloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=8,pin_memory=True,
persistent_workers=True)
    val_dataloader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    print(f"📈 Data Loader Statistics:")
    print(f"   Training windows: {len(train_subset):,}")
    print(f"   Validation windows: {len(val_subset):,}")
    print(f"   Test windows: {len(test_dataset):,}")
    print(f"   Training batches: {len(train_dataloader):,}")
    print(f"   Validation batches: {len(val_dataloader):,}")
    print(f"   Test batches: {len(test_dataloader):,}")
    
    # Test DataLoader functionality
    print("\n🧪 Testing DataLoader functionality...")
    try:
        # Test training dataloader
        train_iter = iter(train_dataloader)
        test_batch = next(train_iter)
        
        print(f"   ✅ Training DataLoader test passed")
        print(f"      Batch input shape: {test_batch['input_values'].shape}")
        print(f"      Batch label shape: {test_batch['label'].shape}")
        print(f"      Batch size: {len(test_batch['file_id'])}")
        print(f"      Sample file IDs: {test_batch['file_id'][:3]}")
        
        # Test validation dataloader
        val_iter = iter(val_dataloader)
        val_test_batch = next(val_iter)
        print(f"   ✅ Validation DataLoader test passed")
        
        del train_iter, val_iter, test_batch, val_test_batch  # Clean up
        
    except Exception as e:
        print(f"   ❌ DataLoader test failed: {e}")
        print("   This indicates a fundamental issue with the dataset")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Initialize model
    print("\n🤖 Initializing Local Boundary Classifier...")
    model_start = time.time()
    
    # Clear GPU cache before model initialization
    if config['device'] == 'cuda':
        clear_gpu_memory()
        print(f"   📊 {get_gpu_memory_info()}")
    
    print("   📥 Loading Wav2Vec2 model (this may take a while)...")
    try:
        # Try with STABLE 128-dim architecture
        model = StableLocalBoundaryClassifier(
            freeze_wav2vec2=True,
            hidden_dim=config['hidden_dim'],        # 128 dimensions
            dropout_rate=config['dropout_rate'],    # Moderate dropout
            use_residual=config['use_residual'],    # CRITICAL: Residual connections
            use_layer_norm=config['use_layer_norm'] # CRITICAL: Layer normalization
        )
        print("   ✅ Model architecture created successfully")
        
        # Enable gradient checkpointing if requested
        if config.get('use_gradient_checkpointing', False):
            if hasattr(model.wav2vec2, 'gradient_checkpointing_enable'):
                model.wav2vec2.gradient_checkpointing_enable()
                print("   ✅ Gradient checkpointing enabled")
        
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        print("   🔄 Trying with even smaller architecture...")
        try:
            # Try with minimal model if base fails
            model = LocalBoundaryClassifier(
                wav2vec2_model_name="facebook/wav2vec2-base", 
                freeze_wav2vec2=True,
                hidden_dim=64,  # Even smaller
                dropout_rate=0.3
            )
            print("   ✅ Model created with minimal architecture")
        except Exception as e2:
            print(f"   ❌ Error with minimal model too: {e2}")
            print("   💡 Suggestion: Your GPU may not have enough memory. Try:")
            print("      - Reducing max_train_samples to 100-200")
            print("      - Using CPU mode by setting device='cpu'") 
            print("      - Closing other applications to free GPU memory")
            return None, None, None
    
    print("   🔄 Moving model to device...")
    try:
        # Clear memory before moving model
        if config['device'] == 'cuda':
            clear_gpu_memory()
        
        model.to(config['device'])
        
        if config['device'] == 'cuda':
            clear_gpu_memory()
            print(f"   📊 After model loading: {get_gpu_memory_info()}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"   ❌ GPU Out of Memory Error!")
            print(f"   💡 Your RTX 3050 6GB doesn't have enough memory for this model")
            print(f"   🔄 Automatically switching to CPU mode...")
            
            # Switch to CPU mode
            config['device'] = 'cpu'
            model.to('cpu')
            clear_gpu_memory()
            print(f"   ✅ Model moved to CPU successfully")
        else:
            raise e
    
    model_time = time.time() - model_start
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📋 Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    print(f"✅ Model initialized in {model_time:.2f}s")
    
    # Training
    print("\n🎯 Starting training...")
    training_start = time.time()
    history = train_local_classifier(
        model, train_dataloader, val_dataloader, 
        config['device'], config,  # Pass config for gradient clipping
        config['num_epochs'], 
        config['learning_rate'], config['pos_weight'], config['mixed_precision'],
        early_stopping_patience=config['early_stopping_patience'],
        min_improvement=config['min_improvement'],
        weight_decay=config['weight_decay'],
        label_smoothing=config['label_smoothing']
    )
    training_time = time.time() - training_start
    print(f"✅ Training completed in {training_time/60:.1f} minutes")
    
    # Debug model predictions with configured threshold
    print(f"\n🔧 Debugging trained model with threshold {config['threshold']}...")
    debug_window_predictions(model, val_dataloader, config['device'], num_samples=10, threshold=config['threshold'])
    analyze_window_confusion_matrix(model, val_dataloader, config['device'])
    
    # Basic evaluation on windows with configured threshold
    print(f"\n🎯 Window-level evaluation with threshold {config['threshold']}...")
    test_loss, test_metrics = evaluate_local_classifier(
        model, test_dataloader, config['device'], 
        threshold=config['threshold']
    )
    
    print(f"\n🏆 WINDOW-LEVEL RESULTS (Threshold: {config['threshold']}):")
    print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"📊 Test Precision: {test_metrics['precision']:.4f}")
    print(f"📊 Test Recall: {test_metrics['recall']:.4f}")
    print(f"📊 Test F1 Score: {test_metrics['f1']:.4f}")
    
    # Comprehensive evaluation on full audio with configured threshold
    print(f"\n🔍 Starting comprehensive full-audio evaluation with threshold {config['threshold']}...")
    eval_start = time.time()
    results = evaluate_on_full_audio(model, test_data, processor, config['device'], config)
    eval_time = time.time() - eval_start
    print(f"✅ Full-audio evaluation completed in {eval_time:.1f}s")
    
    # Plot results
    print("\n📊 Generating comprehensive plots...")
    plot_local_window_results(results, history, save_path='local_window_evaluation.png')
    
    # Print detailed analysis
    print_local_window_analysis(results, history)
    
    # Final summary
    total_pipeline_time = time.time() - pipeline_start_time
    
    print(f"\n🎉 LOCAL WINDOW PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"✨ LOCAL WINDOW APPROACH RESULTS:")
    print(f"   🎯 Clear binary classification task (not sparse sequence)")
    print(f"   🔥 Balanced training data (50/50 vs previous 79/21)")
    print(f"   📈 Simple CNN classifier with global pooling")
    print(f"   ⚡ Direct boundary predictions with post-processing")
    print(f"   🎯 FIXED 0.5 threshold for proper probability calibration")
    print(f"   📊 Standard threshold: 0.5")
    print("=" * 80)
    
    print(f"⏱️  Timing Summary:")
    print(f"   Data loading: {data_time:.1f}s")
    print(f"   STAGE 1 - Preprocessing: {total_preprocessing_time/60:.1f} minutes")
    print(f"   Ultra-fast dataset creation: {dataset_time:.1f}s") 
    print(f"   Model initialization: {model_time:.1f}s")
    print(f"   STAGE 2 - Training: {training_time/60:.1f} minutes")
    print(f"   Model debugging and evaluation: <1 minute")
    print(f"   Full-audio evaluation: {eval_time:.1f}s")
    print(f"   Total pipeline: {total_pipeline_time/60:.1f} minutes")
    print(f"   💡 Next run will skip Stage 1 → ~{(total_pipeline_time - total_preprocessing_time)/60:.1f} min!")
    
    print(f"\n📁 Output Files:")
    print(f"   Model: best_local_model.pth")
    print(f"   Comprehensive plots: local_window_evaluation.png")
    print(f"   Preprocessed train windows: {config['train_preprocessed_dir']}/")
    print(f"   Preprocessed test windows: {config['test_preprocessed_dir']}/")
    
    print(f"\n🚀 REVOLUTIONARY IMPROVEMENTS:")
    print(f"   📦 Stage 1 (preprocessing): {total_preprocessing_time/60:.1f} min (ONCE only)")
    print(f"   ⚡ Stage 2 (training): {training_time/60:.1f} min (every run)")
    print(f"   🎯 Fixed 0.5 threshold: Focuses on proper probability calibration!")
    print(f"   💾 Preprocessed windows saved for instant reuse")
    print(f"   🔄 Subsequent training runs: ~{training_time/60:.1f} min vs {total_pipeline_time/60:.1f} min")
    print(f"   ⚡ Speedup for future runs: {total_pipeline_time/training_time:.1f}x faster!")
    
    print(f"\n📊 Final Performance Summary:")
    print(f"   Window-level F1: {test_metrics['f1']:.4f}")
    print(f"   Full-audio F1: {results['overall_metrics']['mean_f1']:.4f}")
    print(f"   Best validation F1: {max(history['val_f1']):.4f}")
    print(f"   Training samples: {len(train_data):,} audio files → {len(train_dataset):,} windows")
    print(f"   Test samples: {len(test_data):,} audio files → {len(test_dataset):,} windows")
    print(f"   Class balance: ~50/50 (vs previous 79/21 imbalance)")
    
    # Compare with sequence approach
    sequence_f1 = 0.141  # Reference from previous runs
    improvement = (results['overall_metrics']['mean_f1'] - sequence_f1) / sequence_f1 * 100 if sequence_f1 > 0 else 0
    print(f"\n🏆 PARADIGM SHIFT COMPARISON:")
    print(f"   Sequence approach F1: {sequence_f1:.3f}")
    print(f"   Local window F1: {results['overall_metrics']['mean_f1']:.3f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if results['overall_metrics']['mean_f1'] > sequence_f1:
        print(f"   ✅ Local window approach is SUPERIOR!")
    else:
        print(f"   ⚠️ Need further optimization")
    
    print(f"\n🚀 Improved pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Fixed 0.5 threshold + core training improvements!")
    print("=" * 80)
    
    return model, history, results

# ... rest of the file ...

# CRITICAL ANALYSIS OF THE LOCAL WINDOW APPROACH
"""
🧠 CRITICAL ANALYSIS: Local Window vs Sequence Approach

✅ ADVANTAGES:
1. **Much Clearer Task**: Binary classification is simpler than sparse sequence labeling
2. **Balanced Training**: Can easily create 50/50 positive/negative samples
3. **Strong Local Features**: 1.5s context captures phonetic transitions well
4. **No Post-processing Issues**: Direct binary prediction, no complex peak detection
5. **Computational Efficiency**: Simpler model, faster training and inference
6. **Better Interpretability**: Easy to understand what the model is learning
7. **DATA-DRIVEN Thresholds**: No more arbitrary threshold selection!

✅ CORE IMPROVEMENTS:
- Better class balance (40/60 pos/neg vs previous 79/21)
- Fixed 0.5 threshold for proper probability calibration
- Higher learning rate (2e-4) for meaningful weight updates
- Larger exclusion zone (150ms) for truly different negative windows
- Increased model capacity (64 hidden dims) and negative class bias

❌ POTENTIAL LIMITATIONS:
1. **Temporal Resolution**: Limited by window size and stride
2. **Context Dependencies**: Might miss long-range phonetic patterns
3. **Boundary Precision**: 60ms tolerance might be too coarse for some applications
4. **Multiple Boundaries**: If multiple boundaries in 1.5s window, only detects one
5. **Edge Effects**: Boundaries near audio start/end might be missed

📊 EXPECTED PERFORMANCE WITH FIXES:
- **Learning**: Higher learning rate should enable actual weight updates
- **Discrimination**: Larger exclusion zone should create truly different negative samples
- **Capacity**: Larger model should capture more complex boundary patterns
- **Calibration**: Fixed 0.5 threshold should force proper probability calibration
- **Balance**: Slight negative bias should reduce false positive tendency

This approach should be MUCH more effective than the struggling sequence approach!
"""

if __name__ == "__main__":
    model, history, results = main()