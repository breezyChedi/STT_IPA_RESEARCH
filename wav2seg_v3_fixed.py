"""
INTELLIGENT LOCAL WINDOW BOUNDARY DETECTION - V3 FIXED
======================================================

🚨 FIXED: Prosodic feature processing to handle actual keys properly
✅ Consistent dimensions throughout the architecture
✅ Nuclear-grade NaN protection at every step
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
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def create_default_prosodic_features(seq_len=25):
    """
    Create default prosodic features when extraction fails or for dummy samples.
    
    Args:
        seq_len: Sequence length for temporal features (default ~25 for 0.5s at 50Hz)
        
    Returns:
        dict: Default prosodic features with all required keys
    """
    return {
        # Temporal features (5) - sequence-based
        'energy_delta': np.zeros(seq_len, dtype=np.float32),
        'energy_delta2': np.zeros(seq_len, dtype=np.float32),
        'centroid_delta': np.zeros(seq_len, dtype=np.float32),
        'rolloff_delta': np.zeros(seq_len, dtype=np.float32),
        'zcr_delta': np.zeros(seq_len, dtype=np.float32),
        
        # Global features (13) - window-level scalars
        'energy_mean': 0.0,  # 1 feature
        'mfcc_mean': [0.0] * 6,  # 6 features
        'mfcc_std': [0.0] * 6,   # 6 features
    }

def clean_prosodic_features(prosodic_features, verbose=False):
    """
    🔧 ROBUST prosodic feature cleaning to prevent NaN corruption in model.
    """
    if prosodic_features is None:
        if verbose:
            print("⚠️ Prosodic features are None, using defaults")
        return create_default_prosodic_features()
    
    cleaned = {}
    issues_found = []
    
    # Define reasonable value ranges for each feature type
    temporal_range = (-100.0, 100.0)  # Energy/spectral deltas
    energy_range = (0.0, 10.0)        # Energy mean
    mfcc_range = (-50.0, 50.0)        # MFCC coefficients
    
    # Clean temporal features (5 features)
    temporal_keys = ['energy_delta', 'energy_delta2', 'centroid_delta', 'rolloff_delta', 'zcr_delta']
    for key in temporal_keys:
        if key in prosodic_features:
            try:
                feature = np.array(prosodic_features[key], dtype=np.float32)
                
                # Check for NaN/Inf
                nan_mask = ~np.isfinite(feature)
                if np.any(nan_mask):
                    issues_found.append(f"{key}: {np.sum(nan_mask)} NaN/Inf values replaced with 0")
                    feature[nan_mask] = 0.0  # Replace with zeros
                
                # Clip extreme values
                feature = np.clip(feature, temporal_range[0], temporal_range[1])
                
                # Ensure proper length (25 for 0.5s windows)
                if len(feature) != 25:
                    if len(feature) == 0:
                        feature = np.zeros(25, dtype=np.float32)
                    elif len(feature) < 25:
                        # Pad with zeros
                        feature = np.pad(feature, (0, 25 - len(feature)), mode='constant')
                    else:
                        # Truncate to 25
                        feature = feature[:25]
                    issues_found.append(f"{key}: resized to length 25")
                
                cleaned[key] = feature.astype(np.float32)
            except Exception as e:
                # If anything goes wrong, use safe default
                cleaned[key] = np.zeros(25, dtype=np.float32)
                issues_found.append(f"{key}: error {e}, using zeros")
        else:
            # Missing key - use default
            cleaned[key] = np.zeros(25, dtype=np.float32)
            issues_found.append(f"{key}: missing, using zeros")
    
    # Clean global features
    # Energy mean (1 scalar)
    try:
        if 'energy_mean' in prosodic_features:
            energy_mean = float(prosodic_features['energy_mean'])
            if not np.isfinite(energy_mean):
                energy_mean = 0.0
                issues_found.append("energy_mean: NaN/Inf, set to 0")
            energy_mean = np.clip(energy_mean, energy_range[0], energy_range[1])
            cleaned['energy_mean'] = energy_mean
        else:
            cleaned['energy_mean'] = 0.0
            issues_found.append("energy_mean: missing, set to 0")
    except Exception as e:
        cleaned['energy_mean'] = 0.0
        issues_found.append(f"energy_mean: error {e}, set to 0")
    
    # MFCC features (6 + 6 = 12 features)
    for feature_type in ['mfcc_mean', 'mfcc_std']:
        try:
            if feature_type in prosodic_features:
                mfcc_values = prosodic_features[feature_type]
                if isinstance(mfcc_values, (list, tuple, np.ndarray)):
                    mfcc_array = np.array(mfcc_values, dtype=np.float32)
                    
                    # Check for NaN/Inf
                    nan_mask = ~np.isfinite(mfcc_array)
                    if np.any(nan_mask):
                        issues_found.append(f"{feature_type}: {np.sum(nan_mask)} NaN/Inf values replaced with 0")
                        mfcc_array[nan_mask] = 0.0
                    
                    # Clip extreme values
                    mfcc_array = np.clip(mfcc_array, mfcc_range[0], mfcc_range[1])
                    
                    # Ensure exactly 6 values
                    if len(mfcc_array) != 6:
                        if len(mfcc_array) == 0:
                            mfcc_array = np.zeros(6, dtype=np.float32)
                        elif len(mfcc_array) < 6:
                            # Pad with zeros
                            mfcc_array = np.pad(mfcc_array, (0, 6 - len(mfcc_array)), mode='constant')
                        else:
                            # Truncate to 6
                            mfcc_array = mfcc_array[:6]
                        issues_found.append(f"{feature_type}: resized to 6 values")
                    
                    cleaned[feature_type] = mfcc_array.tolist()  # Keep as list for consistency
                else:
                    # Single value or wrong type
                    cleaned[feature_type] = [0.0] * 6
                    issues_found.append(f"{feature_type}: wrong type, using zeros")
            else:
                # Missing key
                cleaned[feature_type] = [0.0] * 6
                issues_found.append(f"{feature_type}: missing, using zeros")
        except Exception as e:
            # If anything goes wrong, use safe default
            cleaned[feature_type] = [0.0] * 6
            issues_found.append(f"{feature_type}: error {e}, using zeros")
    
    # Final validation - test tensor creation to catch any remaining issues
    try:
        # Test tensor creation to catch any remaining issues
        temporal_test = torch.stack([torch.tensor(cleaned[key], dtype=torch.float32) for key in temporal_keys])
        global_test = torch.tensor([cleaned['energy_mean']] + cleaned['mfcc_mean'] + cleaned['mfcc_std'], dtype=torch.float32)
        
        # Check for any remaining NaN/Inf
        if torch.any(~torch.isfinite(temporal_test)) or torch.any(~torch.isfinite(global_test)):
            raise ValueError("NaN/Inf values still present after cleaning")
            
    except Exception as e:
        if verbose:
            print(f"⚠️ Prosodic feature validation failed: {e}")
        # Fall back to completely safe defaults
        return create_default_prosodic_features()
    
    if verbose and issues_found and len(issues_found) > 0:
        print(f"🔧 Cleaned prosodic features: {'; '.join(issues_found[:3])}")
        if len(issues_found) > 3:
            print(f"   ... and {len(issues_found) - 3} more issues")
    
    return cleaned

class CompetitiveMultiScaleBoundaryClassifier(nn.Module):
    """
    🏆 COMPETITIVE BOUNDARY CLASSIFIER - LEARNED ATTENTION + PROSODIC FEATURES
    
    FIXED VERSION with consistent prosodic feature processing
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=256, dropout_rate=0.2, use_prosodic=True):
        """
        Initialize the competitive boundary classifier.
        """
        super().__init__()
        
        self.use_prosodic = use_prosodic
        self.hidden_dim = hidden_dim
        
        print(f"      🏆 Building COMPETITIVE boundary classifier with LEARNED ATTENTION...")
        print(f"      📥 Loading Wav2Vec2 model: {wav2vec2_model_name}")
        
        # Load pretrained Wav2Vec2 model
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                wav2vec2_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
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
        
        # === MULTI-SCALE TEMPORAL CONVOLUTIONS ===
        print(f"      🧠 Building multi-scale temporal convolutions...")
        
        # Fine-scale convolutions (3x3 kernels) - Local patterns
        self.fine_conv1 = nn.Conv1d(wav2vec2_hidden_size, hidden_dim, kernel_size=3, padding=1)
        self.fine_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Coarse-scale convolutions (9x9 kernels) - Broader patterns  
        self.coarse_conv1 = nn.Conv1d(wav2vec2_hidden_size, hidden_dim, kernel_size=9, padding=4)
        self.coarse_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4)
        
        # === PROSODIC FEATURE PROCESSING ===
        if use_prosodic:
            print(f"      🎵 Building prosodic feature processing...")
            # Prosodic feature dimensions: 28 features total
            # 1 (energy) + 13 (spectral) + 1 (delta_energy) + 13 (delta_spectral) = 28
            prosodic_input_dim = 28
            
            self.prosodic_conv1 = nn.Conv1d(prosodic_input_dim, hidden_dim//2, kernel_size=3, padding=1)
            self.prosodic_conv2 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1)
            
            # Total features after concatenation
            total_features = hidden_dim * 2 + hidden_dim//2  # fine + coarse + prosodic
        else:
            total_features = hidden_dim * 2  # fine + coarse only
        
        # === FEATURE FUSION LAYER ===
        print(f"      🔗 Building feature fusion layer...")
        self.fusion_conv = nn.Conv1d(total_features, hidden_dim, kernel_size=5, padding=2)
        
        # === LEARNED ATTENTION MECHANISM ===
        print(f"      🧠 Building LEARNED ATTENTION mechanism...")
        
        # Boundary detection - per-timestep boundary scores
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Learned attention - model learns which parts of window matter
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Layer normalizations for stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        if use_prosodic:
            self.prosodic_layer_norm = nn.LayerNorm(hidden_dim//2)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        if use_prosodic:
            self.prosodic_dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._intelligent_weight_init()
        
        print(f"      ✅ COMPETITIVE architecture built successfully!")
        print(f"      🔧 Features: Multi-scale convs + Prosodic + LEARNED ATTENTION")
        print(f"      🧠 Attention learns which temporal regions matter for boundaries")
        print(f"      🛡️ Nuclear-grade NaN protection throughout")
    
    def _intelligent_weight_init(self):
        """Initialize weights using modern best practices."""
        print(f"      🎯 Applying intelligent weight initialization...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # He initialization for ReLU networks
                fan_in = module.in_channels * module.kernel_size[0]
                if fan_in == 0:
                    fan_in = 1
                std = (2.0 / fan_in) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                fan_in = module.in_features
                fan_out = module.out_features
                if fan_in + fan_out == 0:
                    fan_in, fan_out = 1, 1
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"      ✅ Weight initialization complete")
    
    def forward(self, input_values, prosodic_features=None):
        """
        🏆 COMPETITIVE forward pass with learned attention and prosodic features.
        🚨 FIXED: Properly handles actual prosodic feature keys
        """
        batch_size = input_values.shape[0]
        
        # 🚨 STEP 1: VALIDATE INPUT
        if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Input audio has NaN/Inf, cleaning...")
            input_values = torch.where(
                torch.isnan(input_values) | torch.isinf(input_values),
                torch.zeros_like(input_values), 
                input_values
            )
        
        # === STEP 2: WAV2VEC2 FEATURE EXTRACTION ===
        with torch.no_grad():
            wav2vec2_outputs = self.wav2vec2(input_values)
            hidden_states = wav2vec2_outputs.last_hidden_state.detach()  # [batch, seq_len, 768]
        
        # 🚨 NaN PROTECTION: Validate Wav2Vec2 outputs
        if torch.any(torch.isnan(hidden_states)) or torch.any(torch.isinf(hidden_states)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Wav2Vec2 output has NaN/Inf, cleaning...")
            hidden_states = torch.where(
                torch.isnan(hidden_states) | torch.isinf(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states
            )
        
        # Transpose for convolutions: [batch, hidden_dim, seq_len]
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # === STEP 3: MULTI-SCALE TEMPORAL CONVOLUTIONS ===
        
        # Fine-scale features (local patterns)
        fine_features = F.relu(self.fine_conv1(x))      # [batch, hidden_dim, seq_len]
        fine_features = F.relu(self.fine_conv2(fine_features))
        
        # 🚨 NaN PROTECTION: Check fine features
        if torch.any(torch.isnan(fine_features)) or torch.any(torch.isinf(fine_features)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Fine features have NaN/Inf, cleaning...")
            fine_features = torch.where(
                torch.isnan(fine_features) | torch.isinf(fine_features),
                torch.zeros_like(fine_features),
                fine_features
            )
        
        # Coarse-scale features (broader patterns)
        coarse_features = F.relu(self.coarse_conv1(x))  # [batch, hidden_dim, seq_len]
        coarse_features = F.relu(self.coarse_conv2(coarse_features))
        
        # 🚨 NaN PROTECTION: Check coarse features
        if torch.any(torch.isnan(coarse_features)) or torch.any(torch.isinf(coarse_features)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Coarse features have NaN/Inf, cleaning...")
            coarse_features = torch.where(
                torch.isnan(coarse_features) | torch.isinf(coarse_features),
                torch.zeros_like(coarse_features),
                coarse_features
            )
        
        # === STEP 4: FIXED PROSODIC FEATURE PROCESSING ===
        feature_list = [fine_features, coarse_features]
        
        if self.use_prosodic:
            try:
                # Handle prosodic features - create consistent format regardless of input
                if prosodic_features is not None:
                    # Clean prosodic features first
                    prosodic_features = clean_prosodic_features(prosodic_features, verbose=False)
                    
                    # Get sequence length from existing features
                    seq_len = fine_features.shape[2]
                    
                    # Build prosodic tensor from the actual keys in prosodic_features
                    # Target: 28 features total (to match model architecture)
                    prosodic_components = []
                    
                    # 1. Energy features (1 feature) - use energy_mean as constant across time
                    if 'energy_mean' in prosodic_features:
                        energy_mean = prosodic_features['energy_mean']
                        energy_tensor = torch.full((batch_size, 1, seq_len), energy_mean, 
                                                  device=fine_features.device, dtype=fine_features.dtype)
                    else:
                        energy_tensor = torch.zeros(batch_size, 1, seq_len, 
                                                   device=fine_features.device, dtype=fine_features.dtype)
                    prosodic_components.append(energy_tensor)
                    
                    # 2. Spectral features (13 features) - use MFCC mean/std as constant across time
                    if 'mfcc_mean' in prosodic_features and len(prosodic_features['mfcc_mean']) >= 6:
                        # Extend to 13 features by using MFCC mean + std + padding
                        mfcc_features = []
                        # Use MFCC mean (first 6)
                        mfcc_features.extend(prosodic_features['mfcc_mean'][:6])
                        # Use MFCC std (next 6)
                        if 'mfcc_std' in prosodic_features and len(prosodic_features['mfcc_std']) >= 6:
                            mfcc_features.extend(prosodic_features['mfcc_std'][:6])
                        else:
                            mfcc_features.extend([0.0] * 6)
                        # Add one more to reach 13
                        mfcc_features.append(0.0)
                        
                        # Create tensor [batch, 13, seq_len]
                        spectral_tensor = torch.tensor(mfcc_features[:13], device=fine_features.device, dtype=fine_features.dtype)
                        spectral_tensor = spectral_tensor.unsqueeze(0).unsqueeze(-1).expand(batch_size, 13, seq_len)
                    else:
                        spectral_tensor = torch.zeros(batch_size, 13, seq_len, 
                                                    device=fine_features.device, dtype=fine_features.dtype)
                    prosodic_components.append(spectral_tensor)
                    
                    # 3. Delta energy (1 feature) - use temporal energy_delta
                    if 'energy_delta' in prosodic_features:
                        energy_delta = torch.tensor(prosodic_features['energy_delta'], 
                                                   device=fine_features.device, dtype=fine_features.dtype)
                        # Ensure correct length
                        if energy_delta.shape[0] != seq_len:
                            # Interpolate to correct length
                            if energy_delta.shape[0] > 0:
                                energy_delta = torch.nn.functional.interpolate(
                                    energy_delta.unsqueeze(0).unsqueeze(0), 
                                    size=seq_len, mode='linear', align_corners=False
                                ).squeeze(0).squeeze(0)
                            else:
                                energy_delta = torch.zeros(seq_len, device=fine_features.device, dtype=fine_features.dtype)
                        delta_energy_tensor = energy_delta.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len)
                    else:
                        delta_energy_tensor = torch.zeros(batch_size, 1, seq_len, 
                                                         device=fine_features.device, dtype=fine_features.dtype)
                    prosodic_components.append(delta_energy_tensor)
                    
                    # 4. Delta spectral (13 features) - use combinations of available delta features
                    delta_spectral_components = []
                    temporal_keys = ['energy_delta2', 'centroid_delta', 'rolloff_delta', 'zcr_delta']
                    
                    for i, key in enumerate(temporal_keys):
                        if key in prosodic_features:
                            delta_feature = torch.tensor(prosodic_features[key], 
                                                        device=fine_features.device, dtype=fine_features.dtype)
                            # Ensure correct length
                            if delta_feature.shape[0] != seq_len:
                                if delta_feature.shape[0] > 0:
                                    delta_feature = torch.nn.functional.interpolate(
                                        delta_feature.unsqueeze(0).unsqueeze(0), 
                                        size=seq_len, mode='linear', align_corners=False
                                    ).squeeze(0).squeeze(0)
                                else:
                                    delta_feature = torch.zeros(seq_len, device=fine_features.device, dtype=fine_features.dtype)
                            delta_spectral_components.append(delta_feature.unsqueeze(0).expand(batch_size, seq_len))
                        else:
                            # Add zero features if missing
                            delta_spectral_components.append(torch.zeros(batch_size, seq_len, 
                                                                        device=fine_features.device, dtype=fine_features.dtype))
                    
                    # Fill remaining slots with zeros to reach 13 features
                    while len(delta_spectral_components) < 13:
                        delta_spectral_components.append(torch.zeros(batch_size, seq_len, 
                                                                    device=fine_features.device, dtype=fine_features.dtype))
                    
                    # Stack delta spectral features: [batch, 13, seq_len]
                    delta_spectral_tensor = torch.stack(delta_spectral_components[:13], dim=1)
                    prosodic_components.append(delta_spectral_tensor)
                    
                    # Concatenate all prosodic components: [batch, 28, seq_len]
                    prosodic_tensor = torch.cat(prosodic_components, dim=1)
                    
                else:
                    # Create default prosodic features
                    seq_len = fine_features.shape[2]
                    prosodic_tensor = torch.zeros(batch_size, 28, seq_len, 
                                                device=fine_features.device, dtype=fine_features.dtype)
                
                # 🚨 NaN PROTECTION: Final prosodic check
                if torch.any(torch.isnan(prosodic_tensor)) or torch.any(torch.isinf(prosodic_tensor)):
                    print(f"🚨 NUCLEAR NaN PROTECTION: Final prosodic tensor has NaN/Inf, cleaning...")
                    prosodic_tensor = torch.where(
                        torch.isnan(prosodic_tensor) | torch.isinf(prosodic_tensor),
                        torch.zeros_like(prosodic_tensor),
                        prosodic_tensor
                    )
                
                # Process prosodic features with Conv1d
                prosodic_processed = F.relu(self.prosodic_conv1(prosodic_tensor))  # [batch, hidden_dim//2, seq_len]
                prosodic_processed = F.relu(self.prosodic_conv2(prosodic_processed))
                
                # 🚨 NaN PROTECTION: Check processed prosodic features
                if torch.any(torch.isnan(prosodic_processed)) or torch.any(torch.isinf(prosodic_processed)):
                    print(f"🚨 NUCLEAR NaN PROTECTION: Processed prosodic features have NaN/Inf, cleaning...")
                    prosodic_processed = torch.where(
                        torch.isnan(prosodic_processed) | torch.isinf(prosodic_processed),
                        torch.zeros_like(prosodic_processed),
                        prosodic_processed
                    )
                
                feature_list.append(prosodic_processed)
                
            except Exception as e:
                print(f"🚨 PROSODIC ERROR: {e}, using zero features...")
                # Create default prosodic features with correct dimensions
                seq_len = fine_features.shape[2]
                prosodic_processed = torch.zeros(batch_size, self.hidden_dim//2, seq_len, 
                                               device=fine_features.device, dtype=fine_features.dtype)
                feature_list.append(prosodic_processed)
        
        # === STEP 5: FEATURE FUSION ===
        combined_features = torch.cat(feature_list, dim=1)  # [batch, total_features, seq_len]
        
        # 🚨 NaN PROTECTION: Check combined features
        if torch.any(torch.isnan(combined_features)) or torch.any(torch.isinf(combined_features)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Combined features have NaN/Inf, cleaning...")
            combined_features = torch.where(
                torch.isnan(combined_features) | torch.isinf(combined_features),
                torch.zeros_like(combined_features),
                combined_features
            )
        
        # Fusion convolution
        fused_features = F.relu(self.fusion_conv(combined_features))  # [batch, hidden_dim, seq_len]
        
        # 🚨 NaN PROTECTION: Check fused features
        if torch.any(torch.isnan(fused_features)) or torch.any(torch.isinf(fused_features)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Fused features have NaN/Inf, cleaning...")
            fused_features = torch.where(
                torch.isnan(fused_features) | torch.isinf(fused_features),
                torch.zeros_like(fused_features),
                fused_features
            )
        
        # === STEP 6: LEARNED ATTENTION MECHANISM ===
        # Transpose for linear layers: [batch, seq_len, hidden_dim]
        x = fused_features.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # Get per-timestep boundary scores from entire window
        timestep_scores = self.boundary_detector(x)  # [batch, seq_len, 1]
        timestep_scores = timestep_scores.squeeze(-1)  # [batch, seq_len]
        
        # 🚨 NaN PROTECTION: Check timestep scores
        if torch.any(torch.isnan(timestep_scores)) or torch.any(torch.isinf(timestep_scores)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Timestep scores have NaN/Inf, cleaning...")
            timestep_scores = torch.where(
                torch.isnan(timestep_scores) | torch.isinf(timestep_scores),
                torch.zeros_like(timestep_scores),
                timestep_scores
            )
        
        # LEARNED ATTENTION: Let model learn which parts of window matter
        attention_scores = self.attention_layer(x)  # [batch, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch, seq_len]
        
        # 🚨 NaN PROTECTION: Check attention scores  
        if torch.any(torch.isnan(attention_scores)) or torch.any(torch.isinf(attention_scores)):
            print(f"🚨 NUCLEAR NaN PROTECTION: Attention scores have NaN/Inf, cleaning...") 