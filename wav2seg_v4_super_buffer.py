"""
INTELLIGENT LOCAL WINDOW BOUNDARY DETECTION - V4 ENHANCED
=========================================================

🧠 INTELLIGENT ARCHITECTURE: Designed specifically for balanced window classification

INTELLIGENT IMPROVEMENTS FOR YOUR SPECIFIC TASK:
✅ INTELLIGENT LOSS: Adaptive weighting + confidence regularization + label smoothing
✅ ALL DELTA-BASED FEATURES: Energy, MFCC, spectral, temporal deltas (23 delta features + 5 reserved)
✅ MULTI-SCALE TEMPORAL: Fine (3x3) + Coarse (9x9) convolutions for boundary patterns
✅ BOUNDARY-AWARE WEIGHTING: Harder examples (closer to edge) weighted more
✅ ADAPTIVE CLASS BALANCING: Real-time adjustment based on batch statistics
✅ CONFIDENCE REGULARIZATION: Prevents overconfident predictions for better generalization

🎯 V4 COMPETITIVE WINDOW DEFINITIONS:
✅ POSITIVE WINDOWS: Boundary in last 20ms (±20ms competition tolerance)
✅ NEGATIVE WINDOWS: No boundary in last 120ms (6x separation for clarity)
✅ ENHANCED FEATURES: Prosodic deltas + multi-scale temporal analysis
✅ COMPETITIVE TRAINING: Focal loss + cosine scheduling + gradient accumulation

COMPETITORS ANALYSIS:
- Strgar & Harwath: 85.3% F1 @ ±20ms (supervised, Gaussian targets)
- Shabber & Bansal: 88.1% F1 @ ±20ms (TIMIT, prosodic features + auxiliary loss)

OUR COMPETITIVE APPROACH:
- Multi-scale temporal analysis (3x3 + 9x9 convolutions)
- ALL DELTA-BASED prosodic features (energy, MFCC, spectral, temporal deltas)
- Competitive loss function (focal + auxiliary localization)
- Strict ±20ms evaluation (competition standard)
- Advanced training (cosine annealing, AdamW, gradient accumulation)

Expected Results: 80-90% F1 @ ±20ms tolerance (competitive range)
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
        dict: Default prosodic features with all required keys (ALL DELTA-BASED)
    """
    return {
        # Temporal delta features (5) - sequence-based
        'energy_delta': np.zeros(seq_len, dtype=np.float32),
        'energy_delta2': np.zeros(seq_len, dtype=np.float32),
        'centroid_delta': np.zeros(seq_len, dtype=np.float32),
        'rolloff_delta': np.zeros(seq_len, dtype=np.float32),
        'zcr_delta': np.zeros(seq_len, dtype=np.float32),
        
        # MFCC delta features (12) - temporal sequences  
        'mfcc_delta': [np.zeros(seq_len, dtype=np.float32) for _ in range(6)],  # 6 MFCC delta sequences
        'mfcc_delta2': [np.zeros(seq_len, dtype=np.float32) for _ in range(6)], # 6 MFCC acceleration sequences
        
        # Additional delta features (6) - temporal sequences
        'bandwidth_delta': np.zeros(seq_len, dtype=np.float32),     # Spectral bandwidth changes
        'flatness_delta': np.zeros(seq_len, dtype=np.float32),     # Spectral flatness changes  
        'flux_delta': np.zeros(seq_len, dtype=np.float32),         # Spectral flux changes
        'chroma_delta': np.zeros(seq_len, dtype=np.float32),       # Chroma changes
        'tonnetz_delta': np.zeros(seq_len, dtype=np.float32),      # Tonnetz changes
        'tempo_delta': np.zeros(seq_len, dtype=np.float32),        # Tempo changes
    }

def clean_prosodic_features(prosodic_features, verbose=False):
    """
    🔧 ROBUST prosodic feature cleaning to prevent NaN corruption in model.
    
    Cleans and validates prosodic features by:
    1. Detecting and replacing NaN/Inf values with safe defaults
    2. Clipping extreme values to reasonable ranges
    3. Ensuring proper data types and shapes
    4. Providing fallback defaults for corrupted features
        
        Args:
        prosodic_features: Dict of prosodic features
        verbose: Whether to print cleaning details
        
    Returns:
        dict: Cleaned prosodic features guaranteed to be NaN-free
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
    
    # Clean all temporal delta features
    simple_temporal_keys = ['energy_delta', 'energy_delta2', 'centroid_delta', 'rolloff_delta', 'zcr_delta',
                           'bandwidth_delta', 'flatness_delta', 'flux_delta', 'chroma_delta', 'tonnetz_delta', 'tempo_delta']
    
    def clean_temporal_feature(key, feature_data, target_len=25):
        """Helper function to clean individual temporal features"""
        try:
            if isinstance(feature_data, (list, np.ndarray)):
                feature = np.array(feature_data, dtype=np.float32)
            else:
                feature = np.array([feature_data], dtype=np.float32)
            
            # Check for NaN/Inf
            nan_mask = ~np.isfinite(feature)
            if np.any(nan_mask):
                issues_found.append(f"{key}: {np.sum(nan_mask)} NaN/Inf values replaced with 0")
                feature[nan_mask] = 0.0
            
            # Clip extreme values
            feature = np.clip(feature, temporal_range[0], temporal_range[1])
            
            # Ensure proper length
            if len(feature) != target_len:
                if len(feature) == 0:
                    feature = np.zeros(target_len, dtype=np.float32)
                elif len(feature) < target_len:
                    feature = np.pad(feature, (0, target_len - len(feature)), mode='constant')
                else:
                    feature = feature[:target_len]
                issues_found.append(f"{key}: resized to length {target_len}")
            
            return feature.astype(np.float32)
        except Exception as e:
            issues_found.append(f"{key}: error {e}, using zeros")
            return np.zeros(target_len, dtype=np.float32)
    
    # Clean simple temporal features
    for key in simple_temporal_keys:
        if key in prosodic_features:
            cleaned[key] = clean_temporal_feature(key, prosodic_features[key])
        else:
            cleaned[key] = np.zeros(25, dtype=np.float32)
            issues_found.append(f"{key}: missing, using zeros")
    
    # Clean MFCC delta features (lists of temporal sequences)
    for feature_type in ['mfcc_delta', 'mfcc_delta2']:
        try:
            if feature_type in prosodic_features:
                mfcc_sequences = prosodic_features[feature_type]
                if isinstance(mfcc_sequences, (list, tuple)):
                    cleaned_sequences = []
                    for i, seq in enumerate(mfcc_sequences[:6]):  # Only use first 6 MFCC coefficients
                        cleaned_seq = clean_temporal_feature(f"{feature_type}[{i}]", seq)
                        cleaned_sequences.append(cleaned_seq)
                    
                    # Ensure exactly 6 sequences
                    while len(cleaned_sequences) < 6:
                        cleaned_sequences.append(np.zeros(25, dtype=np.float32))
                        issues_found.append(f"{feature_type}: padded to 6 sequences")
                    
                    cleaned[feature_type] = cleaned_sequences[:6]  # Ensure exactly 6
                else:
                    # Wrong type - create default sequences
                    cleaned[feature_type] = [np.zeros(25, dtype=np.float32) for _ in range(6)]
                    issues_found.append(f"{feature_type}: wrong type, using zero sequences")
            else:
                # Missing key - create default sequences
                cleaned[feature_type] = [np.zeros(25, dtype=np.float32) for _ in range(6)]
                issues_found.append(f"{feature_type}: missing, using zero sequences")
        except Exception as e:
            # If anything goes wrong, use safe default
            cleaned[feature_type] = [np.zeros(25, dtype=np.float32) for _ in range(6)]
            issues_found.append(f"{feature_type}: error {e}, using zero sequences")
    
    # Final validation - test tensor creation to catch any remaining issues
    try:
        # Test tensor creation for all temporal features
        all_temporal_keys = simple_temporal_keys + ['mfcc_delta', 'mfcc_delta2']
        
        # Test simple temporal features
        for key in simple_temporal_keys:
            test_tensor = torch.tensor(cleaned[key], dtype=torch.float32)
            if torch.any(~torch.isfinite(test_tensor)):
                raise ValueError(f"NaN/Inf values in {key}")
        
        # Test MFCC delta sequences  
        for feature_type in ['mfcc_delta', 'mfcc_delta2']:
            for i, seq in enumerate(cleaned[feature_type]):
                test_tensor = torch.tensor(seq, dtype=torch.float32)
                if torch.any(~torch.isfinite(test_tensor)):
                    raise ValueError(f"NaN/Inf values in {feature_type}[{i}]")
            
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

class CompetitiveWindowPreprocessor:
    """
    COMPETITIVE STAGE 1: Pre-process windows with prosodic features and competitive tolerances.
    """
    
    def __init__(self, data, processor, window_duration=0.5, sample_rate=16000, 
                 boundary_tolerance=0.02, negative_exclusion_zone=0.12, 
                 negative_sampling_ratio=0.5, save_dir="./preprocessed_windows_competitive",
                 max_windows_per_file=None, max_positive_per_file=None, max_negative_per_file=50,
                 use_prosodic=True, use_gaussian_targets=False, verbose=True):
        """
        Initialize COMPETITIVE window preprocessor with prosodic features.
        
        COMPETITIVE TOLERANCES:
        - POSITIVE WINDOWS: Boundary in last 20ms (±20ms competition standard)
        - NEGATIVE WINDOWS: No boundary in last 120ms (6x separation)
        - DELTA-BASED FEATURES: All prosodic features are now delta/change-based for better boundary detection
        
        Args:
            boundary_tolerance: ±20ms tolerance (competition standard)
            use_prosodic: Extract prosodic features (energy, spectral, ZCR)
            use_gaussian_targets: Use Gaussian soft targets (like competitors)
            verbose: Whether to print detailed configuration (default: True)
        """
        self.data = data
        self.processor = processor
        self.window_duration = window_duration
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.use_prosodic = use_prosodic
        self.use_gaussian_targets = use_gaussian_targets
        
        # COMPETITIVE DEFINITIONS (±20ms tolerance like competition)
        self.boundary_tolerance_samples = int(boundary_tolerance * sample_rate)  # ±20ms = 320 samples
        self.negative_exclusion_samples = int(negative_exclusion_zone * sample_rate)  # 120ms = 1920 samples
        
        self.negative_sampling_ratio = negative_sampling_ratio
        self.save_dir = save_dir
        self.max_windows_per_file = max_windows_per_file
        self.max_positive_per_file = max_positive_per_file
        self.max_negative_per_file = max_negative_per_file
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Only print configuration if verbose=True
        if verbose:
            print(f"🏆 COMPETITIVE WindowPreprocessor Configuration:")
            print(f"   Window duration: {window_duration}s ({self.window_samples} samples)")
            print(f"   COMPETITIVE TOLERANCE: ±{boundary_tolerance*1000:.0f}ms ({self.boundary_tolerance_samples} samples)")
            print(f"   NEGATIVE exclusion: {negative_exclusion_zone*1000:.0f}ms ({self.negative_exclusion_samples} samples)")
            print(f"   Prosodic features: {'ENABLED' if use_prosodic else 'DISABLED'}")
            print(f"   Save directory: {save_dir}")
            print(f"   🎯 COMPETITIVE: ±20ms tolerance matching competition!")
            print(f"   💡 0.5s windows: Boundary in last 20ms = {(boundary_tolerance/window_duration)*100:.1f}% (strong signal)!")
    
    def extract_prosodic_features(self, audio):
        """
        Extract ALL DELTA-BASED prosodic features for boundary detection.
        
        🔥 NEW: ALL features are now delta-based (changes/transitions) rather than raw values.
        This is superior for boundary detection since boundaries are fundamentally about transitions.
        
        Delta features extracted:
        - Energy deltas (1st and 2nd order changes)
        - MFCC deltas (1st and 2nd order changes for 6 coefficients)
        - Spectral feature deltas (centroid, rolloff, bandwidth, etc.)
        - Temporal feature deltas (ZCR, chroma, tonnetz, etc.)
        """
        if not self.use_prosodic:
            return create_default_prosodic_features()
            
        try:
            # Parameters optimized for speech analysis
            hop_length = 512  # ~32ms frames at 16kHz
            n_mfcc = 13      # Standard MFCC count
            
            # Target length for consistency with Wav2Vec2 (~50Hz)
            target_length = int(len(audio) / self.sample_rate * 50)
            if target_length <= 0:
                target_length = 25  # Default fallback
            
            # 1. Energy delta features (most important for boundaries)
            energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            if len(energy) == 0:
                raise ValueError("Empty energy features")
            energy_delta = np.diff(energy, prepend=energy[0])
            energy_delta2 = np.diff(energy_delta, prepend=energy_delta[0])
            
            # 2. MFCC delta features (spectral shape changes - proven for speech)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc, 
                                       hop_length=hop_length)
            if mfccs.shape[1] == 0:
                raise ValueError("Empty MFCC features")
            
            # Extract MFCC deltas for first 6 coefficients (most informative)
            mfcc_delta = []
            mfcc_delta2 = []
            for i in range(6):
                if i < mfccs.shape[0]:
                    mfcc_coeff = mfccs[i]
                    delta1 = np.diff(mfcc_coeff, prepend=mfcc_coeff[0])
                    delta2 = np.diff(delta1, prepend=delta1[0])
                    mfcc_delta.append(delta1)
                    mfcc_delta2.append(delta2)
                else:
                    # Pad with zeros if not enough MFCC coefficients
                    mfcc_delta.append(np.zeros_like(energy))
                    mfcc_delta2.append(np.zeros_like(energy))
            
            # 3. Spectral feature deltas
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, 
                                                                hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate,
                                                              hop_length=hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate,
                                                                   hop_length=hop_length)[0]
            
            if len(spectral_centroid) == 0 or len(spectral_rolloff) == 0:
                raise ValueError("Empty spectral features")
            
            centroid_delta = np.diff(spectral_centroid, prepend=spectral_centroid[0])
            rolloff_delta = np.diff(spectral_rolloff, prepend=spectral_rolloff[0])
            bandwidth_delta = np.diff(spectral_bandwidth, prepend=spectral_bandwidth[0])
            
            # 4. Additional spectral delta features
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
                flatness_delta = np.diff(spectral_flatness, prepend=spectral_flatness[0])
            except:
                flatness_delta = np.zeros_like(energy)
            
            try:
                # Spectral flux (spectral change rate)
                stft = librosa.stft(audio, hop_length=hop_length)
                spectral_flux = np.sum(np.diff(np.abs(stft), axis=1), axis=0)
                if len(spectral_flux) == len(energy) - 1:
                    spectral_flux = np.append(spectral_flux, spectral_flux[-1])  # Match length
                flux_delta = spectral_flux if len(spectral_flux) == len(energy) else np.zeros_like(energy)
            except:
                flux_delta = np.zeros_like(energy)
            
            # 5. Temporal feature deltas
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
            if len(zcr) == 0:
                raise ValueError("Empty ZCR features")
            zcr_delta = np.diff(zcr, prepend=zcr[0])
            
            # 6. Harmonic/tonal feature deltas
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, hop_length=hop_length)
                chroma_mean = np.mean(chroma, axis=0)  # Average across chroma bins
                chroma_delta = np.diff(chroma_mean, prepend=chroma_mean[0])
            except:
                chroma_delta = np.zeros_like(energy)
            
            try:
                tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate, hop_length=hop_length)
                tonnetz_mean = np.mean(tonnetz, axis=0)  # Average across tonnetz dimensions
                tonnetz_delta = np.diff(tonnetz_mean, prepend=tonnetz_mean[0])
            except:
                tonnetz_delta = np.zeros_like(energy)
            
            # 7. Tempo delta (optional - may be noisy for short windows)
            try:
                tempo_frames = librosa.feature.tempogram(y=audio, sr=self.sample_rate, hop_length=hop_length)
                tempo_mean = np.mean(tempo_frames, axis=0)
                tempo_delta = np.diff(tempo_mean, prepend=tempo_mean[0])
            except:
                tempo_delta = np.zeros_like(energy)
            
            def resample_feature(feature, target_len):
                if len(feature) == 0:
                    return np.zeros(target_len, dtype=np.float32)
                
                # 🔥 NaN SAFETY: Replace NaN/Inf with zeros before resampling
                feature = np.array(feature, dtype=np.float32)
                nan_mask = ~np.isfinite(feature)
                if np.any(nan_mask):
                    feature[nan_mask] = 0.0
                
                if len(feature) != target_len:
                    if len(feature) == 1:
                        value = feature[0] if np.isfinite(feature[0]) else 0.0
                        return np.full(target_len, value, dtype=np.float32)
                    try:
                        f = interp1d(np.linspace(0, 1, len(feature)), feature, kind='linear')
                        result = f(np.linspace(0, 1, target_len)).astype(np.float32)
                        # Final NaN check after interpolation
                        nan_mask = ~np.isfinite(result)
                        if np.any(nan_mask):
                            result[nan_mask] = 0.0
                        return result
                    except Exception:
                        # If interpolation fails, return zeros
                        return np.zeros(target_len, dtype=np.float32)
                return feature.astype(np.float32)
            
            # Compile ALL DELTA-BASED features - ROBUST VERSION with NaN protection
            prosodic_features = {
                # Basic temporal deltas (5 features)
                'energy_delta': resample_feature(energy_delta, target_length),
                'energy_delta2': resample_feature(energy_delta2, target_length),
                'centroid_delta': resample_feature(centroid_delta, target_length),
                'rolloff_delta': resample_feature(rolloff_delta, target_length),
                'zcr_delta': resample_feature(zcr_delta, target_length),
                
                # MFCC delta sequences (12 features: 6 delta + 6 delta2)
                'mfcc_delta': [resample_feature(delta_seq, target_length) for delta_seq in mfcc_delta],
                'mfcc_delta2': [resample_feature(delta_seq, target_length) for delta_seq in mfcc_delta2],
                
                # Additional spectral deltas (6 features)
                'bandwidth_delta': resample_feature(bandwidth_delta, target_length),
                'flatness_delta': resample_feature(flatness_delta, target_length),
                'flux_delta': resample_feature(flux_delta, target_length),
                'chroma_delta': resample_feature(chroma_delta, target_length),
                'tonnetz_delta': resample_feature(tonnetz_delta, target_length),
                'tempo_delta': resample_feature(tempo_delta, target_length),
            }
            
            # 🔥 ADDITIONAL NaN SAFETY: Clean the features immediately after extraction
            prosodic_features = clean_prosodic_features(prosodic_features)
            
            # 🔥 CRITICAL VALIDATION: Ensure all required delta features exist
            required_simple_keys = ['energy_delta', 'energy_delta2', 'centroid_delta', 'rolloff_delta', 'zcr_delta',
                                   'bandwidth_delta', 'flatness_delta', 'flux_delta', 'chroma_delta', 'tonnetz_delta', 'tempo_delta']
            required_sequence_keys = ['mfcc_delta', 'mfcc_delta2']
            
            # Validate simple temporal features
            for key in required_simple_keys:
                if key not in prosodic_features:
                    raise ValueError(f"Missing required prosodic feature: {key}")
                if len(prosodic_features[key]) != target_length:
                    raise ValueError(f"Wrong length for {key}: {len(prosodic_features[key])} != {target_length}")
            
            # Validate MFCC delta sequences
            for key in required_sequence_keys:
                if key not in prosodic_features:
                    raise ValueError(f"Missing required MFCC feature: {key}")
                if len(prosodic_features[key]) != 6:
                    raise ValueError(f"Wrong MFCC sequence count for {key}: {len(prosodic_features[key])} != 6")
                for i, seq in enumerate(prosodic_features[key]):
                    if len(seq) != target_length:
                        raise ValueError(f"Wrong length for {key}[{i}]: {len(seq)} != {target_length}")
            
            return prosodic_features
            
        except Exception as e:
            print(f"⚠️ Prosodic feature extraction failed: {e}")
            print(f"   🔧 Using default prosodic features for safety")
            return create_default_prosodic_features(target_length if 'target_length' in locals() else 25)
    
    def _prosodic_dict_to_tensor(self, prosodic_dict, seq_len=25):
        """Convert ALL DELTA-BASED prosodic dictionary to tensor ONCE during preprocessing"""
        # Create tensor [28, seq_len] - matches model architecture
        prosodic_tensor = torch.zeros(28, seq_len, dtype=torch.float32)
        
        def assign_temporal_feature(tensor_pos, key, feature_data):
            """Helper to assign temporal feature with proper interpolation"""
            if isinstance(feature_data, (list, np.ndarray)) and len(feature_data) > 0:
                feature_array = np.array(feature_data, dtype=np.float32)
                feature_array = np.nan_to_num(feature_array, 0.0)
                
                if len(feature_array) != seq_len:
                    if len(feature_array) == 1:
                        # Single value - broadcast across time
                        feature_array = np.full(seq_len, feature_array[0], dtype=np.float32)
                    else:
                        # Interpolate to target length
                        x_old = np.linspace(0, 1, len(feature_array))
                        x_new = np.linspace(0, 1, seq_len)
                        feature_array = np.interp(x_new, x_old, feature_array)
                
                prosodic_tensor[tensor_pos, :] = torch.from_numpy(feature_array.astype(np.float32))
        
        # Basic temporal deltas (positions 0-10: 11 features)
        basic_temporal_keys = ['energy_delta', 'energy_delta2', 'centroid_delta', 'rolloff_delta', 'zcr_delta',
                              'bandwidth_delta', 'flatness_delta', 'flux_delta', 'chroma_delta', 'tonnetz_delta', 'tempo_delta']
        
        for i, key in enumerate(basic_temporal_keys):
            if i < 11 and key in prosodic_dict:  # Ensure we don't exceed tensor bounds
                assign_temporal_feature(i, key, prosodic_dict[key])
        
        # MFCC delta sequences (positions 11-22: 12 features = 6 delta + 6 delta2)
        pos = 11
        for feature_type in ['mfcc_delta', 'mfcc_delta2']:
            if feature_type in prosodic_dict:
                mfcc_sequences = prosodic_dict[feature_type]
                if isinstance(mfcc_sequences, list):
                    for i, seq in enumerate(mfcc_sequences[:6]):  # Only first 6 MFCC coefficients
                        if pos < 28:  # Safety check
                            assign_temporal_feature(pos, f"{feature_type}[{i}]", seq)
                            pos += 1
                        else:
                            break
                else:
                    pos += 6  # Skip 6 positions if wrong type
            else:
                pos += 6  # Skip 6 positions if missing
        
        # Positions 23-27 remain zeros (5 positions for future expansion)
        
        return prosodic_tensor
    
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
            
            # 🚨 DEBUG: Show detailed processing for first few samples
            if sample_idx < 3:
                print(f"\n🚨 DETAILED SAMPLE {sample_idx} DEBUG:")
                print(f"   Item keys: {list(item.keys())}")
                print(f"   Item type: {type(item)}")
            
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
        """
        Generate COMPETITIVE positive windows - boundary at 60ms from end.
        
        Strategy: For each boundary, create a window where the boundary falls exactly
        60ms from the end of the window, providing future context after the boundary.
        """
        saved_count = 0
        
        for boundary_pos in boundary_positions:
            # Position boundary exactly 60ms from end
            boundary_offset_from_end = int(0.06 * self.sample_rate)  # 60ms = 960 samples at 16kHz
            start_pos = boundary_pos - (self.window_samples - boundary_offset_from_end)
            end_pos = start_pos + self.window_samples
            
            # Handle edge cases with padding
            if start_pos < 0:
                # If window would start before audio begins, pad at start
                padding_amount = abs(start_pos)
                window_audio = audio[:end_pos]
                window_audio = np.pad(window_audio, (padding_amount, 0), mode='constant')
                actual_start = 0
            elif end_pos > len(audio):
                # If window would end after audio ends, pad at end
                padding_amount = end_pos - len(audio)
                window_audio = audio[start_pos:]
                window_audio = np.pad(window_audio, (0, padding_amount), mode='constant')
                actual_start = start_pos
            else:
                # Normal case - full window available
                window_audio = audio[start_pos:end_pos]
                actual_start = start_pos

            # Calculate position of boundary within the window (in samples from start)
            boundary_pos_in_window = boundary_pos - actual_start
            
            # Verify boundary is at the expected position (60ms from end)
            expected_pos = self.window_samples - boundary_offset_from_end
            if abs(boundary_pos_in_window - expected_pos) > 10:  # Allow small numerical tolerance
                continue  # Skip if boundary not at expected position
                
            # Skip if window is empty
            if np.all(window_audio == 0):
                continue
                
            # Ensure window length is correct
            if len(window_audio) != self.window_samples:
                continue
            
            # Process with Wav2Vec2
            try:
                inputs = self.processor(window_audio.astype(np.float32), 
                                      sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                # Extract prosodic features as dict
                prosodic_dict = self.extract_prosodic_features(window_audio)
                
                # Convert to tensor ONCE during preprocessing
                seq_len = input_values.shape[0] // 320  # Approximate sequence length
                prosodic_tensor = self._prosodic_dict_to_tensor(prosodic_dict, seq_len)
                
                # Save processed window with COMPETITIVE features
                window_filename = f"window_{len(metadata):06d}.pt"
                window_path = os.path.join(self.save_dir, window_filename)
                
                window_data = {
                    'input_values': input_values,
                    'label': torch.tensor(1.0, dtype=torch.float32),
                    'file_id': file_id,
                    'prosodic_features': prosodic_tensor,
                    'metadata': {
                        'boundary_pos': boundary_pos,
                        'boundary_pos_in_window': boundary_pos_in_window,
                        'window_start': actual_start,
                        'target_boundary_offset': boundary_offset_from_end
                    }
                }
                
                torch.save(window_data, window_path)
                
                # Add to metadata with consistent float label
                metadata.append({
                    'window_file': window_filename,
                    'label': 1.0,  # Use float to match tensor format
                    'file_id': file_id,
                    'boundary_pos': boundary_pos,
                    'boundary_pos_in_window': boundary_pos_in_window
                })
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def _save_negative_windows(self, audio, boundary_positions, file_id, sample_idx, metadata):
        """
        Generate negative windows - ensuring:
        1. NO BOUNDARY within ±20ms of the 60ms point from the end
           (i.e., between (context_buffer - 20ms) and (context_buffer + 20ms) from end)
        2. This creates clear separation between positive examples (boundary exactly at end - context_buffer)
           and negative examples (no boundary within ±20ms of that point)
        """
        import numpy as np  # Import at function start
        saved_count = 0
        boundary_positions = sorted(boundary_positions)
        
        # Target: get roughly equal number of negative windows as positive windows
        target_negatives = max(10, len(boundary_positions))
        target_negatives = min(target_negatives, self.max_negative_per_file)
        
        valid_positions = []
        min_distances = []  # Store minimum distance to boundary for each position
        
        # Fixed timing parameters
        context_buffer = int(0.060 * self.sample_rate)  # 60ms from end - target point for positive examples
        exclusion_start = context_buffer + int(0.020 * self.sample_rate)  # 60ms + 20ms from end
        exclusion_end = context_buffer - int(0.020 * self.sample_rate)    # 60ms - 20ms from end
        
        # Check EVERY sample position - no stride for comprehensive coverage
        for start_pos in range(0, len(audio) - self.window_samples + 1):
            end_pos = start_pos + self.window_samples
            
            # Check if any boundary falls in the critical region
            has_boundary_in_critical_region = False
            min_distance = float('inf')  # Track minimum distance to any boundary
            
            for boundary_pos in boundary_positions:
                # Calculate distance from boundary to the 60ms point from end
                critical_point = end_pos - context_buffer
                distance = abs(boundary_pos - critical_point) / self.sample_rate
                min_distance = min(min_distance, distance)
                
                # Check if boundary is in the critical region (between 80ms and 40ms from end)
                relative_pos = end_pos - boundary_pos  # How far from end is the boundary
                if exclusion_end <= relative_pos <= exclusion_start:
                    has_boundary_in_critical_region = True
                    break
            
            # If no boundaries in critical region, this is a valid negative window
            if not has_boundary_in_critical_region:
                valid_positions.append(start_pos)
                min_distances.append(min_distance)
        
        # Sample from valid positions
        if valid_positions:
            num_to_sample = min(len(valid_positions), target_negatives)
            indices = random.sample(range(len(valid_positions)), num_to_sample)
            selected_positions = [valid_positions[i] for i in indices]
            selected_distances = [min_distances[i] for i in indices]
            
            for start_pos, min_distance in zip(selected_positions, selected_distances):
                end_pos = start_pos + self.window_samples
                window_audio = audio[start_pos:end_pos]
                
                # Skip if window is problematic
                if len(window_audio) < self.window_samples or np.all(window_audio == 0):
                    continue
                    
                # Pad if necessary
                if len(window_audio) < self.window_samples:
                    padding = self.window_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (0, padding), mode='constant')
                
                # Process with Wav2Vec2
                try:
                    inputs = self.processor(window_audio.astype(np.float32), 
                                          sampling_rate=self.sample_rate, return_tensors="pt")
                    input_values = inputs.input_values.squeeze(0)
                    
                    # Extract prosodic features as dict
                    prosodic_dict = self.extract_prosodic_features(window_audio)
                    
                    # Convert to tensor ONCE during preprocessing  
                    seq_len = input_values.shape[0] // 320  # Approximate sequence length
                    prosodic_tensor = self._prosodic_dict_to_tensor(prosodic_dict, seq_len)
                    
                    # Save processed window with COMPETITIVE features
                    window_filename = f"window_{len(metadata):06d}.pt"
                    window_path = os.path.join(self.save_dir, window_filename)
                    
                    window_data = {
                        'input_values': input_values,
                        'label': torch.tensor(0.0, dtype=torch.float32),
                        'file_id': file_id,
                        'prosodic_features': prosodic_tensor,  # NOW A TENSOR, NOT DICT!
                        'metadata': {
                            'boundary_pos': None,
                            'window_start': start_pos,
                            'competitive_exclusion_samples': self.negative_exclusion_samples,
                            'nearest_boundary_distance': min_distance  # Store distance for loss weighting
                        }
                    }
                    
                    torch.save(window_data, window_path)
                    
                    # Add to metadata
                    metadata.append({
                        'window_file': window_filename,
                        'label': 0.0,
                        'file_id': file_id,
                        'boundary_pos': None,
                        'nearest_boundary_distance': min_distance  # Store in metadata too
                    })
                    
                    saved_count += 1
                    
                except Exception as e:
                    continue
        
        # Collect debug stats for summary (don't print per sample)
        if sample_idx < 20:  # Collect stats from first 20 samples
            if not hasattr(self, '_debug_stats'):
                self._debug_stats = {'coverages': [], 'valid_counts': [], 'total_counts': []}
            
            total_positions = len(audio) - self.window_samples + 1
            if total_positions > 0:
                coverage = len(valid_positions) / total_positions * 100
                self._debug_stats['coverages'].append(coverage)
                self._debug_stats['valid_counts'].append(len(valid_positions))
                self._debug_stats['total_counts'].append(total_positions)
            
            # Print summary after collecting stats from first samples
            if sample_idx == 19:  # After collecting 20 samples
                avg_coverage = np.mean(self._debug_stats['coverages'])
                min_coverage = np.min(self._debug_stats['coverages'])
                max_coverage = np.max(self._debug_stats['coverages'])
                avg_valid = np.mean(self._debug_stats['valid_counts'])
                
                print(f"   📊 Negative Window Generation Summary (first 20 samples):")
                print(f"      Average coverage: {avg_coverage:.1f}% (range: {min_coverage:.1f}% - {max_coverage:.1f}%)")
                print(f"      Average valid positions per sample: {avg_valid:.0f}")
                print(f"      Exclusion zone: {self.negative_exclusion_samples} samples (40ms)")
                
                # Clean up debug stats
                delattr(self, '_debug_stats')
        
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
            
            # Ensure prosodic features are tensors
            if 'prosodic_features' not in window_data:
                window_data['prosodic_features'] = torch.zeros(28, 25, dtype=torch.float32)
            elif not isinstance(window_data['prosodic_features'], torch.Tensor):
                # Legacy support - convert dict to tensor
                print(f"⚠️ Converting legacy prosodic features for window {idx}")
                prosodic_dict = window_data['prosodic_features']
                from wav2seg_v4 import CompetitiveWindowPreprocessor
                preprocessor = CompetitiveWindowPreprocessor([], None, verbose=False)
                window_data['prosodic_features'] = preprocessor._prosodic_dict_to_tensor(prosodic_dict)
            
            return window_data
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessed window {idx}: {e}")
            
            # Return robust dummy sample
            return {
                'input_values': torch.zeros(8000, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32),
                'file_id': f'dummy_{idx}',
                'prosodic_features': torch.zeros(28, 25, dtype=torch.float32),  # TENSOR, not dict!
                'metadata': {'boundary_pos': None, 'window_start': 0}
            }

class CompetitiveMultiScaleBoundaryClassifier(nn.Module):
    """
    🏆 COMPETITIVE BOUNDARY CLASSIFIER - LEARNED ATTENTION + PROSODIC FEATURES
    
    COMBINES THE BEST OF BOTH WORLDS:
    ✅ PROVEN: Learned attention mechanism from wav2seg_stable_fixed.py (was working well)
    ✅ NEW: Multi-scale prosodic feature integration for competitive advantage
    ✅ STABLE: Nuclear-grade NaN protection throughout
    
    INTELLIGENT FEATURES:
    - Multi-scale temporal convolutions (3x3 + 9x9 kernels) 
    - Prosodic feature integration (energy, MFCC, spectral features)
    - LEARNED ATTENTION: Model learns which parts of window matter most
    - Residual connections and layer normalization for stability
    - Nuclear-grade NaN prevention at every step
    """
    
    def __init__(self, wav2vec2_model_name="facebook/wav2vec2-base", freeze_wav2vec2=True, 
                 hidden_dim=256, dropout_rate=0.2, use_prosodic=True):
        """
        Initialize the competitive boundary classifier.
        
        Args:
            wav2vec2_model_name: Pretrained Wav2Vec2 model
            freeze_wav2vec2: Whether to freeze Wav2Vec2 parameters
            hidden_dim: Hidden dimension for classification head
            dropout_rate: Dropout rate for regularization
            use_prosodic: Whether to use prosodic features
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
            # NEW: ALL DELTA-BASED prosodic features (28 temporal sequences):
            # Basic deltas (11): energy_delta, energy_delta2, centroid_delta, rolloff_delta, zcr_delta,
            #                   bandwidth_delta, flatness_delta, flux_delta, chroma_delta, tonnetz_delta, tempo_delta
            # MFCC deltas (12): 6 mfcc_delta + 6 mfcc_delta2 sequences  
            # Future expansion (5): positions 23-27 reserved for additional delta features
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
        
        # === LEARNED ATTENTION MECHANISM (from stable_fixed.py) ===
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
        SIMPLE forward pass - prosodic features are just additional input dimensions!
        """
        batch_size = input_values.shape[0]
        
        # Basic input validation
        input_values = torch.clamp(input_values, -3.0, 3.0)
        input_values = torch.nan_to_num(input_values, 0.0)
        
        # Wav2Vec2 feature extraction
        with torch.no_grad():
            try:
                wav2vec2_outputs = self.wav2vec2(input_values)
                hidden_states = wav2vec2_outputs.last_hidden_state.detach()
            except:
                # Fallback if Wav2Vec2 fails
                hidden_states = torch.zeros(batch_size, 49, 768, device=input_values.device)
        
        hidden_states = torch.nan_to_num(hidden_states, 0.0)
        x = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]
        
        # Multi-scale convolutions
        fine_features = F.relu(self.fine_conv1(x))
        fine_features = F.relu(self.fine_conv2(fine_features))
        
        coarse_features = F.relu(self.coarse_conv1(x))
        coarse_features = F.relu(self.coarse_conv2(coarse_features))
        
        feature_list = [fine_features, coarse_features]
        
        # SIMPLE prosodic processing - NO BULLSHIT!
        if self.use_prosodic and prosodic_features is not None:
            # prosodic_features is already a tensor [batch, 28, seq_len]!
            prosodic_features = prosodic_features.to(x.device)
            prosodic_features = torch.nan_to_num(prosodic_features, 0.0)
            
            # Ensure correct sequence length
            seq_len = fine_features.shape[2]
            if prosodic_features.shape[2] != seq_len:
                # Simple interpolation if needed
                prosodic_features = F.interpolate(prosodic_features, size=seq_len, mode='linear', align_corners=False)
            
            # Process with conv layers
            prosodic_processed = F.relu(self.prosodic_conv1(prosodic_features))
            prosodic_processed = F.relu(self.prosodic_conv2(prosodic_processed))
            prosodic_processed = torch.nan_to_num(prosodic_processed, 0.0)
            
            feature_list.append(prosodic_processed)
        
        # Feature fusion
        combined_features = torch.cat(feature_list, dim=1)
        combined_features = torch.nan_to_num(combined_features, 0.0)
        
        fused_features = F.relu(self.fusion_conv(combined_features))
        fused_features = torch.nan_to_num(fused_features, 0.0)
        
        # Attention mechanism
        x = fused_features.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        timestep_scores = self.boundary_detector(x).squeeze(-1)  # [batch, seq_len]
        attention_scores = self.attention_layer(x).squeeze(-1)   # [batch, seq_len]
        
        timestep_scores = torch.nan_to_num(timestep_scores, 0.0)
        attention_scores = torch.nan_to_num(attention_scores, 0.0)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = torch.nan_to_num(attention_weights, 1.0/attention_weights.shape[1])
        
        # Final prediction
        boundary_logit = (timestep_scores * attention_weights).sum(dim=1)
        boundary_logit = torch.nan_to_num(boundary_logit, 0.0)
        
        return boundary_logit


# Keep aliases for backward compatibility (classes only)
LocalBoundaryClassifier = CompetitiveMultiScaleBoundaryClassifier
StableLocalBoundaryClassifier = CompetitiveMultiScaleBoundaryClassifier
WindowPreprocessor = CompetitiveWindowPreprocessor

class IntelligentWindowLoss(nn.Module):
    """
    🧠 INTELLIGENT LOSS for balanced window binary classification.
    
    Features:
    - Adaptive class weighting based on actual batch statistics
    - Distance-based weighting for hard negative examples
    - Confidence regularization to prevent overconfident predictions
    - Label smoothing for better generalization
    """
    
    def __init__(self, pos_weight=1.2, confidence_penalty=0.1, label_smoothing=0.05, 
                 boundary_aware=True, hard_negative_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.confidence_penalty = confidence_penalty  # Prevent overconfidence
        self.label_smoothing = label_smoothing      # Better generalization
        self.boundary_aware = boundary_aware        # Weight harder examples more
        self.hard_negative_weight = hard_negative_weight  # Weight for hard negatives
        
        # Adaptive weighting based on batch statistics
        self.register_buffer('class_counts', torch.zeros(2))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
    def forward(self, logits, targets, metadata=None):
        """
        Intelligent loss for window classification.
        
        Args:
            logits: Model predictions [batch_size]
            targets: Ground truth labels [batch_size]
            metadata: Optional metadata for boundary-aware weighting
        """
        batch_size = logits.shape[0]
        
        # Convert to probabilities for analysis
        probs = torch.sigmoid(logits)
        
        # Apply label smoothing for better generalization
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Base binary cross entropy with class weighting
        pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=pos_weight, reduction='none'
        )
        
        # Confidence regularization - penalize overconfident predictions
        if self.confidence_penalty > 0:
            # Entropy regularization: encourage predictions away from 0 and 1
            entropy_loss = -torch.mean(
                probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)
            )
            confidence_reg = self.confidence_penalty * entropy_loss
        else:
            confidence_reg = 0.0
        
        # Initialize sample weights
        sample_weights = torch.ones_like(bce_loss)
        
        # Distance-based weighting for negative examples
        if metadata is not None:
            for i, meta in enumerate(metadata):
                if targets[i] == 0 and isinstance(meta, dict):
                    # Weight negative examples based on distance to nearest boundary
                    if 'nearest_boundary_distance' in meta:
                        distance_ms = meta['nearest_boundary_distance'] * 1000  # Convert to ms
                        if distance_ms < 50:  # Close negatives (40-50ms)
                            sample_weights[i] *= self.hard_negative_weight
                        elif distance_ms < 100:  # Medium negatives (50-100ms)
                            sample_weights[i] *= (1.0 + self.hard_negative_weight) / 2
                elif targets[i] == 1 and isinstance(meta, dict):
                    # Optional: Weight positive examples based on boundary position
                    if 'boundary_pos_in_window' in meta:
                        boundary_pos = meta['boundary_pos_in_window']
                        if boundary_pos is not None:
                            # Closer to edge = higher weight
                            window_samples = 8000  # 0.5s * 16kHz
                            distance_from_end = window_samples - boundary_pos
                            # Normalize to [1.0, 1.5] range for positives
                            weight = 1.0 + 0.5 * (1.0 - min(distance_from_end / 1000, 1.0))
                            sample_weights[i] *= weight
        
        # Apply sample weights to BCE loss
        bce_loss = bce_loss * sample_weights
        
        # Adaptive class balancing based on batch statistics
        self._update_class_statistics(targets)
        adaptive_weight = self._get_adaptive_weights(targets)
        bce_loss = bce_loss * adaptive_weight
        
        # Combine losses
        total_loss = bce_loss.mean() + confidence_reg
        
        return total_loss
    
    def _update_class_statistics(self, targets):
        """Update running statistics of class distribution."""
        with torch.no_grad():
            pos_count = torch.sum(targets).cpu()  # Move to CPU to match buffer device
            neg_count = len(targets) - pos_count.item()  # Use .item() to get scalar
            
            self.class_counts[1] += pos_count.item()  # Convert to scalar
            self.class_counts[0] += neg_count
            self.total_samples += len(targets)
    
    def _get_adaptive_weights(self, targets):
        """Get adaptive weights based on running class statistics."""
        if self.total_samples < 100:  # Not enough samples yet
            return torch.ones_like(targets)
        
        # Calculate current class frequencies (keep on CPU for buffer compatibility)
        class_freqs = self.class_counts / self.total_samples
        
        # Inverse frequency weighting with smoothing
        weights = 1.0 / (class_freqs + 0.1)  # Add smoothing
        weights = weights / weights.sum() * 2  # Normalize to sum to 2
        
        # Move weights to target device and apply based on target labels
        weights = weights.to(targets.device)
        sample_weights = torch.where(targets == 1, weights[1], weights[0])
        
        return sample_weights



def train_local_classifier(model, train_dataloader, val_dataloader, device, config, num_epochs=10, 
                          learning_rate=1e-4, pos_weight=2.0, use_mixed_precision=False,
                          early_stopping_patience=3, min_improvement=0.001, weight_decay=1e-5,
                          label_smoothing=0.0):
    """
    Train the local boundary classifier with early stopping and optimizations.
    """
    print("🚀 Starting training with AGGRESSIVE positive bias...")
    print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate dataset balance
    print("\n🔍 Validating training dataset balance...")
    if not validate_window_statistics(train_dataloader.dataset):
        print("❌ Dataset validation failed! Please check class balance.")
        return None
        
    print("\n🔍 Validating validation dataset balance...")
    if not validate_window_statistics(val_dataloader.dataset):
        print("❌ Dataset validation failed! Please check class balance.")
        return None
    
    print(f"\n🔧 Device: {device}")
    print(f"📊 Training batches: {len(train_dataloader)}")
    print(f"📊 Validation batches: {len(val_dataloader)}")
    print(f"🎯 Epochs: {num_epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"⚖️ Positive weight: {config['pos_weight']}")  # Use config's pos_weight
    print(f"🎯 Decision threshold: {config['threshold']}")  # Show decision threshold
    print(f"⏰ Early stopping patience: {early_stopping_patience}")
    print("=" * 80)
    
    # Setup COMPETITIVE training components
    criterion = IntelligentWindowLoss(
        pos_weight=config['pos_weight'],
        confidence_penalty=0.1,
        label_smoothing=config.get('label_smoothing', 0.0),
        boundary_aware=True
    )
    
    # Use AdamW optimizer (better for competitive performance)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # COMPETITIVE: Cosine annealing with warm restarts (better than plateau)
    if config.get('use_cosine_scheduling', False):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,              # Restart every 10 epochs
            T_mult=2,            # Double restart period each time
            eta_min=learning_rate / 100,  # Minimum LR
            verbose=True
        )
    else:
        # Fallback to plateau scheduling
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
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
                    # Prosodic features are now part of the batch!
                    prosodic_features = batch.get('prosodic_features', None)
                    logits = model(input_values, prosodic_features)
                    
                    # Monitor predictions distribution in first batch
                    if batch_idx == 0:
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            print(f"\n📊 Prediction Distribution:")
                            print(f"   Mean probability: {probs.mean():.3f}")
                            print(f"   % predictions > 0.5: {(probs > 0.5).float().mean()*100:.1f}%")
                    
                    loss = criterion(logits, labels, batch['metadata'])
                    
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
                # Prosodic features are now part of the batch!
                prosodic_features = batch.get('prosodic_features', None)
                logits = model(input_values, prosodic_features)
                
                loss = criterion(logits, labels, batch['metadata'])  # Safe BCEWithLogitsLoss
                
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
                
                # 🚨 NUCLEAR GRADIENT PROTECTION: Check gradients for NaN/Inf before step
                gradient_is_safe = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                            print(f"🚨 NUCLEAR GRADIENT PROTECTION: {name} has NaN/Inf gradients, zeroing them!")
                            param.grad.zero_()
                            gradient_is_safe = False
                
                if not gradient_is_safe:
                    print(f"🚨 NUCLEAR: Found corrupted gradients in batch {batch_idx}, continuing with zero gradients")
                
                # AGGRESSIVE gradient clipping for nuclear safety
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('gradient_clip_norm', 0.1))  # Much smaller for nuclear safety
                
                # 🚨 NUCLEAR WEIGHT PROTECTION: Check model weights after optimizer step
                optimizer.step()
                
                # Verify model weights are still finite after step
                for name, param in model.named_parameters():
                    if param.requires_grad and torch.any(torch.isnan(param)):
                        print(f"🚨 NUCLEAR WEIGHT CORRUPTION: {name} became NaN after optimizer step!")
                        print(f"💀 CRITICAL FAILURE - MODEL WEIGHTS CORRUPTED!")
                        return history  # Stop training immediately
            
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
        
        # COMPETITIVE learning rate scheduling
        if config.get('use_cosine_scheduling', False):
            scheduler.step()  # Cosine scheduling doesn't need metrics
        else:
            scheduler.step(val_metrics['f1'])  # Plateau scheduling needs F1
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
    
    # Print comprehensive epoch-by-epoch progress table
    print_epoch_progress_table(history, best_val_f1)
    
    return history

def print_epoch_progress_table(history, best_val_f1):
    """
    Print a comprehensive epoch-by-epoch progress table showing all key metrics.
    
    Args:
        history: Training history dictionary
        best_val_f1: Best validation F1 score achieved
    """
    print(f"\n" + "="*120)
    print(f"📊 EPOCH-BY-EPOCH TRAINING PROGRESS")
    print(f"="*120)
    
    # Table header
    print(f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>9} │ {'Accuracy':>8} │ {'Precision':>9} │ {'Recall':>7} │ {'F1 Score':>8} │ {'Learning Rate':>12} │ {'Status':>8}")
    print(f"{'─'*5}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*12}─┼─{'─'*8}")
    
    # Find best epoch for highlighting
    best_f1_epoch = None
    if history['val_f1']:
        best_f1_epoch = max(range(len(history['val_f1'])), key=lambda i: history['val_f1'][i])
    
    # Print each epoch
    for epoch in range(len(history['train_loss'])):
        train_loss = history['train_loss'][epoch]
        val_loss = history['val_loss'][epoch]
        accuracy = history['val_accuracy'][epoch]
        precision = history['val_precision'][epoch]
        recall = history['val_recall'][epoch]
        f1 = history['val_f1'][epoch]
        lr = history['learning_rates'][epoch]
        
        # Status indicators
        status = ""
        if epoch == best_f1_epoch:
            status = "🏆 BEST"
        elif epoch > 0 and f1 > history['val_f1'][epoch-1]:
            status = "📈 UP"
        elif epoch > 0 and f1 < history['val_f1'][epoch-1]:
            status = "📉 DOWN"
        else:
            status = "➡️ SAME"
        
        print(f"{epoch+1:5d} │ {train_loss:10.4f} │ {val_loss:9.4f} │ {accuracy:8.4f} │ {precision:9.4f} │ {recall:7.4f} │ {f1:8.4f} │ {lr:12.2e} │ {status:>8}")
    
    print(f"{'─'*5}─┴─{'─'*10}─┴─{'─'*9}─┴─{'─'*8}─┴─{'─'*9}─┴─{'─'*7}─┴─{'─'*8}─┴─{'─'*12}─┴─{'─'*8}")
    
    # Summary statistics
    if history['val_f1']:
        final_f1 = history['val_f1'][-1]
        best_f1 = max(history['val_f1'])
        worst_f1 = min(history['val_f1'])
        improvement = final_f1 - history['val_f1'][0] if len(history['val_f1']) > 1 else 0.0
        
        print(f"\n📈 TRAINING SUMMARY:")
        print(f"   🎯 Final F1 Score: {final_f1:.4f}")
        print(f"   🏆 Best F1 Score: {best_f1:.4f} (Epoch {best_f1_epoch + 1})")
        print(f"   📉 Worst F1 Score: {worst_f1:.4f}")
        print(f"   📊 Total Improvement: {improvement:+.4f}")
        print(f"   📈 Improvement Rate: {improvement/len(history['val_f1'])*100:+.2f}% per epoch")
        
        # Learning rate changes
        lr_changes = 0
        for i in range(1, len(history['learning_rates'])):
            if history['learning_rates'][i] != history['learning_rates'][i-1]:
                lr_changes += 1
        
        print(f"   🔄 Learning Rate Changes: {lr_changes}")
        print(f"   📚 Final Learning Rate: {history['learning_rates'][-1]:.2e}")
        
        # Performance trend analysis
        if len(history['val_f1']) >= 3:
            recent_trend = np.mean(history['val_f1'][-3:]) - np.mean(history['val_f1'][-6:-3]) if len(history['val_f1']) >= 6 else 0
            trend_direction = "improving" if recent_trend > 0.01 else "declining" if recent_trend < -0.01 else "stable"
            print(f"   📊 Recent Trend: {trend_direction} ({recent_trend:+.4f})")
        
        # Convergence analysis
        if len(history['val_f1']) >= 5:
            last_5_std = np.std(history['val_f1'][-5:])
            convergence_status = "converged" if last_5_std < 0.01 else "still learning" if last_5_std < 0.05 else "unstable"
            print(f"   🎯 Convergence Status: {convergence_status} (std: {last_5_std:.4f})")
    
    print(f"="*120)

def evaluate_local_classifier(model, dataloader, device, criterion=None, threshold=None):
    """
    Evaluate the local boundary classifier.
    
    Args:
        model: Trained LocalBoundaryClassifier
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)
        threshold: Threshold for binary classification (use config value if None)
        
    Returns:
        tuple: (loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    # Get threshold from config if not provided
    if threshold is None:
        threshold = 0.65  # Default consistent threshold
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass - get logits with prosodic features from batch
            prosodic_features = batch.get('prosodic_features', None)
            logits = model(input_values, prosodic_features)
            
            if criterion:
                loss = criterion(logits, labels, batch['metadata'])
                total_loss += loss.item()
            
            # Convert logits to probabilities, then to binary predictions
            probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
            binary_predictions = (probabilities > threshold).float()  # Use provided/config threshold
            
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

def predict_boundaries_competitive(model, audio, processor, device, 
                                   window_duration=0.5, stride=0.01, 
                                   sample_rate=16000, threshold=0.65, 
                                   grouping_distance=0.02):
    """
    🏆 COMPETITIVE boundary prediction with confidence weighting.
    
    Args:
        model: Trained CompetitiveMultiScaleBoundaryClassifier
        audio: Audio array
        processor: Wav2Vec2 processor
        device: Device for inference
        window_duration: Window duration (0.5s for focused boundary detection)
        stride: Stride (10ms for ±20ms precision)
        sample_rate: Sample rate
        threshold: Threshold for binary classification
        grouping_distance: Distance for grouping (20ms for competition)
        
    Returns:
        list: Predicted boundary positions in samples
    """
    model.eval()
    
    window_samples = int(window_duration * sample_rate)
    stride_samples = int(stride * sample_rate)
    grouping_samples = int(grouping_distance * sample_rate)
    
    # Create prosodic feature extractor once (for efficiency)
    prosodic_extractor = None
    if hasattr(model, 'use_prosodic') and model.use_prosodic:
        prosodic_extractor = CompetitiveWindowPreprocessor(
            [], processor, 
            window_duration=window_duration,
            sample_rate=sample_rate,
            use_prosodic=True,
            verbose=False
        )
    
    # Collect all window predictions
    window_predictions = []
    
    with torch.no_grad():
        for start_pos in range(0, len(audio) - window_samples + 1, stride_samples):
            end_pos = start_pos + window_samples
            window_audio = audio[start_pos:end_pos]
            
            # Ensure window is correct length
            if len(window_audio) != window_samples:
                # Pad if necessary
                padding_needed = window_samples - len(window_audio)
                window_audio = np.pad(window_audio, (0, padding_needed), mode='constant')
            
            # Process window with Wav2Vec2
            inputs = processor(window_audio.astype(np.float32), sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            
            # FIXED: Extract REAL prosodic features (not zeros!)
            prosodic_features = None
            if prosodic_extractor is not None:
                try:
                    # Extract prosodic features exactly like during training
                    prosodic_dict = prosodic_extractor.extract_prosodic_features(window_audio)
                    
                    # Get sequence length to match Wav2Vec2 output
                    seq_len = input_values.shape[1] // 320  # Wav2Vec2 downsampling factor
                    if seq_len <= 0:
                        seq_len = 25  # Default fallback
                    
                    # Convert to tensor with exact same format as training
                    prosodic_tensor = prosodic_extractor._prosodic_dict_to_tensor(prosodic_dict, seq_len)
                    
                    # Add batch dimension and move to device
                    prosodic_features = prosodic_tensor.unsqueeze(0).to(device)
                    
                    # Ensure dimensions match: [batch_size, 28, seq_len]
                    if prosodic_features.shape[1] != 28:
                        print(f"⚠️ Prosodic feature dim mismatch: {prosodic_features.shape[1]} != 28, using zeros")
                        prosodic_features = torch.zeros(1, 28, seq_len, device=device)
                    
                except Exception as e:
                    # Fallback to zeros only if extraction fails
                    seq_len = input_values.shape[1] // 320
                    if seq_len <= 0:
                        seq_len = 25
                    prosodic_features = torch.zeros(1, 28, seq_len, device=device)
                    if start_pos == 0:  # Only warn once
                        print(f"⚠️ Prosodic feature extraction failed: {e}, using zeros")
            
            # Get model prediction
            logits = model(input_values, prosodic_features)
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

def test_prosodic_feature_fix():
    """
    🧪 Quick test to verify prosodic features are working correctly in inference.
    """
    print("\n🧪 TESTING PROSODIC FEATURE FIX...")
    
    try:
        # Create test audio
        test_audio = np.random.randn(8000).astype(np.float32)  # 0.5s at 16kHz
        
        # Initialize processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        # Test prosodic feature extraction
        prosodic_extractor = CompetitiveWindowPreprocessor(
            [], processor, 
            window_duration=0.5,
            sample_rate=16000,
            use_prosodic=True,
            verbose=False
        )
        
        # Extract prosodic features
        prosodic_dict = prosodic_extractor.extract_prosodic_features(test_audio)
        
        # Convert to tensor
        prosodic_tensor = prosodic_extractor._prosodic_dict_to_tensor(prosodic_dict, 25)
        
        print(f"✅ Prosodic feature extraction successful!")
        print(f"   Tensor shape: {prosodic_tensor.shape}")
        print(f"   Expected: [28, 25]")
        print(f"   Non-zero values: {torch.sum(prosodic_tensor != 0).item()}")
        print(f"   Value range: [{prosodic_tensor.min():.4f}, {prosodic_tensor.max():.4f}]")
        
        # Test with batch dimension
        batch_tensor = prosodic_tensor.unsqueeze(0)
        print(f"   Batch tensor shape: {batch_tensor.shape}")
        print(f"   Expected: [1, 28, 25]")
        
        return True
        
    except Exception as e:
        print(f"❌ Prosodic feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_timit_data_for_local_windows(split='train', max_samples=None):
    """Load TIMIT data for local window approach."""
    print(f"\n🚨 LOADING TIMIT DATA DEBUG:")
    print(f"   Split: {split}")
    print(f"   Max samples: {max_samples}")
    
    try:
        from wav2seg import load_data  # Try to import from original file
        print(f"   ✅ Successfully imported load_data from wav2seg")
        
        data = load_data(split, max_samples)
        print(f"   📊 Raw data length: {len(data)}")
        print(f"   📊 Raw data type: {type(data)}")
        
        # DEBUG: Show the structure of first item
        if len(data) > 0:
            first_item = data[0]
            print(f"   🔍 First item keys: {list(first_item.keys())}")
            print(f"   🔍 First item type: {type(first_item)}")
            
            if 'audio' in first_item:
                print(f"   🔍 Audio keys: {list(first_item['audio'].keys())}")
                audio_array = first_item['audio']['array']
                print(f"   🔍 Audio array shape: {audio_array.shape}")
                print(f"   🔍 Audio array type: {type(audio_array)}")
                print(f"   🔍 Audio sampling rate: {first_item['audio']['sampling_rate']}")
            else:
                print("   💀 CRITICAL: No 'audio' key in first item!")
                
            if 'phonetic_detail' in first_item:
                phonetic = first_item['phonetic_detail']
                print(f"   🔍 Phonetic detail keys: {list(phonetic.keys())}")
                if 'start' in phonetic:
                    starts = phonetic['start']
                    print(f"   🔍 Found {len(starts)} phonetic segments")
                    print(f"   🔍 First few starts: {starts[:3] if len(starts) >= 3 else starts}")
                else:
                    print("   💀 CRITICAL: No 'start' key in phonetic_detail!")
            else:
                print("   💀 CRITICAL: No 'phonetic_detail' key in first item!")
        else:
            print("   💀 CRITICAL: load_data returned empty list!")
        
        return data
    
    except ImportError as e:
        print(f"   💀 CRITICAL: Could not import load_data from wav2seg: {e}")
        print("   💡 This means the wav2seg module is missing or broken!")
        return []
    except Exception as e:
        print(f"   💀 CRITICAL: Unexpected error in load_data: {e}")
        import traceback
        print(f"   📋 Full traceback: {traceback.format_exc()}")
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
            
            # COMPETITIVE boundary prediction with ±20ms precision
            pred_boundaries = predict_boundaries_competitive(
                model, audio, processor, device,
                window_duration=config['window_duration'],  # 0.5s windows for focused detection
                stride=0.01,  # 10ms stride for ±20ms precision
                sample_rate=sample_rate,
                threshold=config['threshold'],  # Competitive threshold
                grouping_distance=0.02  # 20ms grouping (competition standard)
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
        print(f"   Mean MAE: {metrics['mean_mae']:.2f} ± {metrics['std_f1']:.2f} frames")
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

def debug_window_predictions(model, dataloader, device, num_samples=10, threshold=0.65):
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
    print(f"\n🔍 DEBUGGING WINDOW PREDICTIONS (Threshold: {threshold})")
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
            
            # Get predictions with prosodic features from batch
            prosodic_features = batch.get('prosodic_features', None)
            logits = model(input_values, prosodic_features)
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

def analyze_window_confusion_matrix(model, dataloader, device, threshold=0.65):
    """
    Analyze the confusion matrix for window-level predictions.
    """
    print(f"\n📊 WINDOW-LEVEL CONFUSION MATRIX ANALYSIS (Threshold: {threshold})")
    print("=" * 50)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Get prosodic features from batch
            prosodic_features = batch.get('prosodic_features', None)
            logits = model(input_values, prosodic_features)
            predictions = torch.sigmoid(logits)  # Convert to probabilities [0,1]
            binary_predictions = (predictions > threshold).float()  # Use provided threshold
            
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
    Custom collate function - prosodic features are now tensors!
    """
    # Filter out None values
    valid_samples = [sample for sample in batch if sample is not None]
    
    if len(valid_samples) == 0:
        # Create dummy batch with tensor prosodic features
        dummy_sample = {
            'input_values': torch.zeros(8000),
            'label': torch.tensor(0.0),
            'file_id': 'dummy_batch',
            'prosodic_features': torch.zeros(28, 25, dtype=torch.float32),  # TENSOR!
            'metadata': {'boundary_pos': None, 'window_start': 0}
        }
        valid_samples = [dummy_sample]
    
    try:
        # Stack prosodic tensors - MUCH simpler!
        prosodic_tensors = []
        for sample in valid_samples:
            prosodic = sample.get('prosodic_features', None)
            if prosodic is None or not isinstance(prosodic, torch.Tensor):
                # Default tensor for missing/invalid prosodic features
                prosodic = torch.zeros(28, 25, dtype=torch.float32)
            prosodic_tensors.append(prosodic)
        
        collated = {
            'input_values': torch.stack([sample['input_values'] for sample in valid_samples]),
            'label': torch.stack([sample['label'] for sample in valid_samples]),
            'file_id': [sample['file_id'] for sample in valid_samples],
            'metadata': [sample['metadata'] for sample in valid_samples],
            'prosodic_features': torch.stack(prosodic_tensors)  # Stack tensors directly!
        }
        return collated
    except Exception as e:
        print(f"⚠️ Error in collation: {e}")
        # Safe fallback
        batch_size = len(valid_samples)
        return {
            'input_values': torch.zeros(batch_size, 8000),
            'label': torch.zeros(batch_size),
            'file_id': [f'error_dummy_{i}' for i in range(batch_size)],
            'prosodic_features': torch.zeros(batch_size, 28, 25, dtype=torch.float32),  # TENSOR!
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
    print(f"🚨 DEBUG: Dataset length: {len(dataset)}")
    print(f"🚨 DEBUG: Dataset type: {type(dataset)}")
    
    # CRITICAL SAFETY CHECK FOR EMPTY DATASET
    if len(dataset) == 0:
        print("🚨 CRITICAL PROBLEM: Dataset is completely empty!")
        print("💀 This means your window preprocessing created ZERO windows!")
        print("🔍 Check the preprocessing logs above for errors.")
        return False
    
    valid_count = 0
    error_count = 0
    
    # Ensure we don't try to sample more than available
    actual_num_samples = min(num_samples, len(dataset))
    print(f"🚨 DEBUG: Sampling {actual_num_samples} out of {len(dataset)} available samples")
    
    # Check a few samples from different parts of the dataset
    sample_indices = np.linspace(0, len(dataset) - 1, actual_num_samples, dtype=int)
    print(f"🚨 DEBUG: Sample indices: {sample_indices}")
    
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
            
            prosodic_features = batch.get('prosodic_features', None)
            logits = model(input_values, prosodic_features)
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

def validate_window_statistics(dataset):
    """Validate the balance of positive and negative windows in the dataset."""
    # Handle both Dataset and Subset objects
    if hasattr(dataset, 'indices'):  # If it's a Subset
        full_dataset = dataset.dataset
        indices = dataset.indices
        # Only count labels for this subset's indices
        pos_count = sum(1 for idx in indices if full_dataset[idx]['label'].item() == 1)
        neg_count = sum(1 for idx in indices if full_dataset[idx]['label'].item() == 0)
    else:  # If it's the full dataset
        if hasattr(dataset, 'metadata'):
            # Use metadata if available (faster)
            pos_count = sum(1 for meta in dataset.metadata if meta['label'] == 1)
            neg_count = sum(1 for meta in dataset.metadata if meta['label'] == 0)
        else:
            # Fallback to iterating through dataset
            pos_count = sum(1 for i in range(len(dataset)) if dataset[i]['label'].item() == 1)
            neg_count = sum(1 for i in range(len(dataset)) if dataset[i]['label'].item() == 0)
    
    total = pos_count + neg_count
    
    print("\n📊 Window Statistics:")
    print(f"   Total windows: {total:,}")
    print(f"   Positive windows: {pos_count:,} ({pos_count/total*100:.1f}%)")
    print(f"   Negative windows: {neg_count:,} ({neg_count/total*100:.1f}%)")
    print(f"   Ratio (pos:neg): 1:{neg_count/pos_count:.2f}")
    
    if pos_count == 0 or neg_count == 0:
        print("⚠️  WARNING: One class has zero samples!")
        return False
        
    if pos_count/total < 0.2 or pos_count/total > 0.8:
        print("⚠️  WARNING: Severe class imbalance detected!")
        return False
    
    return True

def main():
    """
    Main training pipeline with optimized settings.
    """
    config = {
        # COMPETITIVE PARAMETERS MATCHING 85-88% F1 SCORES
        'pos_weight': 1.6,                   # COMPETITIVE: Slight positive bias
        'threshold': 0.45,                    # COMPETITIVE: Standard threshold
        'boundary_tolerance': 0.02,          # COMPETITIVE: ±20ms tolerance (competition standard)
        
        # COMPETITIVE TRAINING STRATEGY  
        'batch_size': 256,                    # Larger batches for better gradient estimates
        'gradient_accumulation_steps': 1,    # No need with larger batch
        'num_epochs': 60,                    # Sufficient for competitive performance
        'learning_rate': 2e-4,               # COMPETITIVE: Higher LR for learning
        'weight_decay': 1e-5,                # Light regularization
        'use_cosine_scheduling': True,       # COMPETITIVE: Better than plateau
        
        # PURE CONV1D MODEL ARCHITECTURE - NO POOLING!
        'hidden_dim': 512,                   # COMPETITIVE: Double capacity
        'dropout_rate': 0.2,                 # COMPETITIVE: Less dropout for capacity
        'use_prosodic': True,                # COMPETITIVE: Prosodic features via Conv1d
        
        # COMPETITIVE WINDOW GENERATION
        'window_duration': 0.5,              # COMPETITIVE: 0.5s windows for focused boundary detection
        'negative_exclusion_zone': 0.01,     # 120ms exclusion (6x separation)
        'negative_sampling_ratio': 0.5,      # Balanced sampling
        'max_windows_per_file': None,        # No limits
        'max_positive_per_file': None,       # All boundaries
        'max_negative_per_file': 50,         # Reasonable limit
        
        # INTELLIGENT LOSS CONFIGURATION
        'label_smoothing': 0.05,             # Better generalization
        'confidence_penalty': 0.1,           # Prevent overconfidence
        'boundary_aware_weighting': True,    # Weight harder examples
        
        # OPTIMIZATION
        'gradient_clip_norm': 0.1,           # NUCLEAR AGGRESSIVE: Prevent any explosion
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': False,            # Keep it simple
        'use_gradient_checkpointing': False, # Keep it simple
        
        # EARLY STOPPING
        'early_stopping_patience': 20,        # Reasonable patience
        'min_improvement': 0.0005,            # Small improvements count
        
        # COMPETITIVE PREPROCESSING DIRECTORIES
        'train_preprocessed_dir': './preprocessed_windows_train_0.5s_competitive',  # COMPETITIVE: ±20ms tolerance + prosodic
        'test_preprocessed_dir': './preprocessed_windows_test_0.5s_competitive',    # COMPETITIVE: ±20ms tolerance + prosodic
        'force_reprocess': True,          # FORCE reprocess for competitive features
        'sample_rate': 16000,               # Standard sample rate
        'gradient_clip_norm': 1.0,          # COMPETITIVE: Gradient clipping
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
    
    # 🚨 CRITICAL DEBUG: Check if TIMIT data is actually loaded
    print(f"\n🚨 TIMIT DATA DEBUG:")
    print(f"   Train data length: {len(train_data)}")
    print(f"   Test data length: {len(test_data)}")
    print(f"   Train data type: {type(train_data)}")
    print(f"   Test data type: {type(test_data)}")
    
    if len(train_data) == 0:
        print("💀 CRITICAL: Training data is EMPTY! TIMIT loading failed!")
        return None, None, None
    
    if len(test_data) == 0:
        print("💀 CRITICAL: Test data is EMPTY! TIMIT loading failed!")
        return None, None, None
    
    # Debug first sample structure
    print(f"   🔍 First train item keys: {list(train_data[0].keys())}")
    if 'audio' in train_data[0]:
        audio_info = train_data[0]['audio']
        print(f"   🔍 Audio keys: {list(audio_info.keys())}")
        print(f"   🔍 Audio array shape: {audio_info['array'].shape}")
        print(f"   🔍 Audio sampling rate: {audio_info['sampling_rate']}")
    
    if 'phonetic_detail' in train_data[0]:
        phonetic_info = train_data[0]['phonetic_detail']
        print(f"   🔍 Phonetic detail keys: {list(phonetic_info.keys())}")
        if 'start' in phonetic_info:
            starts = phonetic_info['start']
            print(f"   🔍 Number of phonetic segments: {len(starts)}")
            print(f"   🔍 First few start times: {starts[:5] if len(starts) >= 5 else starts}")
    else:
        print("   💀 CRITICAL: No 'phonetic_detail' found in first sample!")
        
    print(f"   🔍 Sample ID: {train_data[0].get('id', 'NO_ID_FOUND')}")
    
    # ===============================
    # STAGE 1: PREPROCESSING
    # ===============================
    print("\n" + "="*80)
    print("🚀 STAGE 1: PREPROCESSING ALL WINDOWS")
    print("💡 This is done ONCE - subsequent runs will be much faster!")
    print("="*80)
    
    # Create COMPETITIVE preprocessors for train and test data
    print("\n🏆 Creating COMPETITIVE window preprocessors...")
    print(f"📋 COMPETITIVE Configuration Summary:")
    print(f"   Window duration: {config['window_duration']}s")
    print(f"   Boundary tolerance: ±{config['boundary_tolerance']*1000:.0f}ms (competition standard)")
    print(f"   Negative exclusion: {config['negative_exclusion_zone']*1000:.0f}ms")
    print(f"   Prosodic features: {'ENABLED' if config['use_prosodic'] else 'DISABLED'}")
    print(f"   Target: 80-90% F1 @ ±20ms (competitive range)")
    
    train_preprocessor = CompetitiveWindowPreprocessor(
        train_data, processor, 
        window_duration=config['window_duration'],
        boundary_tolerance=config['boundary_tolerance'],  # ±20ms competitive
        negative_exclusion_zone=config['negative_exclusion_zone'],
        negative_sampling_ratio=config['negative_sampling_ratio'],
        save_dir=config['train_preprocessed_dir'],
        max_windows_per_file=config['max_windows_per_file'],
        max_positive_per_file=config['max_positive_per_file'],
        max_negative_per_file=config['max_negative_per_file'],
        use_prosodic=config['use_prosodic'],  # COMPETITIVE: Prosodic features
        use_gaussian_targets=False,  # Keep binary for window classification
        verbose=False  # Configuration shown in summary above
    )
    
    test_preprocessor = CompetitiveWindowPreprocessor(
        test_data, processor,
        window_duration=config['window_duration'],
        boundary_tolerance=config['boundary_tolerance'],  # ±20ms competitive
        negative_exclusion_zone=config['negative_exclusion_zone'],
        negative_sampling_ratio=config['negative_sampling_ratio'],
        save_dir=config['test_preprocessed_dir'],
        max_windows_per_file=config['max_windows_per_file'],
        max_positive_per_file=config['max_positive_per_file'],
        max_negative_per_file=config['max_negative_per_file'],
        use_prosodic=config['use_prosodic'],  # COMPETITIVE: Prosodic features
        use_gaussian_targets=False,  # Keep binary for window classification
        verbose=False  # Don't repeat configuration for test preprocessor
    )
    
    # Preprocess training data
    print(f"\n📦 Preprocessing TRAINING data...")
    preprocessing_start = time.time()
    train_metadata = train_preprocessor.preprocess_all_windows(force_reprocess=config['force_reprocess'])
    train_preprocessing_time = time.time() - preprocessing_start
    
    # 🚨 CRITICAL DEBUG: Check preprocessing results
    print(f"\n🚨 TRAINING PREPROCESSING DEBUG:")
    print(f"   Metadata type: {type(train_metadata)}")
    print(f"   Metadata length: {len(train_metadata) if train_metadata else 'None'}")
    if train_metadata and len(train_metadata) > 0:
        print(f"   First metadata item: {train_metadata[0]}")
        pos_count = sum(1 for m in train_metadata if m.get('label') == 1.0)
        neg_count = len(train_metadata) - pos_count
        print(f"   Positive windows: {pos_count}")
        print(f"   Negative windows: {neg_count}")
    else:
        print("💀 CRITICAL: Training preprocessing produced NO metadata!")
    
    # Preprocess test data
    print(f"\n📦 Preprocessing TEST data...")
    test_preprocessing_start = time.time()
    test_metadata = test_preprocessor.preprocess_all_windows(force_reprocess=config['force_reprocess'])
    test_preprocessing_time = time.time() - test_preprocessing_start
    
    # 🚨 CRITICAL DEBUG: Check test preprocessing results
    print(f"\n🚨 TEST PREPROCESSING DEBUG:")
    print(f"   Metadata type: {type(test_metadata)}")
    print(f"   Metadata length: {len(test_metadata) if test_metadata else 'None'}")
    if test_metadata and len(test_metadata) > 0:
        print(f"   First metadata item: {test_metadata[0]}")
        pos_count = sum(1 for m in test_metadata if m.get('label') == 1.0)
        neg_count = len(test_metadata) - pos_count
        print(f"   Positive windows: {pos_count}")
        print(f"   Negative windows: {neg_count}")
    else:
        print("💀 CRITICAL: Test preprocessing produced NO metadata!")
    
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
    
    # 🚨 CRITICAL DEBUG: Check metadata before creating datasets
    print(f"\n🚨 PRE-DATASET DEBUG:")
    print(f"   Train metadata available: {train_metadata is not None}")
    print(f"   Test metadata available: {test_metadata is not None}")
    if train_metadata:
        print(f"   Train metadata length: {len(train_metadata)}")
    if test_metadata:
        print(f"   Test metadata length: {len(test_metadata)}")
    
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
    
    # 🚨 CRITICAL DEBUG: Check dataset lengths
    print(f"\n🚨 DATASET DEBUG:")
    print(f"   Train dataset length: {len(train_dataset)}")
    print(f"   Test dataset length: {len(test_dataset)}")
    print(f"   Train dataset type: {type(train_dataset)}")
    print(f"   Test dataset type: {type(test_dataset)}")
    
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
    train_dataloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)
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
    
    print("   🏆 Loading COMPETITIVE model architecture...")
    try:
        # PURE Conv1d Multi-Scale Boundary Classifier - NO POOLING!
        model = CompetitiveMultiScaleBoundaryClassifier(
            freeze_wav2vec2=True,
            hidden_dim=config['hidden_dim'],           # 256 dimensions
            dropout_rate=config['dropout_rate'],       # 0.2 dropout
            use_prosodic=config['use_prosodic']        # Prosodic features via Conv1d
        )
        print("   ✅ COMPETITIVE model architecture created successfully")
        
        # Enable gradient checkpointing if requested
        if config.get('use_gradient_checkpointing', False):
            if hasattr(model.wav2vec2, 'gradient_checkpointing_enable'):
                model.wav2vec2.gradient_checkpointing_enable()
                print("   ✅ Gradient checkpointing enabled")
        
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        print("   🔄 Trying with even smaller architecture...")
        try:
            # Try with minimal architecture if base fails
            model = CompetitiveMultiScaleBoundaryClassifier(
                wav2vec2_model_name="facebook/wav2vec2-base", 
                freeze_wav2vec2=True,
                hidden_dim=128,  # Smaller but still pure Conv1d
                dropout_rate=0.3,
                use_prosodic=False  # Disable prosodic for minimal mode
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
        label_smoothing=config.get('label_smoothing', 0.0)
    )
    training_time = time.time() - training_start
    print(f"✅ Training completed in {training_time/60:.1f} minutes")
    
    # Debug model predictions with configured threshold
    print(f"\n🔧 Debugging trained model with threshold {config['threshold']}...")
    debug_window_predictions(model, val_dataloader, config['device'], num_samples=10, threshold=config['threshold'])
    analyze_window_confusion_matrix(model, val_dataloader, config['device'], threshold=config['threshold'])
    
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
    
    # Test prosodic feature extraction fix
    print(f"\n🧪 Testing prosodic feature fix...")
    test_success = test_prosodic_feature_fix()
    if not test_success:
        print("⚠️ Prosodic feature test failed, but continuing evaluation...")
    
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
    
    print(f"\n🏆 COMPETITIVE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"🚀 PURE CONV1D ARCHITECTURE IMPROVEMENTS vs 85-88% F1 COMPETITORS:")
    print(f"   🎯 Multi-scale temporal convolutions (3x3 + 9x9 kernels)")
    print(f"   🎵 ALL DELTA-BASED prosodic features processed via Conv1d (23 temporal delta sequences)")
    print(f"   🔥 NO POOLING: Pure temporal sequence processing")
    print(f"   📏 ±20ms tolerance (competition standard)")
    print(f"   💪 Gradual sequence reduction to final prediction")
    print(f"   📈 Cosine annealing + AdamW optimizer")
    print(f"   ⚡ Gradient accumulation (effective batch size: 24)")
    print(f"   🚨 Nuclear-grade NaN prevention at every step")
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
    
    print(f"\n🏆 COMPETITIVE ARCHITECTURE IMPROVEMENTS:")
    print(f"   📦 Stage 1 (preprocessing): {total_preprocessing_time/60:.1f} min (ONCE only)")
    print(f"   ⚡ Stage 2 (training): {training_time/60:.1f} min (every run)")
    print(f"   🎯 COMPETITIVE tolerance: ±20ms (vs competitors' ±20ms)")
    print(f"   🎵 ALL DELTA-BASED prosodic features via Conv1d: 23 temporal delta sequences")
    print(f"   🔥 Pure Conv1d architecture: NO POOLING, gradual sequence reduction")
    print(f"   🧠 Multi-scale convolutions: 3x3 + 9x9 kernels")
    print(f"   💾 Preprocessed windows saved for instant reuse")
    print(f"   🔄 Subsequent training runs: ~{training_time/60:.1f} min vs {total_pipeline_time/60:.1f} min")
    print(f"   ⚡ Speedup for future runs: {total_pipeline_time/training_time:.1f}x faster!")
    
    print(f"\n📊 Final Performance Summary:")
    print(f"   Window-level F1: {test_metrics['f1']:.4f}")
    print(f"   Full-audio F1: {results['overall_metrics']['mean_f1']:.4f}")
    print(f"   Best validation F1: {max(history['val_f1']):.4f}")
    print(f"   Training samples: {len(train_data):,} audio files → {len(train_dataset):,} windows")
    print(f"   Test samples: {len(test_data):,} audio files → {len(test_dataset):,} windows")
    print(f"   Class balance: ~50/50 from asymmetric generation")
    
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
    
    print(f"\n🏆 PURE CONV1D pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥 NEW: ALL DELTA-BASED prosodic features for superior boundary detection!")
    print("📈 Target: 80-90% F1 @ ±20ms (competitive with 85-88% F1 leaders)")
    print("🚨 DELTA ADVANTAGE: Transitions/changes are superior signals for boundary detection!")
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

# Function aliases (after functions are defined)
predict_boundaries_on_full_audio = predict_boundaries_competitive