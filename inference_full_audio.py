"""
STANDALONE FULL AUDIO BOUNDARY DETECTION INFERENCE - V4 COMPETITIVE
===================================================================

🏆 COMPETITIVE VERSION - Compatible with wav2seg_v4_super_buffer.py
Uses EXACT SAME preprocessing and feature extraction as competitive training.

Features:
- CompetitiveMultiScaleBoundaryClassifier support
- ALL DELTA-BASED prosodic features (28 temporal sequences)
- Configurable competitive parameters (context buffer, tolerances, etc.)
- ±20ms competition-standard boundary detection
- Multi-scale temporal analysis (3x3 + 9x9 convolutions)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import warnings
import time
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# =============================================================================
# COMPETITIVE CONFIGURATION - All parameters from wav2seg_v4_super_buffer.py
# =============================================================================

# COMPETITIVE INFERENCE CONFIG - Matching training parameters
COMPETITIVE_CONFIG = {
    # MODEL ARCHITECTURE
    'wav2vec2_model_name': "facebook/wav2vec2-base",
    'freeze_wav2vec2': True,
    'hidden_dim': 512,                    # COMPETITIVE: Double capacity
    'dropout_rate': 0.2,                  # COMPETITIVE: Less dropout for capacity
    'use_prosodic': True,                 # COMPETITIVE: ALL DELTA-BASED prosodic features
    
    # WINDOW PROCESSING
    'window_duration': 0.5,               # COMPETITIVE: 0.5s windows for focused boundary detection
    'sample_rate': 16000,                 # Standard sample rate
    'stride': 0.01,                       # 10ms stride for ±20ms precision
    'context_buffer': 0.060,              # 60ms from end - target point for boundary positioning
    
    # COMPETITIVE TOLERANCES & THRESHOLDS
    'boundary_tolerance': 0.02,           # COMPETITIVE: ±20ms tolerance (competition standard)
    'threshold': 0.45,                    # COMPETITIVE: Standard decision threshold
    'grouping_distance': 0.02,            # 20ms grouping (competition standard)
    'negative_exclusion_zone': 0.12,      # 120ms exclusion (6x separation for negatives)
    
    # PROSODIC FEATURES (ALL DELTA-BASED)
    'prosodic_features': {
        'basic_deltas': 11,               # energy_delta, energy_delta2, centroid_delta, etc.
        'mfcc_deltas': 12,                # 6 mfcc_delta + 6 mfcc_delta2 sequences
        'additional_deltas': 5,           # bandwidth_delta, flatness_delta, etc.
        'total_channels': 28,             # Total prosodic feature channels
        'sequence_length': 25             # Default sequence length (~0.5s at 50Hz)
    },
    
    # INFERENCE OPTIMIZATION
    'batch_processing': False,            # Process windows individually for memory efficiency
    'device': 'auto',                     # Auto-detect CUDA/CPU
    'mixed_precision': False,             # Keep inference simple
    'gradient_checkpointing': False,      # Not needed for inference
    
    # OUTPUT CONTROL
    'verbose': True,                      # Detailed logging
    'debug_mode': False,                  # Extended debugging information
    'save_visualizations': True,         # Save prediction plots
    'model_path': 'best_local_model.pth' # Default model checkpoint
}

def visualize_predictions(all_frames, all_predictions, true_boundaries, sample_rate, stride_samples, save_path, threshold=0.45):
    """Create enhanced visualization of predictions with true boundaries marked.
    
    Args:
        all_frames: List of frame positions in samples
        all_predictions: List of prediction values for each frame
        true_boundaries: List of true boundary positions in samples
        sample_rate: Audio sample rate (e.g. 16000)
        stride_samples: Number of samples between frames
        save_path: Path to save the visualization
        threshold: Decision threshold for predictions (default: 0.45)
    """
    try:
        plt.figure(figsize=(20, 6))
        
        # Handle empty data gracefully
        if len(all_frames) == 0 or len(all_predictions) == 0:
            plt.text(0.5, 0.5, 'No prediction data available\n(Check if debug mode or visualization data collection is enabled)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('No Predictions Available')
        else:
            # Convert sample positions to time in seconds
            frame_times = [frame/sample_rate for frame in all_frames]
            true_boundary_times = [tb/sample_rate for tb in true_boundaries]
            
            # Plot prediction values as bars
            plt.bar(frame_times, all_predictions, width=stride_samples/sample_rate, 
                   alpha=0.6, label='Predictions', color='skyblue', edgecolor='navy', linewidth=0.5)
            
            # Plot true boundaries as red vertical lines
            if true_boundary_times:
                for i, tb_time in enumerate(true_boundary_times):
                    label = 'True Boundaries' if i == 0 else ''
                    plt.axvline(x=tb_time, color='red', linestyle='-', alpha=0.8, linewidth=2, label=label)
            
            # Add threshold line using the actual threshold value
            if len(all_predictions) > 0:
                plt.axhline(y=threshold, color='orange', linestyle='--', 
                           alpha=0.8, linewidth=2, label=f'Threshold ({threshold})')
            
            # Add statistics to title
            num_above_threshold = sum(1 for p in all_predictions if p > threshold)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Prediction Value', fontsize=12)
            plt.title(f'Boundary Predictions vs True Boundaries\n'
                     f'Predictions: {len(all_predictions)}, Above threshold: {num_above_threshold}, '
                     f'True boundaries: {len(true_boundaries)}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
            # Set y-axis limits for better visualization
            plt.ylim(-0.05, 1.05)
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the plot with high quality
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Visualization saved: {save_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
        # Ensure we don't leave figures open even if there's an error
        try:
            plt.close()
        except:
            pass

# COMPETITIVE: Import exact same classes from training (wav2seg_v4_super_buffer.py)
try:
    from wav2seg_v4_super_buffer import (
        CompetitiveMultiScaleBoundaryClassifier,
        CompetitiveWindowPreprocessor,
        create_default_prosodic_features,
        clean_prosodic_features,
        load_timit_data_for_local_windows  # For test evaluation
    )
    print("✅ Successfully imported from wav2seg_v4_super_buffer.py")
except ImportError as e:
    print(f"⚠️ Could not import from wav2seg_v4_super_buffer.py: {e}")
    print("🔄 Trying fallback import from wav2seg_v4.py...")
    try:
        from wav2seg_v4 import (
            CompetitiveMultiScaleBoundaryClassifier,
            CompetitiveWindowPreprocessor,
            create_default_prosodic_features,
            clean_prosodic_features,
            load_timit_data_for_local_windows
        )
        print("✅ Successfully imported from wav2seg_v4.py (fallback)")
    except ImportError as e2:
        print(f"❌ Could not import from either module: {e2}")
        print("💡 Make sure wav2seg_v4_super_buffer.py or wav2seg_v4.py is in the same directory")
        raise ImportError("Could not import required classes for competitive inference")

warnings.filterwarnings('ignore')

def predict_boundaries_competitive(model, audio, processor, device, 
                                  config=None, verbose=True, debug_mode=False, 
                                  true_boundaries=None):
    """🏆 COMPETITIVE boundary prediction using EXACT SAME preprocessing as training.
    
    COMPETITIVE FEATURES:
    - Uses CompetitiveMultiScaleBoundaryClassifier
    - ALL DELTA-BASED prosodic features (28 temporal sequences)
    - Context buffer positioning (60ms from end)
    - ±20ms competition-standard tolerance
    - Multi-scale temporal analysis
    
    Args:
        model: Trained CompetitiveMultiScaleBoundaryClassifier
        audio: Audio array
        processor: Wav2Vec2 processor
        device: Device for inference
        config: Configuration dict (uses COMPETITIVE_CONFIG if None)
        verbose: Print detailed progress
        debug_mode: Extended debugging
        true_boundaries: True boundaries for debug analysis (optional)
    
    Returns:
        tuple: (raw_boundaries, raw_confidences, grouped_boundaries, grouped_confidences, 
               all_window_positions, all_prediction_values)
    """
    # Use default competitive config if not provided
    if config is None:
        config = COMPETITIVE_CONFIG.copy()
    
    if verbose:
        print(f"🏆 COMPETITIVE INFERENCE starting...")
        print(f"   Audio length: {len(audio)} samples ({len(audio)/config['sample_rate']:.2f}s)")
        print(f"   Window: {config['window_duration']}s, Stride: {config['stride']}s")
        print(f"   Threshold: {config['threshold']}, Context buffer: {config['context_buffer']*1000:.0f}ms")
        print(f"   Boundary tolerance: ±{config['boundary_tolerance']*1000:.0f}ms (competition standard)")
        print(f"   Prosodic features: {'ENABLED' if config['use_prosodic'] else 'DISABLED'}")
    
    model.eval()
    
    # Extract parameters from config
    window_duration = config['window_duration']
    stride = config['stride']
    sample_rate = config['sample_rate']
    threshold = config['threshold']
    grouping_distance = config['grouping_distance']
    context_buffer = config['context_buffer']
    
    window_samples = int(window_duration * sample_rate)
    stride_samples = int(stride * sample_rate)
    grouping_samples = int(grouping_distance * sample_rate)
    boundary_offset_samples = int(context_buffer * sample_rate)  # COMPETITIVE: 60ms context buffer
    
    # COMPETITIVE: Use EXACT SAME preprocessor from training with ALL competitive parameters
    preprocessor = CompetitiveWindowPreprocessor(
        [], processor,
        window_duration=config['window_duration'],
        sample_rate=config['sample_rate'],
        boundary_tolerance=config['boundary_tolerance'],
        negative_exclusion_zone=config['negative_exclusion_zone'],
        use_prosodic=config['use_prosodic'],
        verbose=False
    )
    
    window_predictions = []
    all_window_positions = []
    all_prediction_values = []
    
    with torch.no_grad():
        # Modified to handle 60ms offset and ensure proper future context
        for window_end in range(window_samples, len(audio) + boundary_offset_samples, stride_samples):
            # Calculate actual audio window position
            window_start = window_end - window_samples
            
            # Get the audio window
            if window_start >= len(audio):
                # We're past the end of the audio
                break
                
            window_audio = np.zeros(window_samples, dtype=np.float32)
            
            if window_start < 0:
                # Handle start padding
                padding_needed = abs(window_start)
                window_audio[padding_needed:] = audio[:window_end-padding_needed]
            else:
                # Get actual audio segment
                actual_end = min(window_end, len(audio))
                actual_window = audio[window_start:actual_end]
                window_audio[:len(actual_window)] = actual_window
            
            # Process exactly like training
            inputs = processor(window_audio, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            
            # CRITICAL FIX: Use training preprocessor for prosodic features
            prosodic_features = None
            if hasattr(model, 'use_prosodic') and model.use_prosodic:
                try:
                    prosodic_dict = preprocessor.extract_prosodic_features(window_audio)
                    seq_len = input_values.shape[1] // 320
                    if seq_len <= 0:
                        seq_len = 25
                    prosodic_tensor = preprocessor._prosodic_dict_to_tensor(prosodic_dict, seq_len)
                    prosodic_features = prosodic_tensor.unsqueeze(0).to(device)
                    
                    if prosodic_features.shape[1] != 28:
                        prosodic_features = torch.zeros(1, 28, seq_len, device=device)
                        
                except Exception as e:
                    seq_len = input_values.shape[1] // 320
                    if seq_len <= 0:
                        seq_len = 25
                    prosodic_features = torch.zeros(1, 28, seq_len, device=device)
            
            # Get model prediction
            logits = model(input_values, prosodic_features)
            prediction = torch.sigmoid(logits).item()
            
            # Calculate boundary position (60ms before the end of window)
            boundary_pos = window_end - boundary_offset_samples
            
            # FIXED: Always collect positions and predictions for visualization
            all_window_positions.append(boundary_pos)
            all_prediction_values.append(prediction)
            
            # Only store predictions for boundaries that would fall within the audio
            if prediction > threshold and boundary_pos < len(audio):
                window_predictions.append({
                    'position': boundary_pos,
                    'confidence': prediction
                })
    
    if verbose:
        print(f"   Raw predictions: {len(window_predictions)}")
    
    # Debug mode: analyze predictions at true boundary positions
    if debug_mode and true_boundaries is not None:
        print("\n🔍 DEBUG: Analyzing predictions at true boundary positions...")
        
        for true_pos in true_boundaries:
            # Find the closest window positions
            distances = np.abs(np.array(all_window_positions) - true_pos)
            closest_indices = np.argsort(distances)[:5]  # Get 5 closest windows
            
            print(f"\n   True boundary at {true_pos} samples ({true_pos/sample_rate:.3f}s):")
            for idx in closest_indices:
                window_pos = all_window_positions[idx]
                pred_value = all_prediction_values[idx]
                time_diff = (window_pos - true_pos) / sample_rate
                print(f"      Window at {window_pos} (Δt={time_diff:+.3f}s): pred={pred_value:.3f}" + 
                      f" {'✓' if pred_value > threshold else '✗'}")
    
    # Return raw predictions with positions and confidences
    raw_boundaries = [pred['position'] for pred in window_predictions]
    raw_confidences = [pred['confidence'] for pred in window_predictions]
    
    # Group predictions for final output
    if not window_predictions:
        return raw_boundaries, raw_confidences, raw_boundaries, raw_confidences, all_window_positions, all_prediction_values
    
    window_predictions.sort(key=lambda x: x['position'])
    
    grouped_boundaries = []
    grouped_confidences = []
    current_group = [window_predictions[0]]
    
    for pred in window_predictions[1:]:
        if pred['position'] - current_group[-1]['position'] <= grouping_samples:
            current_group.append(pred)
        else:
            # Take the highest confidence prediction from the group
            best_pred = max(current_group, key=lambda x: x['confidence'])
            grouped_boundaries.append(best_pred['position'])
            grouped_confidences.append(best_pred['confidence'])
            current_group = [pred]
    
    if current_group:
        best_pred = max(current_group, key=lambda x: x['confidence'])
        grouped_boundaries.append(best_pred['position'])
        grouped_confidences.append(best_pred['confidence'])
    
    if verbose:
        print(f"   Final boundaries: {len(grouped_boundaries)}")
    
    return raw_boundaries, raw_confidences, grouped_boundaries, grouped_confidences, all_window_positions, all_prediction_values

def load_model(model_path=None, device='auto', config=None):
    """Load the trained CompetitiveMultiScaleBoundaryClassifier from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (uses config default if None)
        device: Device to load on ('auto', 'cuda', 'cpu')
        config: Configuration dict (uses COMPETITIVE_CONFIG if None)
        
    Returns:
        tuple: (model, device)
    """
    # Use default competitive config if not provided
    if config is None:
        config = COMPETITIVE_CONFIG.copy()
    
    if model_path is None:
        model_path = config['model_path']
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🤖 Loading COMPETITIVE model from {model_path}")
    print(f"🔧 Using device: {device}")
    print(f"🏆 Architecture: CompetitiveMultiScaleBoundaryClassifier")
    
    # Initialize model with EXACT SAME architecture as competitive training
    model = CompetitiveMultiScaleBoundaryClassifier(
        wav2vec2_model_name=config['wav2vec2_model_name'],
        freeze_wav2vec2=config['freeze_wav2vec2'],
        hidden_dim=config['hidden_dim'],  # COMPETITIVE: Double capacity (512)
        dropout_rate=config['dropout_rate'],  # COMPETITIVE: 0.2 dropout
        use_prosodic=config['use_prosodic']   # COMPETITIVE: ALL DELTA-BASED prosodic features
    )
    
    try:
        # First move model to device
        model = model.to(device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        # Make sure wav2vec2 is on right device
        model.wav2vec2 = model.wav2vec2.to(device)
        
        # Set to eval mode
        model.eval()
        print(f"✅ Model loaded successfully")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def load_audio(audio_path, target_sr=16000):
    """Load and resample audio file."""
    print(f"🎵 Loading audio from {audio_path}")
    
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"   Original: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)")
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            print(f"   Resampled: {len(audio)} samples at {target_sr} Hz ({len(audio)/target_sr:.2f}s)")
        
        return audio
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        raise e

def load_timit_test_data(num_samples=15):
    """Load random TIMIT test data for competitive evaluation."""
    try:
        print(f"📂 Loading TIMIT test data...")
        test_data = load_timit_data_for_local_windows('test', None)
        
        if len(test_data) == 0:
            print("❌ No TIMIT test data found!")
            return []
        
        # Set random seed based on current time to ensure different samples each run
        current_seed = int(time.time() * 1000) % (2**32 - 1)  # Convert to valid seed
        random.seed(current_seed)
        np.random.seed(current_seed)
        print(f"🎲 Using random seed: {current_seed}")
        
        # Randomly select samples
        selected_samples = random.sample(test_data, min(num_samples, len(test_data)))
        print(f"✅ Selected {len(selected_samples)} random TIMIT test samples for competitive evaluation")
        
        # Reset random seed to avoid affecting other parts of the code
        random.seed(None)
        np.random.seed(None)
        
        return selected_samples
        
    except Exception as e:
        print(f"❌ Error loading TIMIT data: {e}")
        print(f"💡 Make sure wav2seg_v4_super_buffer.py is in the same directory")
        return []

def calculate_boundary_metrics(true_boundaries, pred_boundaries, pred_confidences, tolerance_samples, verbose=False):
    """Calculate boundary detection metrics using standard ±20ms tolerance window.
    
    Args:
        true_boundaries: List of true boundary positions in samples
        pred_boundaries: List of predicted boundary positions in samples
        pred_confidences: List of confidence values for predictions
        tolerance_samples: Tolerance window size in samples (typically 320 for ±20ms at 16kHz)
        verbose: Whether to print detailed analysis
    """
    if len(true_boundaries) == 0 and len(pred_boundaries) == 0:
        return 0.0, 1.0, 1.0, 1.0, 0
    
    if len(true_boundaries) == 0:
        return float('inf'), 0.0, 1.0, 0.0, len(pred_boundaries)
    
    if len(pred_boundaries) == 0:
        return float('inf'), 1.0, 0.0, 0.0, 0
    
    true_boundaries = np.array(sorted(true_boundaries))
    pred_boundaries = np.array(sorted(pred_boundaries))
    
    # Simple loop through true boundaries to count successful detections
    true_positives = 0
    for true_bound in true_boundaries:
        # For each true boundary, check if ANY prediction is within ±20ms
        distances = np.abs(pred_boundaries - true_bound)
        if np.any(distances <= tolerance_samples):
            true_positives += 1
    
    # Now find false positives (predictions not within ±20ms of any true boundary)
    false_positives = []
    for i, (pred, conf) in enumerate(zip(pred_boundaries, pred_confidences)):
        # Check if this prediction is near ANY true boundary
        distances = np.abs(true_boundaries - pred)
        min_distance = np.min(distances)
        
        if min_distance > tolerance_samples:
            nearest_true_idx = np.argmin(distances)
            false_positives.append({
                'position': pred,
                'time': pred / 16000,  # Assuming 16kHz
                'confidence': conf,
                'distance_samples': min_distance,
                'distance_ms': (min_distance / 16000) * 1000,  # Convert to ms
                'nearest_true_boundary': true_boundaries[nearest_true_idx],
                'nearest_true_time': true_boundaries[nearest_true_idx] / 16000
            })
    
    # Calculate final metrics
    num_false_positives = len(false_positives)
    precision = true_positives / (true_positives + num_false_positives) if (true_positives + num_false_positives) > 0 else 0.0
    recall = true_positives / len(true_boundaries)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if verbose:
        print("\n   🔍 False Positive Analysis:")
        print(f"   Found {num_false_positives} false positives (predictions with no true boundary within ±20ms)")
        
        if num_false_positives > 0:
            print("\n   Showing distance to nearest true boundary for each false positive:")
            for fp in sorted(false_positives, key=lambda x: x['distance_ms']):
                print(f"      At {fp['time']:.3f}s: conf={fp['confidence']:.3f}, {fp['distance_ms']:.1f}ms to nearest true boundary at {fp['nearest_true_time']:.3f}s")
    
    return 0.0, precision, recall, f1, num_false_positives

def run_test_evaluation(num_samples=15, debug=False, config=None):
    """Run competitive evaluation on random TIMIT test samples.
    
    Args:
        num_samples: Number of random TIMIT samples to evaluate
        debug: Enable debug mode for detailed analysis
        config: Configuration dict (uses COMPETITIVE_CONFIG if None)
    """
    print(f"\n🏆 RUNNING COMPETITIVE TEST EVALUATION ON {num_samples} RANDOM TIMIT SAMPLES")
    print("="*80)
    if debug:
        print("🔍 DEBUG MODE ENABLED - Will show detailed predictions at true boundary positions")
    
    # Use competitive configuration
    if config is None:
        config = COMPETITIVE_CONFIG.copy()
    
    print(f"📋 COMPETITIVE Configuration:")
    print(f"   Window duration: {config['window_duration']}s")
    print(f"   Context buffer: {config['context_buffer']*1000:.0f}ms")
    print(f"   Boundary tolerance: ±{config['boundary_tolerance']*1000:.0f}ms (competition standard)")
    print(f"   Decision threshold: {config['threshold']}")
    print(f"   Prosodic features: {'ENABLED' if config['use_prosodic'] else 'DISABLED'}")
    
    # Load competitive model
    print("📥 Loading trained competitive model...")
    model, device = load_model(config=config)
    
    # Load processor
    print("📥 Loading Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained(config['wav2vec2_model_name'])
    
    # Load test data
    test_data = load_timit_test_data(num_samples)
    if len(test_data) == 0:
        return
    
    # Create output directory for visualizations
    viz_dir = "prediction_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Run evaluation
    results = []
    total_start_time = time.time()
    
    tolerance_samples = int(config['boundary_tolerance'] * config['sample_rate'])
    
    # Create output directory for visualizations
    viz_dir = "prediction_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    for i, item in enumerate(test_data):
        print(f"\n📊 Processing sample {i+1}/{len(test_data)}: {item.get('id', f'sample_{i}')}")
        
        try:
            # Extract audio and resample if necessary
            audio = item['audio']['array']
            original_sr = item['audio']['sampling_rate']
            
            if original_sr != config['sample_rate']:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=config['sample_rate'])
            
            # Get true boundaries
            true_boundaries = []
            if 'phonetic_detail' in item and 'start' in item['phonetic_detail']:
                starts = item['phonetic_detail']['start']
                stops = item['phonetic_detail']['stop']
                
                for start, stop in zip(starts, stops):
                    start_sample = int(start * config['sample_rate'])
                    stop_sample = int(stop * config['sample_rate'])
                    true_boundaries.extend([start_sample, stop_sample])
            
            true_boundaries = sorted(list(set(true_boundaries)))
            true_boundaries = [pos for pos in true_boundaries if 0 <= pos < len(audio)]
            
            # Predict boundaries
            start_time = time.time()
            raw_boundaries, raw_confidences, grouped_boundaries, grouped_confidences, all_frames, all_predictions = predict_boundaries_competitive(
                model, audio, processor, device,
                config=config,
                verbose=False,
                debug_mode=debug,
                true_boundaries=true_boundaries
            )
            inference_time = time.time() - start_time
            
            # Create and save visualization
            viz_path = os.path.join(viz_dir, f"{item.get('id', f'sample_{i}')}_predictions.png")
            visualize_predictions(
                all_frames, 
                all_predictions, 
                true_boundaries,
                config['sample_rate'],
                int(config['stride'] * config['sample_rate']),
                viz_path,
                threshold=config['threshold']
            )
            
            # Calculate metrics with detailed analysis
            raw_mae, raw_precision, raw_recall, raw_f1, raw_fps = calculate_boundary_metrics(
                true_boundaries, raw_boundaries, raw_confidences, tolerance_samples, verbose=True
            )
            
            grouped_mae, grouped_precision, grouped_recall, grouped_f1, grouped_fps = calculate_boundary_metrics(
                true_boundaries, grouped_boundaries, grouped_confidences, tolerance_samples, verbose=True
            )
            
            result = {
                'file_id': item.get('id', f'sample_{i}'),
                'audio_length_s': len(audio) / config['sample_rate'],
                'num_true_boundaries': len(true_boundaries),
                'num_raw_predictions': len(raw_boundaries),
                'num_grouped_predictions': len(grouped_boundaries),
                'raw_false_positives': raw_fps,
                'grouped_false_positives': grouped_fps,
                'raw_precision': raw_precision,
                'raw_recall': raw_recall,
                'raw_f1': raw_f1,
                'grouped_precision': grouped_precision,
                'grouped_recall': grouped_recall,
                'grouped_f1': grouped_f1,
                'inference_time': inference_time
            }
            
            results.append(result)
            
            # Calculate true positives from recall
            raw_true_positives = int(raw_recall * len(true_boundaries))
            grouped_true_positives = int(grouped_recall * len(true_boundaries))
            
            # Print sample results
            print(f"\n   📊 Summary:")
            print(f"   True boundaries: {len(true_boundaries)}")
            print(f"   Raw predictions: {len(raw_boundaries)} (True positives: {raw_true_positives}, False positives: {raw_fps})")
            print(f"   Raw metrics - F1: {raw_f1:.3f}, Precision: {raw_precision:.3f}, Recall: {raw_recall:.3f}")
            print(f"   Grouped predictions: {len(grouped_boundaries)} (True positives: {grouped_true_positives}, False positives: {grouped_fps})")
            print(f"   Grouped metrics - F1: {grouped_f1:.3f}, Precision: {grouped_precision:.3f}, Recall: {grouped_recall:.3f}")
            print(f"   Time: {inference_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error processing sample: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Calculate overall metrics
    if results:
        print(f"\n📈 OVERALL TEST RESULTS ({len(results)} samples)")
        print("="*70)
        print(f"🎯 Raw Performance Metrics:")
        print(f"   Mean F1: {np.mean([r['raw_f1'] for r in results]):.3f} ± {np.std([r['raw_f1'] for r in results]):.3f}")
        print(f"   Mean Precision: {np.mean([r['raw_precision'] for r in results]):.3f}")
        print(f"   Mean Recall: {np.mean([r['raw_recall'] for r in results]):.3f}")
        
        print(f"\n🎯 Grouped Performance Metrics:")
        print(f"   Mean F1: {np.mean([r['grouped_f1'] for r in results]):.3f} ± {np.std([r['grouped_f1'] for r in results]):.3f}")
        print(f"   Mean Precision: {np.mean([r['grouped_precision'] for r in results]):.3f}")
        print(f"   Mean Recall: {np.mean([r['grouped_recall'] for r in results]):.3f}")
        
        print(f"\n📊 Boundary Statistics:")
        print(f"   Avg true boundaries per sample: {np.mean([r['num_true_boundaries'] for r in results]):.1f}")
        print(f"   Avg raw predictions per sample: {np.mean([r['num_raw_predictions'] for r in results]):.1f}")
        print(f"   Avg grouped predictions per sample: {np.mean([r['num_grouped_predictions'] for r in results]):.1f}")
        print(f"   Total raw false positives: {sum([r['raw_false_positives'] for r in results])}")
        print(f"   Total grouped false positives: {sum([r['grouped_false_positives'] for r in results])}")
        
        raw_pred_ratio = np.mean([r['num_raw_predictions'] for r in results]) / np.mean([r['num_true_boundaries'] for r in results])
        grouped_pred_ratio = np.mean([r['num_grouped_predictions'] for r in results]) / np.mean([r['num_true_boundaries'] for r in results])
        print(f"   Raw prediction ratio (pred/true): {raw_pred_ratio:.2f}")
        print(f"   Grouped prediction ratio (pred/true): {grouped_pred_ratio:.2f}")
        
        print(f"\n⏱️ Performance:")
        print(f"   Total evaluation time: {total_time:.1f}s")
        print(f"   Average inference time per sample: {np.mean([r['inference_time'] for r in results]):.2f}s")
        
        # Show best and worst cases
        print(f"\n🏆 BEST 3 SAMPLES (by raw F1):")
        best_samples = sorted(results, key=lambda x: x['raw_f1'], reverse=True)[:3]
        for i, sample in enumerate(best_samples, 1):
            print(f"   {i}. {sample['file_id']}: F1={sample['raw_f1']:.3f}, "
                  f"True={sample['num_true_boundaries']}, Raw Pred={sample['num_raw_predictions']}")
        
        print(f"\n📉 WORST 3 SAMPLES (by raw F1):")
        worst_samples = sorted(results, key=lambda x: x['raw_f1'])[:3]
        for i, sample in enumerate(worst_samples, 1):
            print(f"   {i}. {sample['file_id']}: F1={sample['raw_f1']:.3f}, "
                  f"True={sample['num_true_boundaries']}, Raw Pred={sample['num_raw_predictions']}")
        
        print("="*70)
    else:
        print("❌ No valid results obtained!")

def main():
    """Main competitive inference function with argument parsing."""
    parser = argparse.ArgumentParser(description='🏆 COMPETITIVE FULL AUDIO BOUNDARY DETECTION - V4')
    parser.add_argument('audio_file', nargs='?', help='Path to audio file for inference')
    parser.add_argument('--test', action='store_true', help='Run evaluation on random TIMIT samples')
    parser.add_argument('--num-samples', type=int, default=15, help='Number of test samples (default: 15)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show prediction values at true boundary positions')
    parser.add_argument('--threshold', type=float, help='Override decision threshold (default: from config)')
    parser.add_argument('--context-buffer', type=float, help='Override context buffer in seconds (default: 0.060)')
    parser.add_argument('--model-path', type=str, help='Override model path (default: best_local_model.pth)')
    
    args = parser.parse_args()
    
    print("🏆 COMPETITIVE FULL AUDIO BOUNDARY DETECTION - V4")
    print("="*70)
    print("🎯 Compatible with CompetitiveMultiScaleBoundaryClassifier")
    print("🎵 Includes ALL DELTA-BASED prosodic features (28 channels)")
    print("📏 Competition-standard ±20ms tolerance")
    
    # Use competitive configuration as base
    config = COMPETITIVE_CONFIG.copy()
    
    # Apply command-line overrides
    if args.threshold is not None:
        config['threshold'] = args.threshold
        print(f"🔧 Override threshold: {config['threshold']}")
    
    if args.context_buffer is not None:
        config['context_buffer'] = args.context_buffer
        print(f"🔧 Override context buffer: {config['context_buffer']*1000:.0f}ms")
    
    if args.model_path is not None:
        config['model_path'] = args.model_path
        print(f"🔧 Override model path: {config['model_path']}")
    
    # Handle test mode
    if args.test:
        run_test_evaluation(args.num_samples, debug=args.debug, config=config)
        return
    
    # Regular inference mode
    if not args.audio_file:
        print("❌ Error: Audio file required for inference mode")
        print("\nUsage:")
        print("  Single file: python inference_full_audio.py <audio_file>")
        print("  Test mode:   python inference_full_audio.py --test")
        print("  Advanced:    python inference_full_audio.py --test --num-samples 20 --threshold 0.5")
        print("  Custom:      python inference_full_audio.py <file> --threshold 0.3 --context-buffer 0.080")
        return
    
    audio_path = args.audio_file
    
    # Initialize processor
    print("🔄 Initializing Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained(config['wav2vec2_model_name'])
    print("✅ Processor ready")
    
    # Load competitive model
    model, device = load_model(config=config)
    
    # Load audio
    audio = load_audio(audio_path, config['sample_rate'])
    
    # Predict boundaries
    print(f"\n🔍 Predicting boundaries...")
    
    start_time = time.time()
    
    raw_boundaries, raw_confidences, grouped_boundaries, grouped_confidences, all_frames, all_predictions = predict_boundaries_competitive(
        model, audio, processor, device,
        config=config,
        verbose=True
    )
    
    inference_time = time.time() - start_time
    
    # Results
    print(f"\n🎉 INFERENCE COMPLETE!")
    print(f"⏱️ Inference time: {inference_time:.2f}s")
    print(f"📊 Found {len(raw_boundaries)} boundaries")
    
    # Create visualization for single-file inference
    viz_dir = "prediction_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    viz_path = os.path.join(viz_dir, f"{audio_filename}_predictions.png")
    
    print(f"\n📊 Creating visualization...")
    visualize_predictions(
        all_frames, 
        all_predictions, 
        [],  # No true boundaries for single-file inference
        config['sample_rate'],
        int(config['stride'] * config['sample_rate']),
        viz_path,
        threshold=config['threshold']
    )
    
    if len(raw_boundaries) > 0:
        print(f"\n📍 Boundary positions (samples):")
        for i, pos in enumerate(raw_boundaries):
            time_pos = pos / config['sample_rate']
            print(f"   {i+1:2d}. Sample {pos:6d} ({time_pos:.3f}s)")
        
        print(f"\n📍 Boundary times (seconds):")
        boundary_times = [pos / config['sample_rate'] for pos in raw_boundaries]
        print(f"   {boundary_times}")
    else:
        print("⚠️ No boundaries detected!")
    
    return raw_boundaries, grouped_boundaries

if __name__ == "__main__":
    result = main()
    if result is not None:
        raw_boundaries, raw_confidences, grouped_boundaries, grouped_confidences, all_frames, all_predictions = result

# =============================================================================
# COMPETITIVE INFERENCE DOCUMENTATION
# =============================================================================
"""
🏆 COMPETITIVE INFERENCE FEATURES - V4 COMPATIBILITY

ARCHITECTURE COMPATIBILITY:
✅ CompetitiveMultiScaleBoundaryClassifier (wav2seg_v4_super_buffer.py)
✅ Multi-scale temporal convolutions (3x3 + 9x9 kernels)
✅ ALL DELTA-BASED prosodic features (28 temporal sequences)
✅ Context buffer positioning (60ms from end)
✅ Competition-standard ±20ms tolerance

PROSODIC FEATURES (ALL DELTA-BASED):
- Basic deltas (11): energy_delta, energy_delta2, centroid_delta, rolloff_delta, zcr_delta,
                     bandwidth_delta, flatness_delta, flux_delta, chroma_delta, tonnetz_delta, tempo_delta
- MFCC deltas (12): 6 mfcc_delta + 6 mfcc_delta2 sequences
- Future expansion (5): positions 23-27 reserved for additional delta features

COMPETITIVE PARAMETERS:
- Window duration: 0.5s (focused boundary detection)
- Context buffer: 60ms from end (boundary positioning)
- Boundary tolerance: ±20ms (competition standard)
- Decision threshold: 0.45 (configurable)
- Stride: 10ms (±20ms precision)
- Grouping distance: 20ms (competition standard)

USAGE EXAMPLES:
1. Single file inference:
   python inference_full_audio.py audio.wav

2. Test evaluation (15 random TIMIT samples):
   python inference_full_audio.py --test

3. Advanced test with custom parameters:
   python inference_full_audio.py --test --num-samples 25 --threshold 0.5 --debug

4. Custom inference with parameter overrides:
   python inference_full_audio.py audio.wav --threshold 0.3 --context-buffer 0.080

5. Debug mode (shows predictions at true boundary positions):
   python inference_full_audio.py --test --debug --num-samples 5

CONFIGURATION OVERRIDE OPTIONS:
--threshold FLOAT       Decision threshold (default: 0.45)
--context-buffer FLOAT  Context buffer in seconds (default: 0.060)
--model-path STRING     Model checkpoint path (default: best_local_model.pth)
--num-samples INT       Number of test samples (default: 15)
--debug                 Enable detailed debug output

OUTPUT:
- Raw boundaries: All predictions above threshold
- Grouped boundaries: Nearby predictions grouped within 20ms
- Visualizations: Saved to prediction_visualizations/ directory
- Metrics: Precision, recall, F1 score with ±20ms tolerance

PERFORMANCE TARGETS:
🎯 Competition-level F1: 80-90% @ ±20ms tolerance
🎵 Prosodic advantage: Delta-based features for transition detection
🔥 Multi-scale analysis: Local (3x3) + contextual (9x9) patterns
📏 Precision: 10ms stride for sub-frame accuracy
"""