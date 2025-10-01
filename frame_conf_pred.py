"""
FRAME-LEVEL CONFIDENCE PREDICTION STORAGE
=======================================

Processes entire TIMIT dataset and stores frame-level predictions
for training a post-processing model.

Features:
- Stores predictions for every frame (10ms stride)
- Maintains alignment with original audio
- Includes ground truth boundary information
- Saves results in easily loadable format
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
import pickle
from tqdm import tqdm

# Import from existing inference code
from inference_full_audio import (
    COMPETITIVE_CONFIG,
        CompetitiveMultiScaleBoundaryClassifier,
        CompetitiveWindowPreprocessor,
    load_model,
            load_timit_data_for_local_windows
        )

def process_dataset_to_frame_predictions(split='train', output_dir='frame_predictions', config=None):
    """
    Process entire TIMIT dataset and store frame-level predictions.
    
    Args:
        split: 'train' or 'test'
        output_dir: Directory to save results
        config: Configuration dict
        
    Returns:
        dict: Mapping of utterance IDs to their frame predictions and metadata
    """
    if config is None:
        config = COMPETITIVE_CONFIG.copy()
    
    print(f"🎯 Processing {split} split for frame-level predictions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and processor
    model, device = load_model(config=config)
    processor = Wav2Vec2Processor.from_pretrained(config['wav2vec2_model_name'])
    
    # Load all data
    print(f"📂 Loading TIMIT {split} data...")
    all_data = load_timit_data_for_local_windows(split, None)
    print(f"✅ Loaded {len(all_data)} utterances")
    
    # Storage for results
    dataset_predictions = {}
    
    # Process each utterance
    for item in tqdm(all_data, desc=f"Processing {split} utterances"):
        try:
            # Get audio and true boundaries
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
            
            # Calculate number of frames
            window_samples = int(config['window_duration'] * config['sample_rate'])
            stride_samples = int(config['stride'] * config['sample_rate'])
            num_frames = (len(audio) - window_samples) // stride_samples + 1
            
            # Initialize frame predictions
            frame_predictions = np.zeros(num_frames)
            frame_positions = np.zeros(num_frames, dtype=np.int32)
            
            # Process windows and store predictions
            model.eval()
            with torch.no_grad():
                for frame_idx in range(num_frames):
                    window_start = frame_idx * stride_samples
                    window_end = window_start + window_samples
                    
                    # Get audio window
                    if window_end > len(audio):
                        window_audio = np.zeros(window_samples, dtype=np.float32)
                        window_audio[:len(audio)-window_start] = audio[window_start:]
                    else:
                        window_audio = audio[window_start:window_end]
                    
                    # Process window
                    inputs = processor(window_audio, sampling_rate=config['sample_rate'], return_tensors="pt")
                    input_values = inputs.input_values.to(device)
                    
                    # Get prosodic features
                    prosodic_features = None
                    if hasattr(model, 'use_prosodic') and model.use_prosodic:
                        try:
                            preprocessor = CompetitiveWindowPreprocessor(
                                [], processor,
                                window_duration=config['window_duration'],
                                sample_rate=config['sample_rate'],
                                boundary_tolerance=config['boundary_tolerance'],
                                negative_exclusion_zone=config['negative_exclusion_zone'],
                                use_prosodic=config['use_prosodic'],
                                verbose=False
                            )
                            
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
                    
                    # Get prediction
                    logits = model(input_values, prosodic_features)
                    prediction = torch.sigmoid(logits).item()
                    
                    # Store prediction and position
                    frame_predictions[frame_idx] = prediction
                    frame_positions[frame_idx] = window_start + int(config['context_buffer'] * config['sample_rate'])
            
            # Store results for this utterance
            dataset_predictions[item['id']] = {
                'frame_predictions': frame_predictions,
                'frame_positions': frame_positions,
                'true_boundaries': true_boundaries,
                'audio_length': len(audio),
                'num_frames': num_frames,
                'metadata': {
                    'speaker': item.get('speaker', ''),
                    'utterance': item.get('utterance', ''),
                    'phonetic_detail': item.get('phonetic_detail', {})
                }
            }
            
            # Save periodically
            if len(dataset_predictions) % 100 == 0:
                save_path = os.path.join(output_dir, f'{split}_predictions_{len(dataset_predictions)}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(dataset_predictions, f)
                print(f"\nSaved checkpoint to {save_path}")
            
        except Exception as e:
            print(f"\nError processing {item['id']}: {e}")
            continue
    
    # Save final results
    final_save_path = os.path.join(output_dir, f'{split}_predictions_final.pkl')
    with open(final_save_path, 'wb') as f:
        pickle.dump(dataset_predictions, f)
    print(f"\n✅ Saved final results to {final_save_path}")
    
    return dataset_predictions

def main():
    """Process both train and test splits of TIMIT."""
    print("🎯 FRAME-LEVEL CONFIDENCE PREDICTION STORAGE")
    print("="*50)
    
    # Process train split
    train_predictions = process_dataset_to_frame_predictions('train')
    print(f"\n✅ Processed {len(train_predictions)} training utterances")
    
    # Process test split
    test_predictions = process_dataset_to_frame_predictions('test')
    print(f"\n✅ Processed {len(test_predictions)} test utterances")
    
    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    main()