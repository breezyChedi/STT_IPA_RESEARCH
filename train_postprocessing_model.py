"""
POST-PROCESSING MODEL TRAINING
=============================

Trains a neural network to refine boundary predictions by analyzing
patterns in 8 adjacent frame predictions and outputting binary decisions
for the middle 6 frames.

The model focuses on learning boundary patterns from confidence values
using a simple architecture that matches the straightforward nature of the task.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp
import gc

# Set sharing strategy before any other torch operations
mp.set_sharing_strategy('file_system')

def analyze_dataset_distribution(dataset):
    """Analyze the actual distribution of data in the dataset."""
    print("\n" + "="*50)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("="*50)
    
    total_windows = len(dataset)
    boundary_windows = 0
    total_boundary_frames = 0
    total_frames = 0
    
    # Sample a subset for analysis to avoid memory issues
    sample_size = min(10000, total_windows)
    indices = np.random.choice(total_windows, sample_size, replace=False)
    
    window_boundary_counts = []
    
    for idx in indices:
        _, labels, _ = dataset[idx]
        labels_np = labels.numpy()
        
        has_boundary = np.any(labels_np == 1)
        if has_boundary:
            boundary_windows += 1
        
        boundary_frames_in_window = np.sum(labels_np == 1)
        total_boundary_frames += boundary_frames_in_window
        total_frames += len(labels_np)
        window_boundary_counts.append(boundary_frames_in_window)
    
    # Scale up to full dataset
    scale_factor = total_windows / sample_size
    estimated_boundary_windows = int(boundary_windows * scale_factor)
    estimated_boundary_frames = int(total_boundary_frames * scale_factor)
    estimated_total_frames = int(total_frames * scale_factor)
    
    print(f"Total windows analyzed: {sample_size} (sample) / {total_windows} (full)")
    print(f"Boundary windows: {boundary_windows} ({boundary_windows/sample_size*100:.2f}%)")
    print(f"Non-boundary windows: {sample_size-boundary_windows} ({(sample_size-boundary_windows)/sample_size*100:.2f}%)")
    print(f"Estimated full dataset:")
    print(f"  - Boundary windows: {estimated_boundary_windows} ({estimated_boundary_windows/total_windows*100:.2f}%)")
    print(f"  - Non-boundary windows: {total_windows-estimated_boundary_windows}")
    print(f"Frame-level statistics:")
    print(f"  - Boundary frames: {total_boundary_frames} ({total_boundary_frames/total_frames*100:.4f}%)")
    print(f"  - Non-boundary frames: {total_frames-total_boundary_frames}")
    print(f"Boundary frames per window distribution:")
    print(f"  - 0 boundaries: {np.sum(np.array(window_boundary_counts) == 0)} windows")
    print(f"  - 1 boundary: {np.sum(np.array(window_boundary_counts) == 1)} windows")
    print(f"  - 2+ boundaries: {np.sum(np.array(window_boundary_counts) >= 2)} windows")
    print(f"  - Max boundaries in one window: {max(window_boundary_counts)}")
    
    return {
        'boundary_window_ratio': boundary_windows / sample_size,
        'boundary_frame_ratio': total_boundary_frames / total_frames,
        'avg_boundaries_per_window': np.mean(window_boundary_counts)
    }

def calculate_window_weights(dataset):
    """Calculate weights for windows - now simplified since we use oversampling."""
    # With oversampling, we don't need aggressive weighting
    return torch.FloatTensor([1.0, 1.0])  # Equal weights

class BoundaryFocalLoss(nn.Module):
    """
    Focal loss specifically designed for extreme class imbalance in boundary detection.
    
    Combines:
    1. Focal loss to focus on hard examples
    2. Strong positive weighting to handle class imbalance  
    3. Confidence push to avoid mode collapse
    4. Window-level statistics tracking
    """
    def __init__(self, window_weights, pos_weight=0.1, gamma=2.0, confidence_push=0.5):
        super().__init__()
        self.window_weights = window_weights
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.confidence_push = confidence_push
        self.reset_stats()
        
    def reset_stats(self):
        """Reset accumulated statistics at start of epoch"""
        self.epoch_stats = {
            'boundary_logits': [],
            'boundary_probs': [],
            'non_boundary_logits': [],
            'non_boundary_probs': [],
            'pos_frame_probs': [],
            'total_loss': 0,
            'focal_loss': 0,
            'confidence_loss': 0,
            'n_batches': 0,
            'gradient_norms': []
        }
    
    def log_epoch_stats(self):
        """Print accumulated statistics for the epoch"""
        if self.epoch_stats['n_batches'] == 0:
            return
            
        print("\nEPOCH STATISTICS:")
        print("=================")
        
        # Boundary windows
        if self.epoch_stats['boundary_logits']:
            b_logits = np.concatenate(self.epoch_stats['boundary_logits'])
            b_probs = np.concatenate(self.epoch_stats['boundary_probs'])
            print(f"Boundary windows - Logits: {b_logits.mean():.4f}±{b_logits.std():.4f}, Probs: {b_probs.mean():.4f}±{b_probs.std():.4f}")
        
        # Non-boundary windows
        if self.epoch_stats['non_boundary_logits']:
            nb_logits = np.concatenate(self.epoch_stats['non_boundary_logits'])
            nb_probs = np.concatenate(self.epoch_stats['non_boundary_probs'])
            print(f"Non-boundary windows - Logits: {nb_logits.mean():.4f}±{nb_logits.std():.4f}, Probs: {nb_probs.mean():.4f}±{nb_probs.std():.4f}")
        
        # Positive frames
        if self.epoch_stats['pos_frame_probs']:
            pos_probs = np.concatenate(self.epoch_stats['pos_frame_probs'])
            print(f"Positive frames - Probs: {pos_probs.mean():.4f}±{pos_probs.std():.4f}")
        
        # Gradient analysis
        if self.epoch_stats['gradient_norms']:
            grad_norms = np.array(self.epoch_stats['gradient_norms'])
            print(f"Gradient norms - Mean: {grad_norms.mean():.6f}, Std: {grad_norms.std():.6f}, Max: {grad_norms.max():.6f}")
        
        # Loss components
        avg_loss = self.epoch_stats['total_loss'] / self.epoch_stats['n_batches']
        avg_focal = self.epoch_stats['focal_loss'] / self.epoch_stats['n_batches']
        avg_confidence = self.epoch_stats['confidence_loss'] / self.epoch_stats['n_batches']
        print(f"Average loss: {avg_loss:.4f} (focal: {avg_focal:.4f}, confidence: {avg_confidence:.4f})")
        
        # Check if model is learning
        if self.epoch_stats['boundary_logits'] and self.epoch_stats['non_boundary_logits']:
            b_logits = np.concatenate(self.epoch_stats['boundary_logits'])
            nb_logits = np.concatenate(self.epoch_stats['non_boundary_logits'])
            separation = abs(b_logits.mean() - nb_logits.mean())
            print(f"Logit separation: {separation:.4f} {'✓ LEARNING' if separation > 0.1 else '✗ NOT LEARNING'}")
        
        self.reset_stats()
    
    def add_gradient_norm(self, grad_norm):
        """Add gradient norm to statistics"""
        self.epoch_stats['gradient_norms'].append(grad_norm)
    
    def forward(self, pred, target):
        # Get probabilities
        probs = torch.sigmoid(pred)
        
        # Focal loss component - focuses on hard examples
        # pt is the probability of the correct class
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE with strong positive weighting
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor(self.pos_weight, device=pred.device),
            reduction='none'
        )
        
        # Apply focal weighting to BCE
        focal_loss = focal_weight * bce_loss
        
        # Confidence push: penalize boundary predictions near 0.5
        confidence_loss = torch.tensor(0.0, device=pred.device)
        boundary_mask = (target == 1)
        if boundary_mask.any():
            # Push boundary probabilities away from 0.5 towards 1.0
            boundary_probs = probs[boundary_mask]
            # Quadratic penalty for being close to 0.5
            distance_from_confident = torch.abs(boundary_probs - 0.5)
            confidence_loss = torch.mean((0.5 - distance_from_confident) ** 2)
        
        # Window-level weighting
        has_boundary = (target.sum(dim=1) > 0).float()
        window_weights = has_boundary * self.window_weights[1] + (1 - has_boundary) * self.window_weights[0]
        
        # Combine losses
        window_focal_losses = focal_loss.mean(dim=1)
        weighted_focal_losses = window_weights * window_focal_losses
        
        total_loss = weighted_focal_losses.mean() + self.confidence_push * confidence_loss
        
        # Accumulate statistics
        with torch.no_grad():
            boundary_mask_window = has_boundary == 1
            if boundary_mask_window.any():
                self.epoch_stats['boundary_logits'].append(pred[boundary_mask_window].cpu().numpy())
                self.epoch_stats['boundary_probs'].append(probs[boundary_mask_window].cpu().numpy())
            
            non_boundary_mask = ~boundary_mask_window
            if non_boundary_mask.any():
                self.epoch_stats['non_boundary_logits'].append(pred[non_boundary_mask].cpu().numpy())
                self.epoch_stats['non_boundary_probs'].append(probs[non_boundary_mask].cpu().numpy())
            
            if boundary_mask.any():
                self.epoch_stats['pos_frame_probs'].append(probs[boundary_mask].cpu().numpy())
            
            self.epoch_stats['total_loss'] += total_loss.item()
            self.epoch_stats['focal_loss'] += weighted_focal_losses.mean().item()
            self.epoch_stats['confidence_loss'] += confidence_loss.item()
            self.epoch_stats['n_batches'] += 1
        
        return total_loss

class BoundaryRefinementModel(nn.Module):
    """
    Model that takes 8 adjacent frame predictions and outputs binary decisions for middle 6 frames.
    Uses deep layers to learn patterns in the sequence of predictions.
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Deep network to learn sequential patterns while preserving probability semantics
        self.pattern_learner = nn.Sequential(
            # First layer: Learn basic frame-to-frame relationships
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            
            # Second layer: Learn short sequences
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer: Learn longer sequences
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Fourth layer: Learn full sequence patterns
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Final layer to predict 6 frames
        self.classifier = nn.Linear(hidden_dim, 6)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to preserve probability scales"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Careful initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize final classifier to preserve probability scale
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, -2.0)  # Start with low boundary predictions
        
    def forward(self, x):
        # x: [batch, 8] - sequence of 8 frame predictions
        
        # Learn sequential patterns while preserving probability meaning
        features = self.pattern_learner(x)
        
        # Predict middle 6 frames using learned patterns
        predictions = self.classifier(features)
        
        return predictions

class FramePredictionDataset(Dataset):
    """Dataset for training the post-processing model with SMOTE-like oversampling."""
    
    def __init__(self, data_path, window_size=8, tolerance_samples=0, oversample_target_ratio=0.10):
        self.window_size = window_size
        self.tolerance_samples = tolerance_samples
        self.oversample_target_ratio = oversample_target_ratio
        
        # Load data and immediately convert to numpy arrays
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            # Convert to numpy arrays immediately to close file handles
            self.data = {k: {
                'frame_predictions': np.array(v['frame_predictions'], dtype=np.float32),
                'frame_positions': np.array(v['frame_positions'], dtype=np.int64),
                'true_boundaries': np.array(v['true_boundaries'], dtype=np.int64)
            } for k, v in data_dict.items()}
        
        # Initialize empty lists for windows and labels
        self.windows = []
        self.labels = []
        
        # Process utterances
        print("\nExtracting windows and creating labels...")
        for utterance_id, utterance_data in tqdm(self.data.items()):
            self._process_utterance(utterance_data)
            
        # Convert to numpy arrays
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # Apply oversampling
        self._apply_oversampling()

    def _process_utterance(self, utterance_data):
        # Data is already numpy arrays from initialization
        frame_predictions = utterance_data['frame_predictions']
        frame_positions = utterance_data['frame_positions']
        true_boundaries = utterance_data['true_boundaries']
        
        # Create sliding windows
        for i in range(len(frame_predictions) - self.window_size + 1):
            window = frame_predictions[i:i + self.window_size].copy()
            
            # Get positions of middle 6 frames
            middle_start = i + 1
            middle_end = i + 7
            middle_positions = frame_positions[middle_start:middle_end]
            
            # Create labels for middle 6 frames
            labels = np.zeros(6, dtype=np.float32)
            for j, pos in enumerate(middle_positions):
                if pos in true_boundaries:
                    labels[j] = 1
            
            self.windows.append(window)
            self.labels.append(labels)

    def _apply_oversampling(self):
        """Apply oversampling to balance boundary vs non-boundary windows."""
        
        # Separate boundary and non-boundary windows
        has_boundary = np.any(self.labels == 1, axis=1)
        boundary_indices = np.where(has_boundary)[0]
        non_boundary_indices = np.where(~has_boundary)[0]
        
        n_boundary = len(boundary_indices)
        n_non_boundary = len(non_boundary_indices)
        
        print(f"\nOversampling Analysis:")
        print(f"Boundary windows: {n_boundary}")
        print(f"Non-boundary windows: {n_non_boundary}")
        print(f"Current ratio: {n_non_boundary/(n_boundary + n_non_boundary)*100:.1f}% non-boundary")
        
        # Calculate target numbers
        n_non_boundary_target = n_non_boundary
        n_boundary_target = int(n_non_boundary_target * (1 - self.oversample_target_ratio) / self.oversample_target_ratio)
        
        print(f"Target boundary windows: {n_boundary_target}")
        print(f"Target non-boundary windows: {n_non_boundary_target}")
        print(f"Target ratio: {n_non_boundary_target/(n_boundary_target + n_non_boundary_target)*100:.1f}% non-boundary")
        
        # Pre-allocate arrays for efficiency
        total_size = n_boundary_target + n_non_boundary_target
        final_windows = np.zeros((total_size, self.window_size), dtype=np.float32)
        final_labels = np.zeros((total_size, 6), dtype=np.float32)
        
        # Add non-boundary windows first
        final_windows[:n_non_boundary] = self.windows[non_boundary_indices]
        final_labels[:n_non_boundary] = self.labels[non_boundary_indices]
        
        # Add oversampled boundary windows - just copy them, no noise
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_boundary_target):
            idx = n_non_boundary + i
            source_idx = boundary_indices[i % n_boundary]
            
            # Simple copy without noise
            final_windows[idx] = self.windows[source_idx]
            final_labels[idx] = self.labels[source_idx]
            
            # Free memory periodically
            if i % 1000000 == 0:
                gc.collect()
        
        # Replace old arrays with new ones
        self.windows = final_windows
        self.labels = final_labels
        
        print(f"Oversampling complete: {len(self.windows)} total windows")

    def __del__(self):
        """Cleanup method to ensure memory is freed"""
        self.windows = None
        self.labels = None
        self.data = None
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.windows[idx]), 
                torch.FloatTensor(self.labels[idx]),
                torch.FloatTensor([1.0]))

def analyze_predictions(model, val_loader, device, epoch):
    """Analyze what patterns lead to predictions."""
    model.eval()
    
    # Track statistics
    all_logits = []
    all_probs = []
    all_labels = []
    boundary_window_logits = []
    non_boundary_window_logits = []
    
    n_false_positives = 0
    n_false_negatives = 0
    n_correct_boundaries = 0
    n_total_boundaries = 0
    
    print(f"\nEpoch {epoch} DETAILED ANALYSIS:")
    print("="*40)
    
    with torch.no_grad():
        for windows, labels, _ in val_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model(windows)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            
            # Store for analysis
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Separate boundary and non-boundary windows
            has_boundary = (labels.sum(dim=1) > 0)
            if has_boundary.any():
                boundary_window_logits.append(logits[has_boundary].cpu().numpy())
            if (~has_boundary).any():
                non_boundary_window_logits.append(logits[~has_boundary].cpu().numpy())
            
            # Convert to numpy for analysis
            labels_np = labels.cpu().numpy()
            
            # Update statistics
            n_false_positives += np.sum((preds == 1) & (labels_np == 0))
            n_false_negatives += np.sum((preds == 0) & (labels_np == 1))
            n_correct_boundaries += np.sum((preds == 1) & (labels_np == 1))
            n_total_boundaries += np.sum(labels_np == 1)
    
    # Combine all arrays
    all_logits = np.concatenate(all_logits)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    if boundary_window_logits:
        boundary_window_logits = np.concatenate(boundary_window_logits)
    if non_boundary_window_logits:
        non_boundary_window_logits = np.concatenate(non_boundary_window_logits)
    
    # Detailed probability analysis
    print("\nPROBABILITY DISTRIBUTION ANALYSIS:")
    print(f"Overall prob range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(f"Overall prob mean±std: {all_probs.mean():.4f}±{all_probs.std():.4f}")
    
    # Check for mode collapse
    prob_std = all_probs.std()
    if prob_std < 0.01:
        print("⚠️  WARNING: Probability std < 0.01 - possible mode collapse!")
    
    # Analyze prediction thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        preds_at_thresh = (all_probs > threshold).sum()
        print(f"Predictions > {threshold}: {preds_at_thresh} ({preds_at_thresh/len(all_probs)*100:.2f}%)")
    
    # Boundary detection performance
    print("\nBOUNDARY DETECTION PERFORMANCE:")
    print(f"True Positives: {n_correct_boundaries}")
    print(f"False Positives: {n_false_positives}")
    print(f"False Negatives: {n_false_negatives}")
    print(f"Total true boundaries: {n_total_boundaries}")
    
    if n_total_boundaries > 0:
        recall = n_correct_boundaries / n_total_boundaries
        print(f"Recall: {recall*100:.2f}%")
    
    if n_correct_boundaries + n_false_positives > 0:
        precision = n_correct_boundaries / (n_correct_boundaries + n_false_positives)
        print(f"Precision: {precision*100:.2f}%")
    
    # Compare boundary vs non-boundary windows
    if len(boundary_window_logits) > 0 and len(non_boundary_window_logits) > 0:
        print("\nWINDOW TYPE DISCRIMINATION:")
        b_mean, b_std = boundary_window_logits.mean(), boundary_window_logits.std()
        nb_mean, nb_std = non_boundary_window_logits.mean(), non_boundary_window_logits.std()
        
        print(f"Boundary windows - Logits: {b_mean:.4f}±{b_std:.4f}")
        print(f"Non-boundary windows - Logits: {nb_mean:.4f}±{nb_std:.4f}")
        
        separation = abs(b_mean - nb_mean)
        print(f"Logit separation: {separation:.4f}")
        
        if separation < 0.05:
            print("⚠️  WARNING: Very poor discrimination between window types!")
        elif separation > 0.5:
            print("✓ Good discrimination between window types")
    
    # Overall accuracy
    preds = (all_probs > 0.5).astype(np.int32)
    correct_mask = (preds == all_labels)
    print(f"\nOverall Accuracy: {np.mean(correct_mask)*100:.2f}%")
    
    return {
        'prob_mean': all_probs.mean(),
        'prob_std': all_probs.std(),
        'logit_separation': separation if 'separation' in locals() else 0,
        'recall': recall if 'recall' in locals() else 0,
        'precision': precision if 'precision' in locals() else 0
    }

def debug_model_and_data(model, train_loader, device):
    """Debug what's going wrong with this training."""
    print("\n" + "="*60)
    print("EMERGENCY DEBUGGING SESSION")
    print("="*60)
    
    model.eval()
    
    # Check first few batches for input diversity
    batch_count = 0
    all_windows = []
    all_labels = []
    
    for windows, labels, _ in train_loader:
        windows = windows.to(device)
        labels = labels.to(device)
        
        all_windows.append(windows.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        batch_count += 1
        if batch_count >= 5:  # Just check first 5 batches
            break
    
    # Analyze input diversity
    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Input Analysis (first {all_windows.shape[0]} windows):")
    print(f"Window shape: {all_windows.shape}")
    print(f"Window value range: [{all_windows.min():.6f}, {all_windows.max():.6f}]")
    print(f"Window mean±std: {all_windows.mean():.6f}±{all_windows.std():.6f}")
    
    # Check if all windows are identical
    first_window = all_windows[0]
    identical_count = 0
    different_windows = []
    for i in range(min(1000, len(all_windows))):
        if np.allclose(all_windows[i], first_window, atol=1e-6):
            identical_count += 1
        else:
            different_windows.append(i)
    
    print(f"Windows identical to first: {identical_count}/1000")
    if identical_count > 500:
        print("🚨 CRITICAL: >50% of windows are identical!")
    else:
        print(f"✓ Good input diversity - {len(different_windows)} different windows")
        if len(different_windows) > 0:
            # Show some examples of different windows
            for i in different_windows[:3]:
                diff = np.abs(all_windows[i] - first_window).max()
                print(f"  Window {i} max diff from first: {diff:.6f}")
    
    # Check label diversity
    print(f"\nLabel Analysis:")
    print(f"Label shape: {all_labels.shape}")
    print(f"Positive labels: {np.sum(all_labels == 1)} ({np.mean(all_labels)*100:.2f}%)")
    unique_patterns = set()
    for i in range(min(1000, len(all_labels))):
        pattern = tuple(all_labels[i])
        unique_patterns.add(pattern)
    print(f"Unique label patterns: {len(unique_patterns)}")
    print(f"First 5 label patterns: {list(list(unique_patterns)[:5])}")
    
    # Test model forward pass step by step
    print(f"\nModel Forward Pass Analysis:")
    model.train()
    
    test_windows = torch.FloatTensor(all_windows[:32]).to(device)  # Small batch
    test_labels = torch.FloatTensor(all_labels[:32]).to(device)
    
    # Test feature extractor step by step
    with torch.no_grad():
        # Input to feature extractor
        x_input = test_windows.unsqueeze(1)
        print(f"Input to conv: {x_input.shape}, range: [{x_input.min():.6f}, {x_input.max():.6f}]")
        print(f"Input mean±std: {x_input.mean():.6f}±{x_input.std():.6f}")
        
        # First conv layer
        conv1_weight = model.feature_extractor[0].weight
        conv1_bias = model.feature_extractor[0].bias
        print(f"Conv1 weights: mean±std: {conv1_weight.mean():.6f}±{conv1_weight.std():.6f}")
        
        # Run first conv
        conv1_out = torch.conv1d(x_input, conv1_weight, conv1_bias, padding=1)
        print(f"Conv1 output: {conv1_out.shape}, range: [{conv1_out.min():.6f}, {conv1_out.max():.6f}]")
        print(f"Conv1 output mean±std: {conv1_out.mean():.6f}±{conv1_out.std():.6f}")
        
        # After ReLU
        relu1_out = torch.relu(conv1_out)
        print(f"ReLU1 output: range: [{relu1_out.min():.6f}, {relu1_out.max():.6f}]")
        print(f"ReLU1 output mean±std: {relu1_out.mean():.6f}±{relu1_out.std():.6f}")
        
        # After BatchNorm1
        bn1_out = model.feature_extractor[2](relu1_out)
        print(f"BatchNorm1 output: range: [{bn1_out.min():.6f}, {bn1_out.max():.6f}]")
        print(f"BatchNorm1 output mean±std: {bn1_out.mean():.6f}±{bn1_out.std():.6f}")
        
        # Full feature extractor
        features = model.feature_extractor(x_input)
        print(f"Final features: {features.shape}, range: [{features.min():.6f}, {features.max():.6f}]")
        print(f"Final features mean±std: {features.mean():.6f}±{features.std():.6f}")
        
        # Check if features are identical across batch
        if features.std() < 1e-6:
            print("🚨 CRITICAL: Feature extractor outputs are identical!")
        else:
            # Check variance across different dimensions
            batch_var = features.var(dim=0).mean()
            feature_var = features.var(dim=1).mean()
            spatial_var = features.var(dim=2).mean()
            print(f"✓ Features have variance - Batch: {batch_var:.6f}, Feature: {feature_var:.6f}, Spatial: {spatial_var:.6f}")
        
        # Test frame classifier for multiple frames
        print(f"\nFrame Classifier Analysis:")
        frame_predictions = []
        for frame_idx in range(1, 7):  # Middle 6 frames
            frame_features = features[:, :, frame_idx]
            rel_pos = torch.tensor([(frame_idx-3.5)/3.5, frame_idx/7.0], device=device).expand(32, 2)
            frame_input = torch.cat([frame_features, rel_pos], dim=1)
            
            print(f"Frame {frame_idx} input: {frame_input.shape}, range: [{frame_input.min():.6f}, {frame_input.max():.6f}]")
            print(f"Frame {frame_idx} input mean±std: {frame_input.mean():.6f}±{frame_input.std():.6f}")
            
            # Final prediction for this frame
            pred = model.classifier(frame_input)
            frame_predictions.append(pred)
            print(f"Frame {frame_idx} pred: range: [{pred.min():.6f}, {pred.max():.6f}], mean±std: {pred.mean():.6f}±{pred.std():.6f}")
        
        # Full model prediction
        full_pred = model(test_windows)
        print(f"\nFull model prediction: {full_pred.shape}")
        print(f"Full pred range: [{full_pred.min():.6f}, {full_pred.max():.6f}]")
        print(f"Full pred mean±std: {full_pred.mean():.6f}±{full_pred.std():.6f}")
        
        if full_pred.std() < 1e-6:
            print("🚨 CRITICAL: All predictions are identical!")
        else:
            print("✓ Model outputs have variance")
    
    # Check model weights before training
    print(f"\nModel Weight Analysis:")
    total_params = 0
    zero_weights = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        zero_count = (param.abs() < 1e-8).sum().item()
        total_params += param_count
        zero_weights += zero_count
        
        print(f"{name}: {param.shape}, mean±std: {param.mean():.6f}±{param.std():.6f}, zeros: {zero_count}/{param_count}")
        if param.std() < 1e-6:
            print(f"🚨 {name} has no variance!")
    
    print(f"Total parameters: {total_params}, zero weights: {zero_weights} ({zero_weights/total_params*100:.1f}%)")
    
    # Test gradient computation
    print(f"\nGradient Computation Test:")
    model.train()
    test_windows_grad = test_windows.requires_grad_(True)
    test_logits = model(test_windows_grad)
    
    # Simple loss
    test_loss = torch.nn.functional.binary_cross_entropy_with_logits(test_logits, test_labels)
    test_loss.backward()
    
    # Check gradients
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats.append(grad_norm)
            print(f"{name} grad norm: {grad_norm:.6f}")
        else:
            print(f"{name} has NO gradient!")
    
    if grad_stats:
        print(f"Gradient norms: mean={np.mean(grad_stats):.6f}, std={np.std(grad_stats):.6f}")
        if np.mean(grad_stats) < 1e-6:
            print("🚨 CRITICAL: Gradients are vanishing!")
        else:
            print("✓ Gradients are flowing")
    
    print("="*60)
    return all_windows, all_labels

def train_model(train_loader, val_loader, model, device, window_weights, num_epochs=50):
    """Train the model with enhanced monitoring and analysis."""
    
    # Use the new focal loss with strong positive weighting and confidence push
    criterion = BoundaryFocalLoss(window_weights, pos_weight=25.0, gamma=2.0, confidence_push=0.5)
    
    # Use AdamW with proper weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # More aggressive scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3, min_lr=1e-5)
    
    # Initialize history tracking
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_metrics': [],
        'analysis_data': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 8
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        
        # Training
        model.train()
        train_loss = 0.0
        
        # Training statistics
        epoch_predictions = []
        epoch_labels = []
        
        for windows, labels, _ in tqdm(train_loader, desc='Training', leave=False):
            windows = windows.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(windows)
            
            # Simple BCE loss
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for analysis
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                epoch_predictions.append(probs.cpu().numpy())
                epoch_labels.append(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_losses'].append(avg_train_loss)
        
        # Training epoch statistics
        epoch_predictions = np.concatenate(epoch_predictions)
        epoch_labels = np.concatenate(epoch_labels)
        train_preds = (epoch_predictions > 0.5).astype(int)
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            epoch_labels.flatten(), train_preds.flatten(), 
            average='binary', zero_division=0
        )
        
        print(f"\nTrain Stats: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, "
              f"Precision={train_precision:.4f}, Recall={train_recall:.4f}")
        print(f"Train Prob Range: {epoch_predictions.min():.3f} to {epoch_predictions.max():.3f}")
        print(f"Train Prob Mean±Std: {epoch_predictions.mean():.3f}±{epoch_predictions.std():.3f}")
        
        # Log training epoch statistics from loss function
        criterion.log_epoch_stats()
        
        # Validation with detailed analysis
        model.eval()
        val_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for windows, labels, _ in tqdm(val_loader, desc='Validation', leave=False):
                windows = windows.to(device)
                labels = labels.to(device)
                logits = model(windows)
                
                # Calculate validation loss
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Store predictions and labels for metrics
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_losses'].append(avg_val_loss)
        
        # Combine validation results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Validation statistics
        print(f"\nVal Stats: Loss={avg_val_loss:.4f}")
        print(f"Val Prob Range: {all_probs.min():.3f} to {all_probs.max():.3f}")
        print(f"Val Prob Mean±Std: {all_probs.mean():.3f}±{all_probs.std():.3f}")
        
        # Check for mode collapse
        if all_probs.std() < 0.01:
            print("⚠️  WARNING: Probability std < 0.01 - possible mode collapse!")
        else:
            print("✓ Good probability variance")
        
        # Analyze boundary vs non-boundary discrimination
        boundary_windows = np.any(all_labels == 1, axis=1)
        if boundary_windows.any() and (~boundary_windows).any():
            boundary_probs = all_probs[boundary_windows].mean()
            non_boundary_probs = all_probs[~boundary_windows].mean()
            separation = abs(boundary_probs - non_boundary_probs)
            print(f"Discrimination: B-prob={boundary_probs:.3f}, NB-prob={non_boundary_probs:.3f}, Sep={separation:.3f}")
            
            if separation > 0.05:
                print("✓ Good discrimination between window types")
            else:
                print("⚠️  Poor discrimination between window types")
        
        # Detailed prediction analysis
        analysis_results = analyze_predictions(model, val_loader, device, epoch)
        history['analysis_data'].append(analysis_results)
        
        # Calculate validation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.flatten(), all_preds.flatten(), 
            average='binary', zero_division=0
        )
        
        print(f"\nFINAL METRICS - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Store metrics
        history['val_metrics'].append([precision, recall, f1])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("✓ New best model saved")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return history

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves with metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics
    metrics = np.array(history['val_metrics'])
    ax2.plot(metrics[:, 0], label='Precision', color='blue')
    ax2.plot(metrics[:, 1], label='Recall', color='green')
    ax2.plot(metrics[:, 2], label='F1', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Post-Processing Model')
    parser.add_argument('--train-data', type=str, default='frame_predictions/train_predictions_final.pkl',
                        help='Path to training data pickle file')
    parser.add_argument('--test-data', type=str, default='frame_predictions/test_predictions_final.pkl',
                        help='Path to test data pickle file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,  # Increased default
                        help='Hidden dimension size')
    parser.add_argument('--output-dir', default='postprocessing_model',
                        help='Output directory for saved models and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FramePredictionDataset(args.train_data)
    test_dataset = FramePredictionDataset(args.test_data)
    
    # Analyze dataset distribution
    print("\nAnalyzing training dataset...")
    train_stats = analyze_dataset_distribution(train_dataset)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False  # Disable pinned memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False  # Disable pinned memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False  # Disable pinned memory
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Calculate window weights
    window_weights = calculate_window_weights(train_dataset.dataset)  # Get original dataset from random split
    
    # Create model with larger capacity
    model = BoundaryRefinementModel(hidden_dim=args.hidden_dim).to(device)
    print(f"\nModel architecture (hidden_dim={args.hidden_dim}):")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training with enhanced monitoring...")
    history = train_model(train_loader, val_loader, model, device, window_weights,
                         num_epochs=args.epochs)
    
    # Plot training curves
    plot_training_curves(history, 
                        save_path=os.path.join(args.output_dir, 'training_curves.png'))
    
    # Final evaluation
    model.eval()
    test_preds = []
    test_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for windows, labels, _ in tqdm(test_loader):
            windows = windows.to(device)
            preds = torch.sigmoid(model(windows)) > 0.5
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.numpy())
    
    # Calculate final metrics
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels.flatten(), test_preds.flatten(),
        average='binary', zero_division=0
    )
    
    print("\nTest Set Results:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    results = {
        'test_metrics': {
            'f1': f1,
            'precision': precision,
            'recall': recall
        },
        'training_history': history
    }
    
    results_path = os.path.join(args.output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 