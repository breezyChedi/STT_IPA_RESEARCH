"""
FRAME PREDICTIONS DATA INVESTIGATION
===================================

This script investigates the frame predictions data to understand:
1. What the input windows actually look like
2. How the model processes each frame
3. Why the shared classifier architecture is problematic
4. How position encoding interacts with features
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class BoundaryRefinementModel(nn.Module):
    """Copy of the model from training script for investigation"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # cuDNN-friendly feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # SHARED classifier for all frames - this is the issue
        self.frame_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim * 2),  # +2 for position encoding
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1, bias=True)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.frame_classifier[-1].bias.data.fill_(-0.5)
        
    def forward_detailed(self, x):
        """Forward pass with detailed intermediate outputs for investigation"""
        batch_size = x.shape[0]
        
        print(f"\n=== DETAILED FORWARD PASS ===")
        print(f"Input shape: {x.shape}")
        print(f"Input window example: {x[0].cpu().numpy()}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Input mean±std: {x.mean():.4f}±{x.std():.4f}")
        
        # Extract features (B, 1, 8) -> (B, hidden_dim, 8)
        x = x.unsqueeze(1)  # Add channel dim
        print(f"After unsqueeze: {x.shape}")
        
        features = self.feature_extractor(x)
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Features mean±std: {features.mean():.4f}±{features.std():.4f}")
        
        # Check if features are collapsing
        feature_variance = features.var(dim=0).mean()
        print(f"Feature variance (across batch): {feature_variance:.6f}")
        if feature_variance < 1e-4:
            print("🚨 FEATURE COLLAPSE: Very low variance across batch!")
        
        # Predict for each middle frame
        predictions = []
        frame_inputs_detailed = []
        
        for i in range(1, 7):  # Middle 6 frames
            print(f"\n--- Processing Frame {i} (output index {i-1}) ---")
            
            # Get features for this frame
            frame_features = features[:, :, i]  # (B, hidden_dim)
            print(f"Frame {i} features shape: {frame_features.shape}")
            print(f"Frame {i} features range: [{frame_features.min():.4f}, {frame_features.max():.4f}]")
            print(f"Frame {i} features mean±std: {frame_features.mean():.4f}±{frame_features.std():.4f}")
            
            # Add relative position encoding - THIS IS THE KEY PART
            rel_pos = torch.tensor([(i-3.5)/3.5, i/7.0], 
                                 device=x.device).expand(batch_size, 2)
            print(f"Frame {i} position encoding: {rel_pos[0].cpu().numpy()}")
            
            frame_input = torch.cat([frame_features, rel_pos], dim=1)
            frame_inputs_detailed.append(frame_input.cpu().numpy())
            print(f"Frame {i} classifier input shape: {frame_input.shape}")
            print(f"Frame {i} classifier input range: [{frame_input.min():.4f}, {frame_input.max():.4f}]")
            
            # THE CRITICAL ISSUE: Same classifier processes all frames
            pred = self.frame_classifier(frame_input)
            predictions.append(pred)
            print(f"Frame {i} prediction shape: {pred.shape}")
            print(f"Frame {i} prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"Frame {i} prediction mean±std: {pred.mean():.4f}±{pred.std():.4f}")
        
        # Check if all frame inputs are too similar
        print(f"\n=== FRAME INPUT COMPARISON ===")
        frame_inputs_detailed = np.array(frame_inputs_detailed)  # Shape: (6, batch_size, hidden_dim+2)
        
        for batch_idx in range(min(3, batch_size)):  # Check first 3 examples
            print(f"\nExample {batch_idx}:")
            for frame_idx in range(6):
                frame_input = frame_inputs_detailed[frame_idx, batch_idx]
                features_part = frame_input[:-2]  # All but last 2 (position)
                pos_part = frame_input[-2:]  # Last 2 (position)
                print(f"  Frame {frame_idx+1}: features_mean={features_part.mean():.4f}, "
                      f"features_std={features_part.std():.4f}, pos={pos_part}")
            
            # Calculate similarity between frames for this example
            frame_similarities = []
            for i in range(6):
                for j in range(i+1, 6):
                    features_i = frame_inputs_detailed[i, batch_idx, :-2]
                    features_j = frame_inputs_detailed[j, batch_idx, :-2]
                    similarity = np.corrcoef(features_i, features_j)[0, 1]
                    frame_similarities.append(similarity)
            
            avg_similarity = np.mean(frame_similarities)
            print(f"  Average feature correlation between frames: {avg_similarity:.4f}")
            if avg_similarity > 0.99:
                print(f"  🚨 FEATURES ARE NEARLY IDENTICAL! Position encoding won't help.")
        
        final_predictions = torch.cat(predictions, dim=1)
        print(f"\nFinal predictions shape: {final_predictions.shape}")
        print(f"Final predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
        
        return final_predictions
    
    def forward(self, x):
        """Regular forward pass"""
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        
        predictions = []
        for i in range(1, 7):
            frame_features = features[:, :, i]
            rel_pos = torch.tensor([(i-3.5)/3.5, i/7.0], 
                                 device=x.device).expand(batch_size, 2)
            frame_input = torch.cat([frame_features, rel_pos], dim=1)
            pred = self.frame_classifier(frame_input)
            predictions.append(pred)
            
        return torch.cat(predictions, dim=1)

class FramePredictionDataset(Dataset):
    """Simplified dataset loader for investigation"""
    
    def __init__(self, data_path, max_samples=1000):
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Convert to numpy arrays and limit samples for investigation
        self.windows = []
        self.labels = []
        
        sample_count = 0
        for utterance_id, utterance_data in data_dict.items():
            frame_predictions = np.array(utterance_data['frame_predictions'], dtype=np.float32)
            frame_positions = np.array(utterance_data['frame_positions'], dtype=np.int64)
            true_boundaries = np.array(utterance_data['true_boundaries'], dtype=np.int64)
            
            # Create sliding windows
            for i in range(len(frame_predictions) - 8 + 1):
                if sample_count >= max_samples:
                    break
                    
                window = frame_predictions[i:i + 8].copy()
                
                # Get positions of middle 6 frames
                middle_positions = frame_positions[i + 1:i + 7]
                
                # Create labels for middle 6 frames
                labels = np.zeros(6, dtype=np.float32)
                for j, pos in enumerate(middle_positions):
                    if pos in true_boundaries:
                        labels[j] = 1
                
                self.windows.append(window)
                self.labels.append(labels)
                sample_count += 1
            
            if sample_count >= max_samples:
                break
        
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        print(f"Loaded {len(self.windows)} windows for investigation")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.windows[idx]), 
                torch.FloatTensor(self.labels[idx]))

def investigate_data_structure(data_path):
    """Investigate the structure of frame predictions data"""
    print("="*60)
    print("INVESTIGATING FRAME PREDICTIONS DATA STRUCTURE")
    print("="*60)
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"Number of utterances: {len(data_dict)}")
    
    # Analyze a few utterances
    utterance_ids = list(data_dict.keys())[:5]
    for utterance_id in utterance_ids:
        utterance_data = data_dict[utterance_id]
        
        print(f"\nUtterance: {utterance_id}")
        print(f"Frame predictions shape: {len(utterance_data['frame_predictions'])}")
        print(f"Frame positions shape: {len(utterance_data['frame_positions'])}")
        print(f"True boundaries: {len(utterance_data['true_boundaries'])}")
        
        # Show first few frame predictions
        frame_preds = utterance_data['frame_predictions'][:10]
        print(f"First 10 frame predictions: {frame_preds}")
        print(f"Prediction range: [{min(utterance_data['frame_predictions']):.4f}, {max(utterance_data['frame_predictions']):.4f}]")
        
        # Show boundary information
        print(f"Boundary positions: {utterance_data['true_boundaries'][:10] if len(utterance_data['true_boundaries']) > 0 else 'None'}")

def investigate_model_behavior():
    """Investigate how the model processes different inputs"""
    print("\n" + "="*60)
    print("INVESTIGATING MODEL BEHAVIOR")
    print("="*60)
    
    # Create model
    model = BoundaryRefinementModel(hidden_dim=64)  # Smaller for investigation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create some test inputs
    print("\n--- Testing with identical windows ---")
    identical_window = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]).float()
    identical_batch = identical_window.unsqueeze(0).repeat(4, 1).to(device)
    
    with torch.no_grad():
        pred_identical = model.forward_detailed(identical_batch)
    
    print(f"\nPredictions for identical inputs:")
    for i in range(4):
        print(f"Sample {i}: {pred_identical[i].cpu().numpy()}")
    
    print("\n--- Testing with different windows ---")
    different_windows = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Increasing
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Decreasing
        [0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8],  # Alternating
        [0.4, 0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.4],  # Peak in middle
    ]).float().to(device)
    
    with torch.no_grad():
        pred_different = model.forward_detailed(different_windows)
    
    print(f"\nPredictions for different inputs:")
    for i in range(4):
        print(f"Sample {i}: {pred_different[i].cpu().numpy()}")

def demonstrate_shared_classifier_issue():
    """Demonstrate why shared classifier is problematic"""
    print("\n" + "="*60)
    print("DEMONSTRATING SHARED CLASSIFIER ISSUE")
    print("="*60)
    
    # Create a minimal example
    hidden_dim = 8
    
    # Simulate what happens when features collapse
    print("\n--- Scenario 1: Features collapse (all frames have similar features) ---")
    
    # All frames have nearly identical features
    collapsed_features = torch.tensor([
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 1
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 2 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 3 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 4 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 5 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 6 (identical)
    ]).float()
    
    # Position encodings for frames 1-6
    position_encodings = torch.tensor([
        [(1-3.5)/3.5, 1/7.0],  # Frame 1: [-0.714, 0.143]
        [(2-3.5)/3.5, 2/7.0],  # Frame 2: [-0.429, 0.286]  
        [(3-3.5)/3.5, 3/7.0],  # Frame 3: [-0.143, 0.429]
        [(4-3.5)/3.5, 4/7.0],  # Frame 4: [ 0.143, 0.571]
        [(5-3.5)/3.5, 5/7.0],  # Frame 5: [ 0.429, 0.714]
        [(6-3.5)/3.5, 6/7.0],  # Frame 6: [ 0.714, 0.857]
    ]).float()
    
    print("Features (all identical):")
    for i in range(6):
        print(f"Frame {i+1}: {collapsed_features[i].numpy()}")
    
    print("\nPosition encodings (different):")
    for i in range(6):
        print(f"Frame {i+1}: {position_encodings[i].numpy()}")
    
    # Combine features + position
    frame_inputs = torch.cat([collapsed_features, position_encodings], dim=1)
    print(f"\nCombined inputs shape: {frame_inputs.shape}")
    
    # Create a simple classifier
    classifier = nn.Sequential(
        nn.Linear(10, 4),  # 8 features + 2 position
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    # Initialize with small weights
    for param in classifier.parameters():
        param.data.normal_(0, 0.01)
    
    with torch.no_grad():
        predictions = classifier(frame_inputs)
    
    print(f"\nPredictions when features are identical:")
    for i in range(6):
        print(f"Frame {i+1}: {predictions[i].item():.6f}")
    
    pred_diff = predictions.max() - predictions.min()
    print(f"Prediction range: {pred_diff.item():.6f}")
    
    if pred_diff < 0.01:
        print("🚨 PROBLEM: Even with position encoding, predictions are nearly identical!")
        print("   This is because position encoding is a tiny signal compared to collapsed features.")
    
    print("\n--- Scenario 2: Features are diverse ---")
    
    # Different features for each frame
    diverse_features = torch.tensor([
        [1.0, 0.1, -0.5, 0.2, -0.8, 0.9, 0.3, -0.1],  # Frame 1
        [0.2, 0.8, -0.1, 0.6, -0.3, 0.1, 0.7, -0.9],  # Frame 2 (different)
        [-0.5, 0.3, 0.8, -0.2, 0.6, -0.4, 0.1, 0.9],  # Frame 3 (different)
        [0.7, -0.6, 0.2, 0.9, -0.1, 0.4, -0.8, 0.3],  # Frame 4 (different)
        [-0.2, 0.5, -0.7, 0.1, 0.8, -0.9, 0.4, 0.6],  # Frame 5 (different)
        [0.9, -0.3, 0.4, -0.7, 0.2, 0.6, -0.5, 0.8],  # Frame 6 (different)
    ]).float()
    
    diverse_inputs = torch.cat([diverse_features, position_encodings], dim=1)
    
    with torch.no_grad():
        diverse_predictions = classifier(diverse_inputs)
    
    print(f"\nPredictions when features are diverse:")
    for i in range(6):
        print(f"Frame {i+1}: {diverse_predictions[i].item():.6f}")
    
    diverse_diff = diverse_predictions.max() - diverse_predictions.min()
    print(f"Prediction range: {diverse_diff.item():.6f}")
    
    if diverse_diff > 0.1:
        print("✓ GOOD: With diverse features, we get diverse predictions!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Investigate Frame Predictions Data')
    parser.add_argument('--data-path', type=str, 
                        default='frame_predictions/train_predictions_final.pkl',
                        help='Path to frame predictions data')
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        print("Available frame_predictions files:")
        if os.path.exists('frame_predictions'):
            for f in os.listdir('frame_predictions'):
                if f.endswith('.pkl'):
                    print(f"  - frame_predictions/{f}")
        return
    
    # Investigate data structure
    investigate_data_structure(args.data_path)
    
    # Load small dataset for model testing
    print("\n" + "="*60)
    print("LOADING DATASET FOR MODEL TESTING")
    print("="*60)
    
    dataset = FramePredictionDataset(args.data_path, max_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Show some data examples
    print("\nData examples:")
    for batch_idx, (windows, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Windows shape: {windows.shape}")
        print(f"Labels shape: {labels.shape}")
        
        for i in range(min(2, windows.shape[0])):
            print(f"  Window {i}: {windows[i].numpy()}")
            print(f"  Labels {i}: {labels[i].numpy()}")
        
        if batch_idx >= 2:
            break
    
    # Test model behavior
    investigate_model_behavior()
    
    # Demonstrate the shared classifier issue
    demonstrate_shared_classifier_issue()
    
    print("\n" + "="*60)
    print("SUMMARY: WHY SHARED CLASSIFIER IS PROBLEMATIC")
    print("="*60)
    print("""
The issue is NOT that the model uses position encoding - that's actually good!

The issue is that when the FEATURE EXTRACTOR collapses (outputs similar features
for all frames), the position encoding becomes the ONLY difference between frames.

But position encoding is a tiny 2-dimensional signal compared to the large
feature vector (128 dimensions). When features collapse, the classifier learns
to ignore the tiny position differences and just outputs the same value.

SOLUTION: Either fix feature collapse OR use separate classifiers per frame.
    """)

if __name__ == "__main__":
    main() 
FRAME PREDICTIONS DATA INVESTIGATION
===================================

This script investigates the frame predictions data to understand:
1. What the input windows actually look like
2. How the model processes each frame
3. Why the shared classifier architecture is problematic
4. How position encoding interacts with features
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class BoundaryRefinementModel(nn.Module):
    """Copy of the model from training script for investigation"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # cuDNN-friendly feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # SHARED classifier for all frames - this is the issue
        self.frame_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim * 2),  # +2 for position encoding
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1, bias=True)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.frame_classifier[-1].bias.data.fill_(-0.5)
        
    def forward_detailed(self, x):
        """Forward pass with detailed intermediate outputs for investigation"""
        batch_size = x.shape[0]
        
        print(f"\n=== DETAILED FORWARD PASS ===")
        print(f"Input shape: {x.shape}")
        print(f"Input window example: {x[0].cpu().numpy()}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Input mean±std: {x.mean():.4f}±{x.std():.4f}")
        
        # Extract features (B, 1, 8) -> (B, hidden_dim, 8)
        x = x.unsqueeze(1)  # Add channel dim
        print(f"After unsqueeze: {x.shape}")
        
        features = self.feature_extractor(x)
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Features mean±std: {features.mean():.4f}±{features.std():.4f}")
        
        # Check if features are collapsing
        feature_variance = features.var(dim=0).mean()
        print(f"Feature variance (across batch): {feature_variance:.6f}")
        if feature_variance < 1e-4:
            print("🚨 FEATURE COLLAPSE: Very low variance across batch!")
        
        # Predict for each middle frame
        predictions = []
        frame_inputs_detailed = []
        
        for i in range(1, 7):  # Middle 6 frames
            print(f"\n--- Processing Frame {i} (output index {i-1}) ---")
            
            # Get features for this frame
            frame_features = features[:, :, i]  # (B, hidden_dim)
            print(f"Frame {i} features shape: {frame_features.shape}")
            print(f"Frame {i} features range: [{frame_features.min():.4f}, {frame_features.max():.4f}]")
            print(f"Frame {i} features mean±std: {frame_features.mean():.4f}±{frame_features.std():.4f}")
            
            # Add relative position encoding - THIS IS THE KEY PART
            rel_pos = torch.tensor([(i-3.5)/3.5, i/7.0], 
                                 device=x.device).expand(batch_size, 2)
            print(f"Frame {i} position encoding: {rel_pos[0].cpu().numpy()}")
            
            frame_input = torch.cat([frame_features, rel_pos], dim=1)
            frame_inputs_detailed.append(frame_input.cpu().numpy())
            print(f"Frame {i} classifier input shape: {frame_input.shape}")
            print(f"Frame {i} classifier input range: [{frame_input.min():.4f}, {frame_input.max():.4f}]")
            
            # THE CRITICAL ISSUE: Same classifier processes all frames
            pred = self.frame_classifier(frame_input)
            predictions.append(pred)
            print(f"Frame {i} prediction shape: {pred.shape}")
            print(f"Frame {i} prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"Frame {i} prediction mean±std: {pred.mean():.4f}±{pred.std():.4f}")
        
        # Check if all frame inputs are too similar
        print(f"\n=== FRAME INPUT COMPARISON ===")
        frame_inputs_detailed = np.array(frame_inputs_detailed)  # Shape: (6, batch_size, hidden_dim+2)
        
        for batch_idx in range(min(3, batch_size)):  # Check first 3 examples
            print(f"\nExample {batch_idx}:")
            for frame_idx in range(6):
                frame_input = frame_inputs_detailed[frame_idx, batch_idx]
                features_part = frame_input[:-2]  # All but last 2 (position)
                pos_part = frame_input[-2:]  # Last 2 (position)
                print(f"  Frame {frame_idx+1}: features_mean={features_part.mean():.4f}, "
                      f"features_std={features_part.std():.4f}, pos={pos_part}")
            
            # Calculate similarity between frames for this example
            frame_similarities = []
            for i in range(6):
                for j in range(i+1, 6):
                    features_i = frame_inputs_detailed[i, batch_idx, :-2]
                    features_j = frame_inputs_detailed[j, batch_idx, :-2]
                    similarity = np.corrcoef(features_i, features_j)[0, 1]
                    frame_similarities.append(similarity)
            
            avg_similarity = np.mean(frame_similarities)
            print(f"  Average feature correlation between frames: {avg_similarity:.4f}")
            if avg_similarity > 0.99:
                print(f"  🚨 FEATURES ARE NEARLY IDENTICAL! Position encoding won't help.")
        
        final_predictions = torch.cat(predictions, dim=1)
        print(f"\nFinal predictions shape: {final_predictions.shape}")
        print(f"Final predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
        
        return final_predictions
    
    def forward(self, x):
        """Regular forward pass"""
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        
        predictions = []
        for i in range(1, 7):
            frame_features = features[:, :, i]
            rel_pos = torch.tensor([(i-3.5)/3.5, i/7.0], 
                                 device=x.device).expand(batch_size, 2)
            frame_input = torch.cat([frame_features, rel_pos], dim=1)
            pred = self.frame_classifier(frame_input)
            predictions.append(pred)
            
        return torch.cat(predictions, dim=1)

class FramePredictionDataset(Dataset):
    """Simplified dataset loader for investigation"""
    
    def __init__(self, data_path, max_samples=1000):
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Convert to numpy arrays and limit samples for investigation
        self.windows = []
        self.labels = []
        
        sample_count = 0
        for utterance_id, utterance_data in data_dict.items():
            frame_predictions = np.array(utterance_data['frame_predictions'], dtype=np.float32)
            frame_positions = np.array(utterance_data['frame_positions'], dtype=np.int64)
            true_boundaries = np.array(utterance_data['true_boundaries'], dtype=np.int64)
            
            # Create sliding windows
            for i in range(len(frame_predictions) - 8 + 1):
                if sample_count >= max_samples:
                    break
                    
                window = frame_predictions[i:i + 8].copy()
                
                # Get positions of middle 6 frames
                middle_positions = frame_positions[i + 1:i + 7]
                
                # Create labels for middle 6 frames
                labels = np.zeros(6, dtype=np.float32)
                for j, pos in enumerate(middle_positions):
                    if pos in true_boundaries:
                        labels[j] = 1
                
                self.windows.append(window)
                self.labels.append(labels)
                sample_count += 1
            
            if sample_count >= max_samples:
                break
        
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        print(f"Loaded {len(self.windows)} windows for investigation")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.windows[idx]), 
                torch.FloatTensor(self.labels[idx]))

def investigate_data_structure(data_path):
    """Investigate the structure of frame predictions data"""
    print("="*60)
    print("INVESTIGATING FRAME PREDICTIONS DATA STRUCTURE")
    print("="*60)
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"Number of utterances: {len(data_dict)}")
    
    # Analyze a few utterances
    utterance_ids = list(data_dict.keys())[:5]
    for utterance_id in utterance_ids:
        utterance_data = data_dict[utterance_id]
        
        print(f"\nUtterance: {utterance_id}")
        print(f"Frame predictions shape: {len(utterance_data['frame_predictions'])}")
        print(f"Frame positions shape: {len(utterance_data['frame_positions'])}")
        print(f"True boundaries: {len(utterance_data['true_boundaries'])}")
        
        # Show first few frame predictions
        frame_preds = utterance_data['frame_predictions'][:10]
        print(f"First 10 frame predictions: {frame_preds}")
        print(f"Prediction range: [{min(utterance_data['frame_predictions']):.4f}, {max(utterance_data['frame_predictions']):.4f}]")
        
        # Show boundary information
        print(f"Boundary positions: {utterance_data['true_boundaries'][:10] if len(utterance_data['true_boundaries']) > 0 else 'None'}")

def investigate_model_behavior():
    """Investigate how the model processes different inputs"""
    print("\n" + "="*60)
    print("INVESTIGATING MODEL BEHAVIOR")
    print("="*60)
    
    # Create model
    model = BoundaryRefinementModel(hidden_dim=64)  # Smaller for investigation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create some test inputs
    print("\n--- Testing with identical windows ---")
    identical_window = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]).float()
    identical_batch = identical_window.unsqueeze(0).repeat(4, 1).to(device)
    
    with torch.no_grad():
        pred_identical = model.forward_detailed(identical_batch)
    
    print(f"\nPredictions for identical inputs:")
    for i in range(4):
        print(f"Sample {i}: {pred_identical[i].cpu().numpy()}")
    
    print("\n--- Testing with different windows ---")
    different_windows = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Increasing
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Decreasing
        [0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8],  # Alternating
        [0.4, 0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.4],  # Peak in middle
    ]).float().to(device)
    
    with torch.no_grad():
        pred_different = model.forward_detailed(different_windows)
    
    print(f"\nPredictions for different inputs:")
    for i in range(4):
        print(f"Sample {i}: {pred_different[i].cpu().numpy()}")

def demonstrate_shared_classifier_issue():
    """Demonstrate why shared classifier is problematic"""
    print("\n" + "="*60)
    print("DEMONSTRATING SHARED CLASSIFIER ISSUE")
    print("="*60)
    
    # Create a minimal example
    hidden_dim = 8
    
    # Simulate what happens when features collapse
    print("\n--- Scenario 1: Features collapse (all frames have similar features) ---")
    
    # All frames have nearly identical features
    collapsed_features = torch.tensor([
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 1
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 2 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 3 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 4 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 5 (identical)
        [1.0, 0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4],  # Frame 6 (identical)
    ]).float()
    
    # Position encodings for frames 1-6
    position_encodings = torch.tensor([
        [(1-3.5)/3.5, 1/7.0],  # Frame 1: [-0.714, 0.143]
        [(2-3.5)/3.5, 2/7.0],  # Frame 2: [-0.429, 0.286]  
        [(3-3.5)/3.5, 3/7.0],  # Frame 3: [-0.143, 0.429]
        [(4-3.5)/3.5, 4/7.0],  # Frame 4: [ 0.143, 0.571]
        [(5-3.5)/3.5, 5/7.0],  # Frame 5: [ 0.429, 0.714]
        [(6-3.5)/3.5, 6/7.0],  # Frame 6: [ 0.714, 0.857]
    ]).float()
    
    print("Features (all identical):")
    for i in range(6):
        print(f"Frame {i+1}: {collapsed_features[i].numpy()}")
    
    print("\nPosition encodings (different):")
    for i in range(6):
        print(f"Frame {i+1}: {position_encodings[i].numpy()}")
    
    # Combine features + position
    frame_inputs = torch.cat([collapsed_features, position_encodings], dim=1)
    print(f"\nCombined inputs shape: {frame_inputs.shape}")
    
    # Create a simple classifier
    classifier = nn.Sequential(
        nn.Linear(10, 4),  # 8 features + 2 position
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    # Initialize with small weights
    for param in classifier.parameters():
        param.data.normal_(0, 0.01)
    
    with torch.no_grad():
        predictions = classifier(frame_inputs)
    
    print(f"\nPredictions when features are identical:")
    for i in range(6):
        print(f"Frame {i+1}: {predictions[i].item():.6f}")
    
    pred_diff = predictions.max() - predictions.min()
    print(f"Prediction range: {pred_diff.item():.6f}")
    
    if pred_diff < 0.01:
        print("🚨 PROBLEM: Even with position encoding, predictions are nearly identical!")
        print("   This is because position encoding is a tiny signal compared to collapsed features.")
    
    print("\n--- Scenario 2: Features are diverse ---")
    
    # Different features for each frame
    diverse_features = torch.tensor([
        [1.0, 0.1, -0.5, 0.2, -0.8, 0.9, 0.3, -0.1],  # Frame 1
        [0.2, 0.8, -0.1, 0.6, -0.3, 0.1, 0.7, -0.9],  # Frame 2 (different)
        [-0.5, 0.3, 0.8, -0.2, 0.6, -0.4, 0.1, 0.9],  # Frame 3 (different)
        [0.7, -0.6, 0.2, 0.9, -0.1, 0.4, -0.8, 0.3],  # Frame 4 (different)
        [-0.2, 0.5, -0.7, 0.1, 0.8, -0.9, 0.4, 0.6],  # Frame 5 (different)
        [0.9, -0.3, 0.4, -0.7, 0.2, 0.6, -0.5, 0.8],  # Frame 6 (different)
    ]).float()
    
    diverse_inputs = torch.cat([diverse_features, position_encodings], dim=1)
    
    with torch.no_grad():
        diverse_predictions = classifier(diverse_inputs)
    
    print(f"\nPredictions when features are diverse:")
    for i in range(6):
        print(f"Frame {i+1}: {diverse_predictions[i].item():.6f}")
    
    diverse_diff = diverse_predictions.max() - diverse_predictions.min()
    print(f"Prediction range: {diverse_diff.item():.6f}")
    
    if diverse_diff > 0.1:
        print("✓ GOOD: With diverse features, we get diverse predictions!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Investigate Frame Predictions Data')
    parser.add_argument('--data-path', type=str, 
                        default='frame_predictions/train_predictions_final.pkl',
                        help='Path to frame predictions data')
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        print("Available frame_predictions files:")
        if os.path.exists('frame_predictions'):
            for f in os.listdir('frame_predictions'):
                if f.endswith('.pkl'):
                    print(f"  - frame_predictions/{f}")
        return
    
    # Investigate data structure
    investigate_data_structure(args.data_path)
    
    # Load small dataset for model testing
    print("\n" + "="*60)
    print("LOADING DATASET FOR MODEL TESTING")
    print("="*60)
    
    dataset = FramePredictionDataset(args.data_path, max_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Show some data examples
    print("\nData examples:")
    for batch_idx, (windows, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Windows shape: {windows.shape}")
        print(f"Labels shape: {labels.shape}")
        
        for i in range(min(2, windows.shape[0])):
            print(f"  Window {i}: {windows[i].numpy()}")
            print(f"  Labels {i}: {labels[i].numpy()}")
        
        if batch_idx >= 2:
            break
    
    # Test model behavior
    investigate_model_behavior()
    
    # Demonstrate the shared classifier issue
    demonstrate_shared_classifier_issue()
    
    print("\n" + "="*60)
    print("SUMMARY: WHY SHARED CLASSIFIER IS PROBLEMATIC")
    print("="*60)
    print("""
The issue is NOT that the model uses position encoding - that's actually good!

The issue is that when the FEATURE EXTRACTOR collapses (outputs similar features
for all frames), the position encoding becomes the ONLY difference between frames.

But position encoding is a tiny 2-dimensional signal compared to the large
feature vector (128 dimensions). When features collapse, the classifier learns
to ignore the tiny position differences and just outputs the same value.

SOLUTION: Either fix feature collapse OR use separate classifiers per frame.
    """)

if __name__ == "__main__":
    main() 
 