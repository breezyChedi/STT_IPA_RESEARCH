"""
Demo Script for Speech Segmentation System
==========================================

This script demonstrates how to use individual components of the wav2seg system
for custom applications and experimentation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor
from wav2seg import (
    TIMITSegmentationDataset, 
    Wav2SegModel, 
    load_data, 
    create_dummy_dataset,
    find_boundaries_from_predictions,
    calculate_boundary_metrics
)

def demo_data_loading():
    """Demonstrate data loading and preprocessing."""
    print("=" * 50)
    print("DEMO 1: Data Loading and Preprocessing")
    print("=" * 50)
    
    # Load a small sample of data
    data = load_data('train', max_samples=5)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Create dataset
    dataset = TIMITSegmentationDataset(data, processor, tolerance_ms=20)
    
    # Show sample data
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio shape: {sample['input_values'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Phone boundaries: {sample['phone_boundaries'][:10]}...")
    print(f"File ID: {sample['file_id']}")
    
    # Visualize labels
    labels = sample['labels'].numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(labels[:8000])  # First 0.5 seconds
    plt.title('Frame-level Boundary Labels (First 0.5s)')
    plt.xlabel('Frame')
    plt.ylabel('Boundary Label')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Positive frames: {np.sum(labels)} / {len(labels)} ({np.mean(labels)*100:.1f}%)")

def demo_model_architecture():
    """Demonstrate model architecture and forward pass."""
    print("\n" + "=" * 50)
    print("DEMO 2: Model Architecture")
    print("=" * 50)
    
    # Initialize model
    model = Wav2SegModel(freeze_wav2vec2=True)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Demo forward pass
    dummy_audio = torch.randn(1, 16000)  # 1 second of audio
    print(f"\nInput shape: {dummy_audio.shape}")
    
    with torch.no_grad():
        logits = model(dummy_audio)
        probabilities = torch.sigmoid(logits)
        predictions = probabilities > 0.5
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted boundaries: {torch.sum(predictions).item()} frames")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(probabilities[0].numpy())
    plt.title('Boundary Probabilities')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0].float().numpy())
    plt.title('Binary Predictions')
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_boundary_detection():
    """Demonstrate boundary detection from predictions."""
    print("\n" + "=" * 50)
    print("DEMO 3: Boundary Detection")
    print("=" * 50)
    
    # Create synthetic prediction data
    length = 1000
    predictions = np.zeros(length)
    
    # Add some boundary regions
    true_boundaries = [100, 300, 500, 750, 900]
    for boundary in true_boundaries:
        start = max(0, boundary - 10)
        end = min(length, boundary + 10)
        predictions[start:end] = 1
    
    # Add some noise
    noise_indices = np.random.choice(length, 50, replace=False)
    predictions[noise_indices] = 1
    
    print(f"Input predictions shape: {predictions.shape}")
    print(f"Positive frames: {np.sum(predictions)}")
    
    # Extract boundaries
    detected_boundaries = find_boundaries_from_predictions(predictions, min_distance=50)
    print(f"True boundaries: {true_boundaries}")
    print(f"Detected boundaries: {detected_boundaries}")
    
    # Calculate metrics
    mae, precision, recall, f1 = calculate_boundary_metrics(
        np.array(true_boundaries), detected_boundaries, tolerance=20
    )
    
    print(f"\nMetrics:")
    print(f"MAE: {mae:.2f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(predictions)
    plt.title('Binary Predictions')
    plt.ylabel('Prediction')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.scatter(true_boundaries, [1] * len(true_boundaries), 
               color='blue', label='True Boundaries', s=100, marker='|')
    plt.scatter(detected_boundaries, [0.5] * len(detected_boundaries), 
               color='red', label='Detected Boundaries', s=100, marker='|')
    plt.title('Boundary Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Boundary Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_custom_training():
    """Demonstrate custom training setup."""
    print("\n" + "=" * 50)
    print("DEMO 4: Custom Training Setup")
    print("=" * 50)
    
    # Create small dataset
    data = create_dummy_dataset('train', num_samples=10)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    dataset = TIMITSegmentationDataset(data, processor)
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = Wav2SegModel()
    
    # Setup training components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training setup complete!")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 2")
    print(f"Number of batches: {len(dataloader)}")
    
    # Demo one training step
    model.train()
    batch = next(iter(dataloader))
    
    input_values = batch['input_values']
    labels = batch['labels']
    
    print(f"\nBatch input shape: {input_values.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    # Forward pass
    logits = model(input_values)
    
    # Handle sequence length mismatch
    min_length = min(logits.size(1), labels.size(1))
    logits = logits[:, :min_length]
    labels = labels[:, :min_length]
    
    loss = criterion(logits, labels)
    
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")

def demo_evaluation_metrics():
    """Demonstrate evaluation metrics calculation."""
    print("\n" + "=" * 50)
    print("DEMO 5: Evaluation Metrics")
    print("=" * 50)
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'Perfect Match',
            'true': [100, 200, 300, 400],
            'pred': [100, 200, 300, 400]
        },
        {
            'name': 'Slight Offset',
            'true': [100, 200, 300, 400],
            'pred': [105, 195, 305, 395]
        },
        {
            'name': 'Missing Boundaries',
            'true': [100, 200, 300, 400],
            'pred': [100, 300]
        },
        {
            'name': 'Extra Boundaries',
            'true': [100, 200, 300, 400],
            'pred': [50, 100, 150, 200, 250, 300, 350, 400, 450]
        }
    ]
    
    tolerance = 20
    
    print(f"Tolerance: {tolerance} frames\n")
    
    for scenario in scenarios:
        true_bounds = np.array(scenario['true'])
        pred_bounds = np.array(scenario['pred'])
        
        mae, precision, recall, f1 = calculate_boundary_metrics(
            true_bounds, pred_bounds, tolerance
        )
        
        print(f"{scenario['name']}:")
        print(f"  True: {true_bounds}")
        print(f"  Pred: {pred_bounds}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print()

def main():
    """Run all demos."""
    print("Speech Segmentation System - Component Demos")
    print("=" * 60)
    
    try:
        demo_data_loading()
        demo_model_architecture()
        demo_boundary_detection()
        demo_custom_training()
        demo_evaluation_metrics()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 