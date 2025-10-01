#!/usr/bin/env python3
"""
Check the full TIMIT dataset size
"""

from wav2seg import load_local_timit_data

def check_dataset_size():
    print("Checking full TIMIT dataset size...")
    
    # Load full training set
    train_data = load_local_timit_data('train', max_samples=None)
    print(f"Training samples: {len(train_data)}")
    
    # Load full test set  
    test_data = load_local_timit_data('test', max_samples=None)
    print(f"Test samples: {len(test_data)}")
    
    print(f"\nTotal TIMIT dataset: {len(train_data) + len(test_data)} samples")
    print(f"Train/Test split: {len(train_data)}/{len(test_data)}")

if __name__ == "__main__":
    check_dataset_size() 