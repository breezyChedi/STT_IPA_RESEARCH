#!/usr/bin/env python3
"""
Test script to show the difference between limited and full dataset loading
"""

from wav2seg import load_data

def test_dataset_sizes():
    print("TIMIT Dataset Size Comparison")
    print("=" * 50)
    
    # Test with limits (old configuration)
    print("🔸 With limits (old configuration):")
    train_limited = load_data('train', max_samples=100)
    test_limited = load_data('test', max_samples=50)
    print(f"   Training: {len(train_limited)} samples")
    print(f"   Test: {len(test_limited)} samples")
    print(f"   Total: {len(train_limited) + len(test_limited)} samples")
    
    print("\n🔸 With full dataset (new configuration):")
    train_full = load_data('train', max_samples=None)
    test_full = load_data('test', max_samples=None)
    print(f"   Training: {len(train_full)} samples")
    print(f"   Test: {len(test_full)} samples")
    print(f"   Total: {len(train_full) + len(test_full)} samples")
    
    print(f"\n📈 Improvement:")
    print(f"   Training: {len(train_full) - len(train_limited)} more samples ({len(train_full)/len(train_limited):.1f}x)")
    print(f"   Test: {len(test_full) - len(test_limited)} more samples ({len(test_full)/len(test_limited):.1f}x)")
    print(f"   Total: {(len(train_full) + len(test_full)) - (len(train_limited) + len(test_limited))} more samples")

if __name__ == "__main__":
    test_dataset_sizes() 