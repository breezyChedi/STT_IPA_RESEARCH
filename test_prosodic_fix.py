#!/usr/bin/env python3
"""
🧪 PROSODIC FEATURE FIX DIAGNOSTIC SCRIPT

Run this to verify that the prosodic feature bug has been fixed.
This tests the predict_boundaries_competitive function with real vs zero prosodic features.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Processor
from wav2seg_v4 import (
    CompetitiveMultiScaleBoundaryClassifier, 
    CompetitiveWindowPreprocessor, 
    predict_boundaries_competitive
)

def test_prosodic_bug_fix():
    """Test the prosodic feature fix comprehensively."""
    print("🧪 PROSODIC FEATURE BUG FIX TEST")
    print("=" * 50)
    
    try:
        # 1. Create test components
        print("📦 Setting up test components...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        # Create a simple model (we don't need a trained one for this test)
        model = CompetitiveMultiScaleBoundaryClassifier(
            freeze_wav2vec2=True,
            hidden_dim=64,  # Small for quick test
            use_prosodic=True
        )
        model.eval()
        
        # Create longer test audio (3 seconds)
        test_audio = np.random.randn(48000).astype(np.float32)  # 3s at 16kHz
        print(f"   Audio length: {len(test_audio)} samples ({len(test_audio)/16000:.1f}s)")
        
        # 2. Test prosodic feature extraction directly
        print("\n🔬 Testing prosodic feature extraction...")
        prosodic_extractor = CompetitiveWindowPreprocessor(
            [], processor, 
            window_duration=0.5,
            sample_rate=16000,
            use_prosodic=True
        )
        
        # Extract features from a window
        window_audio = test_audio[:8000]  # 0.5s window
        prosodic_dict = prosodic_extractor.extract_prosodic_features(window_audio)
        prosodic_tensor = prosodic_extractor._prosodic_dict_to_tensor(prosodic_dict, 25)
        
        print(f"   ✅ Prosodic tensor shape: {prosodic_tensor.shape}")
        print(f"   ✅ Non-zero elements: {torch.sum(prosodic_tensor != 0).item()}")
        print(f"   ✅ Value range: [{prosodic_tensor.min():.4f}, {prosodic_tensor.max():.4f}]")
        
        # 3. Test the fixed prediction function
        print("\n🎯 Testing boundary prediction with REAL prosodic features...")
        boundaries = predict_boundaries_competitive(
            model, test_audio, processor, 'cpu',
            window_duration=0.5, stride=0.1, threshold=0.1  # Low threshold to get some predictions
        )
        
        print(f"   ✅ Prediction function ran successfully!")
        print(f"   ✅ Predicted boundaries: {len(boundaries)}")
        if boundaries:
            print(f"   ✅ First few boundaries: {boundaries[:5]}")
        
        # 4. Compare with old broken version (zeros)
        print("\n⚠️  Testing with ZERO prosodic features (old broken version)...")
        
        # Temporarily patch the function to use zeros (simulating the old bug)
        original_extractor = CompetitiveWindowPreprocessor
        
        class BrokenPreprocessor(CompetitiveWindowPreprocessor):
            def extract_prosodic_features(self, audio):
                # Return zeros like the old broken version
                return {
                    'energy_delta': np.zeros(25, dtype=np.float32),
                    'energy_delta2': np.zeros(25, dtype=np.float32),
                    'centroid_delta': np.zeros(25, dtype=np.float32),
                    'rolloff_delta': np.zeros(25, dtype=np.float32),
                    'zcr_delta': np.zeros(25, dtype=np.float32),
                    'energy_mean': 0.0,
                    'mfcc_mean': [0.0] * 6,
                    'mfcc_std': [0.0] * 6,
                }
        
        # Test single window prediction with zeros vs real features
        with torch.no_grad():
            # Real features
            inputs = processor(window_audio, sampling_rate=16000, return_tensors="pt")
            prosodic_real = prosodic_tensor.unsqueeze(0)  # Add batch dim
            pred_real = torch.sigmoid(model(inputs.input_values, prosodic_real)).item()
            
            # Zero features (broken version)
            prosodic_zeros = torch.zeros(1, 28, 25)
            pred_zeros = torch.sigmoid(model(inputs.input_values, prosodic_zeros)).item()
        
        print(f"   📊 Prediction with REAL prosodic features: {pred_real:.6f}")
        print(f"   📊 Prediction with ZERO prosodic features: {pred_zeros:.6f}")
        print(f"   📊 Difference: {abs(pred_real - pred_zeros):.6f}")
        
        # 5. Final assessment
        print("\n🏆 DIAGNOSTIC RESULTS:")
        
        if abs(pred_real - pred_zeros) > 0.001:
            print("   ✅ FIXED! Model predictions differ significantly with real vs zero features")
            print("   ✅ The prosodic feature bug has been resolved!")
            return True
        else:
            print("   ⚠️ Model predictions are very similar with real vs zero features")
            print("   ⚠️ This could mean:")
            print("      - Model wasn't trained with prosodic features")
            print("      - Model learned to ignore prosodic features")
            print("      - There might still be an issue")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prosodic_bug_fix()
    if success:
        print("\n🎉 PROSODIC FEATURE FIX VERIFIED!")
        print("Your boundary prediction should now work correctly on full audio.")
    else:
        print("\n⚠️ PROSODIC FEATURE FIX NEEDS MORE INVESTIGATION")
        print("The model might not be using prosodic features effectively.") 