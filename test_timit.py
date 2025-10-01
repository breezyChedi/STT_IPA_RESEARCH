#!/usr/bin/env python3
"""
Test script to verify local TIMIT dataset loading
"""

from wav2seg import load_local_timit_data, load_data

def test_local_timit():
    """Test loading local TIMIT dataset"""
    print("Testing local TIMIT dataset loading...")
    
    try:
        # Test loading a small sample
        data = load_local_timit_data('train', max_samples=3)
        print(f"✅ Successfully loaded {len(data)} training samples")
        
        if len(data) > 0:
            sample = data[0]
            print(f"✅ Sample keys: {list(sample.keys())}")
            print(f"✅ Audio shape: {sample['audio']['array'].shape}")
            print(f"✅ Sample rate: {sample['audio']['sampling_rate']}")
            print(f"✅ Phone boundaries: {len(sample['phonetic_detail']['start'])} phones")
            print(f"✅ File ID: {sample['id']}")
            print(f"✅ Speaker ID: {sample['speaker_id']}")
            print(f"✅ Transcription: {sample['text']}")
            
            # Test a few phone boundaries
            starts = sample['phonetic_detail']['start'][:5]
            stops = sample['phonetic_detail']['stop'][:5]
            phones = sample['phonetic_detail']['utterance'][:5]
            print(f"✅ First 5 phones: {list(zip(phones, starts, stops))}")
        
        # Test loading test set
        test_data = load_local_timit_data('test', max_samples=2)
        print(f"✅ Successfully loaded {len(test_data)} test samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading local TIMIT: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_data_function():
    """Test the main load_data function"""
    print("\nTesting main load_data function...")
    
    try:
        # This should now use local TIMIT data
        data = load_data('train', max_samples=2)
        print(f"✅ load_data() successfully loaded {len(data)} samples")
        
        if len(data) > 0:
            sample = data[0]
            print(f"✅ Sample structure looks correct: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with load_data(): {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TIMIT Dataset Loading Test")
    print("=" * 50)
    
    success1 = test_local_timit()
    success2 = test_load_data_function()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Local TIMIT dataset is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.") 