#!/usr/bin/env python3
"""
GPU Testing Script for PyTorch CUDA Diagnostics
===============================================

This script tests GPU availability, CUDA installation, and provides
recommendations for fixing GPU detection issues.
"""

import sys
import subprocess

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("🔍 Checking NVIDIA Driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi command not found - NVIDIA driver may not be installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    print("\n🔍 Checking PyTorch CUDA Support...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"🎯 CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
            
            # List all available GPUs
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            
            # Check if PyTorch was compiled with CUDA
            print(f"   PyTorch built with CUDA: {torch.version.cuda is not None}")
            
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_gpu_operations():
    """Test basic GPU operations"""
    print("\n🔍 Testing GPU Operations...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping GPU tests - CUDA not available")
            return False
        
        # Test tensor creation on GPU
        print("   Testing tensor creation on GPU...")
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        print("   ✅ GPU tensor creation successful")
        
        # Test computation on GPU
        print("   Testing computation on GPU...")
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("   ✅ GPU computation successful")
        
        # Test memory usage
        print("   Testing GPU memory...")
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"   📊 GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"   📊 GPU memory reserved: {memory_reserved:.1f} MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        print("   ✅ GPU memory cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU operation failed: {e}")
        return False

def test_model_on_gpu():
    """Test loading a model on GPU"""
    print("\n🔍 Testing Model on GPU...")
    
    try:
        import torch
        import torch.nn as nn
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping model GPU test - CUDA not available")
            return False
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        # Test model on GPU
        device = torch.device('cuda:0')
        model = SimpleModel().to(device)
        print("   ✅ Model moved to GPU successfully")
        
        # Test forward pass
        x = torch.randn(32, 100, device=device)
        output = model(x)
        print("   ✅ Model forward pass on GPU successful")
        print(f"   📊 Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model GPU test failed: {e}")
        return False

def check_transformers_gpu():
    """Test transformers library GPU support"""
    print("\n🔍 Testing Transformers GPU Support...")
    
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping transformers GPU test - CUDA not available")
            return False
        
        print("   Loading Wav2Vec2 model...")
        device = torch.device('cuda:0')
        
        # Load processor and model
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Move model to GPU
        model = model.to(device)
        print("   ✅ Wav2Vec2 model moved to GPU successfully")
        
        # Test with dummy audio
        dummy_audio = torch.randn(16000)  # 1 second of audio
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_values)
        
        print("   ✅ Wav2Vec2 forward pass on GPU successful")
        print(f"   📊 Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Transformers GPU test failed: {e}")
        return False

def provide_recommendations(driver_ok, pytorch_ok, gpu_ops_ok, model_ok, transformers_ok):
    """Provide recommendations based on test results"""
    print("\n" + "="*60)
    print("🔧 RECOMMENDATIONS")
    print("="*60)
    
    if not driver_ok:
        print("❌ NVIDIA Driver Issues:")
        print("   1. Install/update NVIDIA GPU drivers from https://www.nvidia.com/drivers/")
        print("   2. Restart your computer after installation")
        print("   3. Verify installation with 'nvidia-smi' command")
        
    elif not pytorch_ok:
        print("❌ PyTorch CUDA Issues:")
        print("   1. Your PyTorch installation doesn't support CUDA")
        print("   2. Install CUDA-enabled PyTorch:")
        print("      pip uninstall torch torchaudio")
        print("      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Or visit https://pytorch.org/get-started/locally/ for latest instructions")
        
    elif not gpu_ops_ok:
        print("⚠️  GPU Operations Issues:")
        print("   1. GPU detected but operations failing")
        print("   2. Try updating CUDA drivers")
        print("   3. Check GPU memory availability")
        print("   4. Restart Python/system")
        
    elif not model_ok:
        print("⚠️  Model GPU Issues:")
        print("   1. Basic GPU works but model loading fails")
        print("   2. Check GPU memory (may need smaller batch size)")
        print("   3. Try torch.cuda.empty_cache() to free memory")
        
    elif not transformers_ok:
        print("⚠️  Transformers GPU Issues:")
        print("   1. PyTorch GPU works but transformers fails")
        print("   2. Update transformers: pip install --upgrade transformers")
        print("   3. Check GPU memory availability")
        
    else:
        print("🎉 ALL TESTS PASSED!")
        print("   Your GPU setup is working correctly")
        print("   The wav2seg.py script should use GPU automatically")
        print("\n💡 If wav2seg.py still uses CPU, check:")
        print("   1. Make sure you're in the same environment")
        print("   2. Restart your terminal/IDE")
        print("   3. Check for any CUDA_VISIBLE_DEVICES environment variable")

def main():
    """Run all GPU tests"""
    print("GPU Testing Script for PyTorch CUDA")
    print("="*60)
    
    # Run all tests
    driver_ok = check_nvidia_driver()
    pytorch_ok = check_pytorch_cuda()
    gpu_ops_ok = test_gpu_operations()
    model_ok = test_model_on_gpu()
    transformers_ok = check_transformers_gpu()
    
    # Provide recommendations
    provide_recommendations(driver_ok, pytorch_ok, gpu_ops_ok, model_ok, transformers_ok)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   NVIDIA Driver: {'✅' if driver_ok else '❌'}")
    print(f"   PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"   GPU Operations: {'✅' if gpu_ops_ok else '❌'}")
    print(f"   Model on GPU: {'✅' if model_ok else '❌'}")
    print(f"   Transformers GPU: {'✅' if transformers_ok else '❌'}")

if __name__ == "__main__":
    main() 