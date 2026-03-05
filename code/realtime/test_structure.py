"""
Simple test to verify real-time component structure
"""

import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test all real-time component imports"""
    print("Testing real-time component imports...")
    
    try:
        from realtime.memory_manager import GPUMemoryManager, get_memory_manager
        print("✓ Memory manager imported successfully")
    except Exception as e:
        print(f"✗ Memory manager import failed: {e}")
    
    try:
        from realtime.cuda_streams import RealTimeStreams, get_stream_manager
        print("✓ CUDA streams imported successfully")
    except Exception as e:
        print(f"✗ CUDA streams import failed: {e}")
    
    try:
        from realtime.texture_bridge import TextureBridge, RenderBuffer
        print("✓ Texture bridge imported successfully")
    except Exception as e:
        print(f"✗ Texture bridge import failed: {e}")
    
    try:
        from realtime.occupancy_grid import OccupancyGrid, create_occupancy_grid
        print("✓ Occupancy grid imported successfully")
    except Exception as e:
        print(f"✗ Occupancy grid import failed: {e}")
    
    try:
        from realtime.lod_system import AdaptiveRenderer, QualityLevel
        print("✓ LOD system imported successfully")
    except Exception as e:
        print(f"✗ LOD system import failed: {e}")
    
    try:
        from realtime.camera_controls import InstantNGPCamera, CameraMode
        print("✓ Camera controls imported successfully")
    except Exception as e:
        print(f"✗ Camera controls import failed: {e}")

def test_basic_functionality():
    """Test basic functionality of components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test memory manager
        memory_manager = get_memory_manager()
        stats = memory_manager.get_memory_stats()
        print(f"✓ Memory manager stats: {stats['allocated_mb']:.1f}MB allocated")
    except Exception as e:
        print(f"✗ Memory manager test failed: {e}")
    
    try:
        # Test camera controls
        camera = InstantNGPCamera(width=512, height=512)
        camera_info = camera.get_camera_info()
        print(f"✓ Camera controls initialized: {camera_info['mode']} mode")
    except Exception as e:
        print(f"✗ Camera controls test failed: {e}")

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Basic Patch Integration:")
    print("   from realtime.patches import patch_idr_trainer")
    print("   trainer = patch_idr_trainer(trainer, enable_realtime=True)")
    
    print("\n2. Enhanced Training Script:")
    print("   python code/training/enhanced_realtime_train.py")
    print("   --conf ./confs/your_config_realtime.conf")
    
    print("\n3. Configuration:")
    print("   Add realtime_render section to your .conf files:")
    print("   realtime_render {")
    print("       enabled = true")
    print("       target_fps = 60")
    print("   }")

if __name__ == "__main__":
    print("Real-Time IDR Neural Rendering - Structure Test")
    print("="*60)
    
    test_imports()
    test_basic_functionality()
    show_usage_examples()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)