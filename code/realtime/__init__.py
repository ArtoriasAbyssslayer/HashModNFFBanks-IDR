"""
Real-Time Neural Rendering Package for IDR
Provides instant-ngp style real-time visualization during training
"""

from .memory_manager import GPUMemoryManager, get_memory_manager, smart_cache_clear
from .cuda_streams import RealTimeStreams, get_stream_manager, with_stream
from .texture_bridge import TextureBridge, RenderBuffer
from .occupancy_grid import OccupancyGrid, AdaptiveOccupancyGrid, create_occupancy_grid
from .lod_system import AdaptiveRenderer, LevelOfDetailManager, QualityLevel
from .camera_controls import InstantNGPCamera, CameraMode
from .realtime_renderer import RealTimeIDRRenderer
from .patches import patch_idr_trainer

__version__ = "1.0.0"
__author__ = "Real-Time IDR Rendering System"

__all__ = [
    # Core components
    'GPUMemoryManager',
    'RealTimeStreams', 
    'TextureBridge',
    'RenderBuffer',
    'OccupancyGrid',
    'AdaptiveOccupancyGrid',
    'AdaptiveRenderer',
    'LevelOfDetailManager',
    'InstantNGPCamera',
    'RealTimeIDRRenderer',
    
    # Convenience functions
    'get_memory_manager',
    'get_stream_manager',
    'smart_cache_clear',
    'with_stream',
    'create_occupancy_grid',
    'patch_idr_trainer',
    
    # Enums and constants
    'CameraMode',
    'QualityLevel'
]

print(f"Real-Time IDR Rendering Package v{__version__} loaded")
print("Components: Memory Management, CUDA Streams, OpenGL-CUDA Interop, ")
print("            Occupancy Grids, LOD System, Instant-NGP Camera Controls")