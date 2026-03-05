"""
GPU Memory Management for Real-Time Neural Rendering
Fixes critical performance bottlenecks from aggressive cache clearing
"""

import torch
import gc
import os
import threading
from collections import deque
from typing import Optional, Dict, Any


class GPUMemoryManager:
    """
    Advanced GPU memory management for real-time rendering.
    Replaces aggressive torch.cuda.empty_cache() calls with intelligent pooling.
    """
    
    def __init__(self, device='cuda', memory_pool_size_mb=2048, max_fragmentation=0.3):
        self.device = device
        self.memory_pool_size_mb = memory_pool_size_mb
        self.max_fragmentation = max_fragmentation
        
        # Memory monitoring
        self.allocated_history = deque(maxlen=100)
        self.fragmentation_history = deque(maxlen=50)
        self.last_cleanup_time = 0
        self.cleanup_interval = 30.0  # seconds
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Configure CUDA allocator for better memory management
        self._configure_cuda_allocator()
        
        print(f"GPU Memory Manager initialized: {memory_pool_size_mb}MB pool, max fragmentation: {max_fragmentation}")
    
    def _configure_cuda_allocator(self):
        """Configure PyTorch CUDA allocator to prevent fragmentation"""
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
        
        # Enable memory pool for better allocation patterns
        if torch.cuda.is_available():
            torch.cuda.memory._set_allocator_settings(
                {
                    'max_split_size_mb': 128,
                    'garbage_collection_threshold': 0.8
                }
            )
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'fragmentation': 0}
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved
        else:
            fragmentation = 0.0
            
        return {
            'allocated_mb': allocated / (1024**2),
            'reserved_mb': reserved / (1024**2),
            'fragmentation': fragmentation,
            'free_mb': (torch.cuda.get_device_properties(self.device).total_memory - reserved) / (1024**2)
        }
    
    def should_clear_cache(self) -> bool:
        """Intelligently determine if cache clearing is necessary"""
        with self.lock:
            current_time = time.time()
            
            # Don't clear too frequently
            if current_time - self.last_cleanup_time < self.cleanup_interval:
                return False
            
            stats = self.get_memory_stats()
            
            # Store history for trend analysis
            self.allocated_history.append(stats['allocated_mb'])
            self.fragmentation_history.append(stats['fragmentation'])
            
            # Clear conditions
            should_clear = (
                stats['fragmentation'] > self.max_fragmentation or
                stats['free_mb'] < 100 or  # Less than 100MB free
                (len(self.fragmentation_history) >= 10 and 
                 sum(self.fragmentation_history) / len(self.fragmentation_history) > 0.25)
            )
            
            if should_clear:
                self.last_cleanup_time = current_time
                print(f"Memory cleanup triggered - fragmentation: {stats['fragmentation']:.3f}, free: {stats['free_mb']:.1f}MB")
            
            return should_clear
    
    def smart_cache_clear(self, force=False):
        """Smart cache clearing that only runs when necessary"""
        if force or self.should_clear_cache():
            self._perform_cleanup()
    
    def _perform_cleanup(self):
        """Perform actual memory cleanup"""
        try:
            # Force garbage collection first
            gc.collect()
            
            # Clear CUDA cache with synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                
            print("Memory cleanup completed")
        except Exception as e:
            print(f"Memory cleanup failed: {e}")
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 = low, 1.0 = critical)"""
        stats = self.get_memory_stats()
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        
        # Calculate pressure based on usage and fragmentation
        usage_pressure = stats['reserved_mb'] / (total_memory / (1024**2))
        fragmentation_pressure = stats['fragmentation']
        
        return max(usage_pressure, fragmentation_pressure)
    
    def should_reduce_batch_size(self, target_memory_mb=1024) -> bool:
        """Check if batch size should be reduced due to memory pressure"""
        stats = self.get_memory_stats()
        return stats['reserved_mb'] > target_memory_mb
    
    def monitor_memory_health(self) -> Dict[str, Any]:
        """Comprehensive memory health monitoring"""
        stats = self.get_memory_stats()
        
        health_report = {
            'status': 'healthy',
            'stats': stats,
            'recommendations': [],
            'warnings': []
        }
        
        # Check for issues
        if stats['fragmentation'] > 0.4:
            health_report['warnings'].append(f"High fragmentation: {stats['fragmentation']:.2f}")
            health_report['status'] = 'warning'
        
        if stats['free_mb'] < 200:
            health_report['warnings'].append(f"Low free memory: {stats['free_mb']:.1f}MB")
            health_report['status'] = 'critical'
        
        if stats['allocated_mb'] > self.memory_pool_size_mb:
            health_report['recommendations'].append("Consider reducing batch size or model complexity")
        
        return health_report


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager() -> GPUMemoryManager:
    """Get or create global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = GPUMemoryManager()
    return _global_memory_manager


def smart_cache_clear(force=False):
    """Convenient function for smart cache clearing"""
    return get_memory_manager().smart_cache_clear(force)


def get_memory_stats():
    """Convenient function for memory statistics"""
    return get_memory_manager().get_memory_stats()


def get_memory_pressure():
    """Convenient function for memory pressure"""
    return get_memory_manager().get_memory_pressure()


# Import time for interval checking
import time