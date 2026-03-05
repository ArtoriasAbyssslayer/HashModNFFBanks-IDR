"""
CUDA Stream Management for Real-Time Neural Rendering
Enables concurrent training and rendering operations
"""

import torch
import threading
import time
from typing import Optional, Callable, Any, Dict
from enum import Enum


class StreamPriority(Enum):
    """CUDA stream priority levels"""
    LOWEST = -5
    LOW = -3
    NORMAL = 0
    HIGH = 3
    HIGHEST = 5


class RealTimeStreams:
    """
    Manages multiple CUDA streams for concurrent neural rendering operations.
    Provides thread-safe stream management and synchronization.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.streams = {}
        self.events = {}
        self.lock = threading.Lock()
        
        # Create dedicated streams for different operations
        self._create_streams()
        
        # Performance monitoring
        self.timing_history = {}
        self.active_operations = {}
        
        print("Real-Time CUDA Streams initialized")
    
    def _create_streams(self):
        """Create and configure CUDA streams"""
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, streams will be CPU-based")
            return
        
        # Training stream (lower priority for background processing)
        self.streams['training'] = torch.cuda.Stream(
            device=self.device, 
            priority=StreamPriority.LOW.value,
            priority_enabled=True
        )
        
        # Real-time rendering stream (highest priority)
        self.streams['rendering'] = torch.cuda.Stream(
            device=self.device, 
            priority=StreamPriority.HIGHEST.value,
            priority_enabled=True
        )
        
        # Data preprocessing stream
        self.streams['preprocessing'] = torch.cuda.Stream(
            device=self.device, 
            priority=StreamPriority.NORMAL.value,
            priority_enabled=True
        )
        
        # Memory operations stream
        self.streams['memory'] = torch.cuda.Stream(
            device=self.device, 
            priority=StreamPriority.LOW.value,
            priority_enabled=True
        )
        
        # Create events for synchronization
        for name in self.streams:
            self.events[name] = {
                'start': torch.cuda.Event(enable_timing=True),
                'end': torch.cuda.Event(enable_timing=True)
            }
    
    def get_stream(self, name: str) -> torch.cuda.Stream:
        """Get a specific stream by name"""
        with self.lock:
            if name not in self.streams:
                raise ValueError(f"Stream '{name}' not found. Available: {list(self.streams.keys())}")
            return self.streams[name]
    
    def with_stream(self, stream_name: str):
        """Context manager for executing operations on a specific stream"""
        return StreamContext(self.streams[stream_name], stream_name, self)
    
    def synchronize_stream(self, stream_name: str):
        """Synchronize a specific stream"""
        if stream_name in self.streams:
            self.streams[stream_name].synchronize()
    
    def synchronize_all(self):
        """Synchronize all streams"""
        for stream in self.streams.values():
            stream.synchronize()
    
    def wait_for_stream(self, waiting_stream: str, source_stream: str):
        """Make waiting_stream wait for source_stream to complete"""
        waiting = self.streams[waiting_stream]
        source = self.streams[source_stream]
        waiting.wait_stream(source)
    
    def start_timing(self, operation: str, stream_name: str):
        """Start timing an operation on a specific stream"""
        if stream_name in self.events:
            self.events[stream_name]['start'].record(stream=self.streams[stream_name])
            self.active_operations[operation] = {'stream': stream_name, 'start_time': time.time()}
    
    def end_timing(self, operation: str, stream_name: str):
        """End timing an operation and record duration"""
        if stream_name in self.events and operation in self.active_operations:
            self.events[stream_name]['end'].record(stream=self.streams[stream_name])
            
            # Calculate elapsed time
            elapsed_ms = self.events[stream_name]['start'].elapsed_time(
                self.events[stream_name]['end']
            )
            
            # Store in history
            if operation not in self.timing_history:
                self.timing_history[operation] = []
            self.timing_history[operation].append(elapsed_ms)
            
            # Keep only recent history
            if len(self.timing_history[operation]) > 100:
                self.timing_history[operation] = self.timing_history[operation][-100:]
            
            del self.active_operations[operation]
            
            return elapsed_ms
        return None
    
    def get_average_time(self, operation: str) -> float:
        """Get average execution time for an operation"""
        if operation in self.timing_history and self.timing_history[operation]:
            return sum(self.timing_history[operation]) / len(self.timing_history[operation])
        return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'stream_usage': {},
            'operation_times': {},
            'active_operations': len(self.active_operations)
        }
        
        # Operation timing statistics
        for operation, times in self.timing_history.items():
            if times:
                stats['operation_times'][operation] = {
                    'count': len(times),
                    'average_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'last_ms': times[-1]
                }
        
        return stats


class StreamContext:
    """Context manager for executing operations on a CUDA stream"""
    
    def __init__(self, stream: torch.cuda.Stream, stream_name: str, stream_manager: RealTimeStreams):
        self.stream = stream
        self.stream_name = stream_name
        self.stream_manager = stream_manager
        self.operation_timing = {}
    
    def __enter__(self):
        """Enter stream context"""
        self.stream.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit stream context"""
        self.stream.__exit__(exc_type, exc_val, exc_tb)
    
    def time_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute and time an operation on this stream"""
        # Start timing
        self.stream_manager.start_timing(operation_name, self.stream_name)
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # End timing
            elapsed_ms = self.stream_manager.end_timing(operation_name, self.stream_name)
            
            return result, elapsed_ms
        except Exception as e:
            # Clean up timing on error
            if operation_name in self.stream_manager.active_operations:
                del self.stream_manager.active_operations[operation_name]
            raise e


# Global stream manager instance
_global_stream_manager = None


def get_stream_manager() -> RealTimeStreams:
    """Get or create global stream manager instance"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = RealTimeStreams()
    return _global_stream_manager


def get_stream(stream_name: str) -> torch.cuda.Stream:
    """Get a specific stream"""
    return get_stream_manager().get_stream(stream_name)


def with_stream(stream_name: str):
    """Get context manager for a specific stream"""
    return get_stream_manager().with_stream(stream_name)


def render_async(func: Callable, *args, **kwargs):
    """Execute function on rendering stream asynchronously"""
    manager = get_stream_manager()
    with manager.with_stream('rendering') as ctx:
        return ctx.time_operation('render', func, *args, **kwargs)


def train_async(func: Callable, *args, **kwargs):
    """Execute function on training stream asynchronously"""
    manager = get_stream_manager()
    with manager.with_stream('training') as ctx:
        return ctx.time_operation('train', func, *args, **kwargs)


def preprocess_async(func: Callable, *args, **kwargs):
    """Execute function on preprocessing stream asynchronously"""
    manager = get_stream_manager()
    with manager.with_stream('preprocessing') as ctx:
        return ctx.time_operation('preprocess', func, *args, **kwargs)


def synchronize_rendering():
    """Synchronize rendering stream (for real-time constraints)"""
    get_stream_manager().synchronize_stream('rendering')


def wait_for_preprocessing():
    """Make rendering wait for preprocessing to complete"""
    manager = get_stream_manager()
    manager.wait_for_stream('rendering', 'preprocessing')


def get_performance_stats():
    """Get performance statistics from stream manager"""
    return get_stream_manager().get_performance_stats()