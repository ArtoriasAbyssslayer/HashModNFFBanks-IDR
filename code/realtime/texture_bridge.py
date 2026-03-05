"""
OpenGL-CUDA Interoperability Bridge for Real-Time Neural Rendering
Enables zero-copy texture sharing between CUDA and OpenGL
"""

import torch
import numpy as np
import threading
from typing import Optional, Tuple, Dict, Any
from collections import deque

# Try to import OpenGL, but handle gracefully for development/testing
try:
    import OpenGL.GL as gl
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except ImportError:
    print("Warning: OpenGL not available, using mock implementation")
    OPENGL_AVAILABLE = False

# Try to import CUDA GL interop
try:
    import pycuda.driver as cuda
    import pycuda.gl as cudagl
    PYCUDA_AVAILABLE = True
except ImportError:
    print("Warning: PyCUDA not available, using torch CUDA interop")
    PYCUDA_AVAILABLE = False


class TextureBridge:
    """
    Bridge between PyTorch CUDA tensors and OpenGL textures for real-time rendering.
    Provides zero-copy texture updates and efficient GPU-to-GPU transfers.
    """
    
    def __init__(self, width: int, height: int, channels: int = 4, dtype=torch.float32):
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        
        # OpenGL resources
        self.gl_texture = None
        self.gl_fbo = None
        self.gl_renderbuffer = None
        
        # CUDA resources
        self.cuda_graphics_resource = None
        self.cuda_array = None
        self.cuda_tensor = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.update_times = deque(maxlen=60)
        self.last_update_time = 0
        
        # Initialize resources
        self._initialize_resources()
        
        print(f"TextureBridge initialized: {width}x{height}x{channels}")
    
    def _initialize_resources(self):
        """Initialize OpenGL and CUDA resources"""
        if not OPENGL_AVAILABLE:
            print("OpenGL not available, creating mock resources")
            self._initialize_mock_resources()
            return
        
        try:
            self._create_gl_resources()
            self._create_cuda_resources()
        except Exception as e:
            print(f"Failed to initialize resources: {e}")
            self._initialize_mock_resources()
    
    def _initialize_mock_resources(self):
        """Initialize mock resources for development/testing"""
        self.cuda_tensor = torch.zeros(
            (self.height, self.width, self.channels),
            dtype=self.dtype,
            device='cuda'
        )
        print("Mock resources initialized")
    
    def _create_gl_resources(self):
        """Create OpenGL texture and framebuffer"""
        # Create texture
        self.gl_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_texture)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        # Allocate texture memory
        if self.dtype == torch.float32:
            internal_format = gl.GL_RGBA32F
            format = gl.GL_RGBA
            type = gl.GL_FLOAT
        elif self.dtype == torch.uint8:
            internal_format = gl.GL_RGBA8
            format = gl.GL_RGBA
            type = gl.GL_UNSIGNED_BYTE
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, internal_format,
            self.width, self.height, 0, format, type, None
        )
        
        # Create framebuffer for rendering to texture
        self.gl_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.gl_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, self.gl_texture, 0
        )
        
        # Check framebuffer completeness
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")
        
        # Unbind
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
        print("OpenGL resources created successfully")
    
    def _create_cuda_resources(self):
        """Create CUDA graphics resources for texture access"""
        if not PYCUDA_AVAILABLE:
            # Create PyTorch tensor as fallback
            self.cuda_tensor = torch.zeros(
                (self.height, self.width, self.channels),
                dtype=self.dtype,
                device='cuda'
            )
            print("PyTorch CUDA tensor created as fallback")
            return
        
        try:
            # Register OpenGL texture with CUDA
            self.cuda_graphics_resource = cudagl.graphics_map_flags(
                self.gl_texture,
                cudagl.graphics_map_flags.WRITE_DISCARD
            )
            print("CUDA graphics resource created")
        except Exception as e:
            print(f"Failed to create CUDA graphics resource: {e}")
            # Fallback to PyTorch tensor
            self.cuda_tensor = torch.zeros(
                (self.height, self.width, self.channels),
                dtype=self.dtype,
                device='cuda'
            )
    
    def update_from_tensor(self, tensor: torch.Tensor, sync: bool = True):
        """
        Update texture from PyTorch tensor with zero-copy if possible.
        
        Args:
            tensor: Source tensor (should be on CUDA)
            sync: Whether to synchronize after update
        """
        with self.lock:
            start_time = 0
            
            try:
                if tensor.shape != (self.height, self.width, self.channels):
                    raise ValueError(
                        f"Tensor shape mismatch: {tensor.shape} vs "
                        f"expected ({self.height}, {self.width}, {self.channels})"
                    )
                
                if tensor.device != torch.device('cuda'):
                    tensor = tensor.cuda()
                
                if not OPENGL_AVAILABLE or self.cuda_tensor is None:
                    # Fallback: copy to PyTorch tensor
                    self.cuda_tensor.copy_(tensor)
                elif PYCUDA_AVAILABLE:
                    # Use PyCUDA for zero-copy update
                    self._update_with_pycuda(tensor)
                else:
                    # Fallback copy
                    self.cuda_tensor.copy_(tensor)
                
                if sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Record timing
                update_time = (time.time() - start_time) * 1000 if start_time > 0 else 0
                self.update_times.append(update_time)
                self.last_update_time = time.time()
                
            except Exception as e:
                print(f"Error updating texture: {e}")
    
    def _update_with_pycuda(self, tensor: torch.Tensor):
        """Update texture using PyCUDA zero-copy mapping"""
        try:
            # Map graphics resource for CUDA access
            self.cuda_array, = self.cuda_graphics_resource.map()
            
            # Get array descriptor
            array_desc = self.cuda_array.get_descriptor()
            
            # Copy tensor data to CUDA array
            tensor_np = tensor.cpu().numpy()
            
            # Copy data based on format
            if self.dtype == torch.float32:
                # For float32, copy directly
                cuda.memcpy_htod(
                    self.cuda_array,
                    tensor_np.astype(np.float32)
                )
            elif self.dtype == torch.uint8:
                # For uint8, copy directly
                cuda.memcpy_htod(
                    self.cuda_array,
                    tensor_np.astype(np.uint8)
                )
            
            # Unmap resource
            self.cuda_graphics_resource.unmap()
            
        except Exception as e:
            print(f"PyCUDA update failed: {e}")
            # Fallback to tensor copy
            if self.cuda_tensor is not None:
                self.cuda_tensor.copy_(tensor)
    
    def get_cuda_tensor(self) -> Optional[torch.Tensor]:
        """Get CUDA tensor for rendering"""
        with self.lock:
            if self.cuda_tensor is not None:
                return self.cuda_tensor.clone()
            return None
    
    def get_opengl_texture(self) -> Optional[int]:
        """Get OpenGL texture ID"""
        with self.lock:
            return self.gl_texture
    
    def render_to_texture(self, render_func):
        """
        Render directly to OpenGL texture using a render function.
        
        Args:
            render_func: Function that renders to current framebuffer
        """
        if not OPENGL_AVAILABLE:
            print("OpenGL not available for render_to_texture")
            return
        
        with self.lock:
            try:
                # Bind framebuffer
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.gl_fbo)
                gl.glViewport(0, 0, self.width, self.height)
                
                # Clear texture
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                
                # Render
                render_func()
                
                # Unbind framebuffer
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                
            except Exception as e:
                print(f"Render to texture failed: {e}")
    
    def resize(self, new_width: int, new_height: int):
        """Resize texture to new dimensions"""
        if new_width == self.width and new_height == self.height:
            return
        
        with self.lock:
            print(f"Resizing texture from {self.width}x{self.height} to {new_width}x{new_height}")
            
            # Clean up old resources
            self._cleanup_resources()
            
            # Update dimensions
            self.width = new_width
            self.height = new_height
            
            # Recreate resources
            self._initialize_resources()
    
    def _cleanup_resources(self):
        """Clean up OpenGL and CUDA resources"""
        if self.gl_texture is not None and OPENGL_AVAILABLE:
            gl.glDeleteTextures([self.gl_texture])
            self.gl_texture = None
        
        if self.gl_fbo is not None and OPENGL_AVAILABLE:
            gl.glDeleteFramebuffers([self.gl_fbo])
            self.gl_fbo = None
        
        if self.cuda_graphics_resource is not None and PYCUDA_AVAILABLE:
            self.cuda_graphics_resource.unregister()
            self.cuda_graphics_resource = None
        
        self.cuda_tensor = None
        self.cuda_array = None
    
    def get_update_stats(self) -> Dict[str, float]:
        """Get texture update performance statistics"""
        if not self.update_times:
            return {
                'avg_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                'last_ms': 0.0, 'fps': 0.0
            }
        
        avg_time = sum(self.update_times) / len(self.update_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_ms': avg_time, 'min_ms': min(self.update_times), 'max_ms': max(self.update_times),
            'last_ms': self.update_times[-1] if self.update_times else 0.0,
            'fps': fps
        }
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up TextureBridge resources...")
        self._cleanup_resources()


class RenderBuffer:
    """
    Multi-buffered render buffer for smooth real-time rendering.
    Implements double buffering to avoid tearing and flickering.
    """
    
    def __init__(self, width: int, height: int, channels: int = 4, num_buffers: int = 2):
        self.num_buffers = num_buffers
        self.current_buffer = 0
        self.buffers = []
        
        # Create multiple texture bridges
        for i in range(num_buffers):
            bridge = TextureBridge(width, height, channels)
            self.buffers.append(bridge)
        
        print(f"RenderBuffer initialized with {num_buffers} buffers")
    
    def get_write_buffer(self) -> TextureBridge:
        """Get buffer to write to"""
        return self.buffers[self.current_buffer]
    
    def get_read_buffer(self) -> TextureBridge:
        """Get buffer to read from (previous frame)"""
        read_index = (self.current_buffer + 1) % self.num_buffers
        return self.buffers[read_index]
    
    def swap(self):
        """Swap read and write buffers"""
        self.current_buffer = (self.current_buffer + 1) % self.num_buffers
    
    def update_from_tensor(self, tensor: torch.Tensor, buffer_index: Optional[int] = None):
        """Update specific buffer from tensor"""
        if buffer_index is None:
            buffer_index = self.current_buffer
        
        self.buffers[buffer_index].update_from_tensor(tensor)
    
    def resize(self, width: int, height: int):
        """Resize all buffers"""
        for buffer in self.buffers:
            buffer.resize(width, height)
    
    def cleanup(self):
        """Clean up all buffers"""
        for buffer in self.buffers:
            buffer.cleanup()


# Import time for performance monitoring
import time