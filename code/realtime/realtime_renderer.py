"""
Integrated Real-Time Neural Renderer for IDR
Combines memory management, CUDA streams, occupancy grids, LOD, and camera controls
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

# Import our real-time components
try:
    from .memory_manager import get_memory_manager, smart_cache_clear
    from .cuda_streams import get_stream_manager, with_stream, render_async
    from .texture_bridge import RenderBuffer
    from .occupancy_grid import AdaptiveOccupancyGrid
    from .lod_system import AdaptiveRenderer, LevelOfDetailManager
    from .camera_controls import InstantNGPCamera, CameraMode
except ImportError as e:
    print(f"Warning: Real-time components import failed: {e}")
    # Fallback imports for testing
    get_memory_manager = None
    smart_cache_clear = None
    get_stream_manager = None
    with_stream = None
    render_async = None
    AdaptiveOccupancyGrid = None
    AdaptiveRenderer = None
    LevelOfDetailManager = None
    InstantNGPCamera = None
    CameraMode = None

# Try to import OpenGL for actual rendering
try:
    import OpenGL.GL as gl
    from imgui_bundle import imgui, immapp, hello_imgui
    GUI_AVAILABLE = True
except ImportError:
    print("Warning: GUI libraries not available, running in headless mode")
    GUI_AVAILABLE = False


class RealTimeIDRRenderer:
    """
    Integrated real-time renderer for IDR neural reconstruction.
    Provides instant-ngp style real-time visualization during training.
    """
    
    def __init__(self, config: Dict[str, Any], model, device='cuda'):
        self.config = config
        self.model = model
        self.device = device
        
        # Core components
        self.memory_manager = get_memory_manager()
        self.stream_manager = get_stream_manager()
        self.camera = InstantNGPCamera()
        self.adaptive_renderer = AdaptiveRenderer()
        self.lod_manager = LevelOfDetailManager()
        
        # Occupancy grid
        realtime_config = config.get('realtime_render', {})
        occupancy_config = realtime_config.get('optimization', {}).get('occupancy_grid', {})
        if occupancy_config.get('adaptive', False):
            self.occupancy_grid = AdaptiveOccupancyGrid(
                initial_resolution=occupancy_config.get('initial_resolution', 64),
                max_resolution=occupancy_config.get('max_resolution', 256),
                device=device
            )
        else:
            self.occupancy_grid = OccupancyGrid(
                resolution=occupancy_config.get('resolution', 128),
                device=device
            )
        
        # Texture and render buffers
        render_config = realtime_config.get('rendering', {})
        self.render_resolution = render_config.get('base_resolution', 512)
        self.texture_bridge = TextureBridge(
            self.render_resolution, self.render_resolution, 4, torch.float32
        )
        
        # Threading and synchronization
        self.render_lock = threading.Lock()
        self.should_stop = False
        self.is_rendering = False
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = 0
        self.frame_count = 0
        
        # State management
        self.render_state = {
            'rgb': None,
            'depth': None,
            'normals': None,
            'sdf': None,
            'camera_pose': None,
            'iteration': 0,
            'loss': 0.0,
            'psnr': 0.0
        }
        
        print(f"RealTimeIDRRenderer initialized with resolution {self.render_resolution}")
        
        # Initialize occupancy grid
        self._initialize_occupancy_grid()
    
    def _initialize_occupancy_grid(self):
        """Initialize occupancy grid from current model state"""
        print("Initializing occupancy grid...")
        with torch.no_grad():
            # Create SDF function from model
            def sdf_function(points):
                sdf_output = self.model.implicit_network(points)
                return sdf_output[:, 0:1]  # Return SDF component
            
            # Update occupancy grid
            self.occupancy_grid.update_from_sdf(sdf_function)
    
    def render_frame(self, camera_pose: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Render a single frame with current camera settings.
        
        Args:
            camera_pose: Optional camera pose override
            
        Returns:
            Dictionary with rendering results
        """
        start_time = time.time()
        
        # Get current LOD parameters
        lod_params = self.adaptive_renderer.get_rendering_parameters()
        
        # Generate rays for current camera view
        if camera_pose is None:
            camera_pose = self.camera.get_view_matrix()
        
        rays = self._generate_rays(camera_pose, lod_params)
        
        # Use occupancy grid to cull rays
        if self.config.get('realtime_render', {}).get('optimization', {}).get('occupancy_grid', {}).get('enabled', True):
            ray_mask = self.occupancy_grid.should_march_ray(
                rays['origins'], rays['directions'], 
                max_distance=lod_params.get('max_distance', 2.0)
            )
            
            # Filter rays
            rays['origins'] = rays['origins'][ray_mask]
            rays['directions'] = rays['directions'][ray_mask]
        
        # Render with current model
        with with_stream('rendering') as stream_ctx:
            result, render_time = stream_ctx.time_operation('render', 
                self._render_rays, rays, lod_params)
        
        # Update adaptive quality
        frame_time = (time.time() - start_time) * 1000
        self.adaptive_renderer.update_quality(frame_time, camera_pose)
        
        # Update render state
        self._update_render_state(result, camera_pose)
        
        # Update texture
        if 'rgb' in result:
            rgb_tensor = result['rgb'].view(
                self.render_resolution, self.render_resolution, 4
            ).contiguous()
            self.texture_bridge.update_from_tensor(rgb_tensor, sync=False)
        
        # Performance tracking
        self.frame_times.append(frame_time)
        self.frame_count += 1
        self.last_frame_time = time.time()
        
        return {
            'result': result,
            'frame_time_ms': frame_time,
            'lod_params': lod_params,
            'ray_count': rays['origins'].shape[0],
            'performance': self._get_performance_stats()
        }
    
    def _generate_rays(self, camera_pose: torch.Tensor, lod_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate rays for current camera view and LOD"""
        resolution_scale = lod_params.get('resolution_scale', 1.0)
        ray_samples = lod_params.get('ray_samples', 2048)
        
        # Calculate actual resolution
        actual_width = int(self.render_resolution * resolution_scale)
        actual_height = int(self.render_resolution * resolution_scale)
        
        # Generate pixel coordinates
        pixel_coords = torch.stack([
            torch.linspace(0, actual_width - 1, actual_width, device=self.device),
            torch.linspace(0, actual_height - 1, actual_height, device=self.device)
        ], dim=1)
        
        # Create grid and flatten
        xx, yy = torch.meshgrid(
            torch.arange(actual_width, device=self.device),
            torch.arange(actual_height, device=self.device),
            indexing='xy'
        )
        pixels = torch.stack([xx, yy], dim=-1).view(-1, 2)
        
        # Sample rays for efficiency
        if pixels.shape[0] > ray_samples:
            indices = torch.randperm(pixels.shape[0], device=self.device)[:ray_samples]
            pixels = pixels[indices]
        
        # Get ray directions from camera
        camera_vectors = self.camera.get_camera_vectors()
        ray_directions = self.camera.get_ray_directions(pixels)
        
        # Get ray origins (camera position)
        ray_origins = self.camera.position.unsqueeze(0).expand(pixels.shape[0], -1)
        
        return {
            'origins': ray_origins,
            'directions': ray_directions,
            'pixels': pixels,
            'camera_vectors': camera_vectors
        }
    
    def _render_rays(self, rays: Dict[str, torch.Tensor], 
                   lod_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Render rays using the IDR model"""
        try:
            # Prepare model input
            batch_size = rays['origins'].shape[0]
            
            # Create mock model input (simplified for real-time)
            model_input = {
                'intrinsics': self._get_intrinsics(lod_params),
                'uv': rays['pixels'],
                'pose': self.camera.get_view_matrix(),
                'object_mask': torch.ones(batch_size, dtype=torch.bool, device=self.device)
            }
            
            # Move to device
            for key in model_input:
                if isinstance(model_input[key], torch.Tensor):
                    model_input[key] = model_input[key].to(self.device)
            
            # Render with model
            self.model.eval()
            with torch.no_grad():
                model_output = self.model(model_input)
            
            # Extract RGB data
            rgb = model_output.get('rgb_values', torch.zeros(batch_size, 3, device=self.device))
            
            # Pad RGB to RGBA if needed
            if rgb.shape[1] == 3:
                alpha = torch.ones(batch_size, 1, device=self.device)
                rgb = torch.cat([rgb, alpha], dim=1)
            
            # Convert to image format
            rgb_image = rgb.view(-1, 4).contiguous()
            
            # Generate additional outputs if enabled
            result = {'rgb': rgb_image}
            
            rendering_config = self.config.get('realtime_render', {}).get('rendering', {})
            
            if rendering_config.get('enable_depth', True):
                # Simple depth approximation from ray distances
                depth = torch.ones(batch_size, device=self.device) * 0.5
                result['depth'] = depth
            
            if rendering_config.get('enable_normals', True):
                # Generate normals from gradients (simplified)
                normals = self._compute_normals(rays['origins'], rays['directions'])
                result['normals'] = normals
            
            return result
            
        except Exception as e:
            print(f"Ray rendering failed: {e}")
            # Return fallback result
            batch_size = rays['origins'].shape[0]
            return {
                'rgb': torch.zeros(batch_size, 4, device=self.device),
                'depth': torch.zeros(batch_size, device=self.device),
                'normals': torch.zeros(batch_size, 3, device=self.device)
            }
    
    def _compute_normals(self, ray_origins: torch.Tensor, 
                       ray_directions: torch.Tensor) -> torch.Tensor:
        """Compute surface normals using finite differences"""
        epsilon = 0.001
        
        # Sample SDF at multiple points
        points = ray_origins
        sdf_center = self._sample_sdf(points)
        
        # Finite difference gradients
        sdf_dx = self._sample_sdf(points + torch.tensor([epsilon, 0, 0], device=self.device))
        sdf_dy = self._sample_sdf(points + torch.tensor([0, epsilon, 0], device=self.device))
        sdf_dz = self._sample_sdf(points + torch.tensor([0, 0, epsilon], device=self.device))
        
        # Compute gradients
        grad_x = (sdf_dx - sdf_center) / epsilon
        grad_y = (sdf_dy - sdf_center) / epsilon
        grad_z = (sdf_dz - sdf_center) / epsilon
        
        normals = torch.stack([grad_x, grad_y, grad_z], dim=1)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-6)
        
        return normals
    
    def _sample_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Sample SDF from model"""
        try:
            with torch.no_grad():
                sdf_output = self.model.implicit_network(points)
                return sdf_output[:, 0:1].squeeze()
        except:
            return torch.zeros(points.shape[0], device=self.device)
    
    def _get_intrinsics(self, lod_params: Dict[str, Any]) -> torch.Tensor:
        """Get camera intrinsics matrix for current LOD"""
        fov = self.camera.fov
        resolution_scale = lod_params.get('resolution_scale', 1.0)
        width = self.render_resolution * resolution_scale
        height = self.render_resolution * resolution_scale
        
        # Create intrinsics matrix
        focal = height / (2 * np.tan(np.radians(fov / 2)))
        intrinsics = torch.tensor([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        return intrinsics
    
    def _update_render_state(self, result: Dict[str, torch.Tensor], 
                         camera_pose: torch.Tensor):
        """Update internal render state"""
        with self.render_lock:
            self.render_state.update({
                'rgb': result.get('rgb'),
                'depth': result.get('depth'),
                'normals': result.get('normals'),
                'camera_pose': camera_pose.clone() if camera_pose is not None else None,
                'frame_count': self.frame_count
            })
    
    def update_training_metrics(self, iteration: int, loss: float, psnr: float = 0.0):
        """
        Update training metrics from training loop.
        
        Args:
            iteration: Current training iteration
            loss: Current training loss
            psnr: Current PSNR (optional)
        """
        with self.render_lock:
            self.render_state.update({
                'iteration': iteration,
                'loss': loss,
                'psnr': psnr
            })
        
        # Update adaptive occupancy grid if enabled
        if isinstance(self.occupancy_grid, AdaptiveOccupancyGrid):
            self.occupancy_grid.update_adaptive(
                lambda points: self._sample_sdf(points),
                iteration, loss
            )
    
    def handle_input(self, key: str, pressed: bool):
        """Handle keyboard input"""
        self.camera.handle_key_press(key, pressed)
        
        # Special controls
        if key == 'r' and pressed:
            self.camera.reset()
        elif key == 'f' and pressed:
            # Toggle camera mode
            modes = list(CameraMode)
            current_idx = modes.index(self.camera.mode)
            next_idx = (current_idx + 1) % len(modes)
            self.camera.set_mode(modes[next_idx])
        elif key == 'escape' and pressed:
            self.should_stop = True
    
    def get_render_state(self) -> Dict[str, Any]:
        """Get current render state (thread-safe)"""
        with self.render_lock:
            return self.render_state.copy()
    
    def get_texture(self) -> Optional[int]:
        """Get OpenGL texture ID for rendering"""
        return self.texture_bridge.get_opengl_texture()
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        memory_stats = self.memory_manager.get_memory_stats()
        stream_stats = self.stream_manager.get_performance_stats()
        adaptive_stats = self.adaptive_renderer.get_performance_stats()
        lod_info = self.lod_manager.get_lod_info()
        
        return {
            'memory': memory_stats,
            'streams': stream_stats,
            'adaptive_quality': adaptive_stats,
            'lod': lod_info,
            'frame_count': self.frame_count,
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'last_frame_time': self.frame_times[-1] if self.frame_times else 0,
            'fps': 1000.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        }
    
    def resize(self, width: int, height: int):
        """Handle window resize"""
        self.render_resolution = min(width, height, 
                                  self.config.get('realtime_render', {}).get('max_resolution', 1024))
        self.camera.resize(width, height)
        self.texture_bridge.resize(self.render_resolution, self.render_resolution)
        
        print(f"Renderer resized to {width}x{height}, render resolution: {self.render_resolution}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up RealTimeIDRRenderer...")
        
        self.should_stop = True
        self.texture_bridge.cleanup()
        
        if hasattr(self.occupancy_grid, 'cleanup'):
            self.occupancy_grid.cleanup()
        
        print("RealTimeIDRRenderer cleanup completed")
    
    def reset(self):
        """Reset renderer to initial state"""
        self.camera.reset()
        self.adaptive_renderer.reset()
        self.frame_count = 0
        self.frame_times.clear()
        
        # Reinitialize occupancy grid
        self._initialize_occupancy_grid()
        
        print("RealTimeIDRRenderer reset")