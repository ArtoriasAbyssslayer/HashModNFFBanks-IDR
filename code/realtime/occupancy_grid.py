"""
Occupancy Grid Optimization for Real-Time Ray Tracing
Implements sparse voxel grid to skip empty space during sphere tracing
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import math


class OccupancyGrid:
    """
    Sparse voxel occupancy grid for efficient empty space skipping.
    Dramatically reduces sphere tracing iterations by avoiding empty regions.
    """
    
    def __init__(self, resolution: int = 128, bounding_box: Tuple[float, float] = (-1.0, 1.0), 
                 threshold: float = 0.01, device='cuda'):
        self.resolution = resolution
        self.bounding_box = bounding_box
        self.threshold = threshold
        self.device = device
        
        # Voxel grid (1 = occupied, 0 = empty)
        self.grid = torch.zeros(resolution**3, dtype=torch.bool, device=device)
        self.resolution_f = float(resolution)
        
        # Statistics
        self.occupied_voxels = 0
        self.total_voxels = resolution**3
        self.last_update_frame = -1
        
        print(f"OccupancyGrid initialized: {resolution}x{resolution}x{resolution}")
    
    def world_to_voxel(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to voxel coordinates.
        
        Args:
            points: World coordinates [N, 3] in range [bounding_box[0], bounding_box[1]]
            
        Returns:
            Voxel coordinates [N, 3] in range [0, resolution-1]
        """
        # Normalize from world bounds to [0, 1]
        normalized = (points - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        
        # Scale to voxel coordinates
        voxels = (normalized * self.resolution_f).floor().long()
        
        # Clamp to valid range
        voxels = torch.clamp(voxels, 0, self.resolution - 1)
        
        return voxels
    
    def voxel_to_index(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D voxel coordinates to 1D grid index.
        
        Args:
            voxels: Voxel coordinates [N, 3]
            
        Returns:
            Linear indices [N]
        """
        return voxels[:, 0] * self.resolution * self.resolution + \
               voxels[:, 1] * self.resolution + voxels[:, 2]
    
    def index_to_voxel(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert 1D grid indices to 3D voxel coordinates.
        
        Args:
            indices: Linear indices [N]
            
        Returns:
            Voxel coordinates [N, 3]
        """
        x = indices // (self.resolution * self.resolution)
        y = (indices % (self.resolution * self.resolution)) // self.resolution
        z = indices % self.resolution
        
        return torch.stack([x, y, z], dim=1)
    
    def update_from_sdf(self, sdf_network, bounding_box: Optional[Tuple[float, float]] = None):
        """
        Update occupancy grid from SDF network by sampling points on a regular grid.
        
        Args:
            sdf_network: Neural SDF function
            bounding_box: Optional override for bounding box
        """
        if bounding_box is not None:
            self.bounding_box = bounding_box
        
        print("Updating occupancy grid from SDF network...")
        
        # Generate grid points
        points = self._generate_grid_points()
        
        # Evaluate SDF at grid points
        with torch.no_grad():
            sdf_values = sdf_network(points).squeeze()
            
            # Mark voxels as occupied if SDF is within threshold
            self.grid = (sdf_values.abs() < self.threshold)
        
        # Update statistics
        self.occupied_voxels = self.grid.sum().item()
        occupancy_ratio = self.occupied_voxels / self.total_voxels
        
        print(f"Occupancy grid updated: {self.occupied_voxels}/{self.total_voxels} "
              f"({occupancy_ratio:.1%} occupied)")
    
    def _generate_grid_points(self) -> torch.Tensor:
        """Generate 3D grid of points in world coordinates"""
        # Create voxel coordinates
        x = torch.linspace(self.bounding_box[0], self.bounding_box[1], self.resolution)
        y = torch.linspace(self.bounding_box[0], self.bounding_box[1], self.resolution)
        z = torch.linspace(self.bounding_box[0], self.bounding_box[1], self.resolution)
        
        # Create 3D meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Flatten to [N, 3]
        points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        
        return points.to(self.device)
    
    def should_march_ray(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor, 
                       max_distance: float = 2.0) -> torch.Tensor:
        """
        Determine which rays should undergo sphere tracing based on occupancy.
        
        Args:
            ray_origin: Ray origins [N, 3]
            ray_direction: Ray directions [N, 3]
            max_distance: Maximum ray marching distance
            
        Returns:
            Boolean mask [N] where True means ray should be processed
        """
        num_rays = ray_origin.shape[0]
        
        # Sample points along rays
        num_samples = 16  # Reduced samples for efficiency
        t_values = torch.linspace(0, max_distance, num_samples, device=self.device)
        
        # Generate sample points for all rays
        ray_origins_expanded = ray_origin.unsqueeze(1).expand(-1, num_samples, -1)
        ray_directions_expanded = ray_direction.unsqueeze(1).expand(-1, num_samples, -1)
        t_values_expanded = t_values.unsqueeze(0).unsqueeze(-1).expand(num_rays, -1, 1)
        
        sample_points = ray_origins_expanded + ray_directions_expanded * t_values_expanded
        
        # Convert to voxel coordinates and check occupancy
        voxels = self.world_to_voxel(sample_points.view(-1, 3))
        indices = self.voxel_to_index(voxels)
        
        # Check if any sample point is in occupied voxel
        occupied_mask = self.grid[indices].view(num_rays, num_samples)
        rays_hit_occupied = occupied_mask.any(dim=1)
        
        return rays_hit_occupied
    
    def get_occupied_voxels(self) -> torch.Tensor:
        """Get coordinates of all occupied voxels"""
        occupied_indices = self.grid.nonzero(as_tuple=False).squeeze()
        if occupied_indices.dim() == 0:
            occupied_indices = occupied_indices.unsqueeze(0)
        
        return self.index_to_voxel(occupied_indices)
    
    def get_voxel_centers(self, voxels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get world coordinates of voxel centers.
        
        Args:
            voxels: Optional voxel coordinates [N, 3]. If None, returns all voxel centers.
            
        Returns:
            World coordinates of voxel centers [N, 3]
        """
        if voxels is None:
            # Return all voxel centers
            voxels = self._generate_all_voxels()
        
        # Convert to world coordinates
        voxel_size = (self.bounding_box[1] - self.bounding_box[0]) / self.resolution_f
        centers = voxels.float() * voxel_size + voxel_size * 0.5 + self.bounding_box[0]
        
        return centers
    
    def _generate_all_voxels(self) -> torch.Tensor:
        """Generate all voxel coordinates in the grid"""
        x = torch.arange(self.resolution, device=self.device)
        y = torch.arange(self.resolution, device=self.device)
        z = torch.arange(self.resolution, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        return torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
    
    def get_statistics(self) -> dict:
        """Get occupancy grid statistics"""
        occupancy_ratio = self.occupied_voxels / self.total_voxels
        
        return {
            'resolution': self.resolution,
            'total_voxels': self.total_voxels,
            'occupied_voxels': self.occupied_voxels,
            'occupancy_ratio': occupancy_ratio,
            'memory_usage_mb': self.total_voxels * 1 / (1024 * 1024),  # 1 byte per bool
            'bounding_box': self.bounding_box
        }
    
    def clear(self):
        """Clear the occupancy grid"""
        self.grid.zero_()
        self.occupied_voxels = 0
        print("Occupancy grid cleared")
    
    def resize(self, new_resolution: int):
        """Resize the occupancy grid"""
        if new_resolution == self.resolution:
            return
        
        print(f"Resizing occupancy grid from {self.resolution} to {new_resolution}")
        
        self.resolution = new_resolution
        self.resolution_f = float(new_resolution)
        self.total_voxels = new_resolution**3
        
        # Reinitialize grid
        self.grid = torch.zeros(new_resolution**3, dtype=torch.bool, device=self.device)
        self.occupied_voxels = 0
    
    def save_to_file(self, filename: str):
        """Save occupancy grid to file"""
        torch.save({
            'grid': self.grid.cpu(),
            'resolution': self.resolution,
            'bounding_box': self.bounding_box,
            'threshold': self.threshold,
            'occupied_voxels': self.occupied_voxels
        }, filename)
        print(f"Occupancy grid saved to {filename}")
    
    def load_from_file(self, filename: str):
        """Load occupancy grid from file"""
        data = torch.load(filename, map_location=self.device)
        
        self.grid = data['grid'].to(self.device)
        self.resolution = data['resolution']
        self.bounding_box = data['bounding_box']
        self.threshold = data['threshold']
        self.occupied_voxels = data['occupied_voxels']
        self.resolution_f = float(self.resolution)
        self.total_voxels = self.resolution**3
        
        print(f"Occupancy grid loaded from {filename}")


class AdaptiveOccupancyGrid(OccupancyGrid):
    """
    Adaptive occupancy grid that updates dynamically during training.
    Adjusts threshold and resolution based on training progress.
    """
    
    def __init__(self, initial_resolution: int = 64, max_resolution: int = 256,
                 device='cuda'):
        super().__init__(initial_resolution, device=device)
        
        self.initial_resolution = initial_resolution
        self.max_resolution = max_resolution
        self.current_resolution = initial_resolution
        
        # Adaptive parameters
        self.current_threshold = 0.01
        self.min_threshold = 0.001
        self.max_threshold = 0.1
        
        # Training progress tracking
        self.update_frequency = 50  # Update every 50 iterations
        self.update_count = 0
        
    def should_update(self, iteration: int, loss_value: float) -> bool:
        """
        Determine if grid should be updated based on training progress.
        
        Args:
            iteration: Current training iteration
            loss_value: Current loss value
            
        Returns:
            True if grid should be updated
        """
        # Update based on frequency
        if iteration % self.update_frequency == 0:
            # Also consider loss convergence
            if self.update_count == 0 or self._should_adapt_threshold(loss_value):
                return True
        
        return False
    
    def _should_adapt_threshold(self, current_loss: float) -> bool:
        """Determine if threshold should be adapted based on loss"""
        # Adaptive threshold based on training progress
        if hasattr(self, 'last_loss'):
            loss_ratio = current_loss / self.last_loss
            
            # If loss is decreasing significantly, adjust threshold
            if loss_ratio < 0.8:  # 20% reduction
                self.current_threshold = max(self.min_threshold, 
                                        self.current_threshold * 0.9)
                return True
        
        self.last_loss = current_loss
        return False
    
    def update_adaptive(self, sdf_network, iteration: int, loss_value: float):
        """
        Update grid with adaptive resolution and threshold.
        
        Args:
            sdf_network: Neural SDF function
            iteration: Current training iteration
            loss_value: Current loss value
        """
        if not self.should_update(iteration, loss_value):
            return
        
        # Increase resolution as training progresses
        if iteration > 100 and self.current_resolution < self.max_resolution:
            self.current_resolution = min(self.max_resolution, 
                                      self.current_resolution * 2)
            self.resize(self.current_resolution)
        
        # Update grid with current parameters
        self.update_from_sdf(sdf_network)
        self.update_count += 1
        
        print(f"Adaptive grid update #{self.update_count}: "
              f"res={self.current_resolution}, threshold={self.current_threshold:.4f}")


def create_occupancy_grid(config: dict) -> OccupancyGrid:
    """
    Factory function to create occupancy grid from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OccupancyGrid instance
    """
    if config.get('adaptive', False):
        return AdaptiveOccupancyGrid(
            initial_resolution=config.get('initial_resolution', 64),
            max_resolution=config.get('max_resolution', 256),
            device=config.get('device', 'cuda')
        )
    else:
        return OccupancyGrid(
            resolution=config.get('resolution', 128),
            bounding_box=config.get('bounding_box', (-1.0, 1.0)),
            threshold=config.get('threshold', 0.01),
            device=config.get('device', 'cuda')
        )