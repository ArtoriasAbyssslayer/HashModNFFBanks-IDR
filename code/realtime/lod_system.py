"""
Level-of-Detail (LOD) System for Real-Time Neural Rendering
Implements adaptive quality based on performance and scene complexity
"""

import torch
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
from collections import deque
from enum import Enum


class QualityLevel(Enum):
    """Quality levels for adaptive rendering"""
    ULTRA_LOW = 0.25
    LOW = 0.5
    MEDIUM = 0.75
    HIGH = 1.0
    ULTRA = 1.25


class AdaptiveRenderer:
    """
    Adaptive quality system that adjusts rendering parameters based on:
    - Target frame rate
    - Performance history
    - Scene complexity
    - Camera motion
    """
    
    def __init__(self, target_fps: float = 60.0, min_quality: float = 0.25, 
                 max_quality: float = 1.5, adaptation_speed: float = 0.1):
        # Target performance
        self.target_frame_time = 1000.0 / target_fps  # milliseconds
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.adaptation_speed = adaptation_speed
        
        # Current state
        self.current_quality = 1.0
        self.target_quality = 1.0
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)  # Last 60 frames
        self.quality_history = deque(maxlen=30)
        self.last_adaptation_time = 0
        self.adaptation_interval = 0.5  # seconds
        
        # Motion detection
        self.last_camera_pose = None
        self.camera_motion = deque(maxlen=10)
        self.motion_threshold = 0.01  # radians
        
        # Complexity estimation
        self.scene_complexity = 1.0
        self.complexity_samples = deque(maxlen=30)
        
        print(f"AdaptiveRenderer initialized: target_fps={target_fps}, "
              f"quality_range=[{min_quality:.2f}, {max_quality:.2f}]")
    
    def update_quality(self, frame_time: float, camera_pose: Optional[torch.Tensor] = None):
        """
        Update quality based on frame time and other factors.
        
        Args:
            frame_time: Current frame time in milliseconds
            camera_pose: Current camera pose for motion detection
        """
        current_time = time.time()
        
        # Update performance tracking
        self.frame_times.append(frame_time)
        
        # Update motion detection
        self._update_motion(camera_pose)
        
        # Calculate adaptation factors
        performance_factor = self._calculate_performance_factor()
        motion_factor = self._calculate_motion_factor()
        complexity_factor = self._calculate_complexity_factor()
        
        # Combine factors to determine target quality
        combined_factor = performance_factor * motion_factor * complexity_factor
        self.target_quality = np.clip(
            self.current_quality * combined_factor,
            self.min_quality, self.max_quality
        )
        
        # Smooth adaptation
        if current_time - self.last_adaptation_time > self.adaptation_interval:
            quality_delta = self.target_quality - self.current_quality
            self.current_quality += quality_delta * self.adaptation_speed
            self.current_quality = np.clip(self.current_quality, self.min_quality, self.max_quality)
            
            self.quality_history.append(self.current_quality)
            self.last_adaptation_time = current_time
            
            # Log quality changes
            if abs(quality_delta) > 0.05:
                print(f"Quality adapted: {self.current_quality:.3f} "
                      f"(perf={performance_factor:.3f}, motion={motion_factor:.3f}, "
                      f"complex={complexity_factor:.3f})")
    
    def _calculate_performance_factor(self) -> float:
        """Calculate quality adjustment based on frame time performance"""
        if len(self.frame_times) < 5:
            return 1.0
        
        # Use recent average for stability
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        frame_time_ratio = self.target_frame_time / avg_frame_time
        
        # Map ratio to quality adjustment
        if frame_time_ratio > 1.2:  # Running slower than target
            return 0.9  # Reduce quality
        elif frame_time_ratio < 0.8:  # Running faster than target
            return 1.1  # Increase quality
        else:
            return 1.0  # Maintain quality
    
    def _calculate_motion_factor(self) -> float:
        """Calculate quality adjustment based on camera motion"""
        if len(self.camera_motion) < 3:
            return 1.0
        
        avg_motion = sum(self.camera_motion) / len(self.camera_motion)
        
        # High motion = reduce quality for better responsiveness
        if avg_motion > self.motion_threshold * 2:
            return 0.7
        elif avg_motion > self.motion_threshold:
            return 0.85
        else:
            return 1.0
    
    def _calculate_complexity_factor(self) -> float:
        """Calculate quality adjustment based on scene complexity"""
        return 1.0 / np.sqrt(self.scene_complexity)
    
    def _update_motion(self, camera_pose: Optional[torch.Tensor]):
        """Update camera motion detection"""
        if camera_pose is None or self.last_camera_pose is None:
            self.last_camera_pose = camera_pose
            return
        
        # Calculate rotation difference
        if camera_pose.shape[-1] == 7:  # Quaternion format
            current_quat = camera_pose[..., :4]
            last_quat = self.last_camera_pose[..., :4]
            
            # Compute quaternion difference
            dot_product = torch.sum(current_quat * last_quat, dim=-1)
            angle_diff = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            motion_magnitude = angle_diff.item()
        else:  # Matrix format - simplified
            motion_magnitude = torch.mean(
                torch.abs(camera_pose - self.last_camera_pose)
            ).item()
        
        self.camera_motion.append(motion_magnitude)
        self.last_camera_pose = camera_pose.clone()
    
    def update_scene_complexity(self, complexity_metrics: Dict[str, float]):
        """
        Update scene complexity estimation.
        
        Args:
            complexity_metrics: Dictionary with complexity indicators
                - 'ray_count': Number of rays being traced
                - 'occupied_voxels': Number of occupied voxels
                - 'surface_complexity': Surface detail level
        """
        # Normalize and combine metrics
        complexity_score = 0.0
        
        if 'ray_count' in complexity_metrics:
            complexity_score += complexity_metrics['ray_count'] / 100000.0
        
        if 'occupied_voxels' in complexity_metrics:
            complexity_score += complexity_metrics['occupied_voxels'] / 1000000.0
        
        if 'surface_complexity' in complexity_metrics:
            complexity_score += complexity_metrics['surface_complexity']
        
        # Update with smoothing
        self.complexity_samples.append(complexity_score)
        self.scene_complexity = sum(self.complexity_samples) / len(self.complexity_samples)
        self.scene_complexity = max(0.1, min(10.0, self.scene_complexity))
    
    def get_rendering_parameters(self) -> Dict[str, Any]:
        """
        Get current rendering parameters based on quality level.
        
        Returns:
            Dictionary with adaptive rendering parameters
        """
        quality = self.current_quality
        
        # Base parameters at quality=1.0
        base_params = {
            'resolution_scale': 1.0,
            'ray_samples': 2048,
            'sphere_tracing_iters': 10,
            'secant_steps': 8,
            'occupancy_resolution': 128,
            'enable_shadows': True,
            'enable_reflections': True,
        }
        
        # Adjust parameters based on quality
        params = base_params.copy()
        
        # Resolution scaling
        if quality < 0.5:
            params['resolution_scale'] = quality
        elif quality < 0.75:
            params['resolution_scale'] = 0.5 + (quality - 0.5) * 2
        else:
            params['resolution_scale'] = min(1.0, 0.75 + (quality - 0.75))
        
        # Ray sampling
        params['ray_samples'] = int(1024 * quality)
        params['ray_samples'] = max(256, min(4096, params['ray_samples']))
        
        # Iteration counts
        params['sphere_tracing_iters'] = int(10 * quality)
        params['sphere_tracing_iters'] = max(5, min(20, params['sphere_tracing_iters']))
        
        params['secant_steps'] = int(8 * quality)
        params['secant_steps'] = max(4, min(16, params['secant_steps']))
        
        # Occupancy grid resolution
        params['occupancy_resolution'] = int(64 * quality)
        params['occupancy_resolution'] = max(32, min(256, params['occupancy_resolution']))
        
        # Quality-dependent features
        params['enable_shadows'] = quality > 0.5
        params['enable_reflections'] = quality > 0.75
        params['enable_ambient_occlusion'] = quality > 0.6
        
        return params
    
    def get_quality_level(self) -> QualityLevel:
        """Get current quality level as enum"""
        quality = self.current_quality
        
        if quality <= 0.25:
            return QualityLevel.ULTRA_LOW
        elif quality <= 0.5:
            return QualityLevel.LOW
        elif quality <= 0.75:
            return QualityLevel.MEDIUM
        elif quality <= 1.0:
            return QualityLevel.HIGH
        else:
            return QualityLevel.ULTRA
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and quality statistics"""
        if not self.frame_times:
            return {
                'avg_fps': 0.0, 'min_fps': 0.0, 'max_fps': 0.0,
                'avg_frame_time': 0.0, 'current_quality': self.current_quality,
                'target_quality': self.target_quality
            }
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        min_frame_time = min(self.frame_times)
        max_frame_time = max(self.frame_times)
        
        avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        min_fps = 1000.0 / max_frame_time if max_frame_time > 0 else 0.0
        max_fps = 1000.0 / min_frame_time if min_frame_time > 0 else 0.0
        
        return {
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'avg_frame_time': avg_frame_time,
            'min_frame_time': min_frame_time,
            'max_frame_time': max_frame_time,
            'current_quality': self.current_quality,
            'target_quality': self.target_quality,
            'quality_level': self.get_quality_level().name,
            'scene_complexity': self.scene_complexity,
            'recent_motion': sum(self.camera_motion) / len(self.camera_motion) if self.camera_motion else 0.0
        }
    
    def set_manual_quality(self, quality: float):
        """Set quality manually (disables automatic adaptation)"""
        self.current_quality = np.clip(quality, self.min_quality, self.max_quality)
        self.target_quality = self.current_quality
        print(f"Quality set manually to {self.current_quality:.3f}")
    
    def reset(self):
        """Reset the adaptive renderer state"""
        self.current_quality = 1.0
        self.target_quality = 1.0
        self.frame_times.clear()
        self.quality_history.clear()
        self.camera_motion.clear()
        self.complexity_samples.clear()
        self.scene_complexity = 1.0
        self.last_camera_pose = None
        print("AdaptiveRenderer reset to default state")


class LevelOfDetailManager:
    """
    Manages multiple LOD levels for different rendering components.
    Provides smooth transitions between quality levels.
    """
    
    def __init__(self, num_levels: int = 5):
        self.num_levels = num_levels
        self.lod_levels = []
        self.current_lod = 2  # Middle level
        self.transition_progress = 0.0
        self.transition_speed = 0.1
        
        # Initialize LOD levels
        self._initialize_lod_levels()
        
        print(f"LOD Manager initialized with {num_levels} levels")
    
    def _initialize_lod_levels(self):
        """Initialize LOD level configurations"""
        base_quality = 0.25  # Start at 25% quality
        quality_multiplier = 1.5  # Each level is 50% higher quality
        
        for i in range(self.num_levels):
            quality = base_quality * (quality_multiplier ** i)
            quality = min(quality, 2.0)  # Cap at 200% quality
            
            self.lod_levels.append({
                'level': i,
                'quality': quality,
                'resolution_scale': min(quality, 1.0),
                'ray_samples': int(512 * quality),
                'iterations': max(5, int(10 * quality)),
                'enabled': True
            })
    
    def update_lod(self, target_quality: float):
        """
        Update LOD based on target quality with smooth transitions.
        
        Args:
            target_quality: Target quality level
        """
        # Find target LOD level
        target_lod = 0
        for i, lod in enumerate(self.lod_levels):
            if lod['quality'] <= target_quality:
                target_lod = i
        
        # Smooth transition
        if target_lod != self.current_lod:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.current_lod = target_lod
                self.transition_progress = 0.0
    
    def get_current_lod_params(self) -> Dict[str, Any]:
        """
        Get current LOD parameters with transition blending.
        
        Returns:
            Dictionary with blended LOD parameters
        """
        current_params = self.lod_levels[self.current_lod].copy()
        
        # Apply transition blending
        if self.transition_progress > 0.0 and self.current_lod < len(self.lod_levels) - 1:
            next_params = self.lod_levels[self.current_lod + 1]
            
            # Linear interpolation between levels
            t = self.transition_progress
            current_params['resolution_scale'] = (
                current_params['resolution_scale'] * (1 - t) +
                next_params['resolution_scale'] * t
            )
            current_params['ray_samples'] = int(
                current_params['ray_samples'] * (1 - t) +
                next_params['ray_samples'] * t
            )
        
        return current_params
    
    def get_lod_info(self) -> Dict[str, Any]:
        """Get current LOD information"""
        return {
            'current_level': self.current_lod,
            'num_levels': self.num_levels,
            'transition_progress': self.transition_progress,
            'current_quality': self.lod_levels[self.current_lod]['quality'],
            'params': self.get_current_lod_params()
        }