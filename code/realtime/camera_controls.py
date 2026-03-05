"""
Instant-NGP Style Camera Controls for Real-Time Neural Rendering
Implements WASD + mouse navigation with smooth camera movement
"""

import torch
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
from enum import Enum
from collections import deque
import time

class CameraMode(Enum):
    """Camera control modes"""
    ORBIT = "orbit"
    FPS = "fps"
    TURN_TABLE = "turn_table"


class InstantNGPCamera:
    """
    Instant-NGP style camera with WASD + mouse controls.
    Provides smooth, responsive navigation for 3D scenes.
    """
    
    def __init__(self, width: int = 1024, height: int = 768, fov: float = 60.0,
                 near: float = 0.1, far: float = 100.0):
        # Camera parameters
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = width / height
        
        # Camera state
        self.position = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # Pitch, Yaw, Roll
        self.target_position = self.position.clone()
        self.target_rotation = self.rotation.clone()
        
        # Movement parameters
        self.movement_speed = 2.0
        self.rotation_speed = 0.002
        self.zoom_speed = 0.1
        self.smoothing_factor = 0.15  # Camera smoothing
        
        # Input state
        self.keys_pressed = set()
        self.mouse_buttons = {}
        self.last_mouse_pos = (0, 0)
        self.mouse_delta = (0, 0)
        
        # Mode
        self.mode = CameraMode.FPS
        self.orbit_radius = 3.0
        self.orbit_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
        # Performance tracking
        self.update_times = deque(maxlen=60)
        self.last_update_time = time.time()
        
        # Vectors for calculations
        self.world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        
        print(f"InstantNGPCamera initialized: {width}x{height}, fov={fov}°")
    
    def set_mode(self, mode: CameraMode):
        """Switch camera control mode"""
        self.mode = mode
        print(f"Camera mode changed to: {mode.value}")
        
        if mode == CameraMode.ORBIT:
            self._update_orbit_from_position()
    
    def handle_key_press(self, key: str, pressed: bool):
        """Handle keyboard input"""
        if pressed:
            self.keys_pressed.add(key)
        else:
            self.keys_pressed.discard(key)
    
    def handle_mouse_button(self, button: int, pressed: bool, x: int, y: int):
        """Handle mouse button input"""
        self.mouse_buttons[button] = pressed
        self.last_mouse_pos = (x, y)
    
    def handle_mouse_motion(self, x: int, y: int):
        """Handle mouse motion"""
        if self.mode == CameraMode.FPS:
            # Only rotate if left mouse button is pressed
            if self.mouse_buttons.get(0, False):  # Left button
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                
                # Update rotation
                self.target_rotation[1] += dx * self.rotation_speed  # Yaw
                self.target_rotation[0] += dy * self.rotation_speed  # Pitch
                
                # Clamp pitch
                self.target_rotation[0] = np.clip(
                    self.target_rotation[0], -math.pi/2, math.pi/2
                )
                
        elif self.mode == CameraMode.ORBIT:
            if self.mouse_buttons.get(0, False):  # Left button for rotation
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                
                # Update orbit rotation
                self.target_rotation[1] += dx * self.rotation_speed  # Yaw
                self.target_rotation[0] += dy * self.rotation_speed  # Pitch
                
                # Clamp pitch
                self.target_rotation[0] = np.clip(
                    self.target_rotation[0], -math.pi/2, math.pi/2
                )
                
            elif self.mouse_buttons.get(2, False):  # Right button for pan
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                
                # Pan orbit center
                pan_speed = 0.01
                self.orbit_center[0] += dx * pan_speed
                self.orbit_center[1] -= dy * pan_speed
        
        self.last_mouse_pos = (x, y)
        self.mouse_delta = (x - self.last_mouse_pos[0], y - self.last_mouse_pos[1])
    
    def handle_mouse_wheel(self, delta: float):
        """Handle mouse wheel for zoom"""
        if self.mode == CameraMode.FPS:
            self.movement_speed *= (1.1 ** delta)
            self.movement_speed = max(0.1, min(10.0, self.movement_speed))
        elif self.mode == CameraMode.ORBIT:
            self.orbit_radius *= (0.9 ** delta)
            self.orbit_radius = max(0.5, min(20.0, self.orbit_radius))
    
    def update(self, dt: float):
        """
        Update camera state based on input.
        
        Args:
            dt: Time since last update in seconds
        """
        start_time = time.time()
        
        if self.mode == CameraMode.FPS:
            self._update_fps_mode(dt)
        elif self.mode == CameraMode.ORBIT:
            self._update_orbit_mode(dt)
        elif self.mode == CameraMode.TURN_TABLE:
            self._update_turn_table_mode(dt)
        
        # Smooth camera movement
        self.position.lerp_(self.target_position, self.smoothing_factor)
        self.rotation.lerp_(self.target_rotation, self.smoothing_factor)
        
        # Track performance
        update_time = (time.time() - start_time) * 1000
        self.update_times.append(update_time)
    
    def _update_fps_mode(self, dt: float):
        """Update camera in FPS mode"""
        # Calculate movement direction
        movement = torch.zeros(3, dtype=torch.float32)
        
        # Forward/backward (W/S)
        if 'w' in self.keys_pressed:
            movement[2] -= 1.0
        if 's' in self.keys_pressed:
            movement[2] += 1.0
        
        # Left/right (A/D)
        if 'a' in self.keys_pressed:
            movement[0] -= 1.0
        if 'd' in self.keys_pressed:
            movement[0] += 1.0
        
        # Up/down (Q/E for vertical movement)
        if 'q' in self.keys_pressed:
            movement[1] -= 1.0
        if 'e' in self.keys_pressed:
            movement[1] += 1.0
        
        # Apply movement speed and normalize
        if movement.norm() > 0:
            movement = movement / movement.norm() * self.movement_speed * dt
            movement = self.rotate_vector(movement, self.rotation)
            self.target_position += movement
        
        # Speed boost with Shift
        if 'shift' in self.keys_pressed:
            self.movement_speed = min(10.0, self.movement_speed * 1.1)
        else:
            self.movement_speed = max(2.0, self.movement_speed * 0.9)
    
    def _update_orbit_mode(self, dt: float):
        """Update camera in orbit mode"""
        # Calculate orbit position from spherical coordinates
        pitch = self.target_rotation[0]
        yaw = self.target_rotation[1]
        
        # Convert spherical to Cartesian
        x = self.orbit_radius * math.cos(pitch) * math.sin(yaw)
        y = self.orbit_radius * math.sin(pitch)
        z = self.orbit_radius * math.cos(pitch) * math.cos(yaw)
        
        self.target_position = self.orbit_center + torch.tensor([x, y, z])
        
        # Point camera toward orbit center
        direction = (self.orbit_center - self.position)
        direction[1] = 0  # Keep camera level
        direction = direction / direction.norm()
        self.target_rotation[1] = math.atan2(direction[0], direction[2])
    
    def _update_turn_table_mode(self, dt: float):
        """Update camera in turn-table mode"""
        # Auto-rotate around Y axis
        self.target_rotation[1] += dt * 0.5  # Slow rotation
        
        # Calculate position from turn-table
        pitch = self.target_rotation[0]
        yaw = self.target_rotation[1]
        
        x = self.orbit_radius * math.cos(pitch) * math.sin(yaw)
        y = self.orbit_radius * math.sin(pitch)
        z = self.orbit_radius * math.cos(pitch) * math.cos(yaw)
        
        self.target_position = self.orbit_center + torch.tensor([x, y, z])
    
    def _update_orbit_from_position(self):
        """Update orbit parameters from current position"""
        direction = self.position - self.orbit_center
        self.orbit_radius = direction.norm()
        
        # Calculate spherical coordinates
        self.orbit_radius = direction.norm()
        self.target_rotation[0] = math.asin(direction[1] / self.orbit_radius)
        self.target_rotation[1] = math.atan2(direction[0], direction[2])
    
    def rotate_vector(self, vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """
        Rotate a 3D vector by Euler angles.
        
        Args:
            vector: Input vector [3]
            rotation: Euler angles [pitch, yaw, roll] in radians
            
        Returns:
            Rotated vector [3]
        """
        # Create rotation matrices
        pitch, yaw, roll = rotation
        
        # Pitch rotation (around X axis)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        rot_x = torch.tensor([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ], dtype=torch.float32)
        
        # Yaw rotation (around Y axis)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rot_y = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32)
        
        # Roll rotation (around Z axis)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        rot_z = torch.tensor([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Combined rotation: Y * X * Z (note order)
        return rot_y @ (rot_x @ (rot_z @ vector))
    
    def get_view_matrix(self) -> torch.Tensor:
        """
        Get the camera view matrix (4x4).
        
        Returns:
            View matrix [4, 4]
        """
        # Create rotation matrix from Euler angles
        pitch, yaw, roll = self.rotation
        
        # Combined rotation matrix
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        
        # Rotation matrix
        rot_matrix = torch.tensor([
            [cos_y * cos_r, cos_y * sin_r, -sin_y, 0],
            [sin_p * sin_y * cos_r - cos_p * sin_r, 
             sin_p * sin_y * sin_r + cos_p * cos_r, sin_p * cos_y, 0],
            [cos_p * sin_y * cos_r + sin_p * sin_r,
             cos_p * sin_y * sin_r - sin_p * cos_r, cos_p * cos_y, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        # Translation matrix
        trans_matrix = torch.tensor([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        # View matrix = rotation * translation
        return rot_matrix @ trans_matrix
    
    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the camera projection matrix (4x4).
        
        Returns:
            Projection matrix [4, 4]
        """
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        return torch.tensor([
            [f / self.aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), 
             (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)
    
    def get_camera_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Get camera basis vectors.
        
        Returns:
            Dictionary with 'forward', 'right', 'up' vectors
        """
        # Forward vector (looking direction)
        forward = torch.tensor([
            math.sin(self.rotation[1]) * math.cos(self.rotation[0]),
            math.sin(self.rotation[0]),
            math.cos(self.rotation[1]) * math.cos(self.rotation[0])
        ], dtype=torch.float32)
        forward = forward / forward.norm()
        
        # Right vector
        right = torch.tensor([
            math.cos(self.rotation[1]),
            0,
            -math.sin(self.rotation[1])
        ], dtype=torch.float32)
        right = right / right.norm()
        
        # Up vector
        up = torch.cross(right, forward)
        
        return {
            'forward': forward,
            'right': right,
            'up': up
        }
    
    def get_ray_directions(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Generate ray directions for given pixel coordinates.
        
        Args:
            pixel_coords: Pixel coordinates [N, 2] in range [0, width] x [0, height]
            
        Returns:
            Ray directions [N, 3] in world space
        """
        # Convert pixel coordinates to NDC
        x = (2.0 * pixel_coords[:, 0] / self.width) - 1.0
        y = 1.0 - (2.0 * pixel_coords[:, 1] / self.height)
        
        # Apply perspective projection
        fov_rad = math.radians(self.fov)
        tan_half_fov = math.tan(fov_rad / 2.0)
        
        ray_directions = torch.stack([
            x * tan_half_fov * self.aspect,
            y * tan_half_fov,
            -1.0
        ], dim=1)
        
        # Normalize and rotate by camera rotation
        ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
        camera_vectors = self.get_camera_vectors()
        
        # Transform to world space using camera basis
        forward, right, up = camera_vectors['forward'], camera_vectors['right'], camera_vectors['up']
        world_directions = (
            ray_directions[:, 0:1] * right.unsqueeze(0) +
            ray_directions[:, 1:2] * up.unsqueeze(0) +
            ray_directions[:, 2:3] * forward.unsqueeze(0)
        )
        
        return world_directions
    
    def resize(self, width: int, height: int):
        """Handle window resize"""
        self.width = width
        self.height = height
        self.aspect = width / height
        print(f"Camera resized to {width}x{height}")
    
    def reset(self):
        """Reset camera to default position"""
        self.position = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.target_position = self.position.clone()
        self.target_rotation = self.rotation.clone()
        self.orbit_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.orbit_radius = 3.0
        print("Camera reset to default position")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get comprehensive camera information"""
        return {
            'mode': self.mode.value,
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'movement_speed': self.movement_speed,
            'orbit_radius': self.orbit_radius,
            'fov': self.fov,
            'near': self.near,
            'far': self.far,
            'keys_pressed': list(self.keys_pressed),
            'performance': {
                'avg_update_time_ms': sum(self.update_times) / len(self.update_times) if self.update_times else 0,
                'last_update_time': self.last_update_time
            }
        }


