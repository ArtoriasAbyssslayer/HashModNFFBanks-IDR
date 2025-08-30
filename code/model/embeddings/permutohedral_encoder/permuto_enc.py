import torch 
import permutohedral_encoding as permuto_enc 
import numpy as np 


class PermutohedralEncoder(torch.nn.Module):
    def __init__(self, input_dims=3, num_channels=32, include_input=True, bound=1.0):
        super(PermutohedralEncoder, self).__init__()
        self.pos_dim = input_dims
        self.bound = bound
        
        # More conservative capacity - start small for stability
        # The original capacity was too large and causing memory/stability issues
        self.capacity = 4 * num_channels  # Reduced significantly
        
        # Fewer levels for more stable training with IDR
        # IDR needs smooth gradients, too many levels can cause instability
        self.nr_levels = 4  # Reduced from 6
        self.nr_feat_per_level = 2
        
        # Critical: Scale parameters tuned for IDR's coordinate system
        # IDR typically works with object_bounding_sphere=1.0, so coordinates are in [-1,1]
        # We need scales that capture both coarse and fine details within this range
        
        # For IDR, we want to capture:
        # - Coarse features: whole object structure (~1.0 scale)
        # - Medium features: object parts (~0.5 scale) 
        # - Fine features: surface details (~0.1 scale)
        # - Very fine features: texture details (~0.05 scale)
        
        self.coarsest_scale = bound * 1.5   # Slightly larger than object bounds
        self.finest_scale = bound * 0.02    # Fine enough for surface details
        
        # Use geometric progression for better multi-scale representation
        self.scale_list = np.geomspace(self.coarsest_scale, self.finest_scale, num=self.nr_levels)
        print(f"PermutohedralEncoder scales: {self.scale_list}")
        
        self.include_input = include_input
        self.embeddings_dim = (2 * self.nr_levels + input_dims) if include_input else 2 * self.nr_levels
        
        # Cache for the encoding object to avoid recreation
        self._encoding = None

    def _get_encoding(self):
        """Lazy initialization of encoding object"""
        if self._encoding is None:
            self._encoding = permuto_enc.PermutoEncoding(
                self.pos_dim, 
                self.capacity, 
                self.nr_levels, 
                self.nr_feat_per_level, 
                self.scale_list
            )
        return self._encoding

    def forward(self, inputs):
        # Handle empty batch case - CRITICAL for IDR training stability
        if inputs.shape[0] == 0:
            if self.include_input:
                return torch.zeros(0, self.embeddings_dim, device=inputs.device, dtype=inputs.dtype)
            else:
                return torch.zeros(0, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)
        
        # IDR-specific input processing
        # IDR coordinates are typically in world space, centered around object
        # We need to normalize to [-1, 1] range for permutohedral encoding
        
        # First, clamp extreme values that might occur during ray tracing
        inputs_clamped = torch.clamp(inputs, -self.bound * 2.0, self.bound * 2.0)
        
        # Normalize to [-1, 1] range - this is crucial for permutohedral stability
        inputs_normalized = inputs_clamped / self.bound
        
        # Additional safety clamp after normalization
        inputs_normalized = torch.clamp(inputs_normalized, -1.0, 1.0)
        
        try:
            encoding = self._get_encoding()
            permuto_embeds = encoding(inputs_normalized)
            
            # Ensure output has expected shape
            if permuto_embeds.shape[0] != inputs.shape[0]:
                print(f"Warning: Permutohedral encoding shape mismatch. Input: {inputs.shape}, Output: {permuto_embeds.shape}")
                # Fallback to zeros with correct shape
                permuto_embeds = torch.zeros(inputs.shape[0], 2 * self.nr_levels, 
                                           device=inputs.device, dtype=inputs.dtype)
            
            if self.include_input:
                return torch.cat([inputs, permuto_embeds], -1)
            else:
                return permuto_embeds
                
        except Exception as e:
            print(f"PermutohedralEncoder error with {inputs.shape[0]} points: {e}")
            print(f"Input range: [{inputs.min().item():.6f}, {inputs.max().item():.6f}]")
            print(f"Normalized range: [{inputs_normalized.min().item():.6f}, {inputs_normalized.max().item():.6f}]")
            
            # Robust fallback - return input coordinates with zero embeddings
            batch_size = inputs.shape[0]
            if self.include_input:
                zero_embeds = torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)
                return torch.cat([inputs, zero_embeds], -1)
            else:
                return torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)


class PermutohedralEncoderIDROptimized(torch.nn.Module):
    """
    Optimized version specifically for IDR training dynamics
    """
    def __init__(self, input_dims=3, num_channels=16, include_input=True, bound=1.0):
        super(PermutohedralEncoderIDROptimized, self).__init__()
        self.pos_dim = input_dims
        self.bound = bound
        
        # Very conservative settings optimized for IDR
        self.capacity = 2 * num_channels  # Minimal capacity for stability
        self.nr_levels = 3  # Only 3 levels for maximum stability
        self.nr_feat_per_level = 2
        
        # IDR-optimized scale selection
        # Based on typical IDR object_bounding_sphere=1.0
        # Focus on the most important scales for 3D reconstruction
        
        self.coarsest_scale = bound * 1.0   # Object-level features
        self.finest_scale = bound * 0.1     # Surface detail features
        
        # Manual scale selection for IDR optimization
        if self.nr_levels == 3:
            self.scale_list = np.array([
                bound * 1.0,    # Global object structure
                bound * 0.5,    # Object parts/components  
                bound * 0.1     # Surface details
            ])
        else:
            self.scale_list = np.geomspace(self.coarsest_scale, self.finest_scale, num=self.nr_levels)
        
        print(f"IDR-Optimized PermutohedralEncoder scales: {self.scale_list}")
        
        self.include_input = include_input
        self.embeddings_dim = (2 * self.nr_levels + input_dims) if include_input else 2 * self.nr_levels
        
        # Training phase tracking for adaptive behavior
        self.training_step = 0
        self._encoding = None

    def _get_encoding(self):
        if self._encoding is None:
            self._encoding = permuto_enc.PermutoEncoding(
                self.pos_dim, 
                self.capacity, 
                self.nr_levels, 
                self.nr_feat_per_level, 
                self.scale_list
            )
        return self._encoding

    def forward(self, inputs):
        self.training_step += 1
        
        # Handle empty batch case
        if inputs.shape[0] == 0:
            if self.include_input:
                return torch.zeros(0, self.embeddings_dim, device=inputs.device, dtype=inputs.dtype)
            else:
                return torch.zeros(0, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)
        
        # IDR-specific preprocessing with outlier handling
        # During IDR training, ray tracing can produce extreme coordinates
        
        # Detect and handle outliers (beyond reasonable bounds)
        outlier_mask = torch.abs(inputs).max(dim=1)[0] > (self.bound * 3.0)
        if outlier_mask.any():
            print(f"Warning: {outlier_mask.sum()} outlier points detected, clamping...")
        
        # Progressive normalization strategy
        # Early in training: more aggressive clamping for stability
        # Later in training: less aggressive for fine details
        
        if self.training and self.training_step < 1000:  # Early training
            clamp_factor = 1.5
        else:  # Stable training
            clamp_factor = 2.0
        
        inputs_processed = torch.clamp(inputs, -self.bound * clamp_factor, self.bound * clamp_factor)
        
        # Smooth normalization to [-0.9, 0.9] to avoid boundary effects
        inputs_normalized = inputs_processed / (self.bound * 1.1)
        inputs_normalized = torch.clamp(inputs_normalized, -0.9, 0.9)
        
        try:
            encoding = self._get_encoding()
            permuto_embeds = encoding(inputs_normalized)
            
            # Validate output
            if torch.isnan(permuto_embeds).any() or torch.isinf(permuto_embeds).any():
                print("Warning: NaN/Inf in permutohedral embeddings, using fallback")
                permuto_embeds = torch.zeros(inputs.shape[0], 2 * self.nr_levels, 
                                           device=inputs.device, dtype=inputs.dtype)
            
            if self.include_input:
                return torch.cat([inputs, permuto_embeds], -1)
            else:
                return permuto_embeds
                
        except Exception as e:
            if self.training_step % 100 == 0:  # Only print occasionally to avoid spam
                print(f"PermutohedralEncoder fallback at step {self.training_step}: {e}")
            
            # Return safe fallback
            batch_size = inputs.shape[0]
            if self.include_input:
                zero_embeds = torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)
                return torch.cat([inputs, zero_embeds], -1)
            else:
                return torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)


class PermutohedralEncoderMinimal(torch.nn.Module):
    """
    Minimal version for debugging and ensuring basic functionality
    """
    def __init__(self, input_dims=3, num_channels=8, include_input=True, bound=1.0):
        super(PermutohedralEncoderMinimal, self).__init__()
        self.pos_dim = input_dims
        self.bound = bound
        
        # Absolute minimal settings
        self.capacity = num_channels  # Tiny capacity
        self.nr_levels = 2  # Only 2 levels
        self.nr_feat_per_level = 2
        
        # Simple scale selection
        self.scale_list = np.array([bound * 0.8, bound * 0.2])
        
        self.include_input = include_input
        self.embeddings_dim = (2 * self.nr_levels + input_dims) if include_input else 2 * self.nr_levels
        self._encoding = None
        
        print(f"Minimal PermutohedralEncoder: capacity={self.capacity}, levels={self.nr_levels}, scales={self.scale_list}")

    def _get_encoding(self):
        if self._encoding is None:
            self._encoding = permuto_enc.PermutoEncoding(
                self.pos_dim, 
                self.capacity, 
                self.nr_levels, 
                self.nr_feat_per_level, 
                self.scale_list
            )
        return self._encoding

    def forward(self, inputs):
        if inputs.shape[0] == 0:
            if self.include_input:
                return torch.zeros(0, self.embeddings_dim, device=inputs.device, dtype=inputs.dtype)
            else:
                return torch.zeros(0, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)
        
        # Very simple normalization
        inputs_normalized = torch.clamp(inputs / self.bound, -0.8, 0.8)
        
        try:
            encoding = self._get_encoding()
            permuto_embeds = encoding(inputs_normalized)
            
            if self.include_input:
                return torch.cat([inputs, permuto_embeds], -1)
            else:
                return permuto_embeds
                
        except:
            # Always fallback to safe option
            batch_size = inputs.shape[0]
            if self.include_input:
                return torch.cat([inputs, torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)], -1)
            else:
                return torch.zeros(batch_size, 2 * self.nr_levels, device=inputs.device, dtype=inputs.dtype)