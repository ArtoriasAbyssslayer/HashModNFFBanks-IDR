import torch
import numpy as np
import math
import lpips
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMMeasure
import pytorch3d.loss

# ------------------------------
# MSE
# ------------------------------
def mse(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    value = (image_pred - image_ground_truth) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

# ------------------------------
# PSNR
# ------------------------------
def psnr(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_ground_truth, valid_mask, reduction))

# ------------------------------
# Torch PSNR
# ------------------------------
def torch_psnr(rgb_prediction, rgb_ground_truth, valid_mask=None):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, multichannel=True)
    value = psnr_metric(rgb_prediction, rgb_ground_truth)
    if valid_mask is not None:
        value = value[valid_mask]
    return value

# ------------------------------
# Numpy PSNR
# ------------------------------
def calculate_psnr(img1, img2, mask=None):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if mask is not None:
        mse_value = np.mean(((img1 - img2) ** 2)[mask])
    else:
        mse_value = np.mean((img1 - img2) ** 2)
    if mse_value == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse_value))

# ------------------------------
# SSIM
# ------------------------------
def ssim_similarity(image_pred, image_ground_truth, reduction='elementwise_mean'):
    ssim_metric = SSIMMeasure(data_range=1.0, reduction=reduction)
    img1 = torch.from_numpy(image_pred).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.from_numpy(image_ground_truth).permute(2, 0, 1).unsqueeze(0).float()
    ssim_value = ssim_metric(img1, img2)
    return ssim_value.item()

# ------------------------------
# LPIPS
# ------------------------------
def calculate_lpips(img1, img2):
    lpips_model = lpips.LPIPS(net='alex')
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    lpips_distance = lpips_model(img1_tensor, img2_tensor).item()
    return lpips_distance

# ------------------------------
# PLY Reader
# ------------------------------
def read_ply(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_start = lines.index('end_header\n') + 1
    data = np.loadtxt(lines[data_start:], usecols=(0,1,2))
    return torch.from_numpy(data).float()

# ------------------------------
# Chamfer Distance
# ------------------------------
import torch
from pytorch3d.loss import chamfer_distance

def chamfer_distance_pc(point_cloud1, point_cloud2, sample_size=10000):
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        point_cloud1, point_cloud2: torch.Tensor of shape [N,3] and [M,3]
        sample_size: int, max number of points to sample from each cloud
    
    Returns:
        dist1, dist2: torch.Tensor of shape [1,]
    """
    device = point_cloud1.device
    # Subsample point clouds if too large
    if point_cloud1.shape[0] > sample_size:
        idx = torch.randperm(point_cloud1.shape[0], device=device)[:sample_size]
        point_cloud1 = point_cloud1[idx]
    if point_cloud2.shape[0] > sample_size:
        idx = torch.randperm(point_cloud2.shape[0], device=device)[:sample_size]
        point_cloud2 = point_cloud2[idx]
    
    # Add batch dimension
    point_cloud1 = point_cloud1.unsqueeze(0)
    point_cloud2 = point_cloud2.unsqueeze(0)
    
    # Compute Chamfer distance
    dist1, dist2 = chamfer_distance(point_cloud1, point_cloud2)
    
    return dist1, dist2
