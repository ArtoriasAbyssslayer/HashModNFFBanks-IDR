import torch
import numpy as np
import math
import lpips  # Import the lpips library

# Mean Squared Error
def mse(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    value = (image_pred - image_ground_truth) ** 2

    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value
# Peak Signal to Noise Ratio
def psnr(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_ground_truth, valid_mask, reduction))
# Torch PSNR
def torch_psnr(rgb_prediction, rgb_ground_truth, valid_mask=None):
    from torchmetrics import PeakSignalNoiseRatio as psnr
    value = psnr(rgb_prediction, rgb_ground_truth, multichannel=True)
    if valid_mask is not None:
        value = value[valid_mask]
    return value
# Calculate PSNR
def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse_value = np.mean((img1 - img2) ** 2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse_value == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse_value))
# SSIM Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMMeasure
def ssim_similarity(image_pred, image_ground_truth, reduction='elementwise_mean'):
   
    ssim_metric = SSIMMeasure(data_range=1.0, reduction=reduction)

    img1 = torch.from_numpy(image_pred).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.from_numpy(image_ground_truth).permute(2, 0, 1).unsqueeze(0).float()
    
    ssim_value = ssim_metric(img1 , img2)
    return ssim_value.item()

# Calculate LPIPS
def calculate_lpips(img1, img2):
    # Create an instance of the LPIPS model
    lpips_model = lpips.LPIPS(net='alex')
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    # Calculate the LPIPS distance
    lpips_distance = lpips_model(img1_tensor, img2_tensor).item()
    return lpips_distance
# Simple Function Made to calculate Chamfer Distance between two meshes 
import pytorch3d.loss
def read_ply(file_path):
    with open(file_path, 'r') as ply_file:
        lines = ply_file.readlines()

    # Find the data section in the PLY file
    data_start = lines.index('end_header\n') + 1

    # Read the data into a NumPy array
    data = np.genfromtxt(lines[data_start:], delimiter=' ', dtype=None, names=True)

    return data
def chamfer_distance(point_cloud1, point_cloud2):
    
    """
    Calculate the Chamfer Distance between two point clouds.

    Args:
        point_cloud1 (torch.Tensor): The first point cloud of shape (N, 3).
        point_cloud2 (torch.Tensor): The second point cloud of shape (M, 3).

    Returns:
        (torch.Tensor, torch.Tensor): The Chamfer Distance between the two point clouds,
                                      with shape (1,) for each direction.
    """
    gt_file = 'path_to_gt_dtu_scan.ply'
    point_cloud2 = read_ply(gt_file)
    point_cloud1 = point_cloud1 
    point_cloud2 = point_cloud2  
    # Calculate Chamfer Distance
    dist1, dist2 = pytorch3d.loss.chamfer_distance(point_cloud1, point_cloud2)
    return dist1, dist2
