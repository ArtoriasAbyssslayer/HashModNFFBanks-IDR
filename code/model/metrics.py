import torch 

# Mean Squared Error
def mse(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    value = (image_pred-image_ground_truth)**2
    
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

# Peak Signal to Noise Ratio
def psnr(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_ground_truth, valid_mask, reduction))

# Torch PSNR
def torch_psnr(rgb_prediction, rgb_ground_truth, valid_mask=None):
    from torchmetrics import PeakSignalNoiseRatio as psnr 
    value = psnr(rgb_prediction, rgb_ground_truth, multichannel=True)
    if valid_mask is not None:
        value = value[valid_mask]
        return value 
    return value
    
# Structural Similarity Index
def ssim(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    from torchmetrics import StructuralSimilarityIndexMeasure as ssim 
    value = ssim(image_pred, image_ground_truth, multichannel=True)
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

""" 
        Some metrics that are used for validation during training 
        - PSNR(Peak Signal to Noise Ration), 
        - SSIM(structural similarity index measure ), 
        - LPIPS(Learned Perceptual Image Patch Similarity) 
    """
def calculate_psnr(img1, img2, mask):
    import numpy as np 
    import math 
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
def ssim(image_pred, image_ground_truth, valid_mask=None, reduction='mean'):
    from torchmetrics import StructuralSimilarityIndexMeasure as ssim 
    value = ssim(image_pred, image_ground_truth, multichannel=True)
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value
def calculate_lpips(img1, img2):
    import lpips
    # Convert the images to tensors
    img1_tensor = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).float()

    # Normalize the image tensors to the range [0, 1]
    img1_tensor = img1_tensor / 255.0
    img2_tensor = img2_tensor / 255.0

    # Create an instance of the LPIPS model
    lpips_model = lpips.LPIPS(net='vgg')

    # Calculate the LPIPS distance
    lpips_distance = lpips_model(img1_tensor, img2_tensor).item()

    return lpips_distance