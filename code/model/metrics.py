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

# 