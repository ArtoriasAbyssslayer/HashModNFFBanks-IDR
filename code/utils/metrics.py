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


class PSNR_SSIM(Loss):
    default_cfg = {
        'eval_margin_ratio': 1.0,
    }
    def __init__(self, cfg):
        super().__init__([])
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgbs_gt = data_pr['pixel_colors_gt'] # 1,rn,3
        rgbs_pr = data_pr['pixel_colors_nr'] # 1,rn,3
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
        rgbs_pr = rgbs_pr.reshape([h,w,3]).detach().cpu().numpy()
        rgbs_pr=color_map_backward(rgbs_pr)

        rgbs_gt = rgbs_gt.reshape([h,w,3]).detach().cpu().numpy()
        rgbs_gt = color_map_backward(rgbs_gt)

        h, w, _ = rgbs_gt.shape
        h_margin = int(h * (1 - self.cfg['eval_margin_ratio'])) // 2
        w_margin = int(w * (1 - self.cfg['eval_margin_ratio'])) // 2
        rgbs_gt = rgbs_gt[h_margin:h - h_margin, w_margin:w - w_margin]
        rgbs_pr = rgbs_pr[h_margin:h - h_margin, w_margin:w - w_margin]

        psnr = compute_psnr(rgbs_gt,rgbs_pr)
        ssim = structural_similarity(rgbs_gt,rgbs_pr,win_size=11,multichannel=True,data_range=255)
        outputs={
            'psnr_nr': torch.tensor([psnr],dtype=torch.float32),
            'ssim_nr': torch.tensor([ssim],dtype=torch.float32),
        }

        def compute_psnr_prefix(suffix):
            if f'pixel_colors_{suffix}' in data_pr:
                rgbs_other = data_pr[f'pixel_colors_{suffix}'] # 1,rn,3
                # h, w = data_pr['shape']
                rgbs_other = rgbs_other.reshape([h,w,3]).detach().cpu().numpy()
                rgbs_other=color_map_backward(rgbs_other)
                psnr = compute_psnr(rgbs_gt,rgbs_other)
                ssim = structural_similarity(rgbs_gt,rgbs_other,win_size=11,multichannel=True,data_range=255)
                outputs[f'psnr_{suffix}']=torch.tensor([psnr], dtype=torch.float32)
                outputs[f'ssim_{suffix}']=torch.tensor([ssim], dtype=torch.float32)

        # compute_psnr_prefix('nr')
        compute_psnr_prefix('dr')
        compute_psnr_prefix('nr_fine')
        compute_psnr_prefix('dr_fine')
        return outputs

class VisualizeImage(Loss):
    def __init__(self, cfg):
        super().__init__([])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
        def get_img(key):
            rgbs = data_pr[key] # 1,rn,3
            rgbs = rgbs.reshape([h,w,3]).detach().cpu().numpy()
            rgbs = color_map_backward(rgbs)
            return rgbs

        outputs={}
        imgs=[get_img('pixel_colors_gt'), get_img('pixel_colors_nr')]
        if 'pixel_colors_dr' in data_pr: imgs.append(get_img('pixel_colors_dr'))
        if 'pixel_colors_nr_fine' in data_pr: imgs.append(get_img('pixel_colors_nr_fine'))
        if 'pixel_colors_dr_fine' in data_pr: imgs.append(get_img('pixel_colors_dr_fine'))

        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        Path(f'data/vis_val/{model_name}').mkdir(exist_ok=True, parents=True)
        if h<=64 and w<=64:
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.png',concat_images_list(*imgs))
        else:
            imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.jpg', concat_images_list(*imgs))
        return outputs

name2metrics={
    'psnr_ssim': PSNR_SSIM,
    'vis_img': VisualizeImage,
}

def psnr_nr(results):
    return np.mean(results['psnr_nr'])

def psnr_nr_fine(results):
    return np.mean(results['psnr_nr_fine'])

name2key_metrics={
    'psnr_nr': psnr_nr,
    'psnr_nr_fine': psnr_nr_fine,
}