import torch
import math
import random

class ModelEMA:
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        # Update parameters with EMA
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
            
            # Copy buffers (e.g. BN stats) directly
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

from copy import deepcopy

class SyncRandomErasing:
    """
    Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf
    
    Synchronized for RGB and Depth tensors.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, rgb, depth):
        """
        Args:
            rgb (Tensor): Normalized RGB image tensor (C, H, W) or (B, C, H, W).
            depth (Tensor): Normalized Depth image tensor (1, H, W) or (B, 1, H, W).
        Returns:
            Tensor: Erased RGB image.
            Tensor: Erased Depth image.
        """
        if not self.inplace:
            rgb = rgb.clone()
            depth = depth.clone()

        # Handle Batch
        if rgb.dim() == 4:
            B, C, H, W = rgb.shape
            # Loop over batch for now (simple implementation)
            # Vectorized implementation is possible but complex for variable box sizes.
            for i in range(B):
                self._erase_one(rgb[i], depth[i])
            return rgb, depth
        else:
            self._erase_one(rgb, depth)
            return rgb, depth

    def _erase_one(self, rgb, depth):
        if random.random() < self.p:
            # Generate random box
            img_c, img_h, img_w = rgb.shape
            area = img_h * img_w

            for attempt in range(10):
                target_area = random.uniform(*self.scale) * area
                aspect_ratio = random.uniform(*self.ratio)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img_w and h < img_h:
                    x1 = random.randint(0, img_h - h)
                    y1 = random.randint(0, img_w - w)
                    
                    rgb[:, x1:x1+h, y1:y1+w] = self.value
                    depth[:, x1:x1+h, y1:y1+w] = self.value
                    break

