import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import cv2
import random
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Global constants from original file
IMAGE_H = 480
IMAGE_W = 640

class RGBD_Dataset(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None, txt_name='train.txt', use_cutmix=True, use_mixup=False, use_mosaic=True, preload_data=True, target_size=(480, 640)):
        print("Initializing Dataset...")
        """
        Modernized RGBD Dataset with optimized torchvision v2 transforms and CutMix/MixUp/Mosaic.
        """
        self.phase_train = phase_train
        self.data_dir = data_dir
        self.use_cutmix = use_cutmix and phase_train
        self.use_mixup = use_mixup and phase_train
        self.use_mosaic = use_mosaic and phase_train
        self.preload_data = preload_data
        self.target_size = target_size

        
        root = data_dir
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        image_dir = os.path.join(root, 'images')
        depth_dir = os.path.join(root, 'depths')
        mask_dir = os.path.join(root, 'labels')

        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(txt_path, "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            
        self.img_dir_train = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".png") for x in file_names]
        self.label_dir_train = [os.path.join(mask_dir, x + ".png") for x in file_names]

        assert len(self.img_dir_train) == len(self.label_dir_train) == len(self.depth_dir_train)

        # Initialize Transforms
        if transform is None:
            self.transform = self._build_transforms(phase_train)
        else:
            self.transform = transform

        # Preload data if requested
        self.images = []
        self.depths = []
        self.labels = []
        if self.preload_data:
            print(f"Preloading {len(self.img_dir_train)} samples into memory...")
            
            # Use ThreadPoolExecutor for faster I/O on network drives (like GDrive)
            # Reading from GDrive is high latency, so threads help significantly.
            def _load_single_idx(idx):
                return self._read_from_disk(idx)

            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(_load_single_idx, range(len(self.img_dir_train))), 
                                    total=len(self.img_dir_train), 
                                    desc="Loading samples (Multi-threaded)", 
                                    unit="img"))
            
            # Unpack results preserving order
            # results is a list of (img, dep, lab) tuples
            self.images, self.depths, self.labels = [], [], []
            for img, dep, lab in results:
                self.images.append(img)
                self.depths.append(dep)
                self.labels.append(lab)
                
            print("Preloading complete.")

    def _build_transforms(self, is_train):
        h, w = self.target_size
        if is_train:
            return TrainTransform(h=h, w=w)
        else:
            return ValTransform(h=h, w=w)

    def __len__(self):
        return len(self.img_dir_train)

    def _read_from_disk(self, idx, max_retries=3, retry_delay=2.0):
        """
        Read image, depth, and label from disk with retry mechanism.
        
        Args:
            idx: Index of the sample to read
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Seconds to wait between retries (default: 2.0)
        
        Returns:
            Tuple of (image, depth, label) as tv_tensors
        """
        img_path = self.img_dir_train[idx]
        depth_path = self.depth_dir_train[idx]
        label_path = self.label_dir_train[idx]
        
        for attempt in range(max_retries):
            try:
                # Read Image
                # print(f"Reading image: {self.img_dir_train[idx]}")
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise IOError(f"Failed to read image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # H, W, 3

                # Read Depth
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # H, W
                if depth is None:
                    raise IOError(f"Failed to read depth: {depth_path}")
                depth = depth.astype(np.float32)

                # Read Label
                label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED) # H, W
                if label is None:
                    raise IOError(f"Failed to read label: {label_path}")
                
                # Resize raw data if target_size is set (Memory Optimization)
                if hasattr(self, 'target_size') and self.target_size is not None:
                    h, w = self.target_size
                    if image.shape[0] != h or image.shape[1] != w:
                        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

                # Convert to Tensors immediately
                # Image: (3, H, W)
                image = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
                # Depth: (1, H, W) - Treated as Image for geometric transforms (Bilinear)
                depth = tv_tensors.Image(torch.from_numpy(depth).unsqueeze(0))
                # Label: (1, H, W) - Treated as Mask for geometric transforms (Nearest)
                # Cast to int64 to avoid "flip_cpu not implemented for UInt16" error
                label = tv_tensors.Mask(torch.from_numpy(label.astype(np.int64)).unsqueeze(0))

                return image, depth, label
                
            except (IOError, cv2.error, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Failed to read sample {idx} (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to read sample {idx} after {max_retries} attempts: {e}")

    def _load_sample(self, idx):
        if self.preload_data:
            image = self.images[idx]
            depth = self.depths[idx]
            label = self.labels[idx]
        else:
            image, depth, label = self._read_from_disk(idx)
        
        return image, depth, label

    def _load_mosaic(self, idx):
        # Mosaic Augmentation
        h, w = self.target_size
        
        # Center point
        xc = int(random.uniform(w * 0.25, w * 0.75))
        yc = int(random.uniform(h * 0.25, h * 0.75))
        
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        
        # Initialize outputs
        result_img = torch.full((3, h, w), 114, dtype=torch.uint8)
        result_dep = torch.zeros((1, h, w), dtype=torch.float32)
        result_lab = torch.full((1, h, w), 0, dtype=torch.int64)
        
        for i, index in enumerate(indices):
            img, dep, lab = self._load_sample(index)
            
            # Assuming img, dep, lab are (C, H, W) and already resized to target_size 
            # (since _read_from_disk does resize if target_size is set)
            _, h_img, w_img = img.shape
            
            # Define placement coordinates
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w_img, 0), max(yc - h_img, 0), xc, yc
                x1b, y1b, x2b, y2b = w_img - (x2a - x1a), h_img - (y2a - y1a), w_img, h_img
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h_img, 0), min(xc + w_img, w), yc
                x1b, y1b, x2b, y2b = 0, h_img - (y2a - y1a), min(w_img, x2a - x1a), h_img
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w_img, 0), yc, xc, min(yc + h_img, h)
                x1b, y1b, x2b, y2b = w_img - (x2a - x1a), 0, w_img, min(h_img, y2a - y1a)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w_img, w), min(yc + h_img, h)
                x1b, y1b, x2b, y2b = 0, 0, min(w_img, x2a - x1a), min(h_img, y2a - y1a)

            if x2a > x1a and y2a > y1a:
                result_img[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]
                result_dep[:, y1a:y2a, x1a:x2a] = dep[:, y1b:y2b, x1b:x2b]
                result_lab[:, y1a:y2a, x1a:x2a] = lab[:, y1b:y2b, x1b:x2b]

        # Wrap back to tv_tensors
        result_img = tv_tensors.Image(result_img)
        result_dep = tv_tensors.Image(result_dep)
        result_lab = tv_tensors.Mask(result_lab)
        
        return result_img, result_dep, result_lab

    def __getitem__(self, idx):
        # Apply Mosaic or CutMix with probability
        if self.phase_train and self.use_mosaic and random.random() < 0.4:
             image, depth, label = self._load_mosaic(idx)
        else:
             image, depth, label = self._load_sample(idx)

             # Apply CutMix (if not Mosaic)
             if self.phase_train and self.use_cutmix and random.random() < 0.5:
                idx2 = random.randint(0, len(self) - 1)
                image2, depth2, label2 = self._load_sample(idx2)
                image, depth, label = self._apply_cutmix(image, depth, label, image2, depth2, label2)

        # Apply Transforms
        # The transform expects (image, depth, label) and returns dictionary or tuple
        if self.transform:
            sample = self.transform(image, depth, label)
        else:
            sample = {'image': image, 'depth': depth, 'label': label}

        # Compute unique classes (CPU) to avoid GPU sync
        label_t = sample['label']
        u_lbls = torch.unique(label_t)
        u_lbls = u_lbls[(u_lbls > 0) & (u_lbls <= 40)].long() # Filter background and cast
        
        padded_lbls = torch.full((50,), -1, dtype=torch.long)
        n = min(len(u_lbls), 50)
        padded_lbls[:n] = u_lbls[:n]
        
        sample['present_classes'] = padded_lbls
        
        return sample

    def _apply_cutmix(self, img1, dep1, lab1, img2, dep2, lab2):
        """
        Applies CutMix augmentation.
        Pastes a patch from sample 2 onto sample 1.
        """
        # Ensure dimensions match (usually they do in this dataset before resize? 
        # Actually this dataset has raw images, resizing happens in transform.
        # So we should probably resize both first? Or just crop if sizes differ?
        # For simplicity/speed, we assume roughly same size or crop to min.
        
        h = min(img1.shape[1], img2.shape[1])
        w = min(img1.shape[2], img2.shape[2])
        
        # Crop to common size (top-left)
        img1, dep1, lab1 = img1[:, :h, :w], dep1[:, :h, :w], lab1[:, :h, :w]
        img2, dep2, lab2 = img2[:, :h, :w], dep2[:, :h, :w], lab2[:, :h, :w]

        # Generate patch
        lam = np.random.beta(1.0, 1.0)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Paste
        img1[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        dep1[:, bby1:bby2, bbx1:bbx2] = dep2[:, bby1:bby2, bbx1:bbx2]
        lab1[:, bby1:bby2, bbx1:bbx2] = lab2[:, bby1:bby2, bbx1:bbx2]

        return img1, dep1, lab1


class TrainTransform:
    def __init__(self, h=240, w=320):
        print("Initializing Dataset...")
        """
        Lightweight CPU Transform:
        Only performs geometric transforms needed to unify batch sizes (Resize/Crop).
        Heavy pixel-level transforms (Elastic, Color, Norm) are moved to GPU.
        
        Updated for Multi-Scale Training:
        - Removed RandomResize and RandomCrop.
        - Only RandomHorizontalFlip is kept here.
        - Resizing happens per-batch in the training loop.
        """
        # Geometric Transforms (CPU - Lightweight)
        # We only keep Flip here. Resizing is done in training loop for multi-scale batching.
        self.geometric = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10), # Small rotation for indoor scenes
        ])
        
    def __call__(self, image, depth, label):
        # 1. Geometric (All)
        image, depth, label = self.geometric(image, depth, label)
        
        # 2. Type conversions (Prepare for GPU)
        # Convert to float [0, 1] for Image/Depth
        # Keep Label as is (likely Mask) or convert to float if downstream expects it
        
        image = image.float() / 255.0
        depth = depth.float() / 1000.0
        # Note: We keep label as float here to match original logic, 
        # but for GPU transform we'll need to wrap it as Mask to ensure Nearest Interpolation.
        label = label.float() 
        
        # Return Raw (Un-normalized) Tensors
        return {'image': image, 'depth': depth, 'label': label.squeeze(0)}


class GPUAugmentation(torch.nn.Module):
    """
    Heavy augmentations moved to GPU (A100).
    Includes: ElasticTransform, Photometric Distortions, Normalization.
    """
    def __init__(self):
        super().__init__()
        # Elastic Transform (The Bottleneck on CPU)
        # Applied to (Image, Depth, Label)
        self.elastic = v2.RandomApply([v2.ElasticTransform(alpha=50.0)], p=0.2)
        
        # Photometric Transforms (Applied ONLY to Image)
        self.photometric = v2.Compose([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            v2.RandomAutocontrast(p=0.2),
            v2.RandomEqualize(p=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.2),
        ])
        
        # Normalization
        self.norm_img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm_depth = v2.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])

    def forward(self, image, depth, label):
        # inputs are (B, C, H, W) on GPU
        
        # 1. Wrap inputs to ensure correct interpolation mode in v2
        # DataLoader collation strips tv_tensors wrappers, so we re-wrap.
        # Image/Depth -> Bilinear (default for Image)
        # Label -> Nearest (Crucial for Mask)
        
        # image/depth are already float tensors, treated as Image by default
        # label is float tensor, we MUST wrap as Mask to force Nearest interpolation
        # Note: tv_tensors.Mask expects the tensor to be the mask.
        
        # Re-wrapping for v2 geometric transforms
        img_wrap = tv_tensors.Image(image)
        dep_wrap = tv_tensors.Image(depth)
        
        # Label needs to be preserved as discrete values. 
        # Even if it's float, treating as Mask hints v2 to use Nearest.
        lab_wrap = tv_tensors.Mask(label) 
        
        # 2. Elastic Transform (All)
        img_wrap, dep_wrap, lab_wrap = self.elastic(img_wrap, dep_wrap, lab_wrap)
        
        # Unwrap for photometric (or keep wrapped, v2 handles it)
        # 3. Photometric (Image only)
        img_wrap = self.photometric(img_wrap)
        
        # 4. Normalize
        # Normalize expects Tensor or Image
        img_final = self.norm_img(img_wrap)
        dep_final = self.norm_depth(dep_wrap)
        lab_final = lab_wrap # Label doesn't need norm
        
        # Return pure tensors
        return img_final.data, dep_final.data, lab_final.data


class ValTransform:
    def __init__(self, h=240, w=320):
        print("Initializing Dataset...")
        self.resize = v2.Resize(size=(h, w), antialias=True) # Resize to fixed size
        
        self.norm_img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm_depth = v2.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])

    def __call__(self, image, depth, label):
        # Resize
        image, depth, label = self.resize(image, depth, label)
        
        # Normalize
        image = image.float() / 255.0
        depth = depth.float() / 1000.0
        label = label.float()
        
        image = self.norm_img(image)
        depth = self.norm_depth(depth)
        
        return {'image': image, 'depth': depth, 'label': label.squeeze(0)}
