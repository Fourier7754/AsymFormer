import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import shutil
import zipfile
import requests
from tqdm import tqdm
from PIL import Image
try:
    from transformers import AutoModelForDepthEstimation
except Exception as e:
    print(f"WARNING: transformers.AutoModelForDepthEstimation could not be imported ({e}). Depth simulation will use heuristic fallback.")
    AutoModelForDepthEstimation = None

# --- Mirror Configuration ---
# Set to True to use mirrors for GitHub and Model Weights
USE_MIRROR = True
# GitHub Mirror Prefix (e.g. https://ghproxy.com/ or https://mirror.ghproxy.com/)
GITHUB_PROXY = "https://mirror.ghproxy.com/"

def download_with_progress(url, dest_path):
    """Downloads a file with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error while downloading {url}: {e}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.split('/')[-1],
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def load_midas_safely(use_mirror=USE_MIRROR):
    """
    Safely loads MiDaS model, handling download with progress bar and mirrors.
    """
    # Check for user-specified local hub directory
    custom_hub_dirs = [
        '/root/autodl-tmp/torch_hub/hub',
        os.path.expanduser('~/autodl-tmp/torch_hub/hub'),
        'autodl-tmp/torch_hub/hub'
    ]
    
    hub_dir = torch.hub.get_dir()
    for d in custom_hub_dirs:
        if os.path.exists(d):
            hub_dir = d
            print(f"Found custom torch hub directory: {hub_dir}")
            break

    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir, exist_ok=True)
        
    # Define local path for the repo
    # torch.hub uses 'author_repo_branch' format usually, but let's stick to what torch.hub expects.
    # When loading from github, it downloads to hub_dir/author_repo_branch
    # For "intel-isl/MiDaS", it's "intel-isl_MiDaS_master" (if branch is master, default)
    # or just "intel-isl_MiDaS_master".
    repo_dir = os.path.join(hub_dir, 'intel-isl_MiDaS_master')
    
    # 1. Ensure Repo is Downloaded
    if not os.path.exists(os.path.join(repo_dir, 'hubconf.py')):
        print(f"MiDaS repository not found. Downloading to {repo_dir}...")
        
        # We download to a temp zip
        zip_path = os.path.join(hub_dir, 'midas_master.zip')
        
        # URL for the master branch zip
        zip_url = "https://github.com/intel-isl/MiDaS/archive/master.zip"
        if use_mirror:
            zip_url = GITHUB_PROXY + zip_url
            print(f"Using mirror: {zip_url}")
            
        try:
            download_with_progress(zip_url, zip_path)
            
            # Verify file size (Repo zip should be > 10KB)
            file_size = os.path.getsize(zip_path)
            if file_size < 10000: # 10KB
                with open(zip_path, 'r', errors='ignore') as f:
                    content_head = f.read(200)
                raise RuntimeError(f"Downloaded file is too small ({file_size} bytes). Likely an error page. Content head: {content_head}")
            
            # Unzip
            print("Extracting...")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(hub_dir)
            else:
                 raise FileNotFoundError(f"Zip file disappeared: {zip_path}")
            
            # Rename folder (zip usually extracts to MiDaS-master)
            extracted_dir = os.path.join(hub_dir, 'MiDaS-master')
            if os.path.exists(extracted_dir):
                # Ensure target directory is clean before renaming
                if os.path.exists(repo_dir):
                    shutil.rmtree(repo_dir)
                os.rename(extracted_dir, repo_dir)
            else:
                # Sometimes it might be named differently? List dirs to check
                print(f"WARNING: Expected {extracted_dir} not found after extraction.")
                print(f"Dirs in {hub_dir}: {os.listdir(hub_dir)}")
                
            # Cleanup
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("Repository downloaded successfully.")
            
        except Exception as e:
            print(f"Failed to download repository: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            # Clean up repo_dir if it was created but empty/corrupt
            if os.path.exists(repo_dir):
                # Only remove if it doesn't have hubconf.py (broken)
                if not os.path.exists(os.path.join(repo_dir, 'hubconf.py')):
                    shutil.rmtree(repo_dir)
            
            # Fallback to standard torch.hub (might hang)
            print("Falling back to standard torch.hub.load...")
            return torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)

    # 2. Monkeypatch load_state_dict_from_url to use mirror for weights
    original_load = torch.hub.load_state_dict_from_url
    
    def patched_load(url, *args, **kwargs):
        # Check local file first
        filename = os.path.basename(url)
        # Checkpoints are usually in hub_dir/checkpoints
        local_path = os.path.join(hub_dir, 'checkpoints', filename)
        
        if os.path.exists(local_path):
             print(f"Loading weights from local file: {local_path}")
             return torch.load(local_path, map_location='cpu')
             
        if use_mirror and "github.com" in url:
            new_url = GITHUB_PROXY + url
            print(f"Intercepted weight download. Using mirror: {new_url}")
            # torch.hub.load_state_dict_from_url handles check_hash, progress, etc.
            # We just swap the URL.
            # Note: The filename is derived from URL. Mirror might change it? 
            # usually torch.hub uses the filename from the URL path. 
            # ghproxy URLs: https://ghproxy.com/https://github.com/.../file.pt
            # The filename extraction might be tricky if not handled.
            # But let's try passing the mirror URL directly.
            kwargs['map_location'] = 'cpu'
            return original_load(new_url, *args, **kwargs)
        
        kwargs['map_location'] = 'cpu'
        return original_load(url, *args, **kwargs)
        
    # Apply patch temporarily
    torch.hub.load_state_dict_from_url = patched_load
    try:
        # Load from local repo
        print("Loading MiDaS model from local cache...")
        model = torch.hub.load(repo_dir, "MiDaS_small", source='local', trust_repo=True)
    finally:
        # Restore original
        torch.hub.load_state_dict_from_url = original_load
        
    return model

try:
    import scipy.io
except ImportError:
    scipy = None

class SimulateDepth(nn.Module):
    """
    Simulates a depth map from an RGB image.
    Hybrid Approach: Combines lightweight depth estimation (Depth Anything V2 Small) with edge/intensity details.
    
    If model cannot be loaded, falls back to edge+intensity heuristic.
    """
    def __init__(self, target_mean=2.8424503515351494, target_std=0.9932836506164299, use_midas=True, offline_mode=False):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std
        
        # --- Depth Model Setup (Depth Anything V2) ---
        self.depth_model = None
        
        # If offline_mode is True, we SKIP loading the depth model.
        # This prevents network access and memory usage when we already have pre-computed depth maps.
        if offline_mode:
            if use_midas: # Only print if user intended to use it
                 print("Offline mode enabled: Skipping Depth Model loading (using pre-computed depth maps).")
        elif use_midas and AutoModelForDepthEstimation is not None:
            try:
                print("Loading Depth Anything V2 Small for depth simulation...")
                # Load from Hugging Face Hub
                self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
                self.depth_model.eval()
                # Freeze
                for param in self.depth_model.parameters():
                    param.requires_grad = False
                    
                # Normalization (ImageNet)
                self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                
            except Exception as e:
                print(f"WARNING: Could not load Depth Anything V2 ({e}). Falling back to heuristic only.")
                self.depth_model = None
        elif use_midas and AutoModelForDepthEstimation is None:
             print("WARNING: transformers library not found. Falling back to heuristic.")

        # --- Sobel/Heuristic Setup (Structure Preservation) ---
        # Register as buffers so they move to GPU with .to()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def forward(self, rgb_tensor):
        """
        Args:
            rgb_tensor (Tensor): RGB image tensor of shape (C, H, W) or (B, C, H, W).
                                 Assumed to be in [0, 1].
        Returns:
            depth_tensor (Tensor): Simulated depth map (1, H, W) or (B, 1, H, W).
        """
        x = rgb_tensor
        # Ensure input is 4D
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        B, C, H, W = x.shape
        
        # --- 1. Edge / Intensity Component (High Freq) ---
        # Convert to Grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply Sobel
        gray_padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        edge_x = F.conv2d(gray_padded, self.sobel_x)
        edge_y = F.conv2d(gray_padded, self.sobel_y)
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        
        # Raw heuristic depth
        edge_threshold = 0.3 # Threshold to filter out weak texture edges
        edge_clean = F.relu(edge_mag - edge_threshold)
        
        # Normalize edges to 0-1 for mixing
        e_max = edge_clean.amax(dim=[1, 2, 3], keepdim=True)
        edge_norm = edge_clean / (e_max + 1e-6)

        # --- 2. Deep Learning Component (Depth Anything V2) ---
        if self.depth_model is not None:
            # Depth Anything V2 uses 518x518 default
            # Resize
            if self.norm_mean.device != x.device:
                self.norm_mean = self.norm_mean.to(x.device)
                self.norm_std = self.norm_std.to(x.device)
            
            # Resize to 518x518
            x_resized = F.interpolate(x, size=(518, 518), mode='bicubic', align_corners=False)
            
            # Normalize
            input_batch = (x_resized - self.norm_mean) / self.norm_std
            
            with torch.no_grad():
                outputs = self.depth_model(input_batch)
                prediction = outputs.predicted_depth # Output: (B, 518, 518)
                
            # Resize back to original H, W
            prediction = F.interpolate(prediction.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            
            # Normalize prediction to 0-1 range per image
            # Depth Anything outputs relative depth (disparity-like). High = Close.
            # We want High = Far (Depth). So we invert.
            pred_min = prediction.amin(dim=[1, 2, 3], keepdim=True)
            pred_max = prediction.amax(dim=[1, 2, 3], keepdim=True)
            pred_norm = (prediction - pred_min) / (pred_max - pred_min + 1e-6)
            pred_norm = 1.0 - pred_norm # Invert: 0=Close, 1=Far
            
            # Fusion:
            # Depth Anything V2 is very high quality, so we rely on it more.
            # However, preserving some edge mixing as per original design.
            raw_depth = 0.85 * pred_norm + 0.15 * edge_norm
            
        else:
            # Fallback
            raw_depth = edge_norm

        
        # --- Final Normalization to Target Mean/Std ---
        curr_mean = raw_depth.mean(dim=[1, 2, 3], keepdim=True)
        curr_std = raw_depth.std(dim=[1, 2, 3], keepdim=True)
        
        sim_depth = (raw_depth - curr_mean) / (curr_std + 1e-6) * self.target_std + self.target_mean
        
        if rgb_tensor.dim() == 3:
            return sim_depth.squeeze(0)
        return sim_depth

class FlatImageFolder(Dataset):
    """
    Handles ImageNet validation set where images are all in one folder, not separated by class.
    Uses ground truth file and meta.mat if provided to map images to classes.
    """
    def __init__(self, root, gt_file=None, meta_file=None, transform=None):
        self.root = root
        self.transform = transform
        
        # Filter out MacOS metadata files (._*) and ensure valid extensions
        self.images = sorted([
            os.path.join(root, f) 
            for f in os.listdir(root) 
            if f.lower().endswith(('.jpeg', '.jpg', '.png')) and not f.startswith('._')
        ])
        
        self.targets = []
        if gt_file and meta_file and scipy:
             print(f"Loading Metadata from {meta_file}...")
             meta = scipy.io.loadmat(meta_file, squeeze_me=True)
             synsets = meta['synsets']
             
             self.id_to_wnid = {}
             for s in synsets:
                ilsvrc_id = int(s['ILSVRC2012_ID'])
                wnid = str(s['WNID'])
                self.id_to_wnid[ilsvrc_id] = wnid
                
             valid_wnids = []
             for s in synsets:
                if int(s['ILSVRC2012_ID']) <= 1000:
                    valid_wnids.append(str(s['WNID']))
             valid_wnids.sort()
             self.wnid_to_idx = {wnid: i for i, wnid in enumerate(valid_wnids)}
             
             print(f"Loading Ground Truth from {gt_file}...")
             with open(gt_file, 'r') as f:
                raw_ids = [int(line.strip()) for line in f]
                
             for rid in raw_ids:
                if rid in self.id_to_wnid:
                    wnid = self.id_to_wnid[rid]
                    if wnid in self.wnid_to_idx:
                        self.targets.append(self.wnid_to_idx[wnid])
                    else:
                        self.targets.append(0)
                else:
                    self.targets.append(0)
             
             if len(self.images) != len(self.targets):
                 print(f"WARNING: Number of images ({len(self.images)}) and targets ({len(self.targets)}) do not match!")
                 min_len = min(len(self.images), len(self.targets))
                 self.images = self.images[:min_len]
                 self.targets = self.targets[:min_len]
        else:
             print(f"WARNING: Loading FlatImageFolder from {root} without Ground Truth (or missing scipy).")
             if not scipy: print("Scipy not found.")
             if not gt_file: print("No gt_file provided.")
             self.targets = [0] * len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        target = self.targets[index]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error reading {path}: {e}")
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)

import cv2

class ImageNetRGBD(Dataset):
    def __init__(self, root, split='train', transform=None, depth_root=None, gt_file=None, meta_file=None, input_size=224):
        self.transform = transform
        self.depth_root = depth_root
        self.split = split
        self.input_size = input_size
        self.debug_count = 0 # Counter for debug prints
        
        # Simplified Logic: Exclusively use ImageFolder
        # 1. Check root/split
        # 2. Check root
        
        target_path = os.path.join(root, split)
        
        if os.path.exists(target_path) and os.path.isdir(target_path):
            print(f"Loading ImageNet data from: {target_path}")
            try:
                self.ds = ImageFolder(target_path)
            except FileNotFoundError:
                # This happens if 'val' has no subfolders (flat structure)
                if split == 'val':
                    print("Falling back to FlatImageFolder for validation set (no subdirectories found).")
                    self.ds = FlatImageFolder(target_path, gt_file=gt_file, meta_file=meta_file)
                else:
                    raise
        elif os.path.exists(root) and os.path.isdir(root):
            # Check if root contains 'train'/'val' which implies it's the parent dir
            # and we failed to find the split folder inside it.
            subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if split in subdirs:
                # This should have been caught by target_path check, but maybe some path issue?
                # Retry joining
                retry_path = os.path.join(root, split)
                print(f"Retrying found split folder: {retry_path}")
                try:
                    self.ds = ImageFolder(retry_path)
                except FileNotFoundError:
                    if split == 'val':
                        print("Falling back to FlatImageFolder for validation set.")
                        self.ds = FlatImageFolder(retry_path, gt_file=gt_file, meta_file=meta_file)
                    else:
                        raise
            else:
                # Assume root IS the data folder (e.g. user pointed directly to train folder)
                
                # SAFETY CHECK: If we are looking for 'val', and root contains 'train', 
                # we should NOT load root as val (it's likely the parent dir).
                if split == 'val' and 'train' in subdirs:
                    print(f"CRITICAL WARNING: Attempted to load validation set from {root}, but it appears to be the root directory containing 'train'.")
                    print(f"This would load the entire training set (or parent dir) as validation.")
                    print(f"Expected validation path: {target_path}")
                    raise RuntimeError(f"Validation folder '{split}' not found in {root}. Aborting to prevent loading wrong dataset.")

                print(f"Loading ImageNet data from root: {root}")
                # Optional warning if it looks like a parent folder
                if 'train' in subdirs and 'val' in subdirs:
                     print(f"WARNING: Root {root} contains 'train' and 'val'. You are loading {root} as a single dataset.")
                     print(f"If you intended to load the '{split}' split, please check why '{target_path}' was not found.")
                
                try:
                    self.ds = ImageFolder(root)
                except FileNotFoundError:
                     # Could be flat folder
                     print("Falling back to FlatImageFolder (no subdirectories found).")
                     self.ds = FlatImageFolder(root, gt_file=gt_file, meta_file=meta_file)

        else:
            print(f"CRITICAL ERROR: Data directory not found.")
            print(f"Checked: {target_path}")
            print(f"Checked: {root}")
            if os.path.exists(root):
                print(f"Root exists. Contents: {os.listdir(root)[:10]}...")
            else:
                print("Root does not exist.")
            raise RuntimeError(f"Could not find valid ImageNet dataset at {root}")

    def __getitem__(self, index):
        img, target = self.ds[index]
        
        # Determine path for depth loading
        if hasattr(self.ds, 'samples'):
            path = self.ds.samples[index][0]
        elif hasattr(self.ds, 'images'):
             path = self.ds.images[index]
        else:
             path = None

        depth_tensor = None
        if self.depth_root and path:
            # Construct depth path. Assuming depth_root mirrors root structure.
            # root/train/class/img.jpg -> depth_root/train/class/img.png
            try:
                rel_path = os.path.relpath(path, self.ds.root)
                
                # Fix: Append split name (train/val) to depth_root
                # Structure: depth_root/split/class/img.png
                depth_path = os.path.join(self.depth_root, self.split, os.path.splitext(rel_path)[0] + '.png')
                
                if os.path.exists(depth_path):
                     # Try to open with PIL first to check for metadata (uint8 case)
                     depth_pil = Image.open(depth_path)

                     
                     # Check if it has embedded metadata (our custom uint8 format)
                     if "depth_min" in depth_pil.info and "depth_max" in depth_pil.info:
                         d_min = float(depth_pil.info["depth_min"])
                         d_max = float(depth_pil.info["depth_max"])
                         
                         # Convert to numpy float
                         d_np = np.array(depth_pil).astype(np.float32) / 255.0
                         
                         # Restore range
                         d_restored = d_np * (d_max - d_min) + d_min
                         
                         # Convert back to PIL "I;16" (uint16) or "F" (float32) for downstream transforms?
                         # Transforms expect PIL Image.
                         # Since downstream expects "I;16" and divides by 65535, we should map it back to 0-65535 uint16?
                         # OR we can return the float array directly?
                         # The existing pipeline does: depth_np = np.array(depth_pil).astype(np.float32) / 65535.0
                         # So it expects [0, 1] relative depth in a uint16 container.
                         
                         # BUT wait, our restored depth is absolute (or whatever the model output was).
                         # We need to normalize it to [0, 1] per image for the training loop to work as before.
                         # The training loop does: (depth - mean) / std.
                         
                         # Let's map the restored float depth back to [0, 1] for consistency with the "I;16" path.
                         # Wait, if we just want [0, 1], we already have d_np (which is 0-1 quantized).
                         # But d_np lost the relative scale between images if min/max were different?
                         # No, Depth Anything outputs relative depth. Normalizing per-image is standard.
                         # The `generate_depth.py` normalized per-image to 0-1 before saving.
                         # So `d_np` (0-255) IS the relative depth map.
                         # The `depth_min` / `depth_max` are only needed if we care about the absolute values 
                         # (which Depth Anything doesn't provide anyway, it's relative).
                         
                         # HOWEVER, if `generate_depth.py` saved (val - min) / (max - min),
                         # then `d_np` is already 0-1 normalized (just quantized).
                         # So we can just use `d_np` directly?
                         
                         # Let's double check generate_depth.py logic:
                         # d_norm = (d_np - d_min) / (d_max - d_min)
                         # d_out = (d_norm * 255).astype(np.uint8)
                         
                         # So yes, the image content is already 0-1 normalized relative depth.
                         # We don't strictly need min/max to feed the model if we only care about shape.
                         # But if we want to restore the *original* float distribution (which might be important if
                         # the model output wasn't strictly 0-1?), we use min/max.
                         
                         # In `generate_depth.py`, d_np was the raw output.
                         # For Depth Anything, raw output is affine-invariant depth.
                         
                         # If we assume training code re-normalizes anyway:
                         # train_imagenet.py: 
                         # curr_mean = depth.mean(...)
                         # depth = (depth - curr_mean) / curr_std * target_std + target_mean
                         
                         # This re-normalization makes absolute values irrelevant, only the *distribution* matters.
                         # So using the quantized 0-1 (d_np) is fine!
                         # We just need to load it as a float tensor [0, 1].
                         
                         # Let's pretend it's a 16-bit image scaled to 65535 for compatibility
                         # or just handle it as float.
                         
                         # To minimize changes to the "Transforms" block below (which expects PIL and divides by 65535),
                         # let's convert our 0-255 uint8 to 0-65535 uint16 PIL.
                         
                         depth_u16 = (d_np * 65535).astype(np.uint16)
                         depth_pil = Image.fromarray(depth_u16, mode="I;16")
                         
                     else:
                         # Legacy 16-bit or standard PNG
                         # Reload as 'I;16' if it was opened as something else?
                         # Image.open handles formats automatically.
                         if depth_pil.mode != 'I;16':
                             # If it's standard 8-bit png without metadata
                             depth_pil = depth_pil.convert('I') # or keep as is?
                             # Actually cv2.imread(..., UNCHANGED) was used before.
                             # Let's stick to PIL for everything now.
                             pass
                         
                         # If it's the old 16-bit PNG, it's already good.
                else:
                     # DEBUG: Print path if not found (Always print for now to debug)
                     print(f"DEBUG WARNING: Depth file NOT FOUND: {depth_path}")
                     depth_pil = Image.new("I;16", img.size)
            except Exception as e:
                print(f"Error loading depth for {path}: {e}")
                import traceback
                traceback.print_exc()
                depth_pil = Image.new("I;16", img.size)
        else:
            depth_pil = None

        # Apply Transforms
        # We need to manually handle RandomResizedCrop and Flip to sync with Depth
        if self.split == 'train' and self.depth_root and depth_pil:
             # Manual Augmentation
             # 1. Random Resized Crop
             i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3./4., 4./3.))
             img = transforms.functional.resized_crop(img, i, j, h, w, (self.input_size, self.input_size), Image.BICUBIC)
             depth_pil = transforms.functional.resized_crop(depth_pil, i, j, h, w, (self.input_size, self.input_size), Image.NEAREST)
             
             # 2. Random Horizontal Flip
             if torch.rand(1) < 0.5:
                 img = transforms.functional.hflip(img)
                 depth_pil = transforms.functional.hflip(depth_pil)
             
             # 3. Apply other transforms (ColorJitter, TrivialAugment, ToTensor, Normalize) to RGB only
             # Note: self.transform MUST NOT contain RandomResizedCrop or RandomHorizontalFlip
             if self.transform:
                 img = self.transform(img)
                 
             # 4. Process Depth
             depth_np = np.array(depth_pil).astype(np.float32) / 65535.0
             depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) # (1, H, W)
             
        elif self.split == 'val' and self.depth_root and depth_pil:
             # Validation Transform
             # Resize 256, CenterCrop 224 (Standard ImageNet)
             # Calculate resize based on input_size (0.875 crop)
             resize_size = int(self.input_size / 0.875)
             
             img = transforms.functional.resize(img, resize_size, Image.BICUBIC)
             img = transforms.functional.center_crop(img, self.input_size)
             
             depth_pil = transforms.functional.resize(depth_pil, resize_size, Image.NEAREST)
             depth_pil = transforms.functional.center_crop(depth_pil, self.input_size)
             
             if self.transform:
                  # Ensure transform matches (ToTensor, Normalize)
                  # Usually val_transform includes Resize/Crop, so we should avoid double crop?
                  # Ideally self.transform should only be ToTensor+Normalize if we do crop here.
                  # But for compatibility, let's assume self.transform handles tensor conversion.
                  # We might need to bypass self.transform if it has Resize/Crop.
                  # Let's rely on self.transform for RGB and manually match for Depth?
                  # Actually, exact match for Val is easy: CenterCrop is deterministic.
                  # So we can let self.transform do RGB.
                  pass
                  
             if self.transform:
                 img = transforms.functional.to_tensor(img)
                 img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
             depth_np = np.array(depth_pil).astype(np.float32) / 65535.0
             depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        else:
             # Fallback for no depth or test split
             if self.transform:
                 img = self.transform(img)
             if depth_tensor is None:
                 depth_tensor = torch.zeros((1, self.input_size, self.input_size))
            
        return img, depth_tensor, target

    def __len__(self):
        return len(self.ds)
