'''
Enhanced Training Script for AsymFormer with V11 Data Augmentation and EMA
Based on train.py with augmentations aligned to train_asymformer_v11.py
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.optim
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from src.AsymFormer import B0_T
import NYUv2_dataloader as Data
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
import random
import numpy as np

try:
    from src.ema import ModelEmaParamsOnly as ModelEmaV2
    HAS_EMA = True
    EMA_IMPL = "params_only"
except Exception:
    try:
        from timm.utils import ModelEmaV2
        HAS_EMA = True
        EMA_IMPL = "timm_v2"
    except ImportError:
        HAS_EMA = False
        EMA_IMPL = "disabled"
        print("Warning: EMA disabled (no implementation found).")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovász extension w.r.t sorted errors.
    See Alg. 1 in the paper.
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Lovász softmax for flattened predictions.
    Args:
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            prob_c = probas[:, 0]
        else:
            prob_c = probas[:, c]
        errors = (fg - prob_c).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch.
    Args:
        probas: [B, C, H, W] Variable, class probabilities at each prediction
        labels: [B, H, W] Tensor, ground truth labels
        ignore: void class labels
    """
    if probas.dim() == 3:
        probas = probas.unsqueeze(0)
    
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Lovász-Softmax loss.
    Args:
        probas: [B, C, H, W] Variable, class probabilities (after softmax)
        labels: [B, H, W] Tensor, ground truth labels
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
    """
    if per_image:
        loss = 0.
        batch_size = probas.size(0)
        for i in range(batch_size):
            prob = probas[i].unsqueeze(0)
            lab = labels[i].unsqueeze(0)
            loss += lovasz_softmax_flat(*flatten_probas(prob, lab, ignore), classes=classes)
        return loss / batch_size
    else:
        vprobas, vlabels = flatten_probas(probas, labels, ignore)
        return lovasz_softmax_flat(vprobas, vlabels, classes=classes)


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss for semantic segmentation.
    """
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] raw logits (before softmax)
            labels: [B, H, W] ground truth labels
        """
        probas = F.softmax(logits, dim=1)
        return lovasz_softmax(probas, labels, classes='present', ignore=self.ignore_index)


parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation (Enhanced)')
parser.add_argument('--data-dir', default='./data', metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 0.01)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=50, type=int,
                    metavar='N', help='save epoch frequency (default: 50)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt-dir', default='./model_enhanced/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument('--v15', action='store_true', default=False,
                    help='Use AsymFormer v1.5 (mHC backbone)')
parser.add_argument('--use-ema', action='store_true', default=False,
                    help='Use EMA (Exponential Moving Average)')
parser.add_argument('--ema-decay', default=0.9999, type=float,
                    help='EMA decay rate (default: 0.9999)')
parser.add_argument('--seed', default=2333, type=int,
                    help='random seed (default: 2333)')
parser.add_argument('--no-mosaic', action='store_true',
                    help='Disable Mosaic Augmentation (default: enabled)')
parser.add_argument('--no-cutmix', action='store_true',
                    help='Disable CutMix Augmentation (default: disabled)')
parser.add_argument('--use-classmix', action='store_true', default=False,
                    help='Enable ClassMix Augmentation (replaces CutMix, default: disabled)')
parser.add_argument('--use-lovasz', action='store_true', default=False,
                    help='Enable Lovász-Softmax Loss (default: disabled)')
parser.add_argument('--lovasz-weight', default=0.5, type=float,
                    help='Weight for Lovász loss (CE + lovasz_weight * lovasz, default: 0.5)')
parser.add_argument('--depth-norm', default='percentile', type=str,
                    choices=['percentile', 'robust_zscore', 'standard'],
                    help='Depth normalization mode: percentile (2-98%%), robust_zscore (MAD), standard (mean/std)')
parser.add_argument('--depth-percentile-low', default=2, type=float,
                    help='Lower percentile for depth clipping (default: 2)')
parser.add_argument('--depth-percentile-high', default=98, type=float,
                    help='Upper percentile for depth clipping (default: 98)')
parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'],
                    help='Optimizer type (default: adamw)')
parser.add_argument('--muon-momentum', type=float, default=0.95,
                    help='Muon momentum (default: 0.95)')
parser.add_argument('--classifier-lr-factor', default=1.0, type=float,
                    help='Learning rate multiplier for classifier/decoder (default: 1.0)')

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
image_w = 640
image_h = 480


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GPUAugmentationV11(nn.Module):
    """
    V11-style Augmentation:
    Uses Relative Depth Normalization with robust percentile-based clipping.
    """
    def __init__(self, depth_norm_mode='percentile', p_low=2, p_high=98):
        super().__init__()
        self.elastic = v2.RandomApply([v2.ElasticTransform(alpha=50.0)], p=0.2)
        
        self.photometric = v2.Compose([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            v2.RandomAutocontrast(p=0.2),
            v2.RandomEqualize(p=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.2),
        ])
        
        self.norm_img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.depth_norm_mode = depth_norm_mode
        self.p_low = p_low
        self.p_high = p_high

    def forward(self, image, depth, label):
        img_wrap = tv_tensors.Image(image)
        dep_wrap = tv_tensors.Image(depth)
        lab_wrap = tv_tensors.Mask(label) 
        
        img_wrap, dep_wrap, lab_wrap = self.elastic(img_wrap, dep_wrap, lab_wrap)
        
        img_wrap = self.photometric(img_wrap)
        
        img_final = self.norm_img(img_wrap)
        
        dep_final = dep_wrap.clone()
        B = dep_final.shape[0]
        
        for i in range(B):
            d = dep_final[i]
            mask = d > 1e-6
            if mask.sum() > 10:
                valid_pixels = d[mask]
                
                if self.depth_norm_mode == 'percentile':
                    v_min = torch.quantile(valid_pixels, self.p_low / 100.0)
                    v_max = torch.quantile(valid_pixels, self.p_high / 100.0)
                    d_clipped = torch.clamp(d, v_min, v_max)
                    mean = (v_min + v_max) / 2
                    std = (v_max - v_min) / 2 + 1e-6
                    dep_final[i] = (d_clipped - mean) / std
                    dep_final[i][~mask] = 0.0
                    
                elif self.depth_norm_mode == 'robust_zscore':
                    median = torch.median(valid_pixels)
                    mad = torch.median(torch.abs(valid_pixels - median)) + 1e-6
                    dep_final[i] = (d - median) / (1.4826 * mad)
                    dep_final[i][~mask] = 0.0
                    
                else:
                    mean = valid_pixels.mean()
                    std = valid_pixels.std() + 1e-6
                    dep_final[i] = (d - mean) / std
                    dep_final[i][~mask] = 0.0

        lab_final = lab_wrap 
        
        return img_final.data, dep_final.data, lab_final.data


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=4,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def save_ckpt_enhanced(ckpt_dir, model, optimizer, model_ema, global_step, epoch, local_count, num_train):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    state = {
        'global_step': global_step,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'local_count': local_count,
        'num_train': num_train
    }
    
    if model_ema is not None:
        state['state_dict_ema'] = model_ema.module.state_dict() if hasattr(model_ema, 'module') else model_ema.state_dict()
    
    filename = f'ckpt_epoch_{float(epoch):.2f}.pth'
    filepath = os.path.join(ckpt_dir, filename)
    torch.save(state, filepath)
    print(f'Checkpoint saved: {filepath}')


def train():
    setup_seed(args.seed)
    
    # Determine which mix augmentation to use
    use_classmix = args.use_classmix
    use_cutmix = not args.no_cutmix and not use_classmix
    
    train_data = Data.RGBD_Dataset(
        phase_train=True, 
        data_dir=args.data_dir,
        use_mosaic=not args.no_mosaic,
        use_cutmix=use_cutmix,
        use_classmix=use_classmix
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    num_train = len(train_data)

    if args.v15:
        print("Using AsymFormer v1.5 (mHC backbone)...")
        try:
            from src.AsymFormer_v1_5 import AsymFormer_v1_5
        except ImportError:
            from AsymFormer_v1_5 import AsymFormer_v1_5
        model = AsymFormer_v1_5(num_classes=40)
    else:
        model = B0_T(num_classes=40)

    CEL_weighted = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    
    lovasz_loss_fn = None
    if args.use_lovasz:
        lovasz_loss_fn = LovaszSoftmaxLoss(ignore_index=-1)
        print(f"Lovász-Softmax Loss enabled (weight: {args.lovasz_weight})")

    model.train()
    model.to(device)
    CEL_weighted.to(device)
    if lovasz_loss_fn is not None:
        lovasz_loss_fn.to(device)
    
    gpu_aug = GPUAugmentationV11(
        depth_norm_mode=args.depth_norm,
        p_low=args.depth_percentile_low,
        p_high=args.depth_percentile_high
    )
    gpu_aug.to(device)
    gpu_aug.train()
    print(f"Depth normalization: {args.depth_norm} (percentile: {args.depth_percentile_low}-{args.depth_percentile_high}%)")
    if use_classmix:
        print("Using ClassMix augmentation (replaces CutMix)")
    elif use_cutmix:
        print("Using CutMix augmentation")

    if args.optimizer == 'muon':
        from src.muon import MuonWithAuxAdam
        
        # Separate parameters for Muon
        # For B0_T, we consider self.Decoder as the classifier/decoder
        muon_params = []
        adam_params = []
        classifier_params = []
        
        # Decoder parameters (High LR if factor > 1)
        for p in model.Decoder.parameters():
            if p.requires_grad:
                classifier_params.append(p)
        
        # Other parameters
        decoder_param_ids = set(id(p) for p in classifier_params)
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in decoder_param_ids:
                continue
            
            # Muon only for 2D+ parameters (weights of Conv/Linear)
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        muon_lr = args.lr * 10.0
        print(f"Initializing MuonWithAuxAdam (Muon LR: {muon_lr:.2e}, Adam LR: {args.lr:.2e})...")
        
        param_groups = [
            dict(params=muon_params, use_muon=True, lr=muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay),
            dict(params=adam_params, use_muon=False, lr=args.lr, weight_decay=args.weight_decay),
            dict(params=classifier_params, use_muon=False, lr=args.lr * args.classifier_lr_factor, weight_decay=args.weight_decay)
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        print(f"Initializing AdamW (LR: {args.lr:.2e}, Classifier Factor: {args.classifier_lr_factor})...")
        # Fallback to AdamW with layer-wise LR if factor != 1.0
        if args.classifier_lr_factor != 1.0:
            classifier_params = list(model.Decoder.parameters())
            classifier_param_ids = set(id(p) for p in classifier_params)
            other_params = [p for p in model.parameters() if id(p) not in classifier_param_ids and p.requires_grad]
            
            param_groups = [
                {'params': other_params, 'lr': args.lr},
                {'params': classifier_params, 'lr': args.lr * args.classifier_lr_factor}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                          weight_decay=args.weight_decay)
    
    global_step = 0

    model_ema = None
    if args.use_ema and HAS_EMA:
        print(f"Initializing EMA ({EMA_IMPL}) with decay {args.ema_decay}...")
        model_ema = ModelEmaV2(model, decay=args.ema_decay)
    elif args.use_ema and not HAS_EMA:
        print("EMA requested but timm is not installed. EMA disabled.")

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
        if model_ema is not None:
            checkpoint = torch.load(args.last_ckpt, map_location=device, weights_only=False)
            if 'state_dict_ema' in checkpoint:
                print("Loading EMA state dict...")
                model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
                if EMA_IMPL == "params_only":
                    model_bufs = dict(model.named_buffers())
                    for n_ema, b_ema in model_ema.module.named_buffers():
                        if n_ema in model_bufs:
                            b = model_bufs[n_ema]
                            if b_ema.shape == b.shape and b_ema.dtype == b.dtype:
                                b_ema.copy_(b)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    for epoch in range(int(args.start_epoch), args.epochs):

        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt_enhanced(args.ckpt_dir, model, optimizer, model_ema, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].to(device)
            
            with torch.no_grad():
                scale = random.uniform(0.5, 2.0)
                target_h, target_w = image_h, image_w
                cur_h, cur_w = image.shape[-2:]
                new_h = int(cur_h * scale)
                new_w = int(cur_w * scale)
                
                if abs(scale - 1.0) > 0.01:
                    image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    depth = F.interpolate(depth, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    if label.ndim == 3:
                        label = label.unsqueeze(1)
                    label = F.interpolate(label.float(), size=(new_h, new_w), mode='nearest')
                
                cur_h, cur_w = image.shape[-2:]
                pad_h = max(0, target_h - cur_h)
                pad_w = max(0, target_w - cur_w)
                
                if pad_h > 0 or pad_w > 0:
                    image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
                    depth = F.pad(depth, (0, pad_w, 0, pad_h), value=0.0)
                    label = F.pad(label, (0, pad_w, 0, pad_h), value=0.0)
                    
                cur_h, cur_w = image.shape[-2:]
                if cur_h > target_h or cur_w > target_w:
                    y = random.randint(0, max(0, cur_h - target_h))
                    x = random.randint(0, max(0, cur_w - target_w))
                    
                    image = image[..., y:y+target_h, x:x+target_w]
                    depth = depth[..., y:y+target_h, x:x+target_w]
                    label = label[..., y:y+target_h, x:x+target_w]
            
            if label.ndim == 4:
                label = label.squeeze(1)
            
            image, depth, label = gpu_aug(image, depth, label.long())
            
            target_scales = [label]

            optimizer.zero_grad()
            out = model(image, depth)
            
            ce_loss = CEL_weighted(out, (target_scales[0] - 1).long())
            
            if lovasz_loss_fn is not None:
                lovasz_loss = lovasz_loss_fn(out, (target_scales[0] - 1).long())
                loss = ce_loss + args.lovasz_weight * lovasz_loss
            else:
                loss = ce_loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

            local_count += image.data.shape[0]
            global_step += 1

            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                ema_status = " | EMA: Active" if model_ema else ""
                lovasz_status = f" | Lovász: {lovasz_loss.item():.4f}" if lovasz_loss_fn else ""
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                print(f"  [Enhanced] Scale: {scale:.2f}{ema_status}{lovasz_status}")
                end_time = time.time()
                last_count = local_count

    save_ckpt_enhanced(args.ckpt_dir, model, optimizer, model_ema, global_step, args.epochs,
              0, num_train)

    print("Training completed ")
    
    print("\nStarting automatic evaluation...")
    import subprocess
    import sys
    
    last_ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_epoch_{float(args.epochs):.2f}.pth")
    if not os.path.exists(last_ckpt_path):
         print(f"Warning: Expected checkpoint {last_ckpt_path} not found. Trying to find any .pth in {args.ckpt_dir}")
         import glob
         ckpts = glob.glob(os.path.join(args.ckpt_dir, "*.pth"))
         if ckpts:
             last_ckpt_path = sorted(ckpts, key=os.path.getmtime)[-1]
             print(f"Using most recent checkpoint: {last_ckpt_path}")
         else:
             print("No checkpoints found. Skipping evaluation.")
             return

    eval_cmd = [
        sys.executable, "eval_enhanced.py",
        "--last-ckpt", last_ckpt_path,
        "--data-dir", args.data_dir,
        "--save-json",
        "--json-path", os.path.join(args.ckpt_dir, "final_eval_result.json"),
        "--depth-norm", args.depth_norm,
        "--depth-percentile-low", str(args.depth_percentile_low),
        "--depth-percentile-high", str(args.depth_percentile_high)
    ]

    if args.v15:
        eval_cmd.extend(["--model-version", "v1.5"])
    else:
        eval_cmd.extend(["--model-version", "v1"])
    
    if args.use_ema and model_ema is not None:
        eval_cmd.append("--use-ema")
    
    try:
        subprocess.check_call(eval_cmd)
        print("Evaluation finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train()
