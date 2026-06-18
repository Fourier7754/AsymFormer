import argparse
import numpy as np
import os

# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datetime
import cv2
from collections import OrderedDict
import torch.optim
import NYUv2_dataloader as Data
from src.AsymFormer import B0_T
from src.AsymFormer_v1_5 import AsymFormer_v1_5
from utils import utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

pth_dir = './model_M1/ckpt_epoch_500.00.pth'

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation (Enhanced - aligned with train_enhanced.py)')
parser.add_argument('--model-version', default='v1', type=str, choices=['v1', 'v1.5'],
                    help='model version to use: v1 (B0_T) / v1.5 (AsymFormer_v1_5)')
parser.add_argument('--data-dir', default='./data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='./result/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='./model/non_local_5173.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=40, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')
parser.add_argument('--save-json', action='store_true', help='save evaluation results to json')
parser.add_argument('--json-path', default='', type=str, help='path to save json result')
parser.add_argument('--use-ema', action='store_true', default=False,
                    help='Use EMA weights from checkpoint')
parser.add_argument('--deploy', action='store_true', default=False,
                    help='Switch model to deploy structure for inference')
parser.add_argument('--depth-norm', default='percentile', type=str,
                    choices=['percentile', 'robust_zscore', 'standard'],
                    help='Depth normalization mode: percentile (2-98%%), robust_zscore (MAD), standard (mean/std)')
parser.add_argument('--depth-percentile-low', default=2, type=float,
                    help='Lower percentile for depth clipping (default: 2)')
parser.add_argument('--depth-percentile-high', default=98, type=float,
                    help='Upper percentile for depth clipping (default: 98)')
parser.add_argument('--multi-scale', action='store_true', default=False,
                    help='Use multi-scale evaluation (scales: 1.0, 1.2, 1.4)')

args = parser.parse_args()

if args.model_version == 'v1.5':
    model = AsymFormer_v1_5(num_classes=args.num_class)
    print("Using AsymFormer v1.5 model")
else:
    model = B0_T(num_classes=args.num_class)
    print("Using AsymFormer v1 model (B0_T)")

image_w = 640
image_h = 480
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def _load_block_pretrain_weight(model, pretrain_path, use_ema=False):
    if torch.cuda.is_available():
        checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    
    if use_ema and 'state_dict_ema' in checkpoint:
        print("Loading EMA weights (state_dict_ema)...")
        pretrain_dict = checkpoint['state_dict_ema']
    else:
        if use_ema:
            print("Warning: --use-ema set but state_dict_ema not found, falling back to state_dict")
        pretrain_dict = checkpoint['state_dict']
    
    model_keys = set(model.state_dict().keys())

    def _maybe_strip_prefix(sd: dict, prefix: str) -> dict:
        if not sd:
            return sd
        prefixed = [k for k in sd.keys() if k.startswith(prefix)]
        if len(prefixed) == 0:
            return sd
        stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        overlap = sum(1 for k in stripped.keys() if k in model_keys)
        if overlap >= max(1, int(0.5 * len(stripped))):
            return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }
        return sd

    pretrain_dict = _maybe_strip_prefix(pretrain_dict, 'module.')
    pretrain_dict = _maybe_strip_prefix(pretrain_dict, 'model.')

    # Convert old SCC module keys to new format for backward compatibility
    converted_dict = OrderedDict()
    for key, value in pretrain_dict.items():
        new_key = key
        # Old: down_sample_X.SCC.conv1.weight -> New: down_sample_X.SCC.conv_bn.0.weight
        # Old: down_sample_X.SCC.bn.* -> New: down_sample_X.SCC.conv_bn.1.*
        if '.SCC.conv1.' in key:
            new_key = key.replace('.SCC.conv1.', '.SCC.conv_bn.0.')
        elif '.SCC.bn.' in key:
            new_key = key.replace('.SCC.bn.', '.SCC.conv_bn.1.')

        if new_key.endswith('.dwconv_main.weight'):
            cand = new_key.replace('.dwconv_main.weight', '.dwconv_main.branch_kxk.weight')
            if cand in model.state_dict():
                new_key = cand
        elif new_key.endswith('.dwconv_main.bias'):
            cand = new_key.replace('.dwconv_main.bias', '.dwconv_main.branch_kxk.bias')
            if cand in model.state_dict():
                new_key = cand
        elif new_key.endswith('.dwconv_aux.weight'):
            cand = new_key.replace('.dwconv_aux.weight', '.dwconv_aux.branch_kxk.weight')
            if cand in model.state_dict():
                new_key = cand
        elif new_key.endswith('.dwconv_aux.bias'):
            cand = new_key.replace('.dwconv_aux.bias', '.dwconv_aux.branch_kxk.bias')
            if cand in model.state_dict():
                new_key = cand

        converted_dict[new_key] = value

    incompatible = model.load_state_dict(converted_dict, strict=False)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    if missing or unexpected:
        print(f"[ckpt] load_state_dict strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
        if len(missing) <= 20 and len(unexpected) <= 20:
            if missing:
                print("  missing_keys:", missing)
            if unexpected:
                print("  unexpected_keys:", unexpected)
        else:
            print("  (too many keys to print)")


# transform
class scaleNorm(object):
    def __init__(self, multi_scale=True):
        self.multi_scale = multi_scale

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        label = label.astype(np.int16)
        # Bi-linear
        image_base = cv2.resize(image, (image_w, image_h), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth_base = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        label_base = cv2.resize(label, (image_w, image_h), cv2.INTER_NEAREST)

        sample_out = {'image': image_base, 'depth': depth_base, 'label': label_base}

        if self.multi_scale:
            # Scales based on MS5_eval.py:
            # image4: 1.2x -> (576, 768)
            # image5: 1.4x -> (672, 896)
            
            # Scale 1.2
            h4, w4 = 576, 768
            image4 = cv2.resize(image, (w4, h4), cv2.INTER_LINEAR)
            depth4 = cv2.resize(depth, (w4, h4), cv2.INTER_NEAREST)
            sample_out['image4'] = image4
            sample_out['depth4'] = depth4
            
            # Scale 1.4
            h5, w5 = 672, 896
            image5 = cv2.resize(image, (w5, h5), cv2.INTER_LINEAR)
            depth5 = cv2.resize(depth, (w5, h5), cv2.INTER_NEAREST)
            sample_out['image5'] = image5
            sample_out['depth5'] = depth5

        return sample_out


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, multi_scale=True):
        self.multi_scale = multi_scale

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float64)
        
        out = {'image': torch.from_numpy(image).float(),
               'depth': torch.from_numpy(depth).float(),
               'label': torch.from_numpy(label).float()}
               
        if self.multi_scale and 'image4' in sample:
            image4 = sample['image4'].transpose((2, 0, 1))
            depth4 = np.expand_dims(sample['depth4'], 0).astype(np.float64)
            out['image4'] = torch.from_numpy(image4).float()
            out['depth4'] = torch.from_numpy(depth4).float()
            
            image5 = sample['image5'].transpose((2, 0, 1))
            depth5 = np.expand_dims(sample['depth5'], 0).astype(np.float64)
            out['image5'] = torch.from_numpy(image5).float()
            out['depth5'] = torch.from_numpy(depth5).float()
            
        return out


class Normalize(object):
    def __init__(self, depth_norm_mode='percentile', p_low=2, p_high=98, multi_scale=True):
        self.depth_norm_mode = depth_norm_mode
        self.p_low = p_low
        self.p_high = p_high
        self.multi_scale = multi_scale

    def _norm_depth(self, depth):
        # Instance normalization - aligned with train_enhanced.py
        mask = depth > 1e-6
        if mask.sum() > 10:
            valid_pixels = depth[mask]
            
            if self.depth_norm_mode == 'percentile':
                v_min = torch.quantile(valid_pixels, self.p_low / 100.0)
                v_max = torch.quantile(valid_pixels, self.p_high / 100.0)
                d_clipped = torch.clamp(depth, v_min, v_max)
                mean = (v_min + v_max) / 2
                std = (v_max - v_min) / 2 + 1e-6
                depth = (d_clipped - mean) / std
                depth[~mask] = 0.0
                
            elif self.depth_norm_mode == 'robust_zscore':
                median = torch.median(valid_pixels)
                mad = torch.median(torch.abs(valid_pixels - median)) + 1e-6
                depth = (depth - median) / (1.4826 * mad)
                depth[~mask] = 0.0
                
            else: # standard
                mean = valid_pixels.mean()
                std = valid_pixels.std() + 1e-6
                depth = (depth - mean) / std
                depth[~mask] = 0.0
        return depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        
        # Base scale
        image = image / 255
        depth = depth / 1000
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = self._norm_depth(depth)

        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth
        
        if self.multi_scale and 'image4' in sample:
            # Scale 4
            image4 = sample['image4'] / 255
            depth4 = sample['depth4'] / 1000
            image4 = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image4)
            depth4 = self._norm_depth(depth4)
            sample['image4'] = image4
            sample['depth4'] = depth4
            
            # Scale 5
            image5 = sample['image5'] / 255
            depth5 = sample['depth5'] / 1000
            image5 = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image5)
            depth5 = self._norm_depth(depth5)
            sample['image5'] = image5
            sample['depth5'] = depth5

        return sample


class DictCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, label):
        # Convert tv_tensors (C, H, W) to numpy (H, W, C) for legacy transforms
        image = image.permute(1, 2, 0).numpy()
        depth = depth.squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        
        sample = {'image': image, 'depth': depth, 'label': label}
        for t in self.transforms:
            sample = t(sample)
        return sample


def visualize_result(img, depth, label, preds, info, args):
    # segmentation
    img = img.squeeze(0).transpose(0, 2, 1)
    dep = depth.squeeze(0).squeeze(0)
    dep = (dep * 255 / dep.max()).astype(np.uint8)
    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(2, 1, 0)
    seg_color = utils.color_label_eval(label)
    # prediction
    pred_color = utils.color_label_eval(preds)

    # aggregate images and save
    im_vis = np.concatenate((img, dep, seg_color, pred_color),
                            axis=1).astype(np.uint8)
    im_vis = im_vis.transpose(2, 1, 0)

    img_name = str(info)
    # print('write check: ', im_vis.dtype)
    cv2.imwrite(os.path.join(args.output,
                             img_name + '.png'), im_vis)


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.time()


def inference():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Eval config | model={args.model_version} | deploy={args.deploy} | use_ema={args.use_ema} | multi_scale={args.multi_scale}")
    pth_dir = args.last_ckpt
    print(f"Loading weights from {pth_dir}..."); _load_block_pretrain_weight(model, pth_dir, use_ema=args.use_ema)

    use_deploy = bool(args.deploy)
    if use_deploy and hasattr(model, 'switch_to_deploy'):
        model.switch_to_deploy()

    model.eval()
    #model._model_deploy()
    print(f"Using device: {device}"); model.to(device)

    val_data = Data.RGBD_Dataset(transform=DictCompose([scaleNorm(multi_scale=args.multi_scale),
                                                        ToTensor(multi_scale=args.multi_scale),
                                                        Normalize(depth_norm_mode=args.depth_norm,
                                                                  p_low=args.depth_percentile_low,
                                                                  p_high=args.depth_percentile_high,
                                                                  multi_scale=args.multi_scale)]),
                                 phase_train=False,
                                 data_dir=args.data_dir,
                                 txt_name='test.txt'
                                 )
    print("Creating DataLoader (this might take time)..."); val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    t = 0
    acc_collect = []
    if torch.cuda.is_available():
        if torch.cuda.is_available(): torch.cuda.synchronize()
        starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else (None, None)
    else:
        starter, ender = None, None
    timings = np.zeros((len(val_loader), 1))
    dummy_rgb = torch.rand([1, 3, 480, 640], device=device)
    dummy_depth = torch.rand([1, 1, 480, 640], device=device)

    with torch.no_grad():
        # Warmup (only for base scale)
        for _ in range(10):
            if use_deploy:
                _ = model(torch.cat([dummy_rgb, dummy_depth], dim=1))
            else:
                _ = model(dummy_rgb, dummy_depth)

        print("Starting inference loop...")
        for batch_idx, sample in enumerate(val_loader):
            origin_image = sample['origin_image'].numpy()
            origin_depth = sample['origin_depth'].numpy()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()

            if starter:
                starter.record()
            else:
                start_time = time.time()
            
            # Inference
            if use_deploy:
                # Deploy mode + Multi-scale
                if args.multi_scale and 'image4' in sample:
                    image4 = sample['image4'].to(device)
                    depth4 = sample['depth4'].to(device)
                    image5 = sample['image5'].to(device)
                    depth5 = sample['depth5'].to(device)
                    
                    pred1 = model(torch.cat([image, depth], dim=1))
                    pred4 = model(torch.cat([image4, depth4], dim=1))
                    pred5 = model(torch.cat([image5, depth5], dim=1))
                    
                    pred4 = F.interpolate(pred4, size=pred1.shape[-2:], mode='bilinear', align_corners=True)
                    pred5 = F.interpolate(pred5, size=pred1.shape[-2:], mode='bilinear', align_corners=True)
                    
                    pred = pred1 + pred4 + pred5
                else:
                    pred = model(torch.cat([image, depth], dim=1))
            else:
                # Standard mode
                if args.multi_scale and 'image4' in sample:
                    image4 = sample['image4'].to(device)
                    depth4 = sample['depth4'].to(device)
                    image5 = sample['image5'].to(device)
                    depth5 = sample['depth5'].to(device)
                    
                    pred1 = model(image, depth)
                    pred4 = model(image4, depth4)
                    pred5 = model(image5, depth5)
                    
                    pred4 = F.interpolate(pred4, size=pred1.shape[-2:], mode='bilinear', align_corners=True)
                    pred5 = F.interpolate(pred5, size=pred1.shape[-2:], mode='bilinear', align_corners=True)
                    
                    pred = pred1 + pred4 + pred5
                else:
                    pred = model(image, depth)

            if ender:
                ender.record()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

            if not ender:
                end_time = time.time()
            
            if starter:
                curr_time = starter.elapsed_time(ender)
            else:
                curr_time = (end_time - start_time) * 1000
            timings[batch_idx] = curr_time

            output = torch.max(pred, 1)[1] + 1
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, label)
            acc_collect.append(acc)
            intersection, union = intersectionAndUnion(output, label, args.num_class)
            acc_meter.update(acc, pix)
            a_m, b_m = macc(output, label, args.num_class)
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
            print('[{}] iter {}, accuracy: {}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          batch_idx, acc))

            if args.visualize:
                visualize_result(origin_image, origin_depth, label - 1, output - 1, batch_idx, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
    print(mAcc.mean())
    print('[Eval Summary]:')
    miou = iou.mean()
    acc_percent = acc_meter.average() * 100
    avg_inference_time = timings.sum() / len(val_loader)
    
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'.format(miou, acc_percent))
    print('平均推理时间：', avg_inference_time)
    
    # Save results to JSON if requested
    if hasattr(args, 'save_json') and args.save_json:
        import json
        result_data = {
            "mIoU": float(miou),
            "Accuracy": float(acc_percent),
            "Avg_Inference_Time_ms": float(avg_inference_time),
            "Class_IoU": iou.tolist(),
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        json_path = os.path.join(os.path.dirname(args.output), 'eval_result.json')
        if args.json_path:
             json_path = args.json_path
             
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"Evaluation results saved to {json_path}")
        
    np.save('SCC_SRM5', np.array(acc_collect))


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    inference()
