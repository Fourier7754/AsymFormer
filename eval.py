import argparse
import numpy as np
import os

# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import datetime
import cv2
from collections import OrderedDict
import torch.optim
import NYUv2_dataloader as Data
from src.AsymFormer import B0_T
from utils import utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

pth_dir = './model_M1/ckpt_epoch_500.00.pth'
model = B0_T(num_classes=40)

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
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

args = parser.parse_args()

image_w = 640
image_h = 480
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def _load_block_pretrain_weight(model, pretrain_path):
    model_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrain_dict = torch.load(pretrain_path, map_location='cpu')['state_dict']
    else:
        pretrain_dict = torch.load(pretrain_path, map_location='cpu')['state_dict']
    new_state_dict = OrderedDict()
    new_state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    model.load_state_dict(new_state_dict)


# transform
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        label = label.astype(np.int16)
        # Bi-linear
        image = cv2.resize(image, (image_w, image_h), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        label = cv2.resize(label, (image_w, image_h), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float64)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float()}


class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        depth = depth / 1000
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)

        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(
            image)

        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

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
    pth_dir = args.last_ckpt
    print(f"Loading weights from {pth_dir}..."); _load_block_pretrain_weight(model, pth_dir)
    model.eval()
    #model._model_deploy()
    print(f"Using device: {device}"); model.to(device)

    val_data = Data.RGBD_Dataset(transform=DictCompose([scaleNorm(),
                                                        ToTensor(),
                                                        Normalize()]),
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
        for _ in range(10):
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
            pred = model(image, depth)
            if ender:
                ender.record()
            else:
                end_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()
            
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
