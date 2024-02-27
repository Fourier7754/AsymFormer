import argparse
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import skimage.transform
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import datetime
import cv2
from collections import OrderedDict
import torch.optim
import NYUv2_dataloader as ACNet_data
from src.AsymFormer import B0_T
from utils import utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

pth_path='./model_M1/ckpt_epoch_500.00.pth'

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default='./data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='./result/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='./model_M1/ckpt_epoch_450.00.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=40, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=True, action='store_true',
                    help='if output image')

args = parser.parse_args()

image_w = 640
image_h = 480
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def _load_block_pretrain_weight(model, pretrain_path):
    pretrain_dict = torch.load(pretrain_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in pretrain_dict.items():
        name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)


# transform
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        label = label.astype(np.int16)
        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)

        image2 = skimage.transform.resize(image, (288, 384), order=1,
                                          mode='reflect', preserve_range=True)

        image3 = skimage.transform.resize(image, (384, 512), order=1,
                                          mode='reflect', preserve_range=True)

        image4 = skimage.transform.resize(image, (576, 768), order=1,
                                          mode='reflect', preserve_range=True)

        image5 = skimage.transform.resize(image, (672, 896), order=1,
                                          mode='reflect', preserve_range=True)

        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        depth2 = skimage.transform.resize(depth, (288, 384), order=0,
                                          mode='reflect', preserve_range=True)

        depth3 = skimage.transform.resize(depth, (384, 512), order=0,
                                          mode='reflect', preserve_range=True)

        depth4 = skimage.transform.resize(depth, (576, 768), order=0,
                                          mode='reflect', preserve_range=True)

        depth5 = skimage.transform.resize(depth, (672, 896), order=0,
                                          mode='reflect', preserve_range=True)

        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label,
                'image2': image2, 'image3': image3, 'image4': image4, 'image5': image5,
                'depth2': depth2, 'depth3': depth3, 'depth4': depth4, 'depth5': depth5}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        image2, image3, image4, image5 = sample['image2'], sample['image3'], sample['image4'], sample['image5']
        depth2, depth3, depth4, depth5 = sample['depth2'], sample['depth3'], sample['depth4'], sample['depth5']

        image = image.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        image3 = image3.transpose((2, 0, 1))
        image4 = image4.transpose((2, 0, 1))
        image5 = image5.transpose((2, 0, 1))

        depth = np.expand_dims(depth, 0).astype(np.float64)
        depth2 = np.expand_dims(depth2, 0).astype(np.float64)
        depth3 = np.expand_dims(depth3, 0).astype(np.float64)
        depth4 = np.expand_dims(depth4, 0).astype(np.float64)
        depth5 = np.expand_dims(depth5, 0).astype(np.float64)

        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'image2': torch.from_numpy(image2).float(),
                'image3': torch.from_numpy(image3).float(),
                'image4': torch.from_numpy(image4).float(),
                'image5': torch.from_numpy(image5).float(),
                'depth2': torch.from_numpy(depth2).float(),
                'depth3': torch.from_numpy(depth3).float(),
                'depth4': torch.from_numpy(depth4).float(),
                'depth5': torch.from_numpy(depth5).float()}


class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image2, image3, image4, image5 = sample['image2'], sample['image3'], sample['image4'], sample['image5']
        depth2, depth3, depth4, depth5 = sample['depth2'], sample['depth3'], sample['depth4'], sample['depth5']

        origin_image = image.clone()
        origin_depth = depth.clone()

        depth = depth/1000
        depth2 = depth2/1000
        depth3 = depth3/1000
        depth4 = depth4/1000
        depth5 = depth5/1000
        
        image = image / 255
        image2 = image2 / 255
        image3 = image3 / 255
        image4 = image4 / 255
        image5 = image5 / 255

        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)

        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)

        image2 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image2)

        image3 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image3)

        image4 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image4)

        image5 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image5)

        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth)
        depth2 = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth2)
        depth3 = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth3)
        depth4 = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth4)
        depth5 = torchvision.transforms.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])(depth5)

        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['image2'] = image2
        sample['image3'] = image3
        sample['image4'] = image4
        sample['image5'] = image5

        sample['depth'] = depth
        sample['depth2'] = depth2
        sample['depth3'] = depth3
        sample['depth4'] = depth4
        sample['depth5'] = depth5

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


def inference():
    model = B0_T(num_classes=40)
    device = torch.device("cuda:0")
    _load_block_pretrain_weight(model, pth_path)
    model.eval()
    model.to(device)

    val_data = ACNet_data.RGBD_Dataset(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                                 ToTensor(),
                                                                                 Normalize()]),
                                       phase_train=False,
                                       data_dir=args.data_dir,
                                       txt_name='test.txt'
                                       )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            origin_image = sample['origin_image'].numpy()
            origin_depth = sample['origin_depth'].numpy()
            image = sample['image'].to(device)
            image4 = sample['image4'].to(device)
            image5 = sample['image5'].to(device)

            depth = sample['depth'].to(device)
            depth4 = sample['depth4'].to(device)
            depth5 = sample['depth5'].to(device)

            label = sample['label'].numpy()

            with torch.no_grad():
                pred1 = model(image, depth)
                pred4 = model(image4, depth4)
                pred5 = model(image5, depth5)

            pred4 = F.interpolate(pred4, size=pred1.shape[-2:], mode='bilinear', align_corners=True)
            pred5 = F.interpolate(pred5, size=pred1.shape[-2:], mode='bilinear', align_corners=True)

            output1 = pred1.squeeze(0).cpu().numpy()
            output4 = pred4.squeeze(0).cpu().numpy()
            output5 = pred5.squeeze(0).cpu().numpy()

            output = output1+output4+output5
            # output = output1  + output3+output4
            output = output.argmax(0) + 1
            acc, pix = accuracy(output, label)

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

            # img = image.cpu().numpy()
            # print('origin iamge: ', type(origin_image))
            if args.visualize:
                visualize_result(origin_image, origin_depth, label - 1, output - 1, batch_idx, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
    print(mAcc.mean())
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    # imageio.imsave(args.output, output.cpu().numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    inference()
