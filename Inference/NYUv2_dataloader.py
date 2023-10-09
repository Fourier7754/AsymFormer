import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
import cv2

image_h = 480
image_w = 640


class RGBD_Dataset(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None, txt_name='train.txt'):
        self.phase_train = phase_train
        self.transform = transform

        root = data_dir
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        """生成图像文件夹路径与标注(mask)文件夹路径"""
        image_dir = os.path.join(root, 'images')
        depth_dir = os.path.join(root, 'depths')
        mask_dir = os.path.join(root, 'labels40')

        """读取图像列表-txt文件放在根目录"""
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        """读取文件名-图像images，深度depths，标签masks
        不同文件具有不同的扩展名"""
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.img_dir_train = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".npy") for x in file_names]
        self.label_dir_train = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        assert (len(self.img_dir_train) == len(self.label_dir_train) and len(self.img_dir_train) == len(
            self.depth_dir_train))

    def __len__(self):
        return len(self.img_dir_train)

    def __getitem__(self, idx):
        img_dir = self.img_dir_train
        depth_dir = self.depth_dir_train
        label_dir = self.label_dir_train

        label = np.load(label_dir[idx])
        depth = np.load(depth_dir[idx])
        image = np.load(img_dir[idx])

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = cv2.resize(image, (image_w, image_h), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        label = cv2.resize(label, (image_w, image_h), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        # Bi-linear
        image = cv2.resize(image, (target_width, target_height), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (target_width, target_height), cv2.INTER_NEAREST)
        label = cv2.resize(label, (target_width, target_height), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255

        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(
            image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        label = label.astype(np.int16)
        h=label.shape[0]
        w=label.shape[1]
        # Generate different label scales
        label3 = cv2.resize(label, (w//4, h//4), cv2.INTER_NEAREST)
        label4 = cv2.resize(label, (w//8, h//8), cv2.INTER_NEAREST)
        label5 = cv2.resize(label, (w//16, h//16), cv2.INTER_NEAREST)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}
