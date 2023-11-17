import torch.utils.data as data
import cv2
import glob
import os
import numpy as np
from collections import defaultdict
from ..common.datasets import random_rotate, scale_image


def get_gaussian_filter(src_size, dst_size):
    downscale_factor = src_size // dst_size
    sigma = (downscale_factor - 1) / 2

    if downscale_factor % 2 == 0:
        kernel_size = downscale_factor + 1
    else:
        kernel_size = downscale_factor

    return cv2.getGaussianKernel(kernel_size, sigma)


class Dataset(data.Dataset):
    def __init__(self, image_dir, image_size, mask_dir=None, **kwargs):
        '''
        :param image_dir:
        :param image_size:
        :param kwargs:  'random_rotation':[angles_in_degree] or 'random_rotation': True
                        'random_scale': (scale_min, scale_max) or None
        '''

        image_files = glob.glob(os.path.join(image_dir, '*.tiff')) + \
                      glob.glob(os.path.join(image_dir, '*.tif')) + \
                      glob.glob(os.path.join(image_dir, '*.png'))

        if mask_dir:
            mask_files = glob.glob(os.path.join(mask_dir, '*.tiff')) + \
                         glob.glob(os.path.join(mask_dir, '*.tif')) + \
                         glob.glob(os.path.join(mask_dir, '*.png'))
            mask_files = set(mask_files)
        else:
            mask_files = None

        print('%d image files' % len(image_files))

        last_img_width = None

        images = []
        masks = []

        for fn in image_files:
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

            # Low-pass filter images.
            if last_img_width != img.shape[1]:
                # Image size changed. Re-create kernel.
                filter_kernel = get_gaussian_filter(img.shape[1], image_size)
                last_img_width = img.shape[1]

            img = cv2.sepFilter2D(img, -1, filter_kernel, filter_kernel)

            img = cv2.resize(img, dsize=(image_size, image_size))
            img = img[None, :, :]  # 1 x H x W
            images.append(img)

            if mask_dir:
                mask_fn = os.path.join(mask_dir, os.path.basename(fn))
                assert mask_fn in mask_files
                mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, dsize=(image_size, image_size))
                mask = mask[None, :, :]  # 1 x H x W
                masks.append(mask)

        self.images = images
        self.masks = masks

        self.first = True  # Used to initialize random number generator for forked workers.
        self.rng = np.random.RandomState()

        self.opts = defaultdict((lambda: None), kwargs)  # Options (default to None if not specified).

    def __getitem__(self, idx):
        if self.first:
            # Make sure each worker has a different RNG.
            self.rng.seed(idx)
            self.first = False

        seed = self.rng.randint((1 << 32) - 1)
        img = self._transform_img(self.images[idx], np.random.RandomState(seed))
        img = (img / 255.0).astype(np.float32)

        if self.masks:
            mask = self._transform_img(self.masks[idx], np.random.RandomState(seed))
            mask = (mask / 255.0).astype(np.float32)
            return img * mask, mask
        else:
            return img

    def _transform_img(self, img, rng):
        random_scale = self.opts['random_scale']
        if random_scale is not None:
            scale_min, scale_max = random_scale
            scale = rng.uniform(scale_min, scale_max)
            img = scale_image(img, scale, border_mode='zero')

        random_rot = self.opts['random_rotation']
        if random_rot:
            if isinstance(random_rot, list):
                angles = random_rot
            else:
                angles = None
            img = random_rotate(img, rng, angles, border_mode='zero')
        return img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    def _test_dataset():
        dataset = Dataset('../../../data/new_dataset_2019/normalized/', 125,
                          mask_dir='../../../data/new_dataset_2019/mask/',
                          random_scale=(0.5, 2.0), random_rotation=True)

        loader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True)

        for idx, (img, mask) in enumerate(loader):
            cv2.imshow('img', img[0].numpy().transpose((1, 2, 0)))
            cv2.imshow('mask', mask[0].numpy().transpose((1, 2, 0)))
            cv2.waitKey(0)
            # exit(0)

    _test_dataset()
