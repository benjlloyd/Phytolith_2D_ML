import torch.utils.data as data

import cv2
import numpy as np
import os
from collections import defaultdict

from ..preprocess.normalization import resize_and_crop, pad_or_crop
from ..preprocess.parse_dataset import field_map
from skimage import io


class Loader:
    def __init__(self, image_size):
        self.size = image_size

    def load(self, image_dir, list_file, granularity='subfamily', class_dict=None, transforms=None, border_mode='zero'):
        '''
        :param image_dir:
        :param list_file: the file containing the file names to load
        :param granularity:
        :param class_dict: a string->integer map of each class string to an integer label
        :param transforms: a string->ndarray map of each image file name to a 3x3 transform to apply on the image
        :return:
        images: NxHxW array
        labels: length N integer vector
        class_dict: same meaning as the input argument. If not specified in input it will be computed and returned,
        otherwise it is exactly the same as the input argument.
        '''

        classes = set()
        images = []
        text_labels = []

        with open(list_file) as f:
            lines = [l.strip() for l in f.readlines()]

        for fn in lines:
            path = os.path.join(image_dir, fn)
            cls = fn.split('.')[field_map[granularity]]

            if class_dict is not None and cls not in class_dict:
                print('%s is from an unknown class. ignored.' % fn)
                continue

            img = io.imread(path)
            if img.dtype == np.uint16:
                imgf = img / 65535.0
            elif img.dtype == np.uint8:
                imgf = img / 255.0
            else:
                raise Exception('image data type mismatch')

            if transforms is not None and fn in transforms:
                imgf = cv2.warpPerspective(
                    imgf, transforms[fn], imgf.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

            images.append(resize_and_crop(imgf, self.size, border_mode=border_mode))
            text_labels.append(cls)
            classes.add(cls)

        if class_dict is None:
            classes = sorted(list(classes))
            class_dict = dict(zip(classes, range(len(classes))))

        labels = [class_dict[l] for l in text_labels]

        return np.array(images, dtype=np.float32), np.array(labels), class_dict


def affine_transform(imgs, transforms, border_mode='replicate'):
    shape = imgs.shape
    res = np.zeros(imgs.shape, np.float32)
    res = res.reshape((-1,) + res.shape[-3:])
    imgs = imgs.reshape(res.shape)

    n_row = res.shape[2]
    n_col = res.shape[3]

    for i in range(len(imgs)):
        img = np.transpose(imgs[i], (1, 2, 0))

        if border_mode == 'replicate':
            img = cv2.warpAffine(img, transforms[i], (n_col, n_row), borderMode=cv2.BORDER_REPLICATE)
        elif border_mode == 'zero':
            img = cv2.warpAffine(img, transforms[i], (n_col, n_row), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif border_mode == 'reflect':
            img = cv2.warpAffine(img, transforms[i], (n_col, n_row), borderMode=cv2.BORDER_REFLECT)
        else:
            raise RuntimeError('Unsupported border mode %s' % border_mode)

        if len(img.shape) == 2:
            # opencv will strip the color dimension if it is grayscale. Add it back here.
            img = np.expand_dims(img, 2)

        res[i] = np.transpose(img, (2, 0, 1))

    res = np.reshape(res, shape)
    return res


def random_rotate(imgs, rng, angles=None, **kwargs):
    '''
    :param imgs: N x C x H x W
    :param rng:
    :param angles: either a list of angles in degrees, or set to None for uniform rotation angle in [0, 2*pi].
    :return:
    '''
    assert len(imgs.shape) == 4
    width, height = imgs.shape[2:]
    transforms = []

    for i in range(len(imgs)):
        if angles is None:
            angle = rng.uniform(0, 360.0)
        else:
            angle = angles[rng.randint(len(angles))]
        transforms.append(cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1))

    return affine_transform(imgs, transforms, **kwargs)


def random_shift(imgs, rng, shift_min, shift_max, **kwargs):
    assert len(imgs.shape) == 4
    transforms = []
    for i in range(len(imgs)):
        dx = rng.uniform(shift_min, shift_max)
        dy = rng.uniform(shift_min, shift_max)
        transforms.append(np.array([[1, 0, dx], [0, 1, dy]], np.float32))
    return affine_transform(imgs, transforms, **kwargs)


def scale_image(img, scale, **kwargs):
    """
    :param img: C x H x W
    :param rng:
    :param scale_min:
    :param scale_max:
    :return:
    """
    import cv2

    img = img.transpose((1, 2, 0))  # Convert to H x W x C
    size = img.shape[0]  # FIXME: assume that image is of square shape
    # FIXME: INTER_LINEAR may not be good for shrinking images
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    img = pad_or_crop(img, size, **kwargs)

    if len(img.shape) == 2:
        img = img[None, :, :]
    else:
        img = img.transpose((2, 0, 1))

    return img


class Dataset(data.Dataset):
    def __init__(self, image_dir, list_file, granularity, image_size,
                 class_dict=None, transforms=None, border_mode='zero', pair=False, pair_same_prob=0.5,
                 random_rotation=False, random_scale=False, random_shift=False):
        '''
        :param image_dir:
        :param list_file:
        :param granularity:
        :param image_size:
        :param class_dict:
        :param transforms:
        :param pair: if True will return a pair of samples. This is useful for doing metric learning.
        :param kwargs:  'random_rotation':[angles_in_degree] or 'random_rotation': True
                        'random_scale': (scale_min, scale_max) or None
        '''
        print('loading images from %s ...' % list_file)
        loader = Loader(image_size)
        images, labels, class_dict = loader.load(
            image_dir, list_file, granularity, class_dict, transforms, border_mode=border_mode)

        assert images.dtype == np.float32
        if len(images.shape) == 3:
            # Convert N x H x W into N x 1 x H x W
            images = images[:, None, :, :]

        self.images = images
        self.labels = labels.astype(np.int64)

        inverse_counts_per_label = 1.0 / np.bincount(self.labels).astype(np.float32)
        # Used for weighted random sampling
        self.label_weights = [inverse_counts_per_label[_] for _ in self.labels]

        # inverse_labels[i] contains sample indices of class i
        self.inverse_labels = [[] for _ in range(len(class_dict))]
        for idx, l in enumerate(self.labels):
            self.inverse_labels[l].append(idx)

        inverse_label_freq = [1.0 / len(_) for _ in self.inverse_labels]
        s = sum(inverse_label_freq)
        self.normalized_inv_label_freq = [_ / s * len(self.inverse_labels) for _ in inverse_label_freq]

        self.class_dict = class_dict

        self.border_mode = border_mode

        self.pair = pair
        self.pair_same_prob = pair_same_prob

        self.first = True  # Used to initialize random number generator for forked workers.
        self.rng = np.random.RandomState()

        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.random_shift = random_shift

    def _transform(self, img, rng):
        if self.random_rotation:
            if isinstance(self.random_rotation, list):
                angles = self.random_rotation
            else:
                angles = None
            img = random_rotate(img[None], rng, angles, border_mode=self.border_mode)[0]

        if self.random_shift:
            ratio_min, ratio_max = self.random_shift
            shift_min = ratio_min * img.shape[1]
            shift_max = ratio_max * img.shape[1]
            img = random_shift(img[None], rng, shift_min, shift_max, border_mode=self.border_mode)[0]

        if self.random_scale:
            scale_min, scale_max = self.random_scale
            scale = rng.uniform(scale_min, scale_max)
            img = scale_image(img, scale, border_mode=self.border_mode)

        return img

    def __getitem__(self, idx):
        if self.first:
            # Make sure each worker has a different RNG.
            self.rng.seed(idx)
            self.first = False

        if self.pair:
            img = self.images[idx]

            if self.rng.uniform(0.0, 1.0) < self.pair_same_prob:
                # Draw an image of the same class
                idxs = self.inverse_labels[self.labels[idx]]
                assert len(idxs) > 1
                while True:
                    idx2 = idxs[self.rng.randint(len(idxs))]
                    if idx2 != idx:
                        assert self.labels[idx2] == self.labels[idx]
                        break
            else:
                # Draw an image of a different class
                # We first randomly choose a different class, and then draw a sample from that class.
                # Here we simply randomly draw a sample. Repeat if it has the same label as the
                # the first sample.
                # Since each label will be drawn with equal probability, this naturally does weighted random sampling.
                l = self.labels[idx]
                while True:
                    l2 = self.rng.randint(len(self.inverse_labels))
                    if l2 != l:
                        break
                l2_idxs = self.inverse_labels[l2]
                idx2 = l2_idxs[self.rng.randint(len(l2_idxs))]

            img2 = self.images[idx2]

            img, img2 = [self._transform(_, self.rng) for _ in (img, img2)]
            return img, img2, self.labels[idx], self.labels[idx2]
        else:
            img = self.images[idx]
            return self._transform(img, self.rng), self.labels[idx]

    def __len__(self):
        return len(self.labels)


class PairDataset(data.Dataset):
    def __init__(self, image_dir, list_file, granularity, image_size,
                 class_dict=None, transforms=None, **kwargs):
        '''
        :param image_dir:
        :param list_file:
        :param granularity:
        :param image_size:
        :param class_dict:
        :param transforms:
        :param kwargs:  'random_rotation':[angles_in_degree] or 'random_rotation': True
        '''
        loader = Loader(image_size)
        images, labels, class_dict = loader.load(
            image_dir, list_file, granularity, class_dict, transforms)

        # expand to 3-channel image
        images = np.repeat(np.expand_dims(images, 1), 3, 1)
        assert images.dtype == np.float32

        # center
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape((1, 3, 1, 1))
        images = (images - mean) / std

        labels, indices = (list(x) for x in zip(*sorted(zip(labels, range(images.shape[0])))))
        images = images[indices]
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_dict = class_dict

        self.addresses = np.zeros(len(self.class_dict) + 1)
        self.addresses[-1] = self.images.shape[0]
        prev = 0
        for i in range(self.images.shape[0]):
            if prev != self.labels[i]:
                self.addresses[self.labels[i]] = i
                prev = self.labels[i]

        self.first = True  # Used to initialize random number generator for forked workers.
        self.rng = np.random.RandomState()

        self.opts = defaultdict((lambda: None), kwargs)  # Options (default to None if not specified).

    def rand_image_index(self, index):
        return self.rng.randint(self.addresses[index], self.addresses[index + 1])

    def generate_pair(self, idx):

        label = self.labels[idx]
        d_label = label
        while d_label == label:
            d_label = self.rng.randint(0, len(self.class_dict))

        s_index = idx
        while s_index == idx:
            s_index = self.rand_image_index(label)
        d_index = self.rand_image_index(d_label)

        random_rot = self.opts['random_rotation']
        if random_rot:
            if isinstance(random_rot, list):
                angles = random_rot
            else:
                angles = None
            return random_rotate(self.images[idx], self.rng, angles), \
                   random_rotate(self.images[s_index], self.rng, angles), \
                   random_rotate(self.images[d_index], self.rng, angles)
        else:
            return self.images[idx], self.images[s_index], self.images[d_index]

    def __getitem__(self, idx):
        if self.first:
            # Make sure each worker has a different RNG.
            self.rng.seed(idx)
            self.first = False

        return self.generate_pair(idx)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader, WeightedRandomSampler

    def subfamily():
        loader = Loader(224)

        images, labels, class_dict = loader.load('../../data/new_dataset_2019/normalized/',
                                                 '../../data/new_dataset_2019/species_train.txt')

        print(class_dict)
        print('number of subfamily:', len(class_dict))
        print(images.shape, labels.shape)

        for i in range(len(images)):
            cv2.imshow('', images[i])
            cv2.waitKey(0)

    # test loader
    # subfamily()

    # test dataset
    def _test_dataset():
        dataset = Dataset('../../data/new_dataset_2019/normalized/',
                          '../../data/new_dataset_2019/species_train.txt',
                          'species',
                          224,
                          border_mode='reflect',
                          random_rotation=True,
                          random_shift=(-0.1, 0.1))

        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True)

        for idx, (img, label) in enumerate(loader):
            print(img.size(), label.size())
            cv2.imshow('', img[0].numpy().transpose((1, 2, 0)))
            cv2.waitKey(0)
            # exit(0)

    def _test_dataset_pair():
        dataset = Dataset('../../data/new_dataset_2019/normalized/',
                          '../../data/new_dataset_2019/tribe_train.txt',
                          'tribe',
                          224,
                          border_mode='reflect',
                          random_rotation=True,
                          random_shift=(-0.1, 0.1),
                          pair=True)

        print(dataset.normalized_inv_label_freq)

        print('number of samples per class')
        for label, idxs in enumerate(dataset.inverse_labels):
            print('%d: %d' % (label, len(idxs)))

        exit(0)

        sampler = WeightedRandomSampler(dataset.label_weights, 1000)
        loader = DataLoader(dataset,
                            batch_size=8,
                            sampler=sampler,
                            num_workers=1,
                            pin_memory=True)

        for idx, (img1, img2, label1, label2) in enumerate(loader):
            print(label1, label2)
            # cv2.imshow('', img1[0].numpy().transpose((1, 2, 0)))
            # cv2.waitKey(0)
            # exit(0)

    # a, b, c, d = PairDataset('../../data/new_dataset/normalized/',
    #             '../../data/new_dataset/subfamily_train.txt',
    #             'subfamily', 224).__getitem__(10)

    #subfamily()
    _test_dataset_pair()
