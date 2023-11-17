import glob
import os
from skimage import io
import numpy as np
import sys
from preprocess.normalization import resize_and_crop
'''
    from LoadData import LoadData
    ld = LoadData(size) # your images will be size * size
    images, labels = ld.loadTrainData('/home/jifan/Dropbox' -- your dropbox directory,
                                      imageStyle = 'max_images'
                                      -- the folder name of the preprocessed images, 'global_autocontrast_normalized' by default
                                     )
    "images" are (num_of_images, size, size) shaped np array, "labels" are (num_of_iamges,) shaped np array
    ld.loadTestData(dropboxDir, imageStyle) essentially has the same behavior with loadTrainData but loads testing data
'''
class LoadData(object):
    def __init__(self, size):
        self.size = size

    def loadImage(self, dropboxDir, dir, imageStyle):
        f = open(dir, 'r')
        images = []
        labels = []
        for fn in f:
            if fn[0] == "b":
                labels += [0]
            elif fn[0] == "o":
                labels += [1]
            else:
                raise Exception('image not in either family')

            if fn[-5:] != ".tif":
                fn = fn[:-1]

            img = io.imread(os.path.join(dropboxDir, 'phytoliths', imageStyle, fn))
            if img.dtype == np.uint16:
                imgf = img / 65535.0
            elif img.dtype == np.uint8:
                imgf = img / 255.0
            else:
                raise Exception('image data type mismatch')
            images += [resize_and_crop(imgf, self.size)]

        return np.asarray(images), np.asarray(labels)

    def loadTrainData(self, dropboxDir, imageStyle='global_autocontrast_normalized'):
        return self.loadImage(dropboxDir, os.path.join(dropboxDir, 'phytoliths/train_images.txt'), imageStyle)
        
    def loadTrainDatav(self, dropboxDir, imageStyle='global_autocontrast_normalized'):
        return self.loadImage(dropboxDir, os.path.join(dropboxDir, 'phytoliths/train_images_v.txt'), imageStyle)

    def loadValidData(self, dropboxDir, imageStyle='global_autocontrast_normalized'):
        return self.loadImage(dropboxDir, os.path.join(dropboxDir, 'phytoliths/valid_images_v.txt'), imageStyle)

    def loadTestData(self, dropboxDir, imageStyle='global_autocontrast_normalized'):
        return self.loadImage(dropboxDir, os.path.join(dropboxDir, 'phytoliths/test_images.txt'), imageStyle)


class SubfamilyLoader:
    def __init__(self, image_size):
        self.size = image_size

    def get_subfamily(self, filename):
        return filename.split('.')[0]

    def load(self, image_dir, list_file, class_dict=None, transforms=None):
        import cv2

        classes = set()
        images = []
        text_labels = []

        with open(list_file) as f:
            lines = [l.strip() for l in f.readlines()]

        for fn in lines:
            path = os.path.join(image_dir, fn)
            tribe = self.get_subfamily(fn)
            img = io.imread(path)
            if img.dtype == np.uint16:
                imgf = img / 65535.0
            elif img.dtype == np.uint8:
                imgf = img / 255.0
            else:
                raise Exception('image data type mismatch')

            if transforms is not None and fn in transforms:
                imgf = cv2.warpPerspective(imgf, transforms[fn], imgf.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

            images.append(resize_and_crop(imgf, self.size))
            text_labels.append(tribe)
            classes.add(tribe)

        if class_dict is None:
            classes = sorted(list(classes))
            class_dict = dict(zip(classes, range(len(classes))))

        labels = [class_dict[l] for l in text_labels]

        return np.array(images, dtype=np.float32), np.array(labels), class_dict


class TribeLoader:
    def __init__(self, image_size):
        self.size = image_size

    def get_tribe(self, filename):
        return filename.split('.')[1]

    def load(self, image_dir, list_file, tribe_dict=None, transforms=None):
        import cv2

        tribes = set()
        images = []
        text_labels = []

        with open(list_file) as f:
            lines = [l.strip() for l in f.readlines()]

        for fn in lines:
            path = os.path.join(image_dir, fn)
            tribe = self.get_tribe(fn)
            img = io.imread(path)
            if img.dtype == np.uint16:
                imgf = img / 65535.0
            elif img.dtype == np.uint8:
                imgf = img / 255.0
            else:
                raise Exception('image data type mismatch')

            if transforms is not None and fn in transforms:
                imgf = cv2.warpPerspective(imgf, transforms[fn], imgf.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

            images.append(resize_and_crop(imgf, self.size))
            text_labels.append(tribe)
            tribes.add(tribe)

        if tribe_dict is None:
            tribes = sorted(list(tribes))
            tribe_dict = dict(zip(tribes, range(len(tribes))))

        labels = [tribe_dict[l] for l in text_labels]

        return np.array(images, dtype=np.float32), np.array(labels), tribe_dict


class GenusLoader:
    from genus_corrections import misspellings

    def __init__(self, image_size):
        self.size = image_size

    def get_genus(self, filename):
        genus = filename.split('.')[2]
        if genus in GenusLoader.misspellings:
            return GenusLoader.misspellings[genus]
        else:
            return genus

    def load(self, image_dir, list_file, genus_dict=None, transforms=None):
        '''
        :param image_dir:
        :param list_file:
        :param genus_dict:
        :param transforms: a (string, nparray) dictionary that maps an image file to a 3x3 transform. This can be used
        :                  for aligning the orientations of phytoliths.
        :return:
        '''
        import cv2

        genera = set()
        images = []
        text_labels = []

        with open(list_file) as f:
            lines = [l.strip() for l in f.readlines()]

        for fn in lines:
            path = os.path.join(image_dir, fn)
            genus = self.get_genus(fn)
            img = io.imread(path)
            if img.dtype == np.uint16:
                imgf = img / 65535.0
            elif img.dtype == np.uint8:
                imgf = img / 255.0
            else:
                raise Exception('image data type mismatch')

            if transforms is not None and fn in transforms:
                imgf = cv2.warpPerspective(imgf, transforms[fn], imgf.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

            images.append(resize_and_crop(imgf, self.size))
            text_labels.append(genus)
            genera.add(genus)

        if genus_dict is None:
            genera = sorted(list(genera))
            genus_dict = dict(zip(genera, range(len(genera))))

        labels = [genus_dict[l] for l in text_labels]

        return np.array(images, dtype=np.float32), np.array(labels), genus_dict


if __name__ == '__main__':

    def genus():
        loader = GenusLoader(256)

        images, labels, genus_dict = loader.load('../data/max_images', '../data/genus_test.txt')

        import utils
        utils.save_classes(genus_dict, '../data/genus_names.txt')

        with open('../data/genus_test_labels.txt', 'w') as f:
            for l in labels:
                f.write('%d\n' % l)

        print 'number of genus:', len(genus_dict)

    def tribe():
        loader = TribeLoader(32)

        images, labels, class_dict = loader.load('../data/max_images', '../data/tribe_test.txt')

        import utils
        utils.save_classes(class_dict, '../data/tribe_names.txt')

        print 'number of tribes:', len(class_dict)

    tribe()
