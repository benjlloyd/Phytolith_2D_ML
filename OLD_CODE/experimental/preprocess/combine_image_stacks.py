from skimage import io
from parse_dataset import build_tree
import cv2
import os
import numpy as np
import gflags
import sys

gflags.DEFINE_boolean('max', False, '')
gflags.DEFINE_boolean('mean', False, '')
gflags.DEFINE_boolean('median', False, '')
gflags.DEFINE_boolean('downsample_image_stack', False, '')
gflags.DEFINE_integer('max_n_slices', -1,
                      'Specify how many slices are kept in each image stack. -1 means keeping all the slices.')
gflags.DEFINE_integer('resolution', 128, 'Resolution of the downsampled image stacks.')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

OUTPUT_MAX = FLAGS.max
OUTPUT_MEAN = FLAGS.mean
OUTPUT_MEDIAN = FLAGS.median
OUTPUT_DOWNSAMPLED_IMAGE_STACKS = FLAGS.downsample_image_stack
MAX_N_SLICES = FLAGS.max_n_slices
IMAGE_STACK_RESOLUTION = FLAGS.resolution


print 'output max: ', OUTPUT_MAX
print 'output mean: ', OUTPUT_MEAN
print 'output median: ', OUTPUT_MEDIAN
print 'output downsampled image stacks: ', OUTPUT_DOWNSAMPLED_IMAGE_STACKS


root = '../../data/official_data'
max_img_dir = '../../data/max_images'
mean_img_dir = '../../data/mean_images'
median_img_dir = '../../data/median_images'
downsampled_image_stacks_dir = '../../data/downsampled_image_stacks_%d' % IMAGE_STACK_RESOLUTION



def write_max(img, fn):
    maximg = np.max(img, axis=0)
    cv2.imwrite(fn, maximg)


def write_mean(img, fn):
    meanimg = np.mean(img, axis=0).astype(img.dtype)
    cv2.imwrite(fn, meanimg)


def write_median(img, fn):
    medianimg = np.median(img, axis=0).astype(img.dtype)
    cv2.imwrite(fn, medianimg)


def write_downsampled_image_stacks(img, fn):
    dirname = os.path.splitext(fn)[0]
    try:
        os.mkdir(dirname)
    except:
        pass

    _, h, w = img.shape
    longer_side = max(h, w)
    scale = float(IMAGE_STACK_RESOLUTION) / longer_side

    out_h = int(h * scale)
    out_w = int(w * scale)

    step = max(img.shape[0] // MAX_N_SLICES, 1)
    for i in xrange(0, img.shape[0], step):
        out = cv2.resize(img[i], None, fx=scale, fy=scale)
        cv2.imwrite('%s/%03d.png' % (dirname, i), out)


def gen_file_list(root):
    '''
    :param root:
    :return: a list of tuples: (path to image file, subfamily_name, tribe_name, genus_name, index)
    '''
    tree = build_tree(root)
    l = []
    for subfamily_name in tree:
        subfamily_dir = tree[subfamily_name]['dir']
        tribes = tree[subfamily_name]['tribes']
        for tribe_name in tribes:
            tribe_dir = tribes[tribe_name]['dir']
            genus = tribes[tribe_name]['genus']
            image_stacks = genus['image_stack']
            image_stacks_dir = image_stacks['dir']
            image_stack_files = image_stacks['files']
            for genus_name in image_stack_files:
                filenames = image_stack_files[genus_name]
                for i in xrange(len(filenames)):
                    fn = filenames[i]
                    path = os.path.join(root, subfamily_dir, tribe_dir, image_stacks_dir, fn)
                    l.append((path, subfamily_name, tribe_name, genus_name, i))
    return l

if __name__ == '__main__':
    l = gen_file_list(root)

    for path, subfamily_name, tribe_name, genus_name, i in l:
        img = io.imread(path)

        if len(img.shape) < 3:
            print path, 'is not a multi-page tif. skipped.'
            continue

        if len(img.shape) >= 4:
            print path, 'is not a grayscale image.'
            continue

        if img.shape[-1] < 50 or img.shape[-2] < 50:
            print path, 'is small and may not be a valid image. skipped.'
            continue

        filename = '.'.join([subfamily_name, tribe_name, genus_name, str(i)]) + '.tif'

        if OUTPUT_MAX:
            write_max(img, os.path.join(max_img_dir, filename))
        if OUTPUT_MEAN:
            write_mean(img, os.path.join(mean_img_dir, filename))
        if OUTPUT_MEDIAN:
            write_median(img, os.path.join(median_img_dir, filename))
        if OUTPUT_DOWNSAMPLED_IMAGE_STACKS:
            write_downsampled_image_stacks(img, os.path.join(downsampled_image_stacks_dir, filename))

        cv2.waitKey(1)
