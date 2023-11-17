import sys

import cv2
import gflags
import numpy as np
from . import parse_dataset


def resize_and_crop(img, size, return_mask=False, border_mode='replicate'):
    '''
    We keep the whole image content and pad if necessary.
    '''
    h, w = img.shape

    longer_side = max(h, w)

    scale = float(size) / longer_side

    img2 = cv2.resize(img, None, fx=scale, fy=scale)

    padleft = (size - img2.shape[1]) // 2
    padright = size - img2.shape[1] - padleft
    padtop = (size - img2.shape[0]) // 2
    padbottom = size - img2.shape[0] - padtop

    if border_mode == 'replicate':
        padded = cv2.copyMakeBorder(img2, padtop, padbottom, padleft, padright, cv2.BORDER_REPLICATE)
    elif border_mode == 'zero':
        padded = cv2.copyMakeBorder(img2, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT, value=0)
    elif border_mode == 'reflect':
        padded = cv2.copyMakeBorder(img2, padtop, padbottom, padleft, padright, cv2.BORDER_REFLECT)
    elif border_mode == 'none':
        # No padding
        padded = img2
    else:
        raise RuntimeError('Unknown border mode %s' % border_mode)

    if return_mask:
        mask = np.ones(img2.shape, img.dtype)
        mask = cv2.copyMakeBorder(mask, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                  value=0)
        return padded, mask

    return padded


def pad_or_crop(img, size, return_mask=False, border_mode='replicate'):
    """
    Pad or crop the image to make it into specific size
    :param size: an integer
    """
    # Crop the image
    h, w = img.shape[:2]

    org_shape = img.shape

    cropleft = (w - size) // 2
    croptop = (h - size) // 2

    if cropleft >= 0:
        img = img[:, cropleft: cropleft + size]
    if croptop >= 0:
        img = img[croptop: croptop + size, :]

    # Pad if necessary
    h, w = img.shape[:2]

    padleft = (size - w) // 2
    padright = size - w - padleft
    padtop = (size - h) // 2
    padbottom = size - h - padtop

    try:
        mask = np.ones(img.shape, img.dtype)

        if border_mode == 'replicate':
            img = cv2.copyMakeBorder(img, padtop, padbottom, padleft, padright, cv2.BORDER_REPLICATE)
        elif border_mode == 'zero':
            img = cv2.copyMakeBorder(img, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT, value=0)
        elif border_mode == 'reflect':
            img = cv2.copyMakeBorder(img, padtop, padbottom, padleft, padright, cv2.BORDER_REFLECT)
        else:
            raise RuntimeError('Unknown border mode %s' % border_mode)

        mask = cv2.copyMakeBorder(mask, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                  value=0)
    except:
        print(org_shape)
        print(img.shape)
        print(cropleft)
        print(size)
        raise

    if return_mask:
        return img, mask

    return img


def get_gaussian_filter(src_size, dst_size):
    downscale_factor = src_size // dst_size
    sigma = (downscale_factor - 1) / 2

    if downscale_factor % 2 == 0:
        kernel_size = downscale_factor + 1
    else:
        kernel_size = downscale_factor

    return cv2.getGaussianKernel(kernel_size, sigma)


if __name__ == '__main__':
    import glob
    import os
    from collections import defaultdict

    gflags.DEFINE_string('src_dir', '', '')
    gflags.DEFINE_string('dst_dir', '', '')
    gflags.DEFINE_string('mask_dir', '', 'If specified, will output masks to this directory.')
    gflags.DEFINE_integer('size', 512, '')
    gflags.DEFINE_string('border_mode', 'zero', "none: no border; "
                                                "replicate: replicate border pixels; "
                                                "zero: fill in zero values.")
    gflags.DEFINE_boolean('smoothing', True,
                          'Smooth the image before downsampling. '
                          'Note that here we assume original images are larger than the normalized images!')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    SRC_DIR = FLAGS.src_dir
    DST_DIR = FLAGS.dst_dir
    MASK_DIR = FLAGS.mask_dir
    SIZE = FLAGS.size

    files = []
    for root, dirnames, filenames in os.walk(SRC_DIR):
        for fn in filenames:
            if fn.lower().endswith(('.tif', '.jpg', '.png')):
                files.append(os.path.join(root, fn))
    files.sort()

    print('%d files' % len(files))

    counts = defaultdict(int)

    last_img_width = None

    for idx, fn in enumerate(files):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        if img.dtype == np.uint16:
            imgf = img / 65535.0
        elif img.dtype == np.uint8:
            imgf = img / 255.0
        else:
            raise RuntimeError('Unsupported datatype %s' % img.dtype)

        if len(imgf.shape) == 3 and imgf.shape[2] == 2:
            imgf = imgf[:, :, 0]  # Photoshop creates one extra channel that seems to be all ones.

        if FLAGS.smoothing:
            if last_img_width is None or last_img_width != imgf.shape[1]:
                # Image size changed. Re-create kernel.
                filter_kernel = get_gaussian_filter(imgf.shape[1], SIZE)
                last_img_width = imgf.shape[1]
            imgf = cv2.sepFilter2D(imgf, -1, filter_kernel, filter_kernel)

        imgf, mask = resize_and_crop(imgf, SIZE, return_mask=True, border_mode=FLAGS.border_mode)

        subfamily, tribe, genus, species, sample_idx = parse_dataset.parse_filename(fn)
        bio_key = (subfamily, tribe, genus, species)
        key = bio_key + (counts[bio_key],)
        counts[bio_key] += 1

        out_fn = '%s.%s.%s.%s.%d.tif' % key
        cv2.imwrite(os.path.join(DST_DIR, out_fn), (imgf * 255).astype(np.uint8))

        if FLAGS.mask_dir != '':
            cv2.imwrite(os.path.join(MASK_DIR, out_fn), (mask * 255).astype(np.uint8))

        print('[%d/%d] %s -> %s' % (idx + 1, len(files), fn, out_fn))

        # Uncomment to visualize the output
        # cv2.imshow('resized', imgf)
        # cv2.waitKey(1)
