import numpy as np
from skimage import io
import cv2
import gflags
import sys



def autocontrast(img):
    min_value, max_value = img.min(), img.max()
    return (img - min_value) / (max_value - min_value)


def autocontrast2(img):
    pixels = np.sort(img.flatten())
    percentile5 = pixels[int(len(pixels) * 0.05)]
    percentile95 = pixels[int(len(pixels) * 0.95)]
    if percentile95 == percentile5:
        return img
    return np.minimum(np.maximum((img - percentile5) / (percentile95 - percentile5), 0.0), 1.0)


def local_autocontrast(img):
    w = img.shape[1] / 8 * 5
    h = img.shape[0] / 8 * 5

    res = np.zeros(img.shape)

    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            x1 = max(j - w / 2, 0)
            y1 = max(i - h / 2, 0)

            x2 = min(x1 + w, img.shape[1])
            y2 = min(y1 + h, img.shape[0])

            roi = img[y1:y2, x1:x2]
            min_value, max_value = roi.min(), roi.max()

            pixels = np.sort(roi.flatten())

            percentile5 = pixels[int(len(pixels) * 0.05)]
            percentile95 = pixels[int(len(pixels) * 0.95)]

            if percentile5 == percentile95:
                res[i, j] = img[i, j]
            else:
                res[i, j] = np.minimum(np.maximum((img[i, j] - percentile5) / (percentile95 - percentile5), 0.0), 1.0)
                # res[i, j] = (img[i, j] - min_value) / (max_value - min_value)

    return res


def test():
    # fn = '../../data/median_images/bambusoideae.olyreae.pariana.5.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.bambuseae.merostachys.4.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.bambuseae.nastus.3.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.bambuseae.racemo.0.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.olyreae.diandrolyra.17.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/oryzoideae.phyllorachideae.phyllorachis.13.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.arundinarieae.bergbambos.11.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/bambusoideae.arundinarieae.phyllostcahys.1.tif'
    # fn = '/home/xymeng/Dropbox/phd/courses/cse546/phytoliths/median_images/oryzoideae.phyllorachideae.phyllorachis.13.tif'
    fn = '/home/xymeng/dev/phytoliths/data/median_images/bambusoideae.arundinarieae.chimonobambusa.10.tif'

    img = io.imread(fn)
    assert len(img.shape) == 2

    img = resize_and_crop(img, 128)

    cv2.imshow('original', img)

    cv2.imshow('autocontrast2', autocontrast2(img.astype(np.float32) / 65535.0))

    cv2.imshow('local autocontrast', local_autocontrast(img.astype(np.float32) / 65535.0))
    cv2.waitKey(0)

    exit(0)

    img2 = np.array(img, copy=True)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    cl1 = clahe.apply((img / 65535.0 * 255.0).astype(np.uint8))

    cv2.imshow('clahe', cl1)

    # cv2.waitKey(0)
    # exit(0)
    #
    # w = 256
    # h = 256
    #
    # for i in xrange(img.shape[0] - h):
    #     for j in xrange(img.shape[1] - w):
    #         img2[i:i+h, j:j+w] = (img2[i:i+h, j:j+w] + autocontrast(img[i:i+h, j:j+w])) * 0.5

    # img2 = autocontrast(np.array(img, copy=True))

    img3 = autocontrast(img.astype(np.float32) / 65535.0)


    # cv2.imshow('local autocontrast', img2)
    cv2.imshow('global autocontrast', img3)

    cv2.waitKey(0)


def resize_and_crop(img, size):
    '''
    We keep the whole image content and pad if necessary.
    '''
    h, w = img.shape

    longer_side = max(h, w)

    scale = float(size) / longer_side

    img2 = cv2.resize(img, None, fx=scale, fy=scale)

    padleft = (size - img2.shape[1]) / 2
    padright = size - img2.shape[1] - padleft
    padtop = (size - img2.shape[0]) / 2
    padbottom = size - img2.shape[0] - padtop

    padded = cv2.copyMakeBorder(img2, padtop, padbottom, padleft, padright, cv2.BORDER_REPLICATE)
    return padded


if __name__ == '__main__':
    gflags.DEFINE_string('src_dir', '', '')
    gflags.DEFINE_string('dst_dir', '', '')
    gflags.DEFINE_integer('crop_size', 128, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    SRC_DIR = FLAGS.src_dir
    DST_DIR = FLAGS.dst_dir
    CROP_SIZE = FLAGS.crop_size

    #test()

    import glob
    import os

    files = glob.glob(os.path.join(SRC_DIR, '*.tif'))

    print '%d files' % len(files)

    for fn in files:
        img = io.imread(fn)

        if img.dtype == np.uint16:
            imgf = img / 65535.0
        elif img.dtype == np.uint8:
            imgf = img / 255.0
        else:
            assert False

        if len(imgf.shape) == 3 and imgf.shape[2] == 2:
            imgf = imgf[:, :, 0]  # Photoshop creates one extra channel that seems to be all ones.

        imgf = resize_and_crop(imgf, CROP_SIZE)
        assert imgf.shape == (CROP_SIZE, CROP_SIZE)

        cv2.imshow('resized', imgf)

        imgf_normed = local_autocontrast(imgf)

        cv2.imshow('normed', imgf_normed)

        out_fn = os.path.join(DST_DIR, os.path.basename(fn))
        cv2.imwrite(out_fn, (imgf_normed * 255).astype(np.uint8))

        cv2.waitKey(1)

