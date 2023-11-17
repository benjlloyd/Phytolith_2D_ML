import numpy as np


def random_rotate(imgs, rng, angles=None):
    import cv2

    shape = imgs.shape
    res = np.zeros(imgs.shape, np.float32)
    res = res.reshape((-1,) + res.shape[-3:])
    imgs = imgs.reshape(res.shape)

    n_row = res.shape[2]
    n_col = res.shape[3]

    for i in xrange(len(imgs)):
        img = np.transpose(imgs[i], (1, 2, 0))
        if angles is None:
            angle = rng.uniform(0, 360.0)
        else:
            angle = angles[rng.randint(len(angles))]
        rotation_matrix = cv2.getRotationMatrix2D((n_col / 2, n_row / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (n_col, n_row), borderMode=cv2.BORDER_REPLICATE)

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)  # opencv will strip the color dimension if it is grayscale. Add it back here.

        res[i] = np.transpose(img, (2, 0, 1))

    res = np.reshape(res, shape)
    return res
