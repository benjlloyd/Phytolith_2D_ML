import numpy as np
from sklearn.decomposition import PCA
import cv2
import glob


def get_transform(img):
    n_col = img.shape[1]
    n_row = img.shape[0]

    # th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret2, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('thresed', th)

    ys, xs = np.nonzero(255 - th)
    coords = zip(xs, ys)

    pca = PCA(n_components=1)
    pca.fit(coords)

    principal_axis = pca.components_[0]
    translation = np.eye(3)
    translation[:2, 2] = np.array([n_col / 2, n_row / 2]) - pca.mean_

    angle = -np.arccos(np.dot(principal_axis, [0, 1])) * 180 / np.pi
    rot = cv2.getRotationMatrix2D((n_col / 2, n_row / 2), angle, 1)
    R = np.eye(3)
    R[:2, :3] = rot
    transform = np.dot(R, translation)
    return transform


if __name__ == '__main__':
    # img_files = glob.glob('../../data/global_autocontrast_normalized/bambusoideae.olyreae.raddia.25.tif')
    # img_files = glob.glob('../../data/global_autocontrast_normalized/bambusoideae.bambuseae.dinochloa.11.tif')
    # img_files = glob.glob('../../data/global_autocontrast_normalized/bambusoideae.bambuseae.davidsea.0.tif')
    img_files = glob.glob('../../data/misaligned/*.tif')
    import os
    with open('align_transforms.txt', 'w') as f:
        for fn in img_files:
            img = cv2.imread(fn)
            img = img[:, :, 0]
            cv2.imshow('', img)
            transform = get_transform(img)
            aligned = cv2.warpPerspective(img, transform, img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
            cv2.imshow('aligned', aligned)
            f.write('%s,' % os.path.basename(fn))
            f.write(','.join('%f' % v for v in transform.flatten()))
            f.write('\n')
