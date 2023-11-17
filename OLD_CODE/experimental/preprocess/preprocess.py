from skimage import io
import numpy as np
import cv2


def merge_max(stack_file):
    img = io.imread(stack_file)
    return np.max(img, axis=0)


if __name__ == '__main__':
    file_path='../../data/official_data/Bambusoideae- Subfamily/Arundinarieae - Tribe/Image stack files - Arundinarieae/Ampelocalamus_scandens_7_16_Series010_Brightfield.tif'
    maxed = merge_max(file_path)
    cv2.imshow('maxed', maxed)
    cv2.waitKey(0)
