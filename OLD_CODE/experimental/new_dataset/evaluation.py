from .preprocess.normalization import resize_and_crop, get_gaussian_filter
from .inference import ClassifierSimple
import sys
import numpy as np
import cv2
import pandas as pd
import os


MODEL_FILE = sys.argv[1]
IMAGE_DIR = sys.argv[2]


image_files = []
for root, dirnames, filenames in os.walk(IMAGE_DIR):
    for fn in filenames:
        if fn.lower().endswith(('.tif', '.jpg', '.png')):
            image_files.append(os.path.join(root, fn))


classifier = ClassifierSimple(MODEL_FILE)


stats = {}

last_img_width = None
filter_kernel = None

for fn in image_files:
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('failed to read %s. skipped' % fn)
        continue
    img = img.astype(np.float32) / 255.0

    if last_img_width != img.shape[1]:
        # Image size changed. Re-create kernel.
        filter_kernel = get_gaussian_filter(img.shape[1], 224)
        last_img_width = img.shape[1]
    img = cv2.sepFilter2D(img, -1, filter_kernel, filter_kernel)
    img = resize_and_crop(img, 224, border_mode='reflect')

    probs = classifier.predict(img[None, :, :])

    res = {}

    top_indices = np.argsort(probs)[::-1]

    for i in range(5):
        label = top_indices[i]
        res['top-%d' % (i + 1)] = '%s %.3f' % (classifier.inverse_class_dict[label], probs[label])

    rel_path = fn[len(IMAGE_DIR)+1:]
    stats[rel_path] = res


writer = pd.ExcelWriter('output.xlsx')
df = pd.DataFrame.from_dict(stats, orient='index')
df.to_excel(writer, float_format='%.3f')
writer.save()
