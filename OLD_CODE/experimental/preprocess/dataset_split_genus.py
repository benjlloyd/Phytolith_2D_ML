import glob
import os
from collections import defaultdict
import random

rng = random.Random(12345678)

image_files = glob.glob('../../data/global_autocontrast_normalized/*.tif')
n_image_files_used = len(image_files)

TRAIN_RATIO = 0.8

files_by_genus = defaultdict(list)

for fn in image_files:
    bn = os.path.basename(fn)
    genus = bn.split('.')[2]
    files_by_genus[genus].append(bn)

train_files = []
test_files = []

for genus, files in files_by_genus.iteritems():
    files.sort()
    rng.shuffle(files)
    if len(files) == 1:
        print 'genus %s has only one image. Excluded.' % genus
        n_image_files_used -= 1
        continue
    n_train = int(len(files) * TRAIN_RATIO)
    assert 0 < n_train < len(files), files
    train_files.extend(files[:n_train])
    test_files.extend(files[n_train:])
    print 'genus %s train: %d test: %d' % (genus, n_train, len(files) - n_train)

assert len(train_files) + len(test_files) == n_image_files_used


def write_to_file(out_file, file_list):
    with open(out_file, 'w') as f:
        for fn in file_list:
            f.write('%s\n' % fn)


write_to_file('genus_train.txt', train_files)
write_to_file('genus_test.txt', test_files)
