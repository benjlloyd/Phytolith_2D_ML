import glob
import os
from collections import defaultdict
import random

rng = random.Random(12345678)

image_files = glob.glob('../../data/global_autocontrast_normalized/*.tif')

TRAIN_RATIO = 0.8

files_by_tribe = defaultdict(list)

for fn in image_files:
    bn = os.path.basename(fn)
    tribe = bn.split('.')[1]
    files_by_tribe[tribe].append(bn)

train_files = []
test_files = []

for tribe, files in files_by_tribe.iteritems():
    files.sort()
    rng.shuffle(files)
    n_train = int(len(files) * TRAIN_RATIO)
    assert 0 < n_train < len(files)
    train_files.extend(files[:n_train])
    test_files.extend(files[n_train:])
    print 'tribe %s train: %d test: %d' % (tribe, n_train, len(files) - n_train)

assert len(train_files) + len(test_files) == len(image_files)


def write_to_file(out_file, file_list):
    with open(out_file, 'w') as f:
        for fn in file_list:
            f.write('%s\n' % fn)

write_to_file('tribe_train.txt', train_files)
write_to_file('tribe_test.txt', test_files)
