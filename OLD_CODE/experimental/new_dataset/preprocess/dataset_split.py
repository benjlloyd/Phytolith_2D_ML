import glob
import os
from collections import defaultdict
import random
import gflags
import sys
from . parse_dataset import field_map


gflags.DEFINE_string('src_dir', '', '')
gflags.DEFINE_string('splits', '8,1,1', 'Relative ratio between train, validation and test.')
gflags.DEFINE_string('granularity', 'subfamily', 'subfamily, tribe, genus or species')
gflags.DEFINE_integer('fold', 0,
                      'Used for adjusting train validation split. Note that this number must be <= (train+val)/val. '
                      'This does not affect the test split.')
gflags.DEFINE_integer('seed', 12345678, 'Seed for random split')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

rng = random.Random(FLAGS.seed)

rel_ratio = [int(x) for x in FLAGS.splits.split(',')]
split_ratio = [float(x) / sum(rel_ratio) for x in rel_ratio]

assert 0 <= FLAGS.fold <= (rel_ratio[0] + rel_ratio[1]) // rel_ratio[1]

image_files = glob.glob(os.path.join(FLAGS.src_dir, '*.tif'))
n_image_files_used = len(image_files)

print('%d image files' % len(image_files))
if len(image_files) == 0:
    exit(0)

field_idx = field_map[FLAGS.granularity]

file_clusters = defaultdict(list)

for fn in image_files:
    bn = os.path.basename(fn)
    try:
        label = bn.split('.')[field_idx]
    except:
        print(bn)
        raise
    file_clusters[label].append(bn)

print('%d classes' % len(file_clusters))

train_files = []
test_files = []
validation_files = []


for label, files in file_clusters.items():
    files.sort()
    rng.shuffle(files)
    if len(files) == 1:
        print('%s has only one image. Excluded.' % label)
        n_image_files_used -= 1
        continue

    n_train = int(len(files) * split_ratio[0])
    n_val = int(len(files) * split_ratio[1])
    n_test = len(files) - n_train - n_val

    assert 0 <= n_train <= len(files)
    assert 0 <= n_val <= len(files)
    assert 0 <= n_test <= len(files)

    # Split train/validation based on fold index
    val_start = FLAGS.fold * (n_train + n_val) // (rel_ratio[0] + rel_ratio[1])

    train_files.extend(files[0: val_start])
    validation_files.extend(files[val_start: val_start + n_val])
    train_files.extend(files[val_start + n_val: n_train + n_val])

    test_files.extend(files[n_train + n_val:])

    print('%s train: %d val: %d test: %d' % (label, n_train, n_val, n_test))

assert len(train_files) + len(test_files) + len(validation_files) == n_image_files_used


def write_to_file(out_file, file_list):
    with open(out_file, 'w') as f:
        for fn in file_list:
            f.write('%s\n' % fn)

train_files.sort()
test_files.sort()
validation_files.sort()

write_to_file('%s_train.txt' % FLAGS.granularity, train_files)
write_to_file('%s_test.txt' % FLAGS.granularity, test_files)
write_to_file('%s_validation.txt' % FLAGS.granularity, validation_files)
