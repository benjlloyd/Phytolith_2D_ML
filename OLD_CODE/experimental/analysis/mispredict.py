from utils import load_classes
import numpy as np


# Find high off-diagonal values

def select_off_diagonal_values(confm, low_thres, high_thres):
    res = []
    for i in xrange(n_class):
        for j in xrange(n_class):
            if i == j:
                continue
            value = confm[i, j]
            if low_thres <= value <= high_thres:
                res.append((i, j, value))
    res.sort(key=lambda e: e[2])
    return res[::-1]


if __name__ == "__main__":
    import gflags
    import os
    import sys
    import glob
    from shutil import copyfile

    gflags.DEFINE_string('task', 'genus', '')
    gflags.DEFINE_string('model', 'vgg19', '')
    gflags.DEFINE_string('option', 'unweighted', '')
    gflags.DEFINE_string('style', 'mean_global_autocontrast_normalized', '')
    gflags.DEFINE_string('dump_dir', '', 'Dump the test images into a directory')
    gflags.DEFINE_string('image_dir', '', 'Empty string to auto infer the directory.')
    gflags.DEFINE_float('low_thres', 0.5, '')
    gflags.DEFINE_float('high_thres', 1.0, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    TASK = FLAGS.task
    MODEL = FLAGS.model
    OPTION = FLAGS.option
    STYLE = FLAGS.style
    DUMP_DIR = FLAGS.dump_dir
    LOW_THRES = FLAGS.low_thres
    HIGH_THRES = FLAGS.high_thres

    CONFUSION_MATRIX_FILE = '../../data/%s_classification_results/%s/%s/confusion_matrix/%s_confusion_mat.txt' % (
        TASK, MODEL, OPTION, STYLE
    )

    confmat = np.genfromtxt(CONFUSION_MATRIX_FILE, dtype=np.int32, delimiter=',')
    labels = load_classes('../../data/%s_names.txt' % TASK)
    assert confmat.shape[0] == len(labels)

    confmat_normed = confmat.astype(np.float32)
    confmat_normed = confmat_normed / confmat_normed.sum(axis=1, keepdims=True)

    n_class = len(labels)

    entries = select_off_diagonal_values(confmat_normed, LOW_THRES, HIGH_THRES)

    total_mispredicts = sum([confmat[i, j] for i, j, _ in entries])
    print 'total mispredicts:', total_mispredicts

    print '===== most mispredicted classes (groundtruth, prediction) ====='
    print [(labels[i], labels[j], v) for i, j, v in entries]

    if DUMP_DIR != '':
        with open('../../data/%s_test.txt' % TASK) as f:
            test_files = set(filter(lambda x: x != '', [l.strip() for l in f.readlines()]))
        for i, j, v in entries:
            gt = labels[i]
            pred = labels[j]

            if FLAGS.image_dir != '':
                src_dir = FLAGS.image_dir
            else:
                src_dir = os.path.join('../../data/', STYLE)

            gt_files = filter(lambda x: os.path.basename(x) in test_files,
                              glob.glob(os.path.join(src_dir, '*%s.*' % gt)))

            pred_files = filter(lambda x: os.path.basename(x) in test_files,
                                glob.glob(os.path.join(src_dir, '*%s.*' % pred)))

            all_files = gt_files + pred_files
            dst_dir = '%s/%s-%s/' % (DUMP_DIR, gt, pred)
            os.makedirs(dst_dir, 0744)
            for fn in all_files:
                copyfile(fn, os.path.join(dst_dir, os.path.basename(fn)))
