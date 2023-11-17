import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, class_names,
                          title='Confusion matrix',
                          cmap=plt.cm.jet, axis_label=True, color_bar=True, ticks=True):
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if color_bar:
        plt.colorbar()
    plt.grid(True)

    if ticks:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    plt.title(title)

    if axis_label:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


if __name__ == '__main__':
    import gflags
    import sys

    gflags.DEFINE_string('task', 'genus', '')
    gflags.DEFINE_string('model', 'vgg19', '')
    gflags.DEFINE_string('option', 'unweighted', '')
    gflags.DEFINE_string('style', 'mean_global_autocontrast_normalized', '')
    gflags.DEFINE_string('out_file', '', 'save figure to this file')
    gflags.DEFINE_string('title', '', '')
    gflags.DEFINE_boolean('ticks', False, '')
    gflags.DEFINE_boolean('color_bar', False, '')
    gflags.DEFINE_boolean('axis_label', False, '')

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    TASK = FLAGS.task
    MODEL = FLAGS.model
    OPTION = FLAGS.option
    STYLE = FLAGS.style
    OUT_FILE = FLAGS.out_file

    CONFUSION_MATRIX_FILE = '../../data/%s_classification_results/%s/%s/confusion_matrix/%s_confusion_mat.txt' % (
        TASK, MODEL, OPTION, STYLE
    )

    with open('../../data/genus_names.txt') as f:
        class_names = filter(lambda x: x != '', [l.strip() for l in f.readlines()])

    m = np.genfromtxt(CONFUSION_MATRIX_FILE, delimiter=',')

    plot_confusion_matrix(m, class_names, FLAGS.title, ticks=FLAGS.ticks, color_bar=FLAGS.color_bar, axis_label=FLAGS.axis_label)

    if OUT_FILE != '':
        plt.savefig(OUT_FILE)

    plt.show()
