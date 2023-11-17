import numpy as np
from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV  # being deprecated, works for now
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from utils import load_dataset
from transforms import random_rotate
import gflags
import sys
import cv2


gflags.DEFINE_string('task', 'tribe', '')
gflags.DEFINE_boolean('align_orientation', True, 'Align the orientations of phytoliths.')
gflags.DEFINE_string('angles', '0', 'Augment images by rotating using the specified angles.')
gflags.DEFINE_string('styles', 'max_global_autocontrast_normalized', '')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


TASK = FLAGS.task
IMG_SIZE = 32
ALIGN_ORIENTATION = FLAGS.align_orientation

ALL_STYLES = [
    'max_global_autocontrast_normalized',
    'max_images',

    'mean_global_autocontrast_normalized',
    'mean_images',

    'focus_stacking_global_autocontrast_normalized',
    'focus_stacking_images',

    'global_autocontrast_normalized',
    'median_images',
]


if TASK == 'genus':
    # For genus we don't use unnormalized images because every genus seems to captured
    # at different lighting conditions.
    ALL_STYLES = [
        'max_global_autocontrast_normalized',
        'mean_global_autocontrast_normalized',
        'focus_stacking_global_autocontrast_normalized',
        'global_autocontrast_normalized',
    ]

if FLAGS.styles == 'all':
    STYLES = ALL_STYLES
else:
    STYLES = FLAGS.styles.split(',')

ANGLES = [float(s) for s in FLAGS.angles.split(',')] if FLAGS.angles != '' else None


print 'task:', TASK
print 'angles:', ANGLES
print 'styles:', STYLES
print 'align orientation:', ALIGN_ORIENTATION


def rotate(images, angles):
    res = []
    rng = np.random.RandomState()
    for angle in angles:
        # Note that we abuse random_rotate() to make deterministic rotation.
        rotated = random_rotate(np.expand_dims(images, 1), rng, [angle]).squeeze()
        res.append(rotated)
    return np.concatenate(res)


train_images, train_labels, test_images, test_labels = load_dataset(TASK, STYLES, IMG_SIZE, ALIGN_ORIENTATION)

# Since we concatenate different styles we only need one set of labels.
train_labels = train_labels[0]
test_labels = test_labels[0]


if ANGLES is not None:
    augmented_train_images = []
    for i in xrange(train_images.shape[0]):  # For each style
        augmented_train_images.append(rotate(train_images[i], ANGLES))

    train_labels = np.repeat(np.expand_dims(train_labels, 0), len(ANGLES), 0).flatten()
    train_images = np.array(augmented_train_images)

# Assume images of multiple styles into one image.
train_images = np.concatenate(train_images, 1)
test_images = np.concatenate(test_images, 1)

train_size = train_images.shape[0]
test_size = test_images.shape[0]

best_C = None
best_gamma = None
best_accuracy = 0
best_avg_class_accuracy = 0

n_class = int(np.max(train_labels) + 1)
class_counts = np.bincount(test_labels)


def calc_avg_class_accuracy(preds, gts):
    class_corrects = [0] * n_class
    for i in xrange(len(preds)):
        pred = preds[i]
        gt = gts[i]
        if pred == gt:
            class_corrects[gt] += 1
    return np.mean(class_corrects / class_counts.astype(np.float32))


GAMMAS = [1e-2, 1e-3, 1e-4, 1e-5]
CS = [10, 100, 1000, 10000]


def cross_validate():
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': GAMMAS, 'C': CS}]

    scores = ['accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s' % score)
        clf.fit(train_images.reshape(train_size, -1), train_labels)

        print("Best parameters set found on training set:")
        print
        print(clf.best_params_)
        print
        print("Grid scores on training set:")
        print
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print

        print("Detailed classification report:")
        print
        print("The model is trained on the full training set.")
        print("The scores are computed on the full test set.")
        print
        y_true, y_pred = test_labels, clf.predict(test_images.flatten().reshape(test_size, -1))
        print(classification_report(y_true, y_pred))
        print

        accuracy = np.sum(y_pred == test_labels) / float(len(test_labels))
        avg_class_accuracy = calc_avg_class_accuracy(y_pred, test_labels)

        print 'cv score: %s test_accuracy: %f avg_class_accuracy: %f' % (score, accuracy, avg_class_accuracy)


cross_validate()


for C in CS:
    for gamma in GAMMAS:
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(train_images.reshape(train_size, -1), train_labels)
        preds = clf.predict(test_images.reshape(test_size, -1))

        accuracy = np.sum(preds == test_labels) / float(len(test_labels))
        accuracy2 = clf.score(test_images.reshape(test_size, -1), test_labels)
        assert np.allclose(accuracy, accuracy2)

        avg_class_accuracy = calc_avg_class_accuracy(preds, test_labels)

        print 'C: %6d gamma: %6f accuracy: %.2f avg_class_accuracy: %.2f' % (C, gamma, accuracy, avg_class_accuracy)

        if accuracy > best_accuracy:
            best_C = C
            best_gamma = gamma
            best_accuracy = accuracy
            best_avg_class_accuracy = avg_class_accuracy


print 'best C:', best_C
print 'best gamma:', best_gamma
print 'best accuracy:', best_accuracy
print 'best avg class accuracy:', best_avg_class_accuracy
