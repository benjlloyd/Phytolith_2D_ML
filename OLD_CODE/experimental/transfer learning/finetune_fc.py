from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from utils import get_dataset_loader
from transforms import random_rotate
import gflags
import sys


gflags.DEFINE_string('task', 'genus', '')
gflags.DEFINE_boolean('weighted_loss', False, 'Whether we use weighted loss during training.')
gflags.DEFINE_string('model', 'resnet18', '')
gflags.DEFINE_string('model_weights_file', '', 'Load weights from this file. Empty string means using default weights.')
gflags.DEFINE_integer('image_size', 256, '')
gflags.DEFINE_integer('max_epochs', 50, '')
gflags.DEFINE_string('styles', 'all',
                     'comma-separated style names or all. Features from these styles will be augmented.')
gflags.DEFINE_boolean('bootstrap', False, 'Use bootstrap to make number of samples for each class the same.')
gflags.DEFINE_string('augment_strategy', 'select_best', 'How to augment multi style images. Can be "select_best" or "concat".')
gflags.DEFINE_boolean('random_rotation', False, 'Randomly rotate training images.')
gflags.DEFINE_string('angles', '', '')
gflags.DEFINE_boolean('align_orientation', False, 'Align the orientations of phytoliths.')
gflags.DEFINE_string('log_prefix', '', '')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


TASK = FLAGS.task
USE_WEIGHTED_LOSS = FLAGS.weighted_loss
PRE_TRAINED_MODEL = FLAGS.model
MODEL_WEIGHTS_FILE = FLAGS.model_weights_file
IMAGE_SIZE = FLAGS.image_size
MAX_EPOCHS = FLAGS.max_epochs
BOOTSTRAP = FLAGS.bootstrap
RANDOM_ROTATION = FLAGS.random_rotation
AUGMENT_STRATEGY = FLAGS.augment_strategy
ALIGN_ORIENTATION = FLAGS.align_orientation
ANGLES = [float(s) for s in FLAGS.angles.split(',')] if FLAGS.angles != '' else None
LOG_PREFIX = FLAGS.log_prefix


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


print('task: %r' % TASK)
print('model: %r' % PRE_TRAINED_MODEL)
print('image size: %r' % IMAGE_SIZE)
print('styles: %r' % STYLES)
print('augment_strategy: %r' % AUGMENT_STRATEGY)
print('random rotation: %r' % RANDOM_ROTATION)
if RANDOM_ROTATION:
    print('angles: %r' % ANGLES)


def normalize(images):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return (images - mean) / std


def load_dataset(style):
    def make_3_channel_normalized(images):
        return normalize(images.expand(images.size(0), 3, images.size(2), images.size(3)))

    transforms = None
    if ALIGN_ORIENTATION:
        transforms = {}
        with open('../../data/align_transforms.txt') as f:
            for l in f:
                tokens = [e.strip() for e in l.split(',')]
                transform = np.array([float(e) for e in tokens[1:]]).reshape((3, 3))
                transforms[tokens[0]] = transform

    loader = get_dataset_loader(TASK)(IMAGE_SIZE)

    train_images, train_labels, class_dict = loader.load('../../data/%s' % style, '../../data/%s_train.txt' % TASK, None, transforms)
    test_images, test_labels, _ = loader.load('../../data/%s' % style, '../../data/%s_test.txt' % TASK, class_dict, transforms)

    train_images = make_3_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)))
    test_images = make_3_channel_normalized(Variable(torch.from_numpy(test_images.astype(np.float32)).unsqueeze(1)))

    train_labels = Variable(torch.from_numpy(train_labels))
    test_labels = Variable(torch.from_numpy(test_labels))

    return train_images, test_images, train_labels, test_labels, class_dict


def evaluate(model, images, labels, n_class):
    model.train(False)

    confusion_matrix = np.zeros((n_class, n_class), np.int32)

    assert images.size(0) == labels.size(0)
    size = images.size(0)

    batch_size = 1

    n_correct = 0
    n_total = 0

    n_correct_class = [0] * n_class
    total_class = [0] * n_class

    all_scores = []

    def update_per_class_stats(preds, gts):
        preds = preds.data.cpu().numpy()
        gts = gts.data.cpu().numpy()

        for j in xrange(len(preds)):
            pred = preds[j]
            gt = gts[j]

            total_class[gt] += 1
            if pred == gt:
                n_correct_class[gt] += 1

            confusion_matrix[gt, pred] += 1

    for i in range(size // batch_size):
        batch_images = images[i * batch_size:i * batch_size + batch_size].cuda()
        batch_labels = labels[i * batch_size:i * batch_size + batch_size].cuda()

        # Note that scores are without softmax.
        scores = model(batch_images)
        all_scores.append(scores.data.cpu().numpy().flatten())

        _, preds = torch.max(scores, 1)
        n_correct += torch.sum(preds == batch_labels).data[0]
        n_total += batch_size
        update_per_class_stats(preds, batch_labels)

    accuracy = n_correct / float(n_total)
    accuracy_per_class = np.array(n_correct_class) / np.array(total_class)

    return accuracy, accuracy_per_class, confusion_matrix, np.array(all_scores)


def evaluate_select_best(model, images_set, labels, n_class):
    model.train(False)

    confusion_matrix = np.zeros((n_class, n_class), np.int32)

    size = images_set[0].size(0)

    batch_size = 1

    n_correct = 0
    n_total = 0

    n_correct_class = [0] * n_class
    total_class = [0] * n_class

    n_style = len(images_set)

    def update_per_class_stats(preds, gts):
        preds = preds.data.cpu().numpy()
        gts = gts.data.cpu().numpy()

        for j in xrange(len(preds)):
            pred = preds[j]
            gt = gts[j]

            total_class[gt] += 1
            if pred == gt:
                n_correct_class[gt] += 1

            confusion_matrix[gt, pred] += 1

    for i in range(size // batch_size):
        all_max_scores = []
        all_max_indices = []

        for j in xrange(n_style):
            batch_images = images_set[j][i * batch_size:i * batch_size + batch_size].cuda()
            batch_labels = labels[i * batch_size:i * batch_size + batch_size].cuda()

            # Note that scores are without softmax.
            scores = model(batch_images)
            max_scores, max_indices = torch.max(scores, 1)

            all_max_scores.append(max_scores)
            all_max_indices.append(max_indices)

        all_max_scores = torch.cat(all_max_scores)
        all_max_indices = torch.cat(all_max_indices)

        _, score_indices = torch.max(all_max_scores, 0)

        preds = all_max_indices[score_indices]

        update_per_class_stats(preds, batch_labels)
        n_correct += torch.sum(preds == batch_labels).data[0]
        n_total += batch_size

    accuracy = n_correct / float(n_total)
    accuracy_per_class = np.array(n_correct_class) / np.array(total_class)

    return accuracy, accuracy_per_class, confusion_matrix


def bootstrap(labels, rng):
    label_counts = np.bincount(labels)
    most_freq_label = np.argsort(label_counts)[-1]
    max_count = label_counts[most_freq_label]

    label_sorted_indices = np.argsort(labels)

    n_class = len(label_counts)

    bootstrap_indices = np.zeros(n_class * max_count, np.int32)

    i = 0
    start = 0
    for c in label_counts:
        bootstrap_indices[i * max_count: (i + 1) * max_count] = rng.choice(
            label_sorted_indices[start: start + c], max_count, replace=True)
        start += c
        i += 1

    return bootstrap_indices


def train_one_epoch(model, crit, opt, images, labels, batch_size, rng):
    if BOOTSTRAP:
        indices = bootstrap(labels.data.cpu().numpy(), rng)
        indices = Variable(torch.from_numpy(indices).long())
        labels = labels[indices]
        images = images[indices]

    model.train(True)
    size = images.size(0)

    assert images.size(0) == labels.size(0)

    if RANDOM_ROTATION:
        images = images.data.cpu().numpy()
        images = Variable(torch.from_numpy(random_rotate(images, rng, ANGLES)))

    losses = []
    n_correct = 0
    n_total = 0

    indices = np.arange(size)
    rng.shuffle(indices)

    indices = torch.from_numpy(indices).long()
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    for i in range(size // batch_size):
        batch_images = shuffled_images[i * batch_size: (i + 1) * batch_size].cuda()
        batch_labels = shuffled_labels[i * batch_size: (i + 1) * batch_size].cuda()

        opt.zero_grad()

        scores = model(batch_images)
        _, preds = torch.max(scores, 1)
        loss = crit(scores, batch_labels)

        loss.backward()
        opt.step()

        losses.append(loss.data[0])
        n_correct += torch.sum(preds == batch_labels).data[0]
        n_total += batch_size

    avg_loss = np.mean(losses)
    accuracy = float(n_correct) / n_total

    print('training loss: %.4f accuracy: %.4f' % (avg_loss, accuracy))
    return avg_loss, accuracy


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.fd = open(self.log_file, 'w')

    def print_and_log(self, s):
        print(s)
        self.fd.write(s + '\n')


class Identity(nn.Module):
    def forward(self, x):
        return x
        

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.train(False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.n_features = self.model.fc.in_features
        self.model.fc = Identity()

    def forward(self, input):
        return self.model(input)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = torchvision.models.vgg19_bn(pretrained=True).cuda()
        self.model.train(False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.n_features = 4096

    def forward(self, input):
        features = self.model.features(input).view(input.size(0), -1)
        out = self.model.classifier._modules['0'](features)  # First FC
        out = self.model.classifier._modules['1'](out)  # ReLU
        return out


class SingleStyleNet(nn.Module):
    def __init__(self, feature_extractor, n_class):
        super(SingleStyleNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_extractor.n_features, n_class)

    def forward(self, x):
        out = self.feature_extractor(x)
        return self.fc(out)


class MultiStyleConcatNet(nn.Module):
    def __init__(self, feature_extractor, n_class, multiplier=1):
        super(MultiStyleConcatNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_extractor.n_features * multiplier, n_class)

    def forward(self, x):
        # input: batch_size x n_style x C x H x W
        assert x.dim() == 5
        batch_size, n_style, C, H, W = x.size()

        out = self.feature_extractor(x.view(batch_size * n_style, C, H, W))
        out = out.view(batch_size, -1)

        return self.fc(out)


def get_feature_extractor():
    if PRE_TRAINED_MODEL == 'resnet18':
        feature_extractor = ResNet18()
    elif PRE_TRAINED_MODEL == 'vgg19':
        feature_extractor = VGG19()
    else:
        raise RuntimeError('Unknown model %s' % PRE_TRAINED_MODEL)

    return feature_extractor


def get_net(n_class):
    feature_extractor = get_feature_extractor()

    if len(STYLES) > 1:
        if AUGMENT_STRATEGY == 'select_best':
            net = SingleStyleNet(feature_extractor, n_class)
        elif AUGMENT_STRATEGY == 'concat':
            net = MultiStyleConcatNet(feature_extractor, n_class, len(STYLES))
        else:
            raise RuntimeError('unknown augment strategy')
    else:
        net = SingleStyleNet(feature_extractor, n_class)

    if MODEL_WEIGHTS_FILE != '':
        net.feature_extractor.model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE))

    return net


def train_test(train_images, test_images, train_labels, test_labels, class_dict, style, max_epochs=50, batch_size=32):
    n_class = len(class_dict)

    net = get_net(n_class).cuda()

    train_n_labels_per_class = np.bincount(train_labels.data.cpu().numpy())
    test_n_labels_per_class = np.bincount(test_labels.data.cpu().numpy())

    print(net)

    def inverse_freq_ratio(n_labels_per_class):
        total = np.sum(n_labels_per_class)
        freq_ratio = n_labels_per_class / total.astype(np.float32)
        inv = 1.0 / freq_ratio
        r = inv / np.sum(inv)
        return r

    weights = None
    if USE_WEIGHTED_LOSS:
        weights = torch.from_numpy(inverse_freq_ratio(train_n_labels_per_class).astype(np.float32)).cuda()

    crit = nn.CrossEntropyLoss(weight=weights)

    opt = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)

    train_accuracies = []
    test_accuracies = []
    test_per_class_accuracies = []

    rng = np.random.RandomState(123456)

    for i in xrange(max_epochs):
        print('----- epoch %r -----' % i)

        scheduler.step()
        print('learning rate: %.8f' % scheduler.get_lr()[0])

        train_loss, train_accuracy = train_one_epoch(net, crit, opt, train_images, train_labels, batch_size, rng)

        if style == 'multi_select_best':
            test_accuracy, test_accuracy_per_class, _ = evaluate_select_best(net, test_images, test_labels, n_class)
        else:
            test_accuracy, test_accuracy_per_class, _, _ = evaluate(net, test_images, test_labels, n_class)

        print('test accuracy: %.4f' % test_accuracy)
        print('average class accuracy: %.4f' % np.mean(test_accuracy_per_class))
        for tribe, label in class_dict.iteritems():
            print('class test accuracy %16s label: %2d quantity: %3d: %.4f' %
                                 (tribe, label, test_n_labels_per_class[label], test_accuracy_per_class[label]))

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_per_class_accuracies.append(test_accuracy_per_class)

    if style != 'multi_select_best':
        _, _, confusion_matrix, scores = evaluate(net, test_images, test_labels, n_class)
        np.savetxt(LOG_PREFIX + '_confusion_mat.txt', confusion_matrix, fmt='%d', delimiter=',')
        np.savetxt(LOG_PREFIX + '_scores.txt', scores, fmt='%f', delimiter=',')

        _, _, confusion_matrix, _ = evaluate(net, train_images, train_labels, n_class)
        np.savetxt(LOG_PREFIX + '_train_confusion_mat.txt', confusion_matrix, fmt='%d', delimiter=',')
    else:
        _, _, confusion_matrix = evaluate_select_best(net, test_images, test_labels, n_class)
        np.savetxt(LOG_PREFIX + '_confusion_mat.txt', confusion_matrix, fmt='%d', delimiter=',')

    with open(LOG_PREFIX + '.txt', 'w') as f:
        for a in zip(train_accuracies, test_accuracies, test_per_class_accuracies):
            f.write('%r, %r, %r\n' % a)


def run_single_style(style):
    train_images, test_images, train_labels, test_labels, class_dict = load_dataset(style)

    print('train_images: %d' % train_images.size(0))
    print('test_images: %d' % test_images.size(0))

    train_test(train_images, test_images, train_labels, test_labels, class_dict, style, max_epochs=MAX_EPOCHS)


def run_multi_style():
    all_train_images = []
    all_test_images = []
    all_train_labels = []

    for style in STYLES:
        train_images, test_images, train_labels, test_labels, class_dict = load_dataset(style)
        all_train_images.append(train_images)
        all_test_images.append(test_images)
        all_train_labels.append(train_labels)

    print('train_images: %d' % train_images.size(0))
    print('test_images: %d' % test_images.size(0))

    if len(STYLES) > 1:

        if AUGMENT_STRATEGY == 'select_best':
            all_train_images = torch.cat(all_train_images)
            all_train_labels = torch.cat(all_train_labels)

            train_test(all_train_images, all_test_images, all_train_labels, test_labels, class_dict,
                       'multi_' + AUGMENT_STRATEGY, max_epochs=MAX_EPOCHS)

        elif AUGMENT_STRATEGY == 'concat':
            all_train_images = torch.stack(all_train_images, 0).transpose(0, 1)
            all_test_images = torch.stack(all_test_images, 0).transpose(0, 1)

            train_test(all_train_images, all_test_images, train_labels, test_labels, class_dict,
                       'multi_' + AUGMENT_STRATEGY, max_epochs=MAX_EPOCHS)


if len(STYLES) > 1:
    run_multi_style()
else:
    run_single_style(STYLES[0])
