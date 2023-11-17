from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from LoadData import TribeLoader
import gflags
import sys

gflags.DEFINE_boolean('weighted_loss', False, 'Whether we use weighted loss during training.')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

USE_WEIGHTED_LOSS = FLAGS.weighted_loss


def normalize(images):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())
    return (images - mean) / std


def load_images(style):
    def make_3_channel_normalized(images):
        return normalize(images.expand(images.size(0), 3, images.size(2), images.size(3)))

    loader = TribeLoader(256)

    train_images, train_labels, tribe_dict = loader.load('../../data/%s' % style, '../../data/tribe_train.txt')
    test_images, test_labels, _ = loader.load('../../data/%s' % style, '../../data/tribe_test.txt', tribe_dict)

    train_images = make_3_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)).cuda())
    test_images = make_3_channel_normalized(Variable(torch.from_numpy(test_images.astype(np.float32)).unsqueeze(1)).cuda())

    train_labels = Variable(torch.from_numpy(train_labels)).cuda()
    test_labels = Variable(torch.from_numpy(test_labels)).cuda()

    return train_images, test_images, train_labels, test_labels, tribe_dict


def evaluate(model, images, labels, n_class):
    model.train(False)

    assert images.size(0) == labels.size(0)
    size = images.size(0)

    BATCH = 1

    n_correct = 0

    n_correct_class = [0] * n_class
    total_class = [0] * n_class

    def update_per_class_stats(preds, gts):
        preds = preds.data.cpu().numpy()
        gts = gts.data.cpu().numpy()
        for j in xrange(len(preds)):
            pred = preds[j]
            gt = gts[j]

            total_class[gt] += 1
            if pred == gt:
                n_correct_class[gt] += 1

    for i in range(size // BATCH):
        batch_images = images[i * BATCH:i * BATCH + BATCH]
        batch_labels = labels[i * BATCH:i * BATCH + BATCH]
        _, preds = torch.max(model(batch_images), 1)
        n_correct += torch.sum(preds == batch_labels).data[0]
        update_per_class_stats(preds, batch_labels)

    accuracy = n_correct / float(size)
    accuracy_per_class = np.array(n_correct_class) / np.array(total_class)

    return accuracy, accuracy_per_class


def train_one_epoch(model, crit, opt, images, labels, batch_size, rng):
    model.train(True)
    size = images.size(0)

    assert images.size(0) == labels.size(0)

    losses = []
    n_correct = 0

    indices = np.arange(size)
    rng.shuffle(indices)

    indices = torch.from_numpy(indices).cuda().long()
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    for i in range(size // batch_size):
        batch_images = shuffled_images[i * batch_size: (i + 1) * batch_size]
        batch_labels = shuffled_labels[i * batch_size: (i + 1) * batch_size]

        # zero the parameter gradients
        opt.zero_grad()

        # forward
        out = model(batch_images)
        _, preds = torch.max(out, 1)
        loss = crit(out, batch_labels)

        loss.backward()
        opt.step()

        losses.append(loss.data[0])

        n_correct += torch.sum(preds == batch_labels).data[0]

    avg_loss = np.mean(losses)
    accuracy = float(n_correct) / size

    print('training loss: %.4f accuracy: %.4f' % (avg_loss, accuracy))
    return avg_loss, accuracy


def train_test(style, out_file, max_epochs=50, batch_size=32):
    train_images, test_images, train_labels, test_labels, tribe_dict = load_images(style)
    n_class = len(tribe_dict)

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(tribe_dict))

    train_n_labels_per_class = np.bincount(train_labels.data.cpu().numpy())
    test_n_labels_per_class = np.bincount(test_labels.data.cpu().numpy())

    model = model.cuda()

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

    opt = optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)

    train_accuracies = []
    test_accuracies = []

    rng = np.random.RandomState(123456)

    for i in xrange(max_epochs):
        print('----- epoch %r -----' % i)

        scheduler.step()
        print('learning rate: %.8f' % scheduler.get_lr()[0])

        train_loss, train_accuracy = train_one_epoch(model, crit, opt, train_images, train_labels, batch_size, rng)

        test_accuracy, test_accuracy_per_class = evaluate(model, test_images, test_labels, n_class)

        print('test accuracy: %.4f' % test_accuracy)
        print('average class accuracy: %.4f' % np.mean(test_accuracy_per_class))
        for tribe, label in tribe_dict.iteritems():
            print('class test accuracy %16s label: %2d quantity: %3d: %.4f' %
                  (tribe, label, test_n_labels_per_class[label], test_accuracy_per_class[label]))

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    with open(out_file, 'w') as f:
        for a in zip(train_accuracies, test_accuracies):
            f.write('%r, %r\n' % a)


styles = [
    'max_global_autocontrast_normalized',
    # 'max_images',

    # 'mean_global_autocontrast_normalized',
    # 'mean_images',
    #
    # 'focus_stacking_global_autocontrast_normalized',
    # 'focus_stacking_images',
    #
    # 'median_images',
    # 'global_autocontrast_normalized',
]


for style in styles:
    train_test(style, style + '.txt')
