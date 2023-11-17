from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.insert(0, '../')
from LoadData import LoadData
import copy

plt.ion()   # interactive mode

use_gpu = torch.cuda.is_available()


def normalize(images):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())
    return (images - mean) / std


def load_images(style):
    def make_3_channel_normalized(images):
        return normalize(images.expand(images.size(0), 3, images.size(2), images.size(3)))

    ld = LoadData(256)

    train_images, train_labels = ld.loadTrainData('/home/xymeng/Dropbox/phd/courses/cse546/', imageStyle=style)
    valid_images, valid_labels = ld.loadValidData('/home/xymeng/Dropbox/phd/courses/cse546/', imageStyle=style)
    test_images, test_labels = ld.loadTestData('/home/xymeng/Dropbox/phd/courses/cse546', imageStyle=style)

    train_images = make_3_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)).cuda())
    valid_images = make_3_channel_normalized(Variable(torch.from_numpy(valid_images.astype(np.float32)).unsqueeze(1)).cuda())
    test_images = make_3_channel_normalized(Variable(torch.from_numpy(test_images.astype(np.float32)).unsqueeze(1)).cuda())

    train_labels = Variable(torch.from_numpy(train_labels)).cuda()
    valid_labels = Variable(torch.from_numpy(valid_labels)).cuda()
    test_labels = Variable(torch.from_numpy(test_labels)).cuda()

    return train_images, valid_images, test_images, train_labels, valid_labels, test_labels


def evaluate(model, images, labels):
    model.train(False)

    assert images.size(0) == labels.size(0)
    size = images.size(0)

    BATCH = 1

    n_correct = 0

    for i in range(size // BATCH):
        batch_images = images[i * BATCH:i * BATCH + BATCH]
        batch_labels = labels[i * BATCH:i * BATCH + BATCH]
        _, preds = torch.max(model(batch_images), 1)
        n_correct += torch.sum(preds == batch_labels).data[0]

    accuracy = n_correct / float(size)

    return accuracy


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

    print('Training Loss: %.4f Acc: %.4f' % (avg_loss, accuracy))
    return avg_loss, accuracy


def train_val_test(style, out_file, max_epochs=50, batch_size=32):
    train_images, valid_images, test_images, train_labels, valid_labels, test_labels = load_images(style)

    train_size = train_labels.size(0)

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.cuda()

    weight_tensor = torch.cat([torch.sum(train_labels).float() / train_size,
                               (train_size - torch.sum(train_labels).float()) / train_size])
    weight_tensor = weight_tensor.cuda()

    crit = nn.CrossEntropyLoss(weight=weight_tensor)

    opt = optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)

    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    rng = np.random.RandomState(123456)

    for i in xrange(max_epochs):
        print('----- epoch %r -----' % i)

        scheduler.step()
        print('learning rate: %.8f' % scheduler.get_lr()[0])

        train_loss, train_accuracy = train_one_epoch(model, crit, opt, train_images, train_labels, batch_size, rng)

        val_accuracy = evaluate(model, valid_images, valid_labels)
        print('validation accuracy: %.4f' % val_accuracy)

        test_accuracy = evaluate(model, test_images, test_labels)
        print('test accuracy: %.4f' % test_accuracy)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

    with open(out_file, 'w') as f:
        for a in zip(train_accuracies, val_accuracies, test_accuracies):
            f.write('%r, %r, %r\n' % a)


styles = [
    'max_global_autocontrast_normalized',
    'max_images',

    'mean_global_autocontrast_normalized',
    'mean_images',

    'focus_stacking_global_autocontrast_normalized',
    'focus_stacking_images',

    'median_images',
    'global_autocontrast_normalized',
]


for style in styles:
    train_val_test(style, style + '.txt')
