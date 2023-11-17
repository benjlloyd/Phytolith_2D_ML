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

def load_images(styles):
    ld = LoadData(256)
    train_images = None
    valid_images = None
    test_images = None
    loadedFirst = False
    for style in styles:
        train_image, train_labels = ld.loadTrainData('/home/jifan/Dropbox/', imageStyle=style)
        valid_image, valid_labels = ld.loadValidData('/home/jifan/Dropbox/', imageStyle = style)
        test_image, test_labels = ld.loadTrainData('/home/jifan/Dropbox/', imageStyle=style)
        if not loadedFirst:
            loadedFirst = True
            train_images = np.reshape(train_image, (train_image.shape[0], 1, train_image.shape[1], train_image.shape[2]))
            valid_images = np.reshape(valid_image, (valid_image.shape[0], 1, valid_image.shape[1], valid_image.shape[2]))
            test_images = np.reshape(test_image, (test_image.shape[0], 1, test_image.shape[1], test_image.shape[2]))
        else:
            train_images = np.concatenate((train_images, np.reshape(train_image, (train_image.shape[0], 1, train_image.shape[1], train_image.shape[2]))), axis=1)
            train_images = np.asarray(train_images, dtype=np.float32)
            valid_images = np.concatenate((valid_images, np.reshape(valid_image, (valid_image.shape[0], 1, valid_image.shape[1], valid_image.shape[2]))), axis=1)
            valid_images = np.asarray(valid_images, dtype=np.float32)
            test_images = np.concatenate((test_images, np.reshape(test_image, (test_image.shape[0], 1, test_image.shape[1], test_image.shape[2]))), axis=1)
            test_images = np.asarray(test_images, dtype=np.float32)

    return train_images, valid_images, test_images, train_labels, valid_labels, test_labels


def train_model(model, criterion, optimizer, scheduler, data, num_epochs=25, batch_size = 5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
                index = 0
                scheduler.step()
                model.train(True)  # Set model to training mode
                (images, labels) = data[0]
            else:
                model.train(False)
                index = 1
                model.train(False)  # Set model to evaluate mode
                (images, labels) = data[1]

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i in range((data[2][index] - 1) // batch_size + 1):
                # wrap them in Variable
                if i == data[2][index] - 1 // batch_size:
                    if use_gpu:
                        inputs = Variable(torch.from_numpy(images[i * batch_size : data[2][index]]).cuda())
                        label = Variable(torch.from_numpy(labels[i * batch_size : data[2][index]]).cuda())
                    else:
                        inputs, label = Variable(torch.from_numpy(images[i * batch_size : data[2][index]])),\
                                        Variable(torch.from_numpy(labels[i * batch_size : data[2][index]]))
                else:
                    if use_gpu:
                        inputs = Variable(torch.from_numpy(images[i * batch_size : (i + 1) * batch_size]).cuda())
                        label = Variable(torch.from_numpy(labels[i * batch_size : (i + 1) * batch_size]).cuda())
                    else:
                        inputs, label = Variable(torch.from_numpy(images[i * batch_size : (i + 1) * batch_size])),\
                                        Variable(torch.from_numpy(labels[i * batch_size : (i + 1) * batch_size]))

                mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
                std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())
                inputs = (inputs - mean) / std

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, label)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == label.data)

            epoch_loss = running_loss / data[2][index]
            epoch_acc = running_corrects / data[2][index]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_acc, best_model_wts

def train(styles):
    train_images, valid_images, _, train_labels, valid_labels, _ = load_images(styles)
    train_size = train_labels.shape[0]
    valid_size = valid_labels.shape[0]

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_conv = model_conv.cuda()
        weight_tensor = torch.from_numpy(np.array([np.sum(train_labels) / train_size, (train_size - np.sum(train_labels))/train_size], dtype=np.float32))
        weight_tensor = weight_tensor.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.2)
    return train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                       [(train_images, train_labels), (valid_images, valid_labels), (train_size, valid_size)],
                       num_epochs=25)

styles = ['mean_global_autocontrast_normalized', 'focus_stacking_global_autocontrast_normalized',
          'global_autocontrast_normalized', 'max_global_autocontrast_normalized', 'max_images',
          'mean_global_autocontrast_normalized', 'mean_images', 'median_images']

# styles = ['focus_stacking_global_autocontrast_normalized',
#           ]


best_acc = 0
best_model = None
best_styles = []
for i in range(len(styles)):
    for j in range(i, len(styles)):
        for k in range(j, len(styles)):
            style = [styles[i], styles[j], styles[k]]
            print (style)
            acc, model_wts = train(style)
            if acc >= best_acc:
                best_acc = acc
                best_model_wts = model_wts
                best_styles = style
            print ("**********************\n\n")

_, _, test_images, _, _, test_labels = load_images(best_styles)
test_size = test_labels.shape[0]

best_model = torchvision.models.resnet18(pretrained=False)
for param in best_model.parameters():
    param.requires_grad = False

num_ftrs = best_model.fc.in_features
best_model.fc = nn.Linear(num_ftrs, 2)
best_model.fc.weight.data.fill_(0)
best_model.load_state_dict(best_model_wts)


if use_gpu:
    best_model = best_model.cuda()
    weight_tensor = torch.from_numpy(
        np.array([np.sum(test_size) / test_size, (test_size - np.sum(test_size)) / test_size],
                 dtype=np.float32))
    weight_tensor = weight_tensor.cuda()

criterion = nn.CrossEntropyLoss()

running_loss = 0.0
running_corrects = 0

BATCH = 1

best_model.eval()

for i in range(test_size//BATCH):
    if use_gpu:
        inputs = Variable(torch.from_numpy(test_images[i * BATCH:i * BATCH + BATCH]).cuda())
        label = Variable(torch.from_numpy(test_labels[i * BATCH:i * BATCH + BATCH]).cuda())
    else:
        inputs, label = Variable(torch.from_numpy(test_images[i:i + 1])), Variable(torch.from_numpy(test_labels[i:i + 1]))

    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())
    inputs = (inputs - mean) / std

    # forward
    outputs = best_model(inputs)

    _, preds = torch.max(outputs.data, 1)

    loss = criterion(outputs, label)

    # statistics
    running_loss += loss.data[0]
    running_corrects += torch.sum(preds == label.data)

epoch_loss = running_loss / test_size
epoch_acc = running_corrects / test_size

print('{} Loss: {:.4f} Acc: {:.4f} Style: {}'.format(
    'Test', epoch_loss, epoch_acc, best_styles))
