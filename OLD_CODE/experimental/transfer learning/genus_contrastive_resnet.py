from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from LoadData import GenusLoader
ANGLES = [0.,]

def random_rotate(images, rng):
    import cv2

    shape = images.shape
    images = np.reshape(images, (-1,) + images.shape[-3:])

    n_row = images.shape[2]
    n_col = images.shape[3]

    for i in xrange(len(images)):
        img = np.transpose(images[i], (1, 2, 0))
        angle = ANGLES[rng.randint(len(ANGLES))]
        rotation_matrix = cv2.getRotationMatrix2D((n_col / 2, n_row / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (n_col, n_row), borderMode=cv2.BORDER_REPLICATE)
        images[i] = np.transpose(img, (2, 0, 1))

    images = np.reshape(images, shape)
    return images

def rotate(images, angle, rng):
    import cv2

    shape = images.shape
    images = np.reshape(images, (-1,) + images.shape[-3:])

    n_row = images.shape[2]
    n_col = images.shape[3]

    for i in xrange(len(images)):
        img = np.transpose(images[i], (1, 2, 0))
        rotation_matrix = cv2.getRotationMatrix2D((n_col / 2, n_row / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (n_col, n_row), borderMode=cv2.BORDER_REPLICATE)
        images[i] = np.transpose(img, (2, 0, 1))

    images = np.reshape(images, shape)
    return images

def normalize(images):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return (images - mean) / std

def make_3_channel_normalized(images):
    return normalize(images.expand(images.size(0), 3, images.size(2), images.size(3)))

def load_images(styles):

    loader = GenusLoader(256)
    traini = ()
    testi = ()
    trainl = ()
    testl = ()
    for style in styles:
        train_images, train_labels, tribe_dict = loader.load('../../data/%s' % style, '../../data/genus_train.txt')
        test_images, test_labels, _ = loader.load('../../data/%s' % style, '../../data/genus_test.txt', tribe_dict)
        traini += (train_images,)
        testi += (test_images,)
        trainl += (train_labels,)
        testl += (test_labels,)
    train_images = np.concatenate(traini)
    train_labels = np.concatenate(trainl)
    test_images = np.concatenate(testi)

    train_images_cpu = make_3_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)))
    test_images_cpu = make_3_channel_normalized(Variable(torch.from_numpy(test_images.astype(np.float32)).unsqueeze(1)))

    return train_images_cpu, test_images_cpu, train_images, train_labels

def generate_pairs(train_images, train_labels, num = 100):
    size = train_labels.shape[0]
    train_images_similar = []
    train_images_different = []

    train_labels, indices= (list(x) for x in zip(*sorted(zip(train_labels, np.arange(size)))))
    train_images = train_images[indices]
    counts = np.zeros(43)
    for i in range(size):
        counts[train_labels[i]] += 1
    indices = np.random.choice(np.arange(0, counts[0]), num)
    addresses = counts[0]
    for i in range(1, 43):
        indices = np.concatenate((indices, np.random.choice(np.arange(addresses, addresses + counts[i]), num)))
        addresses += counts[i]
    indices = indices.astype(np.int)
    train_images = train_images[indices]
    train_labels = np.asarray(train_labels)[indices]
    for i in range(indices.shape[0]):
        found_similar = False
        found_different = False
        while not found_similar or not found_different:
            j = np.random.choice(np.arange(0, indices.shape[0]))
            if train_labels[j] == train_labels[i] and not found_similar:
                train_images_similar.append(j)
                found_similar = True
            if train_labels[j] != train_labels[i] and not found_different:
                train_images_different.append(j)
                found_different = True
    contrastive_images = np.asarray(train_images_similar + train_images_different)
    labels = np.array([1.] * indices.shape[0] + [-1.] * indices.shape[0])
    labels = Variable(torch.from_numpy(labels))
    images = make_3_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)))
    return images, contrastive_images, labels


def evaluate(model, images):
    model.train(False)

    return model(images.cuda()).cpu().data.numpy()


def train_one_epoch(model, opt, images_np, labels_np, batch_size, criterion, rng):
    model.train(True)

    images, contrastive_images, labels = generate_pairs(images_np, labels_np)
    size = images.size(0)

    losses = []

    indices = np.arange(2 * size)
    rng.shuffle(indices)
    contrastive_images = torch.from_numpy(contrastive_images[indices]).long()
    indices = torch.from_numpy(indices).long()
    labels = labels[indices]

    indices = indices % size
    shuffled_images = images[indices]


    for i in range(size // batch_size):
        batch_images =  Variable(torch.from_numpy(random_rotate(shuffled_images[i * batch_size: (i + 1) * batch_size].data.numpy(), rng))).cuda()
        batch_contrastive_images = Variable(torch.from_numpy(random_rotate(images[contrastive_images[i * batch_size: (i + 1) * batch_size]].data.numpy(), rng))).cuda()

        batch_labels = labels[i * batch_size: (i + 1) * batch_size].cuda()

        # zero the parameter gradients
        opt.zero_grad()

        # forward
        image_out = model(batch_images)
        contrastive_images_out = model(batch_contrastive_images)
        loss = criterion(torch.squeeze(torch.norm(image_out - contrastive_images_out, 2, 1)), batch_labels)

        loss.backward()
        opt.step()
        losses.append(loss.data[0])


    avg_loss = np.mean(losses)

    print('training loss: %.4f' % avg_loss)

def save_eval_rotate(out_file, model, images, length, rng, styles):
    all = ()
    for angle in ANGLES:
        output = evaluate(model, images[0:1])
        for j in range(1, length):
            output = np.concatenate((output, evaluate(model, Variable(torch.from_numpy(rotate(images[j:j + 1].data.numpy(), angle, rng))))))
        output = np.squeeze(output)
        size = length // len(styles)
        concat_output = output[0:size]
        for j in range(1, len(styles)):
            concat_output = np.concatenate((concat_output, output[size*j:size*(j+1)]), axis=1)
        all += (concat_output,)
    np.save(out_file, np.concatenate(all, axis=0))

def save_eval(out_file, model, images, length, rng, styles):
    all = ()
    for angle in [0.]:
        output = evaluate(model, images[0:1])
        for j in range(1, length):
            output = np.concatenate((output, evaluate(model, Variable(torch.from_numpy(rotate(images[j:j + 1].data.numpy(), angle, rng))))))
        output = np.squeeze(output)
        size = length // len(styles)
        concat_output = output[0:size]
        for j in range(1, len(styles)):
            concat_output = np.concatenate((concat_output, output[size*j:size*(j+1)]), axis=1)
        all += (concat_output,)
    np.save(out_file, np.concatenate(all, axis=0))

def train_test(styles, out_file, max_epochs=30, batch_size=64):
    train_images, test_images, train_images_numpy, train_labels_numpy = load_images(styles)
    assert train_images_numpy.shape[0] % len(styles) == 0
    assert test_images.size(0) % len(styles) == 0
    model = torchvision.models.resnet18(pretrained=True)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    for child in list(model.children())[:-2]:
        for param in child.parameters():
            param.requires_grad = False

    model = model.cuda()

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay = 5e-5)
    scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.3)

    rng = np.random.RandomState(123456)
    criterion = nn.HingeEmbeddingLoss(margin=100)

    save_eval_rotate(out_file + '_train', model, train_images, train_images.size(0), rng, styles)
    save_eval(out_file + '_test', model, test_images, test_images.size(0), rng, styles)
    for i in range(max_epochs):
        print('----- epoch %r -----' % i)

        scheduler.step()
        print('learning rate: %.8f' % scheduler.get_lr()[0])

        train_one_epoch(model, opt, train_images_numpy, train_labels_numpy, batch_size, criterion, rng)
        save_eval_rotate(out_file + '_train' + str(i), model, train_images, train_images.size(0),rng, styles)
        save_eval(out_file + '_test' + str(i), model, test_images, test_images.size(0), rng, styles)


styles = [
    'max_global_autocontrast_normalized',
    #'max_images',

    'mean_global_autocontrast_normalized',
    #'mean_images',
    #
    'focus_stacking_global_autocontrast_normalized',
    #'focus_stacking_images',
    #
    #'median_images',
    'global_autocontrast_normalized',
]

for style in styles:
    train_test([style], style, max_epochs=0)
