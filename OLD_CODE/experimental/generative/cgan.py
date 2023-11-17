import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import glob
import cv2
import time
import gflags
import sys


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


TASK = 'tribe'

BATCH_SIZE = 64
Z_DIM = 100
Z_MIN = -1
Z_MAX = 1
IMG_SIZE = 64
TRAINING_SAMPLES_VISUALIZE_FREQ = 100
G_SAMPLES_VISUALIZE_FREQ = 100
FIXED_SAMPLES_VISUALIZE_FREQ = 25

GRID_X = 8
GRID_Y = 8

manualSeed = 12318
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)


def normalize(images):
    mean = Variable(torch.FloatTensor([0.5]).view(1, 1, 1, 1))
    std = Variable(torch.FloatTensor([1.0]).view(1, 1, 1, 1))
    return (images - mean) / std


def get_dataset_loader():
    from LoadData import GenusLoader, TribeLoader

    if TASK == 'tribe':
        return TribeLoader
    elif TASK == 'genus':
        return GenusLoader
    else:
        raise RuntimeError('Cannot find loader for task %r' % TASK)


def random_rotate(images, rng, angles=None):
    import cv2

    shape = images.shape
    images = np.reshape(images, (-1,) + images.shape[-3:])

    n_row = images.shape[2]
    n_col = images.shape[3]

    for i in xrange(len(images)):
        img = np.transpose(images[i], (1, 2, 0))
        if angles is None:
            angle = rng.uniform(0, 360.0)
        else:
            angle = angles[rng.randint(len(angles))]
        rotation_matrix = cv2.getRotationMatrix2D((n_col / 2, n_row / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (n_col, n_row), borderMode=cv2.BORDER_REPLICATE)
        if len(img.shape) == 2:  # OpenCv will strip the color channel if its length is 1 (i.e., grayscale)
            img = img[:, :, np.newaxis]
        images[i] = np.transpose(img, (2, 0, 1))

    images = np.reshape(images, shape)
    return images


def load_dataset(style):

    def make_1_channel_normalized(images):
        return normalize(images.expand(images.size(0), 1, images.size(2), images.size(3)))

    loader = get_dataset_loader()(IMG_SIZE)

    train_images, train_labels, class_dict = loader.load('../../data/%s' % style, '../../data/%s_train.txt' % TASK, None, None)
    test_images, test_labels, _ = loader.load('../../data/%s' % style, '../../data/%s_test.txt' % TASK, class_dict, None)

    train_images = make_1_channel_normalized(Variable(torch.from_numpy(train_images.astype(np.float32)).unsqueeze(1)))
    test_images = make_1_channel_normalized(Variable(torch.from_numpy(test_images.astype(np.float32)).unsqueeze(1)))

    train_labels = Variable(torch.from_numpy(train_labels))
    test_labels = Variable(torch.from_numpy(test_labels))

    return train_images, test_images, train_labels, test_labels, class_dict


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self, n_class):
        super(G, self).__init__()
        # Batch normalization is crucial to prevent mode collapse.
        self.g1 = nn.ConvTranspose2d(Z_DIM + n_class, 512, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.g2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.g3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.g4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.g5 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size()[0], -1, 1, 1)
        x = F.relu(self.bn1(self.g1(x)))
        x = F.relu(self.bn2(self.g2(x)))
        x = F.relu(self.bn3(self.g3(x)))
        x = F.relu(self.bn4(self.g4(x)))
        x = F.tanh(self.g5(x))
        return x


class D(nn.Module):
    def __init__(self, n_class):
        super(D, self).__init__()
        self.n_class = n_class

        # 64 x 64
        self.d1 = nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False)
        # 32 x 32
        self.d2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # 16 x 16
        self.d3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # 8 x 8
        self.d4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        # 4 x 4
        self.d5 = nn.Conv2d(512, 1 + n_class, 4, bias=False)
        # 1 x 1

        self.apply(weights_init)

    def f(self, x):
        x = F.leaky_relu(self.d1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.d2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.d3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.d4(x)), 0.2)
        x = self.d5(x)
        logits = x.view(x.size()[0], 1 + self.n_class)

        real_logits = logits[:, 0: -1]
        fake_logits = logits[:, -1]

        real_max_logits, _ = torch.max(real_logits, 1)
        # Expand dimension
        real_max_logits = real_max_logits.view(-1, 1)

        # This code is based on
        # https://github.com/openai/improved-gan/blob/master/imagenet/discriminator.py
        # This is to compute the logits of real/fake before the sigmoid operation.
        # If you substitute it into sigmoid, it is equivalent to:
        # sum(exp(class_logits)) / (sum(exp(class_logits)) + exp(fake_logit))
        # which is the probability that x is real.
        a = torch.log(torch.sum(torch.exp(real_logits - real_max_logits.expand_as(real_logits)), dim=1))
        d_x_logit = a.view(-1, 1) + real_max_logits - fake_logits

        return real_logits, d_x_logit

    def forward(self, x):
        return self.f(x)


torch.set_default_tensor_type('torch.cuda.FloatTensor')

g_net_class = G
d_net_class = D


def to_onehot(idxs, n_class):
    return torch.zeros(len(idxs), n_class).scatter_(1, idxs.view(len(idxs), 1), 1.0)


def bce_sigmoid(x, z):
    return torch.mean(torch.max(x, Variable(torch.zeros(x.size()))) - x * z + torch.log(1 + torch.exp(-torch.abs(x))))


def gen_img_grid(imgs, grid_x, grid_y):
    imgs = imgs.transpose((0, 2, 3, 1))
    imgs = imgs * 0.5 + 0.5
    imgs = imgs.reshape((grid_y, grid_x, IMG_SIZE, IMG_SIZE))
    out = np.zeros((grid_y * IMG_SIZE, grid_x * IMG_SIZE))
    for i in xrange(grid_y):
        out[i * IMG_SIZE: (i + 1) * IMG_SIZE] = np.concatenate(imgs[i], axis=1)
    return out


def visualize_samples(data, grid_x, grid_y, window_name):
    img_grid = gen_img_grid(data, grid_x, grid_y).astype(np.float32)
    cv2.imshow(window_name, img_grid)
    cv2.waitKey(1)
    return img_grid


def load_training_data():
    train_images, test_images, train_labels, test_labels, class_dict = load_dataset(
        'mean_global_autocontrast_normalized')

    images = torch.cat([train_images, test_images])
    labels = torch.cat([train_labels, test_labels])

    return images, labels, len(class_dict)


def train(epoch, images, labels, n_class, rng):
    images = images.data.cpu().numpy()
    images = random_rotate(images, rng)
    images = Variable(torch.from_numpy(images), requires_grad=False).cuda()

    train_size = images.size(0)

    indices = np.arange(train_size)
    np.random.shuffle(indices)
    indices = torch.from_numpy(indices).long().cuda()

    images = images[indices]
    labels = labels[indices]

    d_losses = []
    g_losses = []

    n_labels_per_class = np.bincount(labels.data.cpu().numpy())

    global n_correct
    global n_total
    n_correct = 0
    n_total = 0

    def inverse_freq_ratio(n_labels_per_class):
        total = np.sum(n_labels_per_class)
        freq_ratio = n_labels_per_class / total.astype(np.float32)
        inv = 1.0 / freq_ratio
        r = inv / np.sum(inv)
        return r

    weights = torch.from_numpy(inverse_freq_ratio(n_labels_per_class).astype(np.float32)).cuda()

    cross_entropy_crit = nn.CrossEntropyLoss(weights).cuda()

    def update_d(idx, x, z, y):
        d_net.zero_grad()

        class_logits, d_x = d_net.forward(x)
        real_loss = bce_sigmoid(d_x, Variable(torch.ones(d_x.size()[0])))
        class_loss = cross_entropy_crit(class_logits, y)
        (class_loss + real_loss).backward()

        g_samples = g_net(z)
        _, d_g_z = d_net.forward(g_samples.detach())
        fake_loss = bce_sigmoid(d_g_z, Variable(torch.zeros(d_g_z.size()[0])))
        fake_loss.backward()

        d_opt.step()
        d_loss = (real_loss + fake_loss).data.cpu().numpy()[0]
        d_losses.append(d_loss)

        return g_samples

    def update_g(idx, g_samples, y):
        g_net.zero_grad()

        class_logits, d_g_z = d_net.forward(g_samples)
        g_loss = bce_sigmoid(d_g_z, Variable(torch.ones(d_g_z.size()[0])))
        class_loss = cross_entropy_crit(class_logits, y)
        (class_loss + g_loss).backward()
        g_opt.step()

        l = g_loss.data.cpu().numpy()[0]
        g_losses.append(l)

        _, preds = torch.max(class_logits, 1)

        global n_correct, n_total
        n_correct += torch.sum(preds == batch_labels).data[0]
        n_total += x.size(0)

        if idx % G_SAMPLES_VISUALIZE_FREQ == 0:
            visualize_samples(g_samples.data.cpu().numpy(), GRID_X, GRID_Y, 'G samples')

    for idx in xrange(train_size // BATCH_SIZE):
        batch_images = images[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
        batch_labels = labels[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]

        start_time = time.time()

        x = batch_images

        batch_labels_onehot = to_onehot(batch_labels.data, n_class)

        z_batch = torch.zeros(BATCH_SIZE, Z_DIM)
        z_batch.uniform_(Z_MIN, Z_MAX)
        z = Variable(z_batch.cuda(async=True), requires_grad=False)
        z = torch.cat([z, batch_labels_onehot], 1)

        g_samples = update_d(idx, x, z, batch_labels)
        update_g(idx, g_samples, batch_labels)

        end_time = time.time()

        if idx % FIXED_SAMPLES_VISUALIZE_FREQ == 0:
            samples = g_net(fixed_z)
            visualize_samples(samples.data.cpu().numpy(), GRID_X, n_class, 'fixed samples')

        if idx % TRAINING_SAMPLES_VISUALIZE_FREQ == 0:
            visualize_samples(batch_images.data.cpu().numpy(), GRID_X, GRID_Y, 'training samples')

        print 'step %d (%.2f sec) g_loss %.3f d_loss %.3f' % (idx, end_time - start_time, g_losses[-1], d_losses[-1])

    print 'epoch %d avg g_loss %.3f avg d_loss %.3f G accuracy %.3f' % (
        epoch, np.mean(g_losses), np.mean(d_losses), float(n_correct) / n_total)


images, labels, n_class = load_training_data()
images = images.cuda()
labels = labels.cuda()

g_net = g_net_class(n_class).cuda()
d_net = d_net_class(n_class).cuda()

# Adam should not use a high learning rate. The effective learning rate is already high at the beginning.
# Too high learning rate may result in subpar quality.
g_opt = optim.Adam(g_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(d_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

fixed_z = torch.from_numpy(np.random.uniform(Z_MIN, Z_MAX, (GRID_X * n_class, Z_DIM)).astype(np.float32))
fixed_z = Variable(fixed_z, volatile=True).cuda()
fixed_z_labels = torch.from_numpy(np.repeat(np.arange(n_class), GRID_X)).cuda()
fixed_z_labels_onehot = to_onehot(fixed_z_labels, n_class)
fixed_z = torch.cat((fixed_z, fixed_z_labels_onehot), 1)


rng = np.random.RandomState(412093)
epoch = 0
while True:
    train(epoch, images, labels, n_class, rng)
    epoch += 1
