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
    def __init__(self):
        super(G, self).__init__()
        # Batch normalization is crucial to prevent mode collapse.
        self.g1 = nn.ConvTranspose2d(Z_DIM, 512, 4, bias=False)
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
    def __init__(self):
        super(D, self).__init__()
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
        self.d5 = nn.Conv2d(512, 1, 4, bias=False)
        # 1 x 1

        self.apply(weights_init)

    def f(self, x):
        x = F.leaky_relu(self.d1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.d2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.d3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.d4(x)), 0.2)
        x = self.d5(x)
        x = x.view(x.size()[0], 1)
        return x

    def forward(self, x):
        return self.f(x)


torch.set_default_tensor_type('torch.cuda.FloatTensor')

g_net_class = G
d_net_class = D

g_net = g_net_class().cuda() #nn.DataParallel(g_net_class().cuda(), device_ids=[0, 1])
d_net = d_net_class().cuda() #nn.DataParallel(d_net_class().cuda(), device_ids=[0, 1])

# Adam should not use a high learning rate. The effective learning rate is already high at the beginning.
# Too high learning rate may result in subpar quality.
g_opt = optim.Adam(g_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(d_net.parameters(), lr=0.0001, betas=(0.5, 0.999))


def bce_sigmoid(x, z):
    return torch.mean(torch.max(x, Variable(torch.zeros(x.size()))) - x * z + torch.log(1 + torch.exp(-torch.abs(x))))


def gen_img_grid(imgs, grid_x, grid_y):
    imgs = imgs.transpose((0, 2, 3, 1))
    imgs = imgs * 0.5 + np.array([0.5, 0.5, 0.5])
    img_grid = np.concatenate(imgs.reshape((grid_x, grid_y, IMG_SIZE, IMG_SIZE, 3)), axis=2)
    img_grid = np.concatenate(img_grid, axis=0)
    return img_grid


def visualize_samples(data, grid_x, grid_y, window_name):
    img_grid = gen_img_grid(data, grid_x, grid_y).astype(np.float32)
    img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img_grid)
    cv2.waitKey(1)
    return img_grid


fixed_z = Variable(torch.from_numpy(np.random.uniform(Z_MIN, Z_MAX, (BATCH_SIZE, Z_DIM)).astype(np.float32)),
                       volatile=True).cuda()


def train():
    train_images, test_images, train_labels, test_labels, class_dict = load_dataset('focus_stacking_global_autocontrast_normalized')

    images = torch.cat([train_images, test_images])
    train_size = images.size(0)
    indices = np.arange(train_size)
    np.random.shuffle(indices)
    images = images[torch.from_numpy(indices).long()]
    images = images.cuda()

    n_epoch = 0

    d_losses = []
    g_losses = []

    def update_d(idx, x, z):
        d_net.zero_grad()

        d_x = d_net.forward(x)
        d1_loss = bce_sigmoid(d_x, Variable(torch.ones(d_x.size()[0])))
        d1_loss.backward()

        g_samples = g_net(z)
        d_g_z = d_net.forward(g_samples.detach())
        d2_loss = bce_sigmoid(d_g_z, Variable(torch.zeros(d_g_z.size()[0])))
        d2_loss.backward()

        d_opt.step()
        d_loss = (d1_loss + d2_loss).data.cpu().numpy()[0]
        d_losses.append(d_loss)

        return g_samples

    def update_g(idx, g_samples):
        g_net.zero_grad()

        d_g_z = d_net.forward(g_samples)
        g_loss = bce_sigmoid(d_g_z, Variable(torch.ones(d_g_z.size()[0])))
        g_loss.backward()
        g_opt.step()

        l = g_loss.data.cpu().numpy()[0]
        g_losses.append(l)

        if idx % G_SAMPLES_VISUALIZE_FREQ == 0:
            visualize_samples(g_samples.data.cpu().numpy(), GRID_X, GRID_Y, 'G samples')

    for idx in xrange(train_size // BATCH_SIZE):
        batch_images = images[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE].cuda()

        start_time = time.time()

        # x = Variable(batch_images.cuda(async=True), requires_grad=False)
        x = batch_images

        z_batch = torch.zeros(BATCH_SIZE, Z_DIM)
        # z_batch.normal_(0.0, 1.0)
        z_batch.uniform_(Z_MIN, Z_MAX)
        z = Variable(z_batch.cuda(async=True), requires_grad=False)

        g_samples = update_d(idx, x, z)
        update_g(idx, g_samples)

        end_time = time.time()

        if idx % FIXED_SAMPLES_VISUALIZE_FREQ == 0:
            samples = g_net(fixed_z)
            visualize_samples(samples.data.cpu().numpy(), GRID_X, GRID_Y, 'fixed samples')

        if idx % TRAINING_SAMPLES_VISUALIZE_FREQ == 0:
            visualize_samples(batch_images.data.cpu().numpy(), GRID_X, GRID_Y, 'training samples')

        print 'step %d (%.2f sec) g_loss %.3f d_loss %.3f' % (idx, end_time - start_time, g_losses[-1], d_losses[-1])

    print 'epoch %d avg g_loss %.3f avg d_loss %.3f' % (n_epoch, np.mean(g_losses), np.mean(d_losses))


while True:
    train()
