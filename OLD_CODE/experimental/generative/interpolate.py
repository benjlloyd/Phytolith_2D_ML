import torch
import torch.nn as nn
from torch import autograd
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
import utils


gflags.DEFINE_string('model_dir', 'cwgan_all_logs', '')
gflags.DEFINE_string('type', '', '')
gflags.DEFINE_integer('grid_x', 16, '')
gflags.DEFINE_integer('grid_y', 8, '')
gflags.DEFINE_string('out_dir', '', '')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

MODEL_DIR = FLAGS.model_dir

Z_DIM = 100
Z_MIN = -1
Z_MAX = 1
IMG_SIZE = 64
GRID_X = FLAGS.grid_x
GRID_Y = FLAGS.grid_y


def to_onehot(idxs, n_class):
    return torch.zeros(len(idxs), n_class).scatter_(1, idxs.view(len(idxs), 1), 1.0)


def load_model(dir, filename, step=None):
    '''
    :param model:
    :param dir:
    :param filename:
    :param step: if None. Load the latest.
    :return: the saved state dict
    '''
    import parse
    if not step:
        files = glob.glob(os.path.join(dir, '%s.*' % filename))
        parsed = []
        for fn in files:
            r = parse.parse('{}.{:d}', fn)
            if r:
                parsed.append((r, fn))
        if not parsed:
            return None

        step, path = max(parsed, key=lambda (r, _): r[1])
    else:
        path = os.path.join(dir, '%s.%d' % (filename, step))

    if os.path.isfile(path):
        return torch.load(path)

    raise Exception('Failed to load model')


class G(nn.Module):

    def __init__(self, n_class, z_dim):
        super(G, self).__init__()
        self.g1 = nn.ConvTranspose2d(z_dim + n_class, 512, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.g2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.g3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.g4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.g5 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1, 1, 1)
        x = F.relu(self.bn1(self.g1(x)))
        x = F.relu(self.bn2(self.g2(x)))
        x = F.relu(self.bn3(self.g3(x)))
        x = F.relu(self.bn4(self.g4(x)))
        x = F.tanh(self.g5(x))
        return x


def gen_img_grid(imgs, grid_x, grid_y):
    imgs = imgs.transpose((0, 2, 3, 1))
    imgs = imgs * 0.5 + 0.5
    imgs = imgs.reshape((grid_y, grid_x, IMG_SIZE, IMG_SIZE))
    out = np.zeros((grid_y * IMG_SIZE, grid_x * IMG_SIZE))
    for i in xrange(grid_y):
        out[i * IMG_SIZE: (i + 1) * IMG_SIZE] = np.concatenate(imgs[i], axis=1)
    return out


class_names = utils.load_classes('../../data/tribe_names.txt')
n_class = len(class_names)

g_net = G(n_class, Z_DIM).cuda()

state = load_model(MODEL_DIR, 'model', None)
if state:
    epoch = int(state['epoch'])
    g_net.load_state_dict(state['g_net'])
    print 'loaded saved state. n_epoch: %d' % epoch
else:
    print 'no model available.'
    exit(0)

g_net.train(False)


def make_inter_class_video():
    PAUSE_FRAMES = 30
    STEPS = 30

    noise = torch.from_numpy(np.random.uniform(Z_MIN, Z_MAX, (GRID_X * GRID_Y, Z_DIM)).astype(np.float32))
    noise = Variable(noise, volatile=True).cuda()

    BANNER_HEIGHT = 150
    TEXT_VERTICAL_DISTANCE = 15

    def write_frame(idx, frame):
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%03d.png' % idx), frame)

    frame_idx = 0
    for i, label in enumerate(list(np.linspace(0.0, n_class - 1, (n_class - 1) * STEPS + 1))):
        onehot_labels = np.zeros((GRID_X * GRID_Y, n_class), np.float32)

        prob1 = 1.0 + int(label) - label
        onehot_labels[:, int(label)] = prob1

        prob2 = 1.0 - prob1

        if int(label) < n_class - 1:
            onehot_labels[:, int(label)+1] = prob2

        z = torch.cat((noise, Variable(torch.from_numpy(onehot_labels), volatile=True).cuda()), 1)
        samples = g_net(z)

        samples = samples.data.cpu().numpy()
        grid = gen_img_grid(samples, GRID_X, GRID_Y)
        grid = (grid * 256).astype(np.uint8)

        frame = np.zeros((grid.shape[0] + BANNER_HEIGHT, grid.shape[1]), np.uint8)
        frame[:grid.shape[0], :grid.shape[1]] = grid

        captions = ['%s: %.2f' % (class_names[i], onehot_labels[0][i]) for i in xrange(n_class)]
        for j in xrange(n_class):
            cv2.putText(frame, captions[j], (10, frame.shape[0] - TEXT_VERTICAL_DISTANCE * (n_class - j)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)

        cv2.imshow('', frame)

        write_frame(frame_idx, frame)
        frame_idx += 1

        if label == int(label):
            for _ in xrange(PAUSE_FRAMES):
                write_frame(frame_idx, frame)
                frame_idx += 1


def make_intra_class_video():
    rng = np.random.RandomState(99)

    STEPS = 30
    N_SAMPLES = 900

    v = rng.uniform(Z_MIN, Z_MAX, (N_SAMPLES // STEPS + 1, Z_DIM))

    lambs = np.linspace(0.0, 1.0, STEPS)[:, np.newaxis]

    all_samples = []

    for i in xrange(n_class):
        class_samples = []

        for k in xrange(N_SAMPLES // STEPS):
            v2 = rng.uniform(Z_MIN, Z_MAX, Z_DIM)

            onehot_labels = np.zeros((STEPS, n_class), np.float32)
            onehot_labels[:, i] = 1.0

            v1 = v[k]
            v2 = v[k+1]

            noise = np.repeat(v1[np.newaxis, :], STEPS, 0) * (1.0 - lambs) + \
            np.repeat(v2[np.newaxis, :], STEPS, 0) * lambs

            z = np.concatenate([noise, onehot_labels], axis=1)
            z = Variable(torch.from_numpy(z.astype(np.float32)), volatile=True).cuda()

            samples = g_net(z)
            samples = samples.data.cpu().numpy()

            class_samples.append(samples)

        class_samples = np.concatenate(class_samples, 0)
        all_samples.append(class_samples)
    all_samples = np.array(all_samples)

    def write_frame(idx, frame):
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%03d.png' % idx), frame)

    for i in xrange(N_SAMPLES):
        samples = all_samples[:, i]

        frame = np.zeros((IMG_SIZE * 2, n_class * IMG_SIZE + (n_class + 1) * IMG_SIZE), np.uint8)

        for j in xrange(samples.shape[0]):
            sample = samples[j][0] * 0.5 + 0.5
            frame[0:IMG_SIZE, (j * 2 + 1)*IMG_SIZE:(j * 2 + 2)*IMG_SIZE] = (sample * 256).astype(np.uint8)

            cv2.putText(frame, class_names[j], ((j * 2 + 1) * IMG_SIZE, IMG_SIZE + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)

        write_frame(i, frame)
        cv2.imshow('', frame)
        cv2.waitKey(1)


def make_intra_class_images():
    GRID_X = 10

    rng = np.random.RandomState(9999)
    v1 = rng.uniform(Z_MIN, Z_MAX, Z_DIM)
    v2 = rng.uniform(Z_MIN, Z_MAX, Z_DIM)

    lambs = np.linspace(0.0, 1.0, GRID_X)[:, np.newaxis]

    noise = np.repeat(v1[np.newaxis, :], GRID_X, 0) * (1.0 - lambs) + np.repeat(v2[np.newaxis, :], GRID_X, 0) * lambs

    for i in xrange(n_class):
        onehot_labels = np.zeros((GRID_X, n_class), np.float32)
        onehot_labels[:, i] = 1.0

        z = np.concatenate([noise, onehot_labels], axis=1)
        z = Variable(torch.from_numpy(z.astype(np.float32)), volatile=True).cuda()

        samples = g_net(z)

        print samples.size()

        samples = samples.data.cpu().numpy()
        print samples.shape
        grid = gen_img_grid(samples, GRID_X, 1)

        grid = (grid * 256).astype(np.uint8)
        cv2.imwrite('%d.png' % i, grid)


def make_inter_class_images():
    GRID_X = 10
    rng = np.random.RandomState(9999)

    noise = rng.uniform(Z_MIN, Z_MAX, Z_DIM)

    for i in xrange(n_class - 1):
        soft_labels = list(np.linspace(0, 1, GRID_X))
        onehot_labels = np.zeros((GRID_X, n_class), np.float32)
        for j in xrange(len(soft_labels)):
            prob = 1 - soft_labels[j]
            onehot_labels[j, i] = prob
            onehot_labels[j, i + 1] = 1 - prob

        z = np.concatenate([np.repeat(np.expand_dims(noise, 0), GRID_X, 0), onehot_labels], axis=1)
        z = Variable(torch.from_numpy(z.astype(np.float32)), volatile=True).cuda()

        samples = g_net(z)

        print samples.size()

        samples = samples.data.cpu().numpy()
        print samples.shape
        grid = gen_img_grid(samples, GRID_X, 1)

        cv2.imshow('', grid)
        cv2.waitKey(0)

        grid = (grid * 256).astype(np.uint8)
        cv2.imwrite('%d.png' % i, grid)


if FLAGS.type == 'interclass_video':
    make_inter_class_video()

elif FLAGS.type == 'intraclass_video':
    make_intra_class_video()

elif FLAGS.type == 'intraclass_imgs':
    make_intra_class_images()

elif FLAGS.type == 'interclass_imgs':
    make_inter_class_images()
else:
    raise RuntimeError('Unknown type: %r' % FLAGS.type)