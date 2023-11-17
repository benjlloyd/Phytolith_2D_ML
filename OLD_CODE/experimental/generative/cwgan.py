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


gflags.DEFINE_boolean('clip_weights', False, '')
gflags.DEFINE_boolean('independent_loss', True, 'No coupling between classification loss and discrimination loss.')
gflags.DEFINE_boolean('free_dloss', True, 'No sigmoid if True.')
gflags.DEFINE_float('class_loss_weight', 1.0, '')
gflags.DEFINE_float('gradient_penalty_weight', 10.0, '')
gflags.DEFINE_boolean('D_batch_norm', False, '')
gflags.DEFINE_integer('max_epochs', 2000, '')
gflags.DEFINE_string('log_dir', '', '')
gflags.DEFINE_string('style', 'all', 'comma-separated style names or all.')

gflags.DEFINE_boolean('gen_real_grid', True, 'Generates an image grid with real samples and exit.')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)

ALL_STYLES = [
    'max_global_autocontrast_normalized',
    'mean_global_autocontrast_normalized',
    'focus_stacking_global_autocontrast_normalized',
    'global_autocontrast_normalized',
]

if FLAGS.style == 'all':
    STYLES = ALL_STYLES
else:
    STYLES = FLAGS.style.split(',')


CLIP_WEIGHTS = FLAGS.clip_weights
INDEPENDENT_LOSS = FLAGS.independent_loss
FREE_DLOSS = FLAGS.free_dloss
GRADIENT_PENALTY_WEIGHT = FLAGS.gradient_penalty_weight
D_BATCH_NORM = FLAGS.D_batch_norm
MAX_EPOCHS = FLAGS.max_epochs
LOG_DIR = FLAGS.log_dir

CLASS_LOSS_WEIGHT = FLAGS.class_loss_weight
# if FREE_DLOSS:
#     CLASS_LOSS_WEIGHT = 1.1

print 'clip_weights:', CLIP_WEIGHTS
print 'independent_loss:', INDEPENDENT_LOSS
print 'free_dloss:', FREE_DLOSS
print 'gradient penalty weight:', GRADIENT_PENALTY_WEIGHT
print 'D batch norm:', D_BATCH_NORM
print 'class loss weight:', CLASS_LOSS_WEIGHT
print 'styles:', STYLES

if LOG_DIR != '':
    try:
        os.makedirs(LOG_DIR, 0744)
    except:
        pass

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


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class D(nn.Module):
    def __init__(self, n_class, batch_norm=True):
        super(D, self).__init__()
        self.n_class = n_class
        self.batch_norm = batch_norm

        # 64 x 64
        self.d1 = nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False)
        # 32 x 32
        self.d2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = self.maybe_batch_norm(128)
        # 16 x 16
        self.d3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn3 = self.maybe_batch_norm(256)
        # 8 x 8
        self.d4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)
        self.bn4 = self.maybe_batch_norm(512)
        # 4 x 4
        self.d5 = nn.Conv2d(512, 1 + n_class, 4, bias=False)
        # 1 x 1

        self.apply(weights_init)

    def maybe_batch_norm(self, *args):
        if self.batch_norm:
            return nn.BatchNorm2d(*args)
        else:
            return Identity(*args)

    def f(self, x):
        x = F.leaky_relu(self.d1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.d2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.d3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.d4(x)), 0.2)
        x = self.d5(x)
        logits = x.view(x.size()[0], 1 + self.n_class)

        real_logits = logits[:, 0: -1]
        fake_logits = logits[:, -1]

        if INDEPENDENT_LOSS:
            return real_logits, fake_logits

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
    all_images = []
    all_labels = []
    for style in STYLES:
        train_images, test_images, train_labels, test_labels, class_dict = load_dataset(style)
        all_images.append(train_images)
        all_images.append(test_images)
        all_labels.append(train_labels)
        all_labels.append(test_labels)

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    return all_images, all_labels, len(class_dict)


def calc_gradient_penalty(netD, real_data, fake_data, lamb):
    alpha = torch.rand(BATCH_SIZE, 1).cuda()
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    _, disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty


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

        class_logits, real_scores = d_net.forward(x)

        if not FREE_DLOSS:
            real_loss = bce_sigmoid(real_scores, Variable(torch.ones(real_scores.size()[0])))
        else:
            real_loss = -torch.mean(real_scores)

        class_loss = cross_entropy_crit(class_logits, y)
        (class_loss * CLASS_LOSS_WEIGHT + real_loss).backward()

        fake_samples = g_net(z)
        _, fake_scores = d_net.forward(fake_samples.detach())

        if not FREE_DLOSS:
            fake_loss = bce_sigmoid(fake_scores, Variable(torch.zeros(fake_scores.size()[0])))
        else:
            fake_loss = torch.mean(fake_scores)

        fake_loss.backward()

        gradient_penalty = 0
        if GRADIENT_PENALTY_WEIGHT > 0:
            gradient_penalty = calc_gradient_penalty(d_net, x, fake_samples, GRADIENT_PENALTY_WEIGHT)
            gradient_penalty.backward()

        d_opt.step()

        d_loss = (real_loss + fake_loss + gradient_penalty).data[0]
        d_losses.append(d_loss)

        return fake_samples

    def update_g(idx, g_samples, y):
        g_net.zero_grad()

        class_logits, fake_scores = d_net.forward(g_samples)

        if not FREE_DLOSS:
            fake_loss = bce_sigmoid(fake_scores, Variable(torch.ones(fake_scores.size()[0])))
        else:
            fake_loss = -torch.mean(fake_scores)

        class_loss = cross_entropy_crit(class_logits, y)
        g_loss = (class_loss * CLASS_LOSS_WEIGHT + fake_loss)
        g_loss.backward()
        g_opt.step()

        g_losses.append(g_loss.data[0])

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

        if CLIP_WEIGHTS:
            for param in d_net.parameters():
                param.data.clamp_(-0.02, 0.02)

        g_samples = update_d(idx, x, z, batch_labels)
        update_g(idx, g_samples, batch_labels)

        end_time = time.time()

        if idx % FIXED_SAMPLES_VISUALIZE_FREQ == 0:
            samples = g_net(fixed_z)
            visualize_samples(samples.data.cpu().numpy(), GRID_X, n_class, 'fixed samples')

        if idx % TRAINING_SAMPLES_VISUALIZE_FREQ == 0:
            visualize_samples(batch_images.data.cpu().numpy(), GRID_X, GRID_Y, 'training samples')

        print 'step %d (%.2f sec) d_loss: %.2f g_loss: %.2f' % (
            idx, end_time - start_time, np.mean(d_losses), np.mean(g_losses))

    print 'epoch %d G accuracy %.3f' % (
        epoch, float(n_correct) / n_total)

    if epoch % 100 == 0 and LOG_DIR != '':
        samples = g_net(fixed_z)
        img_grid = visualize_samples(samples.data.cpu().numpy(), GRID_X, n_class, 'fixed samples')
        cv2.imwrite(os.path.join(LOG_DIR, 'fixed_z_epoch_%d.png' % epoch), (img_grid * 255.0).astype(np.uint8))


def save_model(state, step, dir, filename):
    path = os.path.join(dir, '%s.%d' % (filename, step))
    torch.save(state, path)


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


images, labels, n_class = load_training_data()

if FLAGS.gen_real_grid:
    samples = np.zeros((GRID_X * n_class, 1, IMG_SIZE, IMG_SIZE), np.float32)
    for i in xrange(n_class):
        for j in xrange(GRID_X):
            while True:
                idx = np.random.randint(labels.size(0))
                if labels[idx].data[0] == i:
                    samples[i * GRID_X + j] = images[idx].data.numpy()
                    break
    samples = random_rotate(samples, np.random.RandomState(12314))
    grid = visualize_samples(samples, GRID_X, n_class, '')
    cv2.imwrite('real_samples.png', (grid * 256).astype(np.uint8))
    cv2.waitKey(0)
    exit(0)


images = images.cuda()
labels = labels.cuda()


g_net = g_net_class(n_class).cuda()
d_net = d_net_class(n_class).cuda()

g_opt = optim.Adam(g_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(d_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

fixed_z = torch.from_numpy(np.random.uniform(Z_MIN, Z_MAX, (GRID_X * n_class, Z_DIM)).astype(np.float32))
fixed_z = Variable(fixed_z, volatile=True).cuda()
fixed_z_labels = torch.from_numpy(np.repeat(np.arange(n_class), GRID_X)).cuda()
fixed_z_labels_onehot = to_onehot(fixed_z_labels, n_class)
fixed_z = torch.cat((fixed_z, fixed_z_labels_onehot), 1)


rng = np.random.RandomState(412093)

epoch = 0

state = load_model(LOG_DIR, 'model', None)
if state:
    epoch = int(state['epoch'])
    g_net.load_state_dict(state['g_net'])
    d_net.load_state_dict(state['d_net'])
    g_opt.load_state_dict(state['g_optim'])
    d_opt.load_state_dict(state['d_optim'])
    print 'loaded saved state. n_epoch: %d' % epoch
    epoch += 1

while True:
    train(epoch, images, labels, n_class, rng)
    epoch += 1
    if epoch >= MAX_EPOCHS:
        break

    if epoch % 100 == 0:
        state = {
            'epoch': epoch,
            'g_net': g_net.state_dict(),
            'd_net': d_net.state_dict(),
            'g_optim': g_opt.state_dict(),
            'd_optim': d_opt.state_dict()
        }
        save_model(state, 0, LOG_DIR, 'model')
