import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import numpy as np
from LoadData import LoadData


class Identity(nn.Module):
    def forward(self, x):
        return x


def extract_features_vgg19(images):
    images = Variable(torch.from_numpy(images.astype(np.float32)).cuda(), volatile=True)
    images = torch.unsqueeze(images, dim=1)
    images = images.expand(images.size(0), 3, 256, 256).contiguous()

    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    model = torchvision.models.vgg19_bn(pretrained=True).cuda()
    model.train(False)

    images = (images - mean) / std

    all_features = []

    for i in xrange(images.size(0)):
        all_features.append(model.features(images[i].view(1, 3, 256, 256)).view(1, -1))

    all_features = torch.cat(all_features, dim=0)
    print all_features.size()

    return all_features.data.cpu().numpy()


def extract_features_resnet152(images):
    images = Variable(torch.from_numpy(images.astype(np.float32)).cuda(), volatile=True)
    images = torch.unsqueeze(images, dim=1)
    images = images.expand(images.size(0), 3, 256, 256).contiguous()

    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    model = torchvision.models.resnet152(pretrained=True).cuda()
    model.fc = Identity()
    model.train(False)

    images = (images - mean) / std

    all_features = []

    for i in xrange(images.size(0)):
        all_features.append(model(images[i].view(1, 3, 256, 256)).view(1, -1))

    all_features = torch.cat(all_features, dim=0)
    print all_features.size()

    return all_features.data.cpu().numpy()


def extract(style='global_autocontrast_normalized'):
    ld = LoadData(256)
    train_images, train_labels = ld.loadTrainData('/home/xymeng/Dropbox/phd/courses/cse546/', imageStyle=style)
    test_images, test_labels = ld.loadTestData('/home/xymeng/Dropbox/phd/courses/cse546/', imageStyle=style)

    np.save('train_features_resnet152.npy', extract_features_resnet152(train_images))
    np.save('test_features_resnet152.npy', extract_features_resnet152(test_images))


#extract('focus_stacking_global_autocontrast_normalized')
extract('max_global_autocontrast_normalized')
