import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class Identity(nn.Module):
    def forward(self, input):
        return input


class ResNet18Feature(nn.Module):
    def __init__(self, unfreeze_last_k_layer=0):
        super(ResNet18Feature, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.train(False)  # Freeze batch_norm, dropout, etc.

        for child in list(self.model.children())[:-unfreeze_last_k_layer]:
            for param in child.parameters():
                param.requires_grad = False

        self.n_features = self.model.fc.in_features
        self.model.fc = Identity()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).reshape((1, 3, 1, 1)))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).reshape((1, 3, 1, 1)))

    def train(self, mode=True):
        return  # do nothing because the weights are frozen.

    def forward(self, input):
        # Input: batch_size x C x H x W where C can be either 1 or 3
        batch_size, c, h, w = input.size()
        input = (input.expand(batch_size, 3, h, w) - self.mean.data) / self.std.data
        return self.model(input)


class ResNet34Feature(nn.Module):
    def __init__(self, unfreeze_last_k_layer=0):
        super(ResNet34Feature, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.train(False)  # Freeze batch_norm, dropout, etc.

        for child in list(self.model.children())[:-unfreeze_last_k_layer]:
            for param in child.parameters():
                param.requires_grad = False

        self.n_features = self.model.fc.in_features
        self.model.fc = Identity()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).reshape((1, 3, 1, 1)))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).reshape((1, 3, 1, 1)))

    def train(self, mode=True):
        return  # do nothing because the weights are frozen.

    def forward(self, input):
        # Input: batch_size x C x H x W where C can be either 1 or 3
        batch_size, c, h, w = input.size()
        input = (input.expand(batch_size, 3, h, w) - self.mean.data) / self.std.data
        return self.model(input)


# class ResNet18Metric(nn.Module):
#     def __init__(self):
#         super(ResNet18Metric, self).__init__()
#
#         self.model = torchvision.models.resnet18(pretrained=True)
#         self.model.train(False)
#         self.n_features = self.model.fc.in_features
#         self.model.fc = Identity()
#
#         for child in list(self.model.children())[:-3]:
#             for param in child.parameters():
#                 param.requires_grad = False
#
#     '''def train(self, mode=True):
#         if mode:
#             return self
#         else:
#             return'''
#
#     def forward(self, input):
#         return self.model(input)


class VGG19Feature(nn.Module):
    def __init__(self):
        super(VGG19Feature, self).__init__()

        self.model = torchvision.models.vgg19_bn(pretrained=True).cuda()
        self.model.train(False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.n_features = 4096

    def train(self, mode=True):
        return  # do nothing because the weights are freezed

    def forward(self, input):
        features = self.model.features(input).view(input.size(0), -1)
        out = self.model.classifier._modules['0'](features)  # First FC
        out = self.model.classifier._modules['1'](out)  # ReLU
        return out


class LinearNet(nn.Module):
    '''
    Linear classification using the provided feature extractor.
    '''
    def __init__(self, feature_extractor, n_class):
        super(LinearNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_extractor.n_features, n_class)

    def forward(self, x):
        out = self.feature_extractor(x)
        return self.fc(out)


class LinearClassifier(nn.Module):
    def __init__(self, n_class=1, input_dim=2048, init_scale=1.0, weight_init_method='orthogonal'):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, n_class)

        for layer in (self.fc,):
            if weight_init_method == 'orthogonal':
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

            elif weight_init_method == 'default':
                pass

            else:
                raise RuntimeError('Unknown weight_init_method %s' % weight_init_method)

    def forward(self, x):
        return self.fc(x)


class LinearClassifier2(nn.Module):
    def __init__(self, n_class=1, input_dim=2048, init_method='orthogonal', init_scale=1.0):
        super(LinearClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, n_class)

        for layer in (self.fc1, self.fc2):
            if init_method == 'orthogonal':
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

            elif init_method == 'default':
                pass

            else:
                raise RuntimeError('Unknown weight_init_method %s' % init_method)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Encoder125(nn.Module):
    def __init__(self, init_scale=1.0, output_dim=512, no_weight_init=False):
        super(Encoder125, self).__init__()

        # Input: 1 x 125 x 125
        self.conv1 = nn.Conv2d(1, output_dim // 16, kernel_size=5, stride=2)
        # 32 x 62 x 62
        self.conv2 = nn.Conv2d(output_dim // 16, output_dim // 8, kernel_size=5, stride=2)
        # 64 x 29 x 29
        self.conv3 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=5, stride=2)
        # 128 x 13 x 13
        self.conv4 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=5, stride=2)
        # 256 x 5 x 5
        self.conv5 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=5, stride=1)
        # 512 x 1 x 1

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs):
        # Input: BATCH_SIZE x 1 x H x W
        x = F.relu(self.conv1(src_imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x.flatten(1)


class Decoder125(nn.Module):
    def __init__(self, init_scale=1.0, input_dim=512, no_weight_init=False):
        super(Decoder125, self).__init__()

        # Input: 512 x 1 x 1
        self.conv1 = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=5, stride=1)
        self.conv2 = nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d(input_dim // 8, input_dim // 16, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(input_dim // 16, 1, kernel_size=5, stride=2)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, embeddings):
        # Input: BATCH_SIZE x input_dim
        # Output is NOT clamped into [0.0, 1.0]
        batch_size, dim = embeddings.size()
        x = F.relu(self.conv1(embeddings.view(batch_size, dim, 1, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class Decoder64(nn.Module):
    def __init__(self, init_scale=1.0, input_dim=512, no_weight_init=False):
        super(Decoder64, self).__init__()

        # Input: 512 x 1 x 1
        self.conv1 = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=1)
        self.conv2 = nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=4, stride=2)
        self.conv3 = nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=4, stride=2)
        self.conv4 = nn.ConvTranspose2d(input_dim // 8, input_dim // 16, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(input_dim // 16, 1, kernel_size=4, stride=2)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, embeddings):
        # Input: BATCH_SIZE x input_dim
        # Output is NOT clamped into [0.0, 1.0]
        batch_size, dim = embeddings.size()
        x = F.relu(self.conv1(embeddings.view(batch_size, dim, 1, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class Encoder224(nn.Module):
    def __init__(self, init_scale=1.0, output_dim=512, no_weight_init=False):
        super(Encoder224, self).__init__()

        # Input: 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, output_dim // 32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(output_dim // 32, output_dim // 16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(output_dim // 16, output_dim // 8, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=4, stride=1)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs):
        # Input: BATCH_SIZE x 1 x H x W
        x = F.relu(self.conv1(src_imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x.flatten(1)


class Decoder224(nn.Module):
    def __init__(self, init_scale=1.0, input_dim=512, no_weight_init=False):
        super(Decoder224, self).__init__()

        # Input: 512 x 1 x 1
        self.conv1 = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=4, stride=1)
        self.conv2 = nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(input_dim // 4, input_dim // 8, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d(input_dim // 8, input_dim // 16, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(input_dim // 16, input_dim // 32, kernel_size=5, stride=2)
        self.conv6 = nn.ConvTranspose2d(input_dim // 32, 1, kernel_size=8, stride=2)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, embeddings):
        # Input: BATCH_SIZE x input_dim
        # Output is NOT clamped into [0.0, 1.0]
        batch_size, dim = embeddings.size()
        x = F.relu(self.conv1(embeddings.view(batch_size, dim, 1, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


if __name__ == '__main__':
    # encoder = Encoder224(output_dim=512)
    # out = encoder(torch.rand((1, 1, 224, 224)))
    # print(out.size())
    # exit(0)
    decoder = Decoder64(input_dim=512)
    out = decoder(torch.rand((1, 512)))
    print(out.size())
