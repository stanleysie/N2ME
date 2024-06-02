import torch
import torch.nn as nn
import numpy as np

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
    
    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self.weights_init_fn)
    
    def weights_init_fn(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            model.weight.data.normal_(0.0, 0.02)
            if hasattr(model.bias, 'data'):
                model.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            model.weight.data.normal_(1.0, 0.02)
            model.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=False)
        )
    
    def forward(self, x):
        return x + self.main(x)


class Generator(BaseNetwork):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=False))
        layers.append(nn.ReLU(inplace=True))

        # Down sampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2
        
        # Bottleneck layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        # Up sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        self.main = nn.Sequential(*layers)

        # color mask
        self.img_reg = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

        # attention mask
        self.attention_reg = nn.Sequential(
            nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        # initializing weights
        self.init_weights()
    
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        features = self.main(x)
    
        return self.attention_reg(features), self.img_reg(features)


class Discriminator(BaseNetwork):
    def __init__(self, img_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim *= 2
        
        self.main = nn.Sequential(*layers)

        kernel_size = int(img_size / np.power(2, repeat_num))
        # patch discriminator top
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # au regressor
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
    
        # initializing weights
        self.init_weights()

    def forward(self, x):
        h = self.main(x)

        out_real = self.conv1(h)
        out_aux = self.conv2(h)

        return out_real.squeeze(), out_aux.squeeze()