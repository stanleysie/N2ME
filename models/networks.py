import torch
import torch.nn as nn
import torch.nn.functional as F
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

            
class ResBlock(nn.Module):
    # https://github.com/mindslab-ai/hififace/blob/master/model/hififace.py
    def __init__(self, dim_in, dim_out, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample

        layers = []
        layers.append(nn.InstanceNorm2d(dim_in))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
        
        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2))
        elif upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        layers.append(nn.InstanceNorm2d(dim_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))
        
        self.main = nn.Sequential(*layers)

        layers = [nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)]
        if self.downsample:
            layers.append(nn.AvgPool2d(kernel_size=2))
        elif self.upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x) + self.side(x)


class AdaIn(nn.Module):
    # https://github.com/mindslab-ai/hififace/blob/master/model/hififace.py
    def __init__(self, in_channel, vector_size):
        super(AdaIn, self).__init__()
        self.eps = 1e-5
        self.std_style_fc = nn.Linear(vector_size, in_channel)
        self.mean_style_fc = nn.Linear(vector_size, in_channel)

    def forward(self, x, style_vector):
        std_style = self.std_style_fc(style_vector)
        mean_style = self.mean_style_fc(style_vector)

        std_style = std_style.unsqueeze(-1).unsqueeze(-1)
        mean_style = mean_style.unsqueeze(-1).unsqueeze(-1)

        x = F.instance_norm(x)
        x = std_style * x + mean_style
        return x


class AdaInResBlock(nn.Module):
    # https://github.com/mindslab-ai/hififace/blob/master/model/hififace.py
    def __init__(self, dim_in, dim_out, vector_size):
        super(AdaInResBlock, self).__init__()

        self.adain1 = AdaIn(dim_in, vector_size)
        self.adain2 = AdaIn(dim_out, vector_size)

        self.part1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        )
        self.part2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x, c):
        adain1 = self.adain1(x, c)
        out = self.part1(adain1)
        adain2 = self.adain2(out, c)
        out = self.part2(adain2)

        return x + out
    

class Generator(BaseNetwork):
    def __init__(self, conv_dim=64, c_dim=5):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1),
        )

        self.down1 = ResBlock(conv_dim, conv_dim*2, downsample=True)
        self.down2 = ResBlock(conv_dim*2, conv_dim*4, downsample=True)

        self.resblock1 = ResBlock(conv_dim*4, conv_dim*4)
        self.resblock2 = ResBlock(conv_dim*4, conv_dim*4)
        self.resblock3 = ResBlock(conv_dim*4, conv_dim*4)
        self.resblock4 = AdaInResBlock(conv_dim*4, conv_dim*4, c_dim)
        self.resblock5 = ResBlock(conv_dim*4, conv_dim*4)
        self.resblock6 = ResBlock(conv_dim*4, conv_dim*4)

        self.up1 = ResBlock(conv_dim*4*2, conv_dim*2, upsample=True)
        self.up2 = ResBlock(conv_dim*2*2, conv_dim, upsample=True)

        self.out = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(conv_dim*2, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, c):
        conv1 = self.conv1(x)
#         print('conv1', conv1.shape)
        down1 = self.down1(conv1)
#         print('down1', down1.shape)
        down2 = self.down2(down1)
#         print('down2', down2.shape)

        resblock1 = self.resblock1(down2)
#         print('res_block1', resblock1.shape)
        resblock2 = self.resblock2(resblock1)
#         print('res_block2', resblock2.shape)
        resblock3 = self.resblock3(resblock2)
#         print('res_block3', resblock3.shape)
        resblock4 = self.resblock4(resblock3, c)
#         print('res_block4', resblock3.shape)
        resblock5 = self.resblock5(resblock4)
#         print('res_block5', resblock5.shape)
        resblock6 = self.resblock6(resblock5)
#         print('res_block6', resblock6.shape)

        out = torch.cat([resblock6, down2], dim=1)
#         print('cat', out.shape)
        up1 = self.up1(out)
#         print('up1', up1.shape)
        out = torch.cat([up1, down1], dim=1)
#         print('cat', out.shape)
        up2 = self.up2(out)
#         print('up2', up2.shape)
        out = torch.cat([up2, conv1], dim=1)
#         print('cat', out.shape)
        out = self.out(out)
#         print(out.shape)

        return out


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