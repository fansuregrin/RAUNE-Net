import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torchvision import models


"""This code is from "https://github.com/xahidbuffon/FUnIE-GAN".
"""

def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


"""
 > Network architecture of UGAN / UGAN-P model
   * Original paper: https://arxiv.org/pdf/1801.04011.pdf
   * Original repo: github.com/cameronfabbri/Underwater-Color-Correction
 > Maintainer: https://github.com/xahidbuffon
"""
class GeneratorUNet(nn.Module):
    """ Standard UNet generator with skip connections used by  
        - Pix2Pix (https://phillipi.github.io/pix2pix/)
        - UGAN (https://github.com/cameronfabbri/Underwater-Color-Correction)  
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """ Standard PatchGAN discriminator (cGAN)  
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)


class UGAN_Nets:
    def __init__(self, base_model='pix2pix'):
        if base_model=='pix2pix': # default
            self.netG = GeneratorUNet() 
            self.netD = Discriminator()
        elif base_model=='resnet':
            #TODO: add ResNet support
            pass
        else: 
            pass


class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
        self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        # get graduent of pred and true
        gradX_true, gradY_true = self.get_gradients(true)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
        # compute and return GDL
        return 0.5 * torch.mean(grad_true - grad_pred_a)

