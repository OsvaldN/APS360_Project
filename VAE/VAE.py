import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

#TODO:
#     try removing pooling and increasing depth
#     only if model capacity turns out to be insufficient
#     regardless this should eventually be parameterized

class Encoder(nn.Module):
    '''
    '''
    def __init__(self, in_channels=3, d_factor=4, latent_variable_size=100, droprate=0):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.latent_variable_size = latent_variable_size
        self.d_factor = d_factor
        
        self.nonlinear = nn.SELU()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, d_factor, kernel_size=4, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(d_factor)
        self.conv2 = nn.Conv2d(d_factor, d_factor*2, kernel_size=4, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(d_factor*2)
        self.conv3 = nn.Conv2d(d_factor*2, d_factor*4, kernel_size=4, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(d_factor*4)
        self.conv4 = nn.Conv2d(d_factor*4, d_factor*8, kernel_size=4, stride=2, padding=1)
        
        # depth factor * final h * final w
        self.fc1 = nn.Linear(d_factor*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(d_factor*8*4*4, latent_variable_size)

    def forward(self, X):
        print(X.shape)
        X = self.nonlinear(self.conv1(self.bn1(X)))
        print(X.shape)
        X = self.nonlinear(self.conv2(self.bn2(X)))
        print(X.shape)
        X = self.nonlinear(self.conv3(self.bn3(X)))
        print(X.shape)
        X = self.nonlinear(self.conv4(self.bn4(X)))
        print(X.shape)
        #TODO: should be parameterized and not hardcoded
        X = X.view(-1, self.d_factor*8*4*4)
        return self.fc1(X), self.fc2(X)

class Decoder(nn.Module):
    '''
    '''
    def __init__(self, out_channels=3, c_factor=4, latent_variable_size=100, droprate=0):
        super(Decoder, self).__init__()

        self.out_channels = out_channels
        self.latent_variable_size = latent_variable_size
        self.c_factor = c_factor
        
        self.nonlinear = nn.SELU()

        self.drop0 = nn.Dropout1d(droprate)
        self.fc1 = nn.Linear(latent_variable_size, c_factor*8*4*4)

        self.drop1 = nn.Dropout2d(droprate)
        self.conv1 = nn.ConvTranspose2d(c_factor*8, c_factor*4, padding=1, kernel_size=4, stride=2)

        self.drop2 = nn.Dropout2d(droprate)
        self.conv2 = nn.ConvTranspose2d(c_factor*4, c_factor*2, padding=1, kernel_size=4, stride=2)

        self.drop3 = nn.Dropout2d(droprate)
        self.conv3 = nn.ConvTranspose2d(c_factor*2, c_factor, padding=1, kernel_size=4, stride=2)

        self.drop4 = nn.Dropout2d(droprate)
        self.conv4 = nn.ConvTranspose2d(c_factor, out_channels, padding=1, kernel_size=4, stride=2)

    def forward(self, Z):
        Z = self.nonlinear(self.fc1(self.drop0(Z)))
        Z = Z.view(-1, self.c_factor*8, 4, 4)
        Z = self.conv1(self.drop1(Z))
        Z = self.conv2(self.drop2(Z))
        Z = self.conv3(self.drop3(Z))
        Z = self.conv4(self.drop4(Z))
        return Z

class VAE(nn.Module):
    def __init__(self, in_channels=3, d_factor=4, latent_variable_size=100, droprate=0, cuda=False):
        super(VAE, self).__init__()
        
        self.cuda = cuda
        self.encode = Encoder(in_channels, d_factor, latent_variable_size, droprate=droprate)
        self.decode = Decoder(in_channels, d_factor, latent_variable_size, droprate=droprate)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar