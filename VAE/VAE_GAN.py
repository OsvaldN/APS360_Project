import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Encoder(nn.Module):
    '''
    '''
    def __init__(self, in_channels=3, d_factor=4, latent_variable_size=100, activation='ReLU'):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.latent_variable_size = latent_variable_size
        self.d_factor = d_factor
        
        if activation == 'leakyrelu':
            self.nonlinear = nn.LeakyReLU()
        elif activation == 'ReLU':
            self.nonlinear = nn.ReLU()
        elif activation == 'SELU':
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
        X = self.nonlinear(self.conv1(self.bn1(X)))
        X = self.nonlinear(self.conv2(self.bn2(X)))
        X = self.nonlinear(self.conv3(self.bn3(X)))
        X = self.nonlinear(self.conv4(self.bn4(X)))
        #TODO: should be parameterized and not hardcoded
        X = X.view(-1, self.d_factor*8*4*4)
        return self.fc1(X), self.fc2(X)

class Decoder(nn.Module):
    '''
    '''
    def __init__(self, out_channels=3, c_factor=4, latent_variable_size=100, droprate=0, activation='ReLU'):
        super(Decoder, self).__init__()

        self.out_channels = out_channels
        self.latent_variable_size = latent_variable_size
        self.c_factor = c_factor
        
        if activation == 'leakyrelu':
            self.nonlinear = nn.LeakyReLU()
        elif activation == 'ReLU':
            self.nonlinear = nn.ReLU()
        elif activation == 'SELU':
            self.nonlinear = nn.SELU()

        self.drop0 = nn.Dropout(droprate)
        self.fc1 = nn.Linear(latent_variable_size, c_factor*8*4*4)

        self.drop1 = nn.Dropout2d(droprate)
        self.conv1 = nn.ConvTranspose2d(c_factor*8, c_factor*4, padding=1, kernel_size=4, stride=2)

        self.drop2 = nn.Dropout2d(droprate)
        self.conv2 = nn.ConvTranspose2d(c_factor*4, c_factor*2, padding=1, kernel_size=4, stride=2)

        self.drop3 = nn.Dropout2d(droprate)
        self.conv3 = nn.ConvTranspose2d(c_factor*2, c_factor, padding=1, kernel_size=4, stride=2)

        self.drop4 = nn.Dropout2d(droprate)
        self.conv4 = nn.ConvTranspose2d(c_factor, out_channels, padding=1, kernel_size=4, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, Z):
        Z = self.nonlinear(self.fc1(self.drop0(Z)))
        Z = Z.view(-1, self.c_factor*8, 4, 4)
        Z = self.conv1(self.drop1(Z))
        Z = self.conv2(self.drop2(Z))
        Z = self.conv3(self.drop3(Z))
        Z = self.conv4(self.drop4(Z))
        return self.sigmoid(Z)

class VAE(nn.Module):
    '''
    '''
    def __init__(self, in_channels=3, d_factor=4, latent_variable_size=100, droprate=0, cuda=False, activation='SELU'):
        super(VAE, self).__init__()
        
        self.cuda = cuda
        self.encode = Encoder(in_channels, d_factor, latent_variable_size, activation=activation)
        self.decode = Decoder(in_channels, d_factor, latent_variable_size, droprate=droprate, activation=activation)

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

def reparametrize(mu, logvar, cuda=False):
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

class SRNet(nn.Module):
    def __init__(self, in_channels=3, d_factor=4, activation='SELU'):
        super(SRNet, self).__init__()

        self.in_channels = in_channels
        self.d_factor = d_factor
        
        if activation == 'leakyrelu':
            self.nonlinear = nn.LeakyReLU()
        elif activation == 'ReLU':
            self.nonlinear = nn.ReLU()
        elif activation == 'SELU':
            self.nonlinear = nn.SELU()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.downconv1 = nn.Conv2d(in_channels, d_factor, kernel_size=4, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(d_factor)
        self.downconv2 = nn.Conv2d(d_factor, d_factor*2, kernel_size=4, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(d_factor*2)
        self.downconv3 = nn.Conv2d(d_factor*2, d_factor*4, kernel_size=4, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(d_factor*4)
        self.downconv4 = nn.Conv2d(d_factor*4, d_factor*8, kernel_size=4, stride=2, padding=1)

        self.upconv1 = nn.ConvTranspose2d(d_factor*8, d_factor*4, padding=1, kernel_size=4, stride=2)

        self.upconv2 = nn.ConvTranspose2d(d_factor*4*2, d_factor*2, padding=1, kernel_size=4, stride=2)

        self.upconv3 = nn.ConvTranspose2d(d_factor*2*2, d_factor, padding=1, kernel_size=4, stride=2)

        self.upconv4 = nn.ConvTranspose2d(d_factor*2, in_channels, padding=1, kernel_size=4, stride=2)
        
    def forward(self, X):
        # down convs
        X1 = self.nonlinear(self.downconv1(self.bn1(X)))
        X2 = self.nonlinear(self.downconv2(self.bn2(X1)))
        X3 = self.nonlinear(self.downconv3(self.bn3(X2)))
        X4 = self.nonlinear(self.downconv4(self.bn4(X3)))
        # up convs
        X = self.upconv1(X4)
        X = self.upconv2(torch.cat((X3,X),axis=2))
        X = self.upconv3(torch.cat((X2,X),axis=2))
        X = self.upconv4(torch.cat((X1,X),axis=2))
        return self.sigmoid(X)

class Discriminator(nn.Module):
    '''
    '''
    def __init__(self, in_channels=3, d_factor=4, fcl_size=32, droprate=0.5):
        super(Discriminator, self).__init__()
       
        self.d_factor = d_factor
        self.nonlinear = nn.SELU()
        #self.nonlinear = nn.LeakyReLU(0.2)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, d_factor, kernel_size=4, stride=2, padding=1)

        self.drop2 = nn.Dropout2d(p=droprate)
        self.bn2 = nn.BatchNorm2d(d_factor)
        self.conv2 = nn.Conv2d(d_factor, d_factor*2, kernel_size=4, stride=2, padding=1)

        self.drop3 = nn.Dropout2d(p=droprate)
        self.bn3 = nn.BatchNorm2d(d_factor*2)
        self.conv3 = nn.Conv2d(d_factor*2, d_factor*4, kernel_size=4, stride=2, padding=1)

        self.drop4 = nn.Dropout2d(p=droprate)
        self.bn4 = nn.BatchNorm2d(d_factor*4)
        self.conv4 = nn.Conv2d(d_factor*4, d_factor*8, kernel_size=4, stride=2, padding=1)

        self.drop5 = nn.Dropout2d(p=droprate)
        self.bn5 = nn.BatchNorm1d(d_factor*8*4*4)
        self.fcl = nn.Linear(d_factor*8*4*4, fcl_size)
        self.out = nn.Linear(fcl_size, 1)
        self.simgoid = nn.Sigmoid()

    def forward(self, X):
        X = self.nonlinear(self.conv1(self.bn1(X)))
        X = self.nonlinear(self.conv2(self.bn2(self.drop2(X))))
        X = self.nonlinear(self.conv3(self.bn3(self.drop3(X))))
        X = self.nonlinear(self.conv4(self.bn4(self.drop4(X))))
        X = self.drop5(X)
        X = X.view(-1, self.d_factor*8*4*4)
        X = self.out(self.nonlinear(self.fcl(self.bn5(X))))
        return self.simgoid(X)

