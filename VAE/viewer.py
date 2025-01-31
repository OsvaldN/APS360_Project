import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from VAE_GAN import Encoder, Decoder, VAE
from data_loader import get_data_loader
from util import plotter, save_prog, show_prog
import matplotlib.pyplot as plt
   
def show_samples(loader='train'):
    #TODO: ensure no transforms on these

    loader = get_data_loader(batch_size=16, set=loader, shuffle=False)
    plt.clf()
    plt.subplot('481')
    for data, _ in loader:
        inputs = data
        outputs,_,_ = model(inputs)
        for i in range(16):
            output = np.transpose(outputs[i].detach().cpu().numpy(), [1,2,0])
            original = np.transpose(inputs[i].detach().cpu().numpy(), [1,2,0])
            plt.subplot(4, 8, i+1)
            plt.axis('off')
            plt.imshow(original)  
            plt.subplot(4, 8, 16+i+1)
            plt.axis('off')
            plt.imshow(output)
        
        if save:
            plt.savefig(save_path+'faces.png')
        plt.show()

        break

def generate():

    plt.clf()
    plt.subplot('281')

    Guassi_boi = torch.cat((torch.randn(8, latent) * 0.25, torch.randn(8, latent) * 0.50,
                            torch.randn(8, latent) * 0.75, torch.randn(8, latent) * 1.00,
                            torch.randn(8, latent) * 1.25, torch.randn(8, latent) * 1.50,
                            torch.randn(8, latent) * 1.75, torch.randn(8, latent) * 2.00), dim=0)
    outputs = model.decode(Guassi_boi)

    for i in range(64):
        output = np.transpose(outputs[i].detach().cpu().numpy(), [1,2,0])
        plt.subplot(8, 8, i+1)
        plt.axis('off')
        plt.imshow(output)  
    
    if save:
        plt.savefig(save_path+'faces.png')
    plt.show()

if __name__ == '__main__':

    save = False

    latent = 100
    dilation=16
    folder = '/home/osvald/Projects/APS360/APS360_Project/VAE/GAN_models/l_200_df_16_kld_0.5_b1_0.5_b2_0.999_lr_0.001_g_0.99_db_2_gw_1.5'
    state = '/gen_epoch20'
    model = VAE(d_factor=dilation, latent_variable_size=latent, cuda=False, activation='SELU').to('cpu')
    model.load_state_dict(torch.load(folder + state))
    model.eval()

    generate()
    show_samples()

'''
def generate_rand_sample():
  sample = torch.randn(1, 64, 4, 4)
  sample = sample.cuda()
  recon = model.decoder(sample)
  recon = np.transpose(recon[0].detach().cpu().numpy(), [1,2,0])
  plt.imshow(recon)

generate_rand_sample()
'''