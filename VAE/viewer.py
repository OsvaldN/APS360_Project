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
   

if __name__ == '__main__':

    save = False

    model = VAE(d_factor=4, latent_variable_size=500, cuda=False)
    model.load_state_dict(torch.load('/home/osvald/Projects/APS360/APS360_Project/VAE/GAN_models/db_2.0_l_500_df_4_kld_0.001_b1_0.9_b2_0.999_lr_0.001_g_0.99/gen_epoch80'))
    model.eval()
    
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