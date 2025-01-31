import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from VAE_GAN import Encoder, Decoder, VAE
from data_loader import get_data_loader, raw_loader
from viewer import show_samples, generate
import matplotlib.pyplot as plt
   
def get_latent_dist(model, n=1000, b_size=64):
    '''
    Gets approx. distribution of n random samples in latent space
    '''
    loader = raw_loader(batch_size=b_size)
    latent = torch.tensor([])

    with torch.no_grad():
        for batch, _ in loader:
            
            # run network
            mu, logvar = model.encode(batch)
            z = model.reparametrize(mu, logvar)
            latent = torch.cat((latent, z), dim=0)

            if (n != None) and (latent.shape[0] >= n):
                break

    return latent.cpu().numpy()

if __name__ == '__main__':

    save = False

    latent = 1000
    dilation= 16
    #folder = '/home/osvald/Projects/APS360/APS360_Project/VAE/GAN_models/l_200_df_16_kld_0.5_b1_0.5_b2_0.999_lr_0.001_g_0.99_db_2_gw_1.5'
    #folder = "C:\\Users\\osval\\Documents\\School\\APS360\\APS360_Project\\VAE\\GAN_models\\db_1.0_l_1000_df_16_kld_0.001_b1_0.5_b2_0.999_lr_0.0005_g_0.99"
    folder = "/home/osvald/Projects/APS360/APS360_Project/VAE/GAN_models/db_1.0_l_1000_df_16_kld_0.001_b1_0.5_b2_0.999_lr_0.0005_g_0.99"
    #folder = "C:\\Users\\osval\\Documents\\School\\APS360\\APS360_Project\\VAE\\VAE_models\\l_500_df_20_kld_0.01_b1_0.9_b2_0.999_lr_0.001_g_0.99"
    state = '/gen_epoch120'
    model = VAE(d_factor=dilation, latent_variable_size=latent, cuda=False, activation='SELU')#.to('cpu')
    model.load_state_dict(torch.load(folder + state))#, map_location='cpu'))
    model.eval()

    sc = StandardScaler()
    pca = PCA(n_components=30)

    # Get latent distribution
    latent = get_latent_dist(model, n=None)
    # standardize latent distribution
    latent = sc.fit_transform(latent)
    # save standardized latent distribution
    pickle.dump(sc, open( folder+'/std_scaler30.p', "wb" ) )

    # perform PCA
    pca = pca.fit(latent)
    # save PCA
    pickle.dump(pca, open( folder+'/pca30.p', "wb" ) )
    pickle.dump(pca.components_, open( folder+'/components30.p', "wb" ) )
    pickle.dump(pca.mean_, open( folder+'/mean30.p', "wb" ) )
    print(pca.explained_variance_ratio_)