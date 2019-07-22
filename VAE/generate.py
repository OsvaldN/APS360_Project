import matplotlib.pyplot as plt
import numpy as np
import torch

#TODO: load model and have some sort of navigation here
#TODO: load some (deterministic) pics from train, test
#TODO: generate new images by random sample in latent space

def show_samples(sample_number = 5):
  train_loader = get_data_loader(sample_number)
  for data, _ in train_loader:
    data = data.cuda()
    recon_emb = model.encoder(data)
    rad = torch.randn(sample_number, 64, 4, 4).cuda()
    mod_emb = recon_emb + rad 
    mod_emb =  model.decoder(mod_emb)
    recon_outputs = model.decoder(recon_emb)
    for i in range(sample_number):
      images = mod_emb
      recon_images = recon_outputs
      ori_images = data 
      image = np.transpose(images[i].detach().cpu().numpy(), [1,2,0])
      recon_image = np.transpose(recon_images[i].detach().cpu().numpy(), [1,2,0])
      ori_image = np.transpose(ori_images[i].detach().cpu().numpy(), [1,2,0])
      plt.subplot(3, sample_number, i+1)
      plt.imshow(image)  
      plt.subplot(3, sample_number, sample_number+i+1)
      plt.imshow(recon_image)
      plt.subplot(3, sample_number, sample_number*2+i+1)
      plt.imshow(ori_image)
    break

show_samples(sample_number = 5)

def generate_rand_sample():
  sample = torch.randn(1, 64, 4, 4)
  sample = sample.cuda()
  recon = model.decoder(sample)
  recon = np.transpose(recon[0].detach().cpu().numpy(), [1,2,0])
  plt.imshow(recon)

generate_rand_sample()