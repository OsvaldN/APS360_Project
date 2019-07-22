import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from VAE_GAN import Encoder, Decoder, VAE, Discriminator, reparametrize
from data_loader import get_data_loader
from util import plotter, save_prog, show_prog
import matplotlib.pyplot as plt

save_path = os.path.dirname(os.path.realpath(__file__)) + '\\models\\'

######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true', default=True,
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=10, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=32, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __VAEGAN__ ########
parser.add_argument('-l', '--latent', action='store', default=500, type=int,
                    help='latent embedding size')
parser.add_argument('-fcl', action='store', default=32, type=int,
                    help='discriminator fcl size')
parser.add_argument('-beta', action='store', default=5, type=float,
                    help='Encoder loss param')
parser.add_argument('-df', '--dilation', action='store', default=32, type=int,
                    help='depth dilation factor')
parser.add_argument('-dr', '--drop', action='store', default=0, type=float,
                    help='droprate')

######## __OPTIM__ ########
parser.add_argument('-lr', action='store', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('-b1', action='store', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-b2', action='store', default=0.999, type=float,
                    help='momentum')
parser.add_argument('-gamma', action='store', default=0.9, type=float,
                    help='learning rate')
args = parser.parse_args()


model_name = '_'.join(['b_'+str(args.batch),'dr_'+str(args.drop),
                          'l_'+str(args.latent), 'df_'+str(args.dilation),
                          'b1_'+str(args.b1), 'b2_'+str(args.b2),
                          'lr_'+str(args.lr), 'g_'+str(args.gamma)])
                       

if __name__ == '__main__':

    # Create target Directory if don't exist
    if args.nosave: print('WARNING: MODEL AND DATA ARE NOT BEING SAVED')
    elif not args.nosave:
        if not os.path.exists(save_path+model_name):
            os.mkdir(save_path+model_name)
        else:
            print('WARNING: overwriting existing directory:', model_name)
    save_path = save_path + model_name + '/'

    ######## __GPU_SETUP__ ########
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    epochs = args.epochs
    batch_size = args.batch
    save = not args.nosave

    G_losses = np.zeros(epochs)
    D_losses = np.zeros(epochs)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    #train_loader = get_data_loader(batch_size=batch_size, set='strain')
    train_loader = get_data_loader(batch_size=batch_size, set='stest')
    valid_loader = get_data_loader(batch_size=batch_size, set='svalid')

    generator = VAE(d_factor=args.dilation, latent_variable_size=args.latent, cuda=(not args.disable_cuda))
    discriminator = Discriminator(d_factor=args.dilation, fcl_size=args.fcl)

    #TODO: add loss control MSE/BCE
    criterion = nn.MSELoss()
    VAE_criterion = nn.MSELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    #TODO: patience loss
    lr_lambda = lambda epoch: args.gamma ** (epoch)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda)
    D_scheduler = optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda)  
    Discriminator = Discriminator()

    def train():
        G_losses, D_losses, VAE_losses = 0, 0, 0
        for batch, _ in train_loader:
            ones_label = Variable(torch.ones(batch_size))
            zeros_label = Variable(torch.zeros(batch_size))
            
            rec_enc, mu, logvar = generator(batch)
            
            noisev = Variable(torch.randn(batch_size, args.latent))
            rec_noise = generator.decode(noisev)
            
            # train discriminator
            #   real photo
            output = discriminator(batch)
            DR_loss = criterion(output, ones_label)
            #   reconstructed photo
            output = discriminator(rec_enc)
            DF_loss = criterion(output, zeros_label)
            #   Decoded noise
            output = discriminator(rec_noise)
            DN_loss = criterion(output, zeros_label)
            
            D_loss = DR_loss + DF_loss + DN_loss
            D_losses += D_loss

            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()
            
            # train decoder
            #   real photo
            output = discriminator(batch)
            DR_err = criterion(output, ones_label)
            #   generated photo
            output = discriminator(rec_enc)
            DF_err = criterion(output, zeros_label)
            #   decoded noise
            output = discriminator(rec_noise)
            DN_err = criterion(output, zeros_label)
            
            dis_loss = DR_err + DF_err + DN_err
            G_loss = -(dis_loss)
            G_losses += G_loss

            rec_loss = VAE_criterion(rec_enc, batch)

            G_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            G_optimizer.step()
            
            # train encoder 
            # #TODO: understand this lol
            prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu)
            VAE_loss = prior_loss + args.beta * rec_loss
            VAE_losses += VAE_loss

            G_optimizer.zero_grad()
            VAE_loss.backward(retain_graph=True)
            G_optimizer.step()
        
        #TODO: update this loss tracking
        # this is just to match VAE train.py
        train_losses[epoch] = VAE_losses/len(train_loader)

    def valid():
        running_loss = 0
        for batch, _ in valid_loader:
            # pass to GPU if available
            batch = batch.to(args.device)
            
            # run network
            output, mu, logvar = generator(batch)
            loss = criterion(output, batch)
            
            # store loss
            running_loss += loss.cpu().data.numpy()
    
        val_losses[epoch] = running_loss/len(valid_loader)

    #TODO: return lowest validation loss for hp tuning with hyperopt
    start = time.time()
    for epoch in range(epochs):
        train_loader = get_data_loader(batch_size=batch_size, set='strain')
        generator.train()
        discriminator.train()
        train()
        generator.eval()
        discriminator.eval()
        valid()
        #TODO: what type of LR decay for GAN?
        #scheduler.step()

        if args.prog:
            show_prog(epoch, train_losses[epoch], val_losses[epoch], time.time()-start)
        
        best_loss = val_losses[epoch] == min(val_losses[:epoch+1])
        best_t_loss = train_losses[epoch] == min(train_losses[:epoch+1])
        
        #TODO: fix saving for GAN, model is no longer single entity
        if save:
            save_prog(model, save_path, train_losses, val_losses, epoch, save_rate=10, best_loss=best_loss)
        
    # PLOT GRAPHS
    if save:
        plotter(model_name, train_losses, val_losses, save=save_path, show=False)
    else:
        plotter(model_name, train_losses, val_losses, save=False, show=True)

    print('Model:', model_name, 'completed ; ', epochs, 'epochs', 'in %ds' % (time.time()-start))
    print('min vl_loss: %0.5f at epoch %d' % (min(val_losses), val_losses.argmin()+1))
    print('min tr_loss: %0.5f at epoch %d' % (min(train_losses), train_losses.argmin()+1))

    def show_samples(loader='train'):
        #TODO: ensure no transforms on these
        loader = get_data_loader(batch_size=16, set=loader, shuffle=False)
        for data, _ in loader:
            inputs = data.to(args.device)
            outputs,_,_ = generator(data)
            for i in range(16):
                plt.subplot('481')
                output = np.transpose(outputs[i].detach().cpu().numpy(), [1,2,0])
                original = np.transpose(inputs[i].detach().cpu().numpy(), [1,2,0])
                plt.subplot(4, 8, i+1)
                plt.imshow(original)  
                plt.subplot(4, 8, 16+i+1)
                plt.imshow(output)
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