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
from util import GANplotter, GANsave_prog, GANshow_prog
import matplotlib.pyplot as plt

#   windows
#save_path = os.path.dirname(os.path.realpath(__file__)) + '\\GAN_models\\'
#   linux
save_path = os.path.dirname(os.path.realpath(__file__)) + '/GAN_models/'

######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=10, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=128, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __VAE-GAN__ ########
parser.add_argument('-l', '--latent', action='store', default=500, type=int,
                    help='latent embedding size')
parser.add_argument('-fcl', action='store', default=32, type=int,
                    help='discriminator fcl size')
parser.add_argument('-db', '--dboost', action='store', default=2, type=float,
                    help='discriminator lr factor')
parser.add_argument('-beta', action='store', default=0.001, type=float,
                    help='Encoder loss param')
parser.add_argument('-df', '--dilation', action='store', default=4, type=int,
                    help='depth dilation factor')
parser.add_argument('-dr', '--drop', action='store', default=0, type=float,
                    help='droprate')

######## __OPTIM__ ########
parser.add_argument('-lr', action='store', default=0.0005, type=float,
                    help='learning rate')
parser.add_argument('-b1', action='store', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-b2', action='store', default=0.999, type=float,
                    help='momentum')
parser.add_argument('-gamma', action='store', default=0.99, type=float,
                    help='learning rate')
args = parser.parse_args()

model_name = '_'.join(['db_'+str(args.dboost), 'l_'+str(args.latent), 'df_'+str(args.dilation), 'kld_'+str(args.beta),
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
    G_acc = np.zeros(epochs)
    D_acc = np.zeros(epochs)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_sim_losses = np.zeros(epochs)
    val_sim_losses = np.zeros(epochs)

    train_loader = get_data_loader(batch_size=batch_size, set='train')
    valid_loader = get_data_loader(batch_size=batch_size, set='valid')

    generator = VAE(d_factor=args.dilation, latent_variable_size=args.latent, cuda=(not args.disable_cuda)).to(args.device)
    generator.load_state_dict(torchload(os.path.dirname(os.path.realpath(__file__)) + '/VAE_models/l_500_df_4_kld_0.01_b1_0.9_b2_0.999_lr_0.001_g_0.99/best_loss'))
    discriminator = Discriminator(d_factor=args.dilation, fcl_size=args.fcl).to(args.device)

    #TODO: add loss control MSE/BCE
    criterion = nn.MSELoss()
    VAE_criterion = nn.MSELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr / args.dboost)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    #TODO: patience loss
    lr_lambda = lambda epoch: args.gamma ** (epoch)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda)
    D_scheduler = optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda)  

    def train():
        G_loss, D_loss, VAE_loss, sim_loss = 0, 0, 0, 0
        D_acc_epoch = 0
        for batch, _ in train_loader:
            batch = batch.to(args.device)
            ones_label = Variable(torch.ones(batch.shape[0], 1))
            zeros_label = Variable(torch.zeros(batch.shape[0], 1))
            
            rec_enc, mu, logvar = generator(batch)
            
            noisev = Variable(torch.randn(batch.shape[0], args.latent))
            rec_noise = generator.decode(noisev)
            
            # train discriminator
            #   real photo
            output = discriminator(batch)
            DR_loss = criterion(output, ones_label)
            D_acc_epoch += ((torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(ones_label) )/6
            G_acc_epoch += 1 - ((torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(ones_label) )/6

            #   reconstructed photo
            output = discriminator(rec_enc)
            DF_loss = criterion(output, zeros_label)
            D_acc_epoch += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6
            G_acc_epoch += 1 - ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6

            #   Decoded noise
            output = discriminator(rec_noise)
            DN_loss = criterion(output, zeros_label)
            D_acc_epoch += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6
            G_acc_epoch += 1- ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6

            D_l = DR_loss + DF_loss + DN_loss
            D_loss += D_l

            D_optimizer.zero_grad()
            D_l.backward(retain_graph=True)
            D_optimizer.step()
            
            # train decoder
            #   real photo
            output = discriminator(batch)
            DR_err = criterion(output, zeros_label)
            G_acc_epoch += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6
            D_acc_epoch += 1- ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6

            #   generated photo
            output = discriminator(rec_enc)
            DF_err = criterion(output, ones_label)
            G_acc_epoch += ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6
            D_acc_epoch += 1- ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6

            #   decoded noise
            output = discriminator(rec_noise)
            DN_err = criterion(output, ones_label)
            G_acc_epoch += ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6
            D_acc_epoch += 1- ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/6

            G_l = DR_err + DF_err + DN_err
            G_loss += G_l

            rec_loss = VAE_criterion(rec_enc, batch)
            sim_loss += rec_loss

            G_optimizer.zero_grad()
            G_l.backward(retain_graph=True)
            G_optimizer.step()
            
            # train encoder 
            prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu)
            VAE_l = (args.beta * prior_loss) + rec_loss
            VAE_loss += VAE_l

            G_optimizer.zero_grad()
            VAE_l.backward(retain_graph=True)
            G_optimizer.step()
        
        #TODO: update this loss tracking
        # this is just to match VAE train.py
        train_losses[epoch] = VAE_loss/len(train_loader)
        train_sim_losses[epoch] = sim_loss/len(train_loader)
        D_acc[epoch] = D_acc_epoch/len(train_loader)
        G_acc[epoch] = G_acc_epoch/len(train_loader)
        D_losses[epoch] = D_loss/len(train_loader)
        G_losses[epoch] = G_loss/len(train_loader)

    def valid():
        VAE_loss, sim_loss = 0, 0
        for batch, _ in valid_loader:
            batch = batch.to(args.device)
            # pass to GPU if available
            batch = batch.to(args.device)
            
            # run network
            rec_enc, mu, logvar = generator(batch)
            rec_loss = VAE_criterion(rec_enc, batch)
            sim_loss += rec_loss

            prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu)
            VAE_l = (args.beta * prior_loss) + rec_loss
            VAE_loss += VAE_l
    
        val_losses[epoch] = VAE_loss/len(valid_loader)
        val_sim_losses[epoch] = sim_loss/len(valid_loader)

    #TODO: return lowest validation loss for hp tuning with hyperopt
    start = time.time()
    for epoch in range(epochs):
        train_loader = get_data_loader(batch_size=batch_size, set='train')
        generator.train()
        discriminator.train()
        train()
        generator.eval()
        discriminator.eval()
        valid()
        #TODO: what type of LR decay for GAN?
        G_scheduler.step()
        D_scheduler.step()

        if args.prog:
            GANshow_prog(epoch, G_losses[epoch], G_acc[epoch], D_losses[epoch], D_acc[epoch],
                         train_losses[epoch], train_sim_losses[epoch], val_losses[epoch], val_sim_losses[epoch], time.time()-start)
        
        best_loss = val_losses[epoch] == min(val_losses[:epoch+1])
        best_t_loss = train_losses[epoch] == min(train_losses[:epoch+1])
        
        #TODO: fix saving for GAN, model is no longer single entity
        if save:
            GANsave_prog(generator, discriminator, save_path, train_losses, val_losses, epoch, save_rate=10, best_loss=best_loss)
        
    # PLOT GRAPHS
    if save:
        GANplotter(model_name, G_losses, G_acc, D_losses, D_acc, train_losses, train_sim_losses, val_losses, val_sim_losses, save=save_path, show=False)
    else:
        GANplotter(model_name, G_losses, G_acc, D_losses, D_acc, train_losses, train_sim_losses, val_losses, val_sim_losses, save=False, show=True)

    print('Model:', model_name, 'completed ; ', epochs, 'epochs', 'in %ds' % (time.time()-start))
    print('min vl_loss: %0.5f at epoch %d' % (min(val_losses), val_losses.argmin()+1))
    print('min tr_loss: %0.5f at epoch %d' % (min(train_losses), train_losses.argmin()+1))

    def show_samples(loader='train'):
        #TODO: ensure no transforms on these
        loader = get_data_loader(batch_size=16, set=loader, shuffle=False)
        for data, _ in loader:
            inputs = data.to(args.device)
            outputs,_,_ = generator(inputs)
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