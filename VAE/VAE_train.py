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

#   windows
#save_path = os.path.dirname(os.path.realpath(__file__)) + '\\VAE_models\\'
#   linux
save_path = os.path.dirname(os.path.realpath(__file__)) + '/VAE_models/'

######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=50, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=128, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __VAE__ ########
parser.add_argument('-l', '--latent', action='store', default=500, type=int,
                    help='latent embedding size')
parser.add_argument('-kld', action='store', default=0.05, type=float,
                    help='KLD loss weight')
parser.add_argument('-df', '--dilation', action='store', default=4, type=int,
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
parser.add_argument('-gamma', action='store', default=0.99, type=float,
                    help='learning rate')
args = parser.parse_args()


model_name = '_'.join(['l_'+str(args.latent), 'df_'+str(args.dilation), 'kld_'+str(args.kld),
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

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_sim_losses = np.zeros(epochs)
    val_sim_losses = np.zeros(epochs)

    train_loader = get_data_loader(batch_size=batch_size, set='train')
    valid_loader = get_data_loader(batch_size=batch_size, set='valid')

    model = VAE(d_factor=args.dilation, latent_variable_size=args.latent, cuda=(not args.disable_cuda)).to(args.device)
    #TODO: add loss control MSE/BCE
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #TODO: patience loss
    lr_lambda = lambda epoch: args.gamma ** (epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  

    def train():
        running_loss = 0
        running_sim_loss = 0
        for batch, _ in train_loader:
            # pass to GPU if available
            batch = batch.to(args.device)
            
            # run network
            output, mu, logvar = model(batch)
            loss = criterion(output, batch)
            running_sim_loss += loss.cpu().data.numpy()
            #   add KLD loss
            loss += -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1)) * args.kld

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # store loss
            running_loss += loss.cpu().data.numpy()
        
        train_losses[epoch] = running_loss/len(train_loader)
        train_sim_losses[epoch] = running_sim_loss/len(train_loader)

    def valid():
        running_loss = 0
        running_sim_loss = 0
        for batch, _ in valid_loader:
            # pass to GPU if available
            batch = batch.to(args.device)
            
            # run network
            output, mu, logvar = model(batch)
            loss = criterion(output, batch)
            running_sim_loss += loss.cpu().data.numpy()
            '''just MSE loss on valid to compare with AE'''
            #   add KLD loss
            loss += -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1)) * args.kld
            
            # store loss
            running_loss += loss.cpu().data.numpy()
    
        val_losses[epoch] = running_loss/len(valid_loader)
        val_sim_losses[epoch] = running_sim_loss/len(valid_loader)

    #TODO: return lowest validation loss for hp tuning with hyperopt
    start = time.time()
    for epoch in range(epochs):
        train_loader = get_data_loader(batch_size=batch_size, set='train')
        model.train()
        train()
        model.eval()
        valid()
        scheduler.step()

        if args.prog:
            show_prog(epoch, train_losses[epoch], val_losses[epoch], time.time()-start)
        
        best_loss = val_losses[epoch] == min(val_losses[:epoch+1])
        best_t_loss = train_losses[epoch] == min(train_losses[:epoch+1])

        if save:
            save_prog(model, save_path, train_losses, val_losses, epoch, save_rate=10, best_loss=best_loss)
        
    # PLOT GRAPHS
    if save:
        plotter(model_name, train_losses, train_sim_losses, val_losses, val_sim_losses, save=save_path, show=False)
    else:
        plotter(model_name, train_losses, train_sim_losses, val_losses, val_sim_losses, save=False, show=True)

    print('Model:', model_name, 'completed ; ', epochs, 'epochs', 'in %ds' % (time.time()-start))
    print('min vl_loss: %0.5f at epoch %d' % (min(val_losses), val_losses.argmin()+1))
    print('min tr_loss: %0.5f at epoch %d' % (min(train_losses), train_losses.argmin()+1))
    print('min tr_sim_loss: %0.5f at epoch %d' % (min(train_sim_losses), train_sim_losses.argmin()+1))
    print('min vl_sim_loss: %0.5f at epoch %d' % (min(val_sim_losses), val_sim_losses.argmin()+1))

    def show_samples(loader='train'):
        #TODO: ensure no transforms on these

        loader = get_data_loader(batch_size=16, set=loader, shuffle=False)
        plt.clf()
        plt.subplot('481')
        for data, _ in loader:
            inputs = data.to(args.device)
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