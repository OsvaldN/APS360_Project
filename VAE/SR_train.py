import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from VAE_GAN import Encoder, Decoder, VAE, Discriminator, reparametrize, SRNet
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
parser.add_argument('-l', '--latent', action='store', default=200, type=int,
                    help='latent embedding size')
parser.add_argument('-fcl', action='store', default=32, type=int,
                    help='discriminator fcl size')
parser.add_argument('-df', '--dilation', action='store', default=20, type=int,
                    help='depth dilation factor')
parser.add_argument('-beta', action='store', default=0.1, type=float,
                    help='Encoder loss param')
parser.add_argument('-db', '--dboost', action='store', default=1, type=int,
                    help='Discminator to Generator train ratio')
parser.add_argument('-ganweight', action='store', default=1, type=float,
                    help='GAN loss weight relative to VAE')
parser.add_argument('-dr', '--drop', action='store', default=0, type=float,
                    help='droprate in generator')


######## __OPTIM__ ########
parser.add_argument('-lr', action='store', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('-b1', action='store', default=0.5, type=float,
                    help='momentum')
parser.add_argument('-b2', action='store', default=0.999, type=float,
                    help='momentum')
parser.add_argument('-gamma', action='store', default=0.99, type=float,
                    help='learning rate')
parser.add_argument("--clip_value", default=0.1, type=float,
                    help="lower and upper clip value for disc. weights")
args = parser.parse_args()

model_name = '_'.join(['l_'+str(args.latent), 'df_'+str(args.dilation), 'kld_'+str(args.beta),
                        'b1_'+str(args.b1), 'b2_'+str(args.b2),
                        'lr_'+str(args.lr), 'g_'+str(args.gamma),
                        'db_'+str(args.dboost), 'gw_'+str(args.ganweight)])
                       
                       
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
    discriminator = Discriminator(d_factor=args.dilation, fcl_size=args.fcl).to(args.device)

    #TODO: add loss control MSE/BCE
    criterion = nn.BCELoss()
    VAE_criterion = nn.MSELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    #TODO: patience loss
    lr_lambda = lambda epoch: args.gamma ** (epoch)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda)
    D_scheduler = optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda)  

    def pretrain(cutoff=0.6):
        D_acc = 0
        while D_acc < cutoff:
            train_loader = get_data_loader(batch_size=batch_size, set='train')
            D_acc = 0
            parity = 0
            for i, (batch, _) in enumerate(train_loader):
                batch = batch.to(args.device)

                ones_label = Variable(torch.ones(batch.shape[0], 1))
                zeros_label = Variable(torch.zeros(batch.shape[0], 1))
                
                rec_enc, mu, logvar = generator(batch)
                
                noisev = Variable(torch.randn(batch.shape[0], args.latent))
                rec_noise = generator.decode(noisev)
                
                ''' train discriminator '''
                #   real photo
                output = discriminator(batch)
                
                DR_loss = criterion(output, ones_label * 0.9)
                D_acc += ((torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(ones_label) )/2

                if parity:
                    #   reconstructed photo
                    output = discriminator(rec_enc)
                    
                    DF_loss = criterion(output, ones_label * 0.1)
                    D_acc+= ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2

                elif not parity:
                    #   Decoded noise
                    output = discriminator(rec_noise)
                    
                    DF_loss = criterion(output, ones_label * 0.1)
                    D_acc += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2

                D_l = DR_loss + DF_loss

                D_optimizer.zero_grad()
                D_l.backward()
                D_optimizer.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

                # switch train type next D batch
                parity = not parity

                if D_acc/(i+1) > cutoff:
                    print('pretrain D_acc: %.3f' % (D_acc/(i+1)))
                    return
            
            D_acc = D_acc/len(train_loader)
            print('pretrain D_acc: %.3f' % (D_acc))

    def train():
        G_loss, D_loss, VAE_loss, sim_loss = [0,0], [0,0], 0, 0
        out_one, out_total = 0, 0
        D_acc_epoch, G_acc_epoch = 0, 0

        parity = 0 # controls whether G or D is trained
        parity2 = 0 # controls whether D is trained on reconstruction or noise
        for batch, _ in train_loader:
            batch = batch.to(args.device)

            ones_label = Variable(torch.ones(batch.shape[0], 1))
            zeros_label = Variable(torch.zeros(batch.shape[0], 1))
            
            rec_enc, mu, logvar = generator(batch)
            
            noisev = Variable(torch.randn(batch.shape[0], args.latent))
            rec_noise = generator.decode(noisev)
            
            
            if not (parity % (args.dboost +1)):
                ''' train discriminator '''
                #   real photo
                output = discriminator(batch)
                out_one += torch.round(output).sum()
                out_total += output.numel()
                
                DR_loss = criterion(output, ones_label * 0.9)
                D_acc_epoch += ((torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(ones_label) )/2
                G_acc_epoch += (1 - ((torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(ones_label) ))/2

                if parity2:
                    #   reconstructed photo
                    output = discriminator(rec_enc)
                    out_one += torch.round(output).sum()
                    out_total += output.numel()
                    
                    DF_loss = criterion(output, ones_label * 0.1)
                    D_acc_epoch += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2
                    G_acc_epoch += (1 - ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) ))/2

                elif not parity2:
                    #   Decoded noise
                    output = discriminator(rec_noise)
                    out_one += torch.round(output).sum()
                    out_total += output.numel()
                    
                    DF_loss = criterion(output, ones_label * 0.1)
                    D_acc_epoch += ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2
                    G_acc_epoch += (1- ( (torch.round(output) == zeros_label).sum().cpu().numpy() / torch.numel(zeros_label) ))/2

                D_l = DR_loss + DF_loss
                D_loss[0] += D_l / 2
                D_loss[1] += 1

                D_optimizer.zero_grad()
                D_l.backward(retain_graph=True)
                D_optimizer.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

                # switch train type next D batch
                parity2 = not parity2
            
            elif (parity % (args.dboost + 1)):
                ''' train generator '''
                #   generated photo
                output = discriminator(rec_enc)
                out_one += torch.round(output).sum()
                out_total += output.numel()
                
                DF_err = criterion(output, ones_label * 0.9)
                G_acc_epoch += ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2
                D_acc_epoch += (1- ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) ))/2

                #   decoded noise
                output = discriminator(rec_noise)
                out_one += torch.round(output).sum()
                out_total += output.numel()
                
                DN_err = criterion(output, ones_label * 0.1)
                G_acc_epoch += ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) )/2
                D_acc_epoch += (1- ( (torch.round(output) == ones_label).sum().cpu().numpy() / torch.numel(zeros_label) ))/2

                # loss consolidation
                G_l =  DF_err + DN_err
                G_loss[0] += G_l / 2
                G_loss[1] += 1

                
                # optim step
                G_optimizer.zero_grad()
                G_l.backward(retain_graph=True)
                G_optimizer.step()
                
            ''' train VAE '''
            rec_loss = VAE_criterion(rec_enc, batch)
            sim_loss += rec_loss

            prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu)
            #VAE_l = ((args.beta * prior_loss) + rec_loss) / args.ganweight
            VAE_l = (args.beta * prior_loss) / args.ganweight
            VAE_loss += VAE_l * args.ganweight
            
            # encoding step
            G_optimizer.zero_grad()
            VAE_l.backward()
            G_optimizer.step()
            
            parity += 1
            torch.cuda.empty_cache()

        print('E %03d fake guess ratio: %3f' % (epoch+1, float(out_one/out_total)))
        train_losses[epoch] = VAE_loss/len(train_loader)
        train_sim_losses[epoch] = sim_loss/len(train_loader)
        D_acc[epoch] = D_acc_epoch/len(train_loader)
        G_acc[epoch] = G_acc_epoch/len(train_loader)
        D_losses[epoch] = D_loss[0]/D_loss[1]
        G_losses[epoch] = G_loss[0]/G_loss[1]

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

    start = time.time()
    discriminator.train()
    pretrain()
    for epoch in range(epochs):
        train_loader = get_data_loader(batch_size=batch_size, set='train')
        generator.train()
        discriminator.train()
        train()
        generator.eval()
        discriminator.eval()
        valid()
        #TODO: what type of LR decay for GAN? is it even necessary?
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
