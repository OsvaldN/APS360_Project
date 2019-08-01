import matplotlib.pyplot as plt
import numpy as np
import torch

#TODO: split into VAE util and GAN util

def plotter(model_name, train_losses, train_sim_losses, val_losses, val_sim_losses, save=False, show=True, loss_type='MSELoss'):
    '''
    Plots loss curves
    or saves plot of loss curves
    '''
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.plot(train_sim_losses, label='train_sim')
    plt.plot(val_sim_losses, label='val_sim')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title('Loss')
    plt.grid()
    if show:
        plt.show()
    if bool(save):
        plt.savefig(save+'curves.png')
    plt.clf()
    return

def GANplotter(model_name, G_loss, G_acc, D_loss, D_acc, train_losses, train_sim_losses, val_losses, val_sim_losses, save=False, show=True, loss_type='MSELoss'):
    '''
    Plots loss curves
    or saves plot of loss curves
    #TODO: clean up lol this is really ugly
    '''
    if show:
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.plot(train_sim_losses, label='train_sim')
        plt.plot(val_sim_losses, label='val_sim')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(loss_type + ' + KLD loss')
        plt.title('Reconstruction Loss')
        plt.grid()

        plt.subplot(3,1,2)
        plt.plot(G_loss, c='g', label='G_loss')
        plt.plot(D_loss, c='r', label='D_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(loss_type)
        plt.title('GAN Loss')
        plt.grid()

        plt.subplot(3,1,3)
        plt.plot(G_acc, c='g', label='G_acc')
        plt.plot(D_acc, c='r', label='D_acc')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.grid()
        plt.show()
    if bool(save):
        plt.clf()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.plot(train_sim_losses, label='train_sim')
        plt.plot(val_sim_losses, label='val_sim')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(loss_type + ' + KLD loss')
        plt.title('Reconstruction Loss')
        plt.grid()
        plt.savefig(save+'reconstruction_loss.png')

        plt.clf()
        plt.plot(G_loss, c='g', label='G_loss')
        plt.plot(D_loss, c='r', label='D_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(loss_type)
        plt.title('GAN Loss')
        plt.grid()
        plt.savefig(save+'GAN_loss.png')

        plt.clf()
        plt.plot(G_acc, c='g', label='G_acc')
        plt.plot(D_acc, c='r', label='D_acc')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.grid()
        plt.savefig(save+'GAN_accuracy.png')
    plt.clf()
    return

def GANshow_prog(epoch, G_loss, G_acc, D_loss, D_acc, train_loss, train_sim_loss, val_loss, val_sim_loss, time_elapsed):
    '''
    Prints current epoch's losses and runtime
    '''
    print('E %03d --- RUNTIME: %ds' % (epoch+1, time_elapsed))
    print('GNRTR  |  loss: %.4f   |  acc: %.4f' % (G_loss, G_acc))
    print('DSCRM  |  loss: %.4f   |  acc: %.4f' % (D_loss, D_acc))
    print('TRAIN  |  loss: %.4f   |  sim: %.4f' % (train_loss, train_sim_loss))
    print('VALID  |  loss: %.4f   |  sim: %.4f' % (val_loss, val_sim_loss))

def show_prog(epoch, train_loss, train_sim_loss, val_loss, val_sim_loss, time_elapsed, kld_weight):
    '''
    Prints current epoch's losses and runtime
    '''
    print('E %03d --- RUNTIME: %ds' % (epoch+1, time_elapsed))
    print('TRAIN  |  loss: %.4f  |  mse_loss: %.4f  |  kld: %.4f' % (train_loss, train_sim_loss, (train_loss-train_sim_loss)/kld_weight))
    print('VALID  |  loss: %.4f  |  mse_loss: %.4f  |  kld: %.4f' % (val_loss, val_sim_loss, (val_loss-val_sim_loss)/kld_weight))
    
def save_prog(model, model_path, train_losses, val_losses, epoch, save_rate, best_loss):
    '''
    Saves losses to model folder
    Saves model state dict every save_rate epochs
    '''
    np.save(model_path +'train_losses', train_losses)
    np.save(model_path +'val_losses', val_losses)

    if (epoch+1) % save_rate == 0: #save model dict at save_rate epochs
        torch.save(model.state_dict(), model_path + 'model_epoch%s' % (epoch+1))

    if best_loss: #save model dict at best loss
        torch.save(model.state_dict(), model_path + 'best_loss')

def GANsave_prog(generator, discriminator, model_path, train_losses, val_losses, epoch, save_rate, best_loss):
    '''
    Saves losses to model folder
    Saves model state dict every save_rate epochs
    '''
    np.save(model_path +'train_losses', train_losses)
    np.save(model_path +'val_losses', val_losses)

    if (epoch+1) % save_rate == 0: #save model dict at save_rate epochs
        torch.save(generator.state_dict(), model_path + 'gen_epoch%s' % (epoch+1))
        torch.save(discriminator.state_dict(), model_path + 'dis_epoch%s' % (epoch+1))

    if best_loss: #save model dict at best loss
        torch.save(generator.state_dict(), model_path + 'gen_best_loss')
        torch.save(discriminator.state_dict(), model_path + 'dis_best_loss')