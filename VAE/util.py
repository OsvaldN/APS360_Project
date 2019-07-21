import matplotlib.pyplot as plt
import numpy as np
import torch

def plotter(model_name, train_losses, val_losses, save=False, show=True, loss_type='MSELoss'):
    '''
    Plots loss curves
    or saves plot of loss curves
    '''
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title('Loss')
    plt.grid()
    if show:
        plt.show()
    if bool(save):
        plt.savefig(save+'curves.png')
    plt.cls()
    return

def show_prog(epoch, train_loss, val_loss, time_elapsed):
    '''
    Prints current epoch's losses and runtime
    '''
    print('E %03d --- RUNTIME: %ds' % (epoch+1, time_elapsed))
    print('TRAIN  |  loss: %.3f' % train_loss)
    print('VALID  |  loss: %.3f' % val_loss)
    
def save_prog(model, model_path, train_losses, val_losses, epoch, save_rate, best_loss):
    '''
    Saves losses to model folder
    Saves model state dict every save_rate epochs
    '''
    np.save(model_path +'train_losses', train_losses)
    np.save(model_path +'val_losses', val_losses)

    if (epoch+1) % save_rate ==0 or best_loss: #save model dict
        torch.save(model.state_dict(), model_path + 'model_epoch%s' % (epoch+1))