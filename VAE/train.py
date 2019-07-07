import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from VAE import Encoder, Decoder, VAE
from data_loader import get_data_loader


######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=30, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=256, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __VAE__ ########
parser.add_argument('-embedding', action='store', default=32, type=int,
                    help='static embedding size')
parser.add_argument('-layers', nargs='+', type=int, default=[16, 16],
                    help='layer depths')
parser.add_argument('-fcl', action='store', default=16, type=int,
                    help='fully connected size')
parser.add_argument('-attention', action='store', default=1, type=int,
                    help='attention head count')
parser.add_argument('-drop', action='store', default=0.25, type=float,
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


model_name = '_'.join(['b_'+str(args.batch),'d_'+str(args.drop), str(args.layers),
                          'fcl_'+str(args.fcl), 'a_'+str(args.attention),
                          'b1_'+str(args.b1), 'b2_'+str(args.b2),
                          'lr_'+str(args.lr), 'g_'+str(args.gamma)])
                       

# Create target Directory if don't exist
#if not os.path.exists(save_path+model_name):
#    os.mkdir(save_path+model_name)
#elif not args.nosave:
#    print('WARNING: overwriting existing directory:', model_name)
#save_path = save_path + model_name + '/'
#if args.nosave: print('WARNING: MODEL AND DATA ARE NOT BEING SAVED')

######## __GPU_SETUP__ ########
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

if __name__ == '__main__':
    epochs = args.epochs
    batch_size = args.batch
    save = not args.nosave


    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    train_loader = get_data_loader(batch_size=4)

    model = VAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start = time.time()
    for epoch in range(2):
        loss = 0
        for batch,_ in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss += criterion(output, batch)
        print('time:', time.time()-start)
        print(loss/len(train_loader))
    