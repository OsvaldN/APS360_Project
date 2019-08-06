import random
import matplotlib
import pickle
import numpy as np
import tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from VAE_GAN import Encoder, Decoder, VAE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch

'''
    This function is called everytime the slider values are changed
    The all the slider values are stored in a list called "val", and
    a face will be generated based on "val" and plotted on the canvas    
'''
def get_model():
    '''
    TODO: parameterize all of this
    '''
    latent = 500
    dilation=20
    folder = '/home/osvald/Projects/APS360/APS360_Project/VAE/GAN_models/l_200_df_16_kld_0.5_b1_0.5_b2_0.999_lr_0.001_g_0.99_db_2_gw_1.5'
    folder = "C:\\Users\\osval\\Documents\\School\\APS360\\APS360_Project\\VAE\\VAE_models\\l_500_df_20_kld_0.01_b1_0.9_b2_0.999_lr_0.001_g_0.99"
    state = '\\model_epoch150'
    model = VAE(d_factor=dilation, latent_variable_size=latent, cuda=False, activation='SELU').to('cpu')
    model.load_state_dict(torch.load(folder + state, map_location='cpu'))
    model.eval()
    sc = pickle.load(open(folder+'\\std_scaler.p', 'rb'))
    pca_components = pickle.load(open(folder+'\\components.p', 'rb'))
    pca_mean = pickle.load(open(folder+'\\mean.p', 'rb'))
    return model, sc, pca_components, pca_mean

model, sc, pca_components, pca_mean = get_model()

def pca_to_img(val, model=model, sc=sc, components=pca_components, mean=pca_mean):
    '''
    Transforms Principle components to latent distrribution
    '''
    z = np.dot(val, components) + mean
    #z = sc.inverse_transform(z)
    print(z.shape)
    img = model.decode(torch.Tensor(z))
    return(img)

def update(val):
    val = []
    for i in range(30):
        val.append(s_time[i].val)
        print(val[i])

    with torch.no_grad():
        image = pca_to_img(val).squeeze().permute(1,2,0)

    ax.imshow(image)



matplotlib.use('TkAgg')
root = Tk.Tk()
root.wm_title("VAE face generation")
fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


with torch.no_grad():
    image = model.decode(torch.zeros(500)).squeeze().permute(1,2,0)

ax=fig.add_subplot(122)
ax.imshow(image)

s_time = []

for i in range (30):
    ax_time = fig.add_axes([0.05, 0.1+0.03*i, 0.4, 0.02])
    s_time.append(Slider(ax_time, str(i), -10, 10, valinit=0))
    s_time[i].on_changed(update)

Tk.mainloop()

