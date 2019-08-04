import random
import matplotlib
import tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

def update(val):
    val = []
    for i in range(40):
        val.append(s_time[i].val)
        print(val[i])
    #TODO: use PCA to create a new face
    #TODO: plot the image using ax.imshow(image)
    #An example of plotting an image is shown below
    image = torch.randn(64, 64, 3)
    ax.imshow(image)

matplotlib.use('TkAgg')
root = Tk.Tk()
root.wm_title("VAE face generation")
fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

image = torch.randn(64, 64, 3)

ax=fig.add_subplot(122)
ax.imshow(image)

s_time = []

for i in range (40):
    ax_time = fig.add_axes([0.05, 0.1+0.02*i, 0.4, 0.01])
    s_time.append(Slider(ax_time, str(i), 0, 30, valinit=0))
    s_time[i].on_changed(update)



Tk.mainloop()
