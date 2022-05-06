from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import *

df = pd.read_excel('../../data/paramcompilation.xlsx', sheet_name=None)
bm1s = df['Bm1']
bm2 = df['Bm2']
b2 = df['B2']
wl = df['wl']

markers = {'MNWA': ('s', radish),
           'MNA': ('d', green),
           'BRIG': ('*', orange),
           'CEP': ('^', vermillion),
           'CSP': ('v', blue),
           'MWAP1': ('h', sky),
           'MWAP2': ('D', black),
           'MBATS': ('X', 'r'),
           'MSARG': ('p', 'g')}

def plot_percentage(percent, x, y, ax, name):
    
    i = min(range(len(x)), key=lambda i:abs(x[i] - percent))
    ax.hlines(y[i], 0, x[i], color='k', ls='--')
    ax.vlines(x[i], 0, y[i], color='k', ls='--')
    print(f'{percent}, {name}: {y[i]:.6f}')
        
def cdf(df, title, name):
    
    sorted = df.sort_values('val').copy()
    sorted.reset_index(inplace=True)
    y = sorted['val']
    x = [sum(j <= i for j in y)/len(y) for i in y]
    m = sorted['id']
    
    fig, ax = plt.subplots(1,1,tight_layout=True)
    for i, _ in enumerate(y):
        marker, color = markers[m[i]]
        ax.scatter(x[i], y[i], marker=marker, c=color, label=m[i])
        ax.set_yscale('log')
    ax.set_ylabel(title, fontsize=16)
    ax.set_xticks(np.arange(0, 1.1, 0.1))

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=10, ncol=1, frameon=False, loc='lower right')
    
    plot_percentage(0.5, x, y, ax, name)
    plot_percentage(0.75, x, y, ax, name)
    plot_percentage(0.25, x, y, ax, name)
    
    fig.savefig(f'../../results/geotraces/ranges_{name}')
    plt.close()

cdf(bm1s, '$\\beta_{-1}$ (d$^{-1}$)', 'bm1')
cdf(bm2, '$\\beta_{-2}$ (d$^{-1}$)', 'bm2')
cdf(b2, '$\\beta_{2}$ (d$^{-1}$)', 'b2')
cdf(wl, '$w_L$ (m d$^{-1}$)', 'wl')
