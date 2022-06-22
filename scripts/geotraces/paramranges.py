from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import *

df = pd.read_excel('../../../geotraces/paramcompilation.xlsx', sheet_name=None)
bm1s = df['Bm1s']
bm1l = df['Bm1l']
bm2 = df['Bm2']
b2 = df['B2']
wl = df['wl']
ws = df['ws']

markers = {'MNWA': ('s', radish, 60),
           'MNA': ('d', green, 60),
           'BRIG': ('*', orange, 60),
           'CEP': ('^', vermillion, 60),
           'CSP': ('v', blue, 60),
           'MWAP1': ('h', sky, 60),
           'MWAP2': ('D', black, 60),
           'MBATS': ('X', 'r', 60),
           'MSARG': ('p', 'g', 60),
           'XIANG': ('o', 'grey', 4),
           'A16': ('s', radish, 60),
           'L17': ('d', green, 60),
           'G20': ('*', orange, 60),
           'K76': ('^', vermillion, 60),
           'K81': ('v', blue, 60),
           'S95': ('h', sky, 60),
           'RVDL93': ('D', black, 60),
           'V08': ('X', 'r', 60),
           'ENA': ('h', sky, 60),
           'ENP': ('D', black, 60),
           'SBB': ('X', 'r', 60)}

# def plot_percentage(percent, x, y, ax, name):
    
#     i = min(range(len(x)), key=lambda i:abs(x[i] - percent))
#     ax.hlines(y[i], 0, x[i], color='k', ls='--')
#     ax.vlines(x[i], 0, y[i], color='k', ls='--')
#     print(f'{percent}, {name}: {y[i]:.6f}')
        
def cdf(df, title, name):
    
    sorted = df.sort_values('val').copy()
    sorted.reset_index(inplace=True)
    y = sorted['val']
    x = [sum(j <= i for j in y)/len(y) for i in y]
    m = sorted['id']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(12,4))
    for i, _ in enumerate(y):
        marker, color, size = markers[m[i]]
        ax1.scatter(x[i], y[i], marker=marker, c=color, label=m[i], s=size)
        ax1.set_yscale('log')
    ax1.set_ylabel(title, fontsize=16)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))

    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax1.legend(*zip(*unique), fontsize=10, ncol=1, frameon=False, loc='lower right')
    ax2.hist(y, bins=30)
    
    fig.savefig(f'../../results/geotraces/ranges/ranges_{name}')
    plt.close()

cdf(bm1l, '$\\beta_{-1,L}$ (d$^{-1}$)', 'bm1l')
cdf(bm1s, '$\\beta_{-1,S}$ (d$^{-1}$)', 'bm1s')
cdf(bm2, '$\\beta_{-2}$ (d$^{-1}$)', 'bm2')
cdf(b2, '$\\beta_{2}$ (d$^{-1}$)', 'b2')
cdf(wl, '$w_L$ (m d$^{-1}$)', 'wl')
cdf(ws, '$w_S$ (m d$^{-1}$)', 'ws')
