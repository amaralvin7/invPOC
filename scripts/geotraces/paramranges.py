from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import *

df = pd.read_excel('../../data/paramestimates.xlsx', sheet_name=None)
bm1s = df['Bm1']
bm2 = df['Bm2']
b2 = df['B2']

markers = {'MNWA': ('s', radish),
           'MNA': ('d', green),
           'BRIG': ('*', orange),
           'CEP': ('^', vermillion),
           'CSP': ('v', blue)}

leg_elements = [
    Line2D([0], [0], marker='s', mec=black, c=white,
            label='Murnane et al. (1994)\nNWAO',
            markerfacecolor=radish, ms=9),
    Line2D([0], [0], marker='d', mec=black, c=white,
            label='Murnane et al. (1996)\nNABE',
            markerfacecolor=green, ms=9),
    Line2D([0], [0], marker='*', mec=black, c=white,
            label='Briggs et al. (2020)\nSNAO, SO',
            markerfacecolor=orange, ms=12),
    Line2D([0], [0], marker='^', mec=black, c=white,
            label='Clegg et al. (1991)\nEPO',
            markerfacecolor=vermillion, ms=9),
    Line2D([0], [0], marker='v', mec=black, c=white,
            label='Clegg et al. (1991)\nStation P',
            markerfacecolor=blue, ms=9)]

def cdf(df, title, name):
    
    sorted = df.sort_values('val').copy()
    sorted.reset_index(inplace=True)
    y = sorted['val']
    x = np.arange(len(y))
    m = sorted['id']
    
    fig, ax = plt.subplots(1,1,tight_layout=True)
    for i, _ in enumerate(y):
        marker, color = markers[m[i]]
        ax.scatter(x[i], y[i], marker=marker, c=color)
        ax.set_yscale('log')
    ax.set_ylabel(title, fontsize=16)
    ax.axes.xaxis.set_visible(False)

    ax.legend(handles=leg_elements, fontsize=10, ncol=1, frameon=False)
    
    fig.savefig(f'../../results/geotraces/ranges_{name}')
    plt.close()

cdf(bm1s, '$\\beta_{-1}$ (d$^{-1}$)', 'bm1')
cdf(bm2, '$\\beta_{-2}$ (d$^{-1}$)', 'bm2')
cdf(b2, '$\\beta_{2}$ (d$^{-1}$)', 'b2')
