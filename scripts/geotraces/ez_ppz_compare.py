#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors

import src.geotraces.data as data

poc_data = data.load_poc_data()
ppz_data = data.load_ppz_data()
Lp_priors = data.get_Lp_priors(poc_data)

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(wspace=0.5)
ax.set_xlabel('EZ from $K_d$ (m)', fontsize=14)
ax.set_ylabel('PPZ (m)', fontsize=14)
axcb = colorbar.make_axes(ax)[0]
normfac = colors.Normalize(1, 39)
scheme = plt.cm.viridis
colorbar.ColorbarBase(axcb, cmap=scheme, norm=normfac)


for s in Lp_priors:
    ez = Lp_priors[s]*np.log(1000)
    ax.scatter(ez, ppz_data[s], c=s, norm=normfac, cmap=scheme)
ax.plot(np.linspace(40, 300), np.linspace(40, 300), c='k')

plt.savefig(f'../../results/geotraces/ez_ppz_compare')
plt.close()