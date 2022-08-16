#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import src.geotraces.data as data

poc_data = data.poc_by_station()
param_uniformity = data.define_param_uniformity()
Lp_priors = data.get_Lp_priors(poc_data)
ez_depths = data.get_ez_depths(Lp_priors)
station_data = data.get_station_data(poc_data, param_uniformity, ez_depths)
stations = list(station_data.keys())

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(wspace=0.5)
ax.set_xlabel('EZ from $K_d$ (m)', fontsize=14)
ax.set_ylabel('zg (m)', fontsize=14)
scheme = plt.cm.viridis
norm = Normalize(min(stations), max(stations))
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)

for s in stations:  
    ax.scatter(ez_depths[s], station_data[s]['zg'], c=s, norm=norm, cmap=scheme)
ax.plot(np.linspace(20, 300), np.linspace(20, 300), c='k')

plt.savefig(f'../../results/geotraces/ez_zg_compare')
plt.close()