#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colorbar
import pandas as pd
import pickle
import numpy as np
import os

import sys
import src.geotraces.data as data
from src.colors import *

plt.ion()

poc_data = data.load_poc_data()

results_path = f'../../results/geotraces/'
pickled_files = [f for f in os.listdir(results_path) if f.endswith('NA.pkl')]

file_path = os.path.join(results_path, pickled_files[0])
with open(file_path, 'rb') as file:
    unpickled = pickle.load(file)
    _, params, *_ = unpickled
dv_params = [p for p in params if params[p]['dv']]

    
for p in dv_params:

    df = pd.DataFrame(columns=('depth', 'latitude', 'estimate'))
    latitudes = []
    mlds = []
    ppzs = []
    stations = []

    for f in pickled_files:

        station, priors_from = f.split('_')
        s = station[3:]
        pf = priors_from[:2]
        file_path = os.path.join(results_path, f)

        with open(file_path, 'rb') as file:
            unpickled = pickle.load(file)
            _, params, residuals, grid, ppz, mld, layers, _, _ = unpickled

        station_poc = poc_data.loc[poc_data['station'] == int(s)]
        latitude = round(station_poc.iloc[0]['latitude'], 1)
        
        latitudes.append(latitude)
        mlds.append(mld)
        ppzs.append(ppz)
        stations.append(s)

        for l in layers:
            if 'w' in p:
                depth = grid[l]
            else:
                zi = grid[l]
                zim1 = grid[grid.index(zi) - 1] if l > 0 else 0
                depth = np.mean((zi, zim1))
            df.loc[df.shape[0]] = [depth, latitude, params[p]['posterior'][l]]

    scheme = plt.cm.viridis
    normfac = Normalize(min(df['estimate']), max(df['estimate']))

    fig, ax = plt.subplots(figsize=(10,4))
    ax.invert_xaxis()
    ax.invert_yaxis()
    axcb = colorbar.make_axes(ax)[0]
    cbar = colorbar.ColorbarBase(axcb, cmap=scheme, norm=normfac)
    ax.scatter(df['latitude'], df['depth'], c=df['estimate'], norm=normfac, cmap=scheme, zorder=10)
    
    stations = [s for _, s in sorted(zip(latitudes, stations))]
    mlds = [mld for _, mld in sorted(zip(latitudes, mlds))]
    ppzs = [ppz for _, ppz in sorted(zip(latitudes, ppzs))]
    latitudes.sort()

    for i, lat in enumerate(latitudes):
        ax.text(lat, -30, stations[i], ha='center')
    ax.plot(latitudes, mlds, c=black, zorder=1)
    ax.plot(latitudes, ppzs, c=black, zorder=1)

    plt.savefig(f'../../results/geotraces/section_{p}')
    plt.close()
