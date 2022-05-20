#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import colorbar
import pandas as pd
import pickle
import numpy as np
import os

import sys
import src.geotraces.data as data
from src.colors import *
from src.tools import get_layer_bounds

text = {'ws': ('Settling speed, 1-51 µm (m d$^{-1}$)',),
        'wl': ('Settling speed, > 51 µm (m d$^{-1}$)',),
        'B2p': ('Aggregation (m$^{3}$ mmol$^{-1}$ d$^{-1}$)',),
        'Bm2': ('Disaggregation (d$^{-1}$)',),
        'Bm1s': ('Remineralization, 1-51 µm (d$^{-1}$)',),
        'Bm1l': ('Remineralization, > 51 µm (d$^{-1}$)',),
        'sinkdiv_S': ('Sinking flux divergence, 1-51 µm (mmol m$^{-2}$ d$^{-1}$)', (-2, 2)),
        'sinkdiv_L': ('Sinking flux divergence, > 51 µm (mmol m$^{-2}$ d$^{-1}$)', (-2, 2)),
        'remin_S': ('Remineralization, 1-51 µm (mmol m$^{-2}$ d$^{-1}$)', (2, 5)),
        'remin_L': ('Remineralization, > 51 µm (mmol m$^{-2}$ d$^{-1}$)'),
        'aggregation': ('Aggregation (mmol m$^{-2}$ d$^{-1}$)', (0, 0.1)),
        'disaggregation': ('Disaggregation (mmol m$^{-2}$ d$^{-1}$)',),
        'production': ('Production (mmol m$^{-2}$ d$^{-1}$)',),
        'dvm': ('Diel vertical migration (mmol m$^{-2}$ d$^{-1}$)', (-2, 2)),
        'sink_S': ('Sinking flux, 1-51 µm (mmol m$^{-2}$ d$^{-1}$)', (0, 8)),
        'sink_L': ('Sinking flux, > 51 µm (mmol m$^{-2}$ d$^{-1}$)', (0, 8)),
        'sink_T': ('Sinking flux, >1 µm (mmol m$^{-2}$ d$^{-1}$)', (0, 8))}

poc_data = data.poc_by_station()
results_path = f'../../results/geotraces/'
pickled_files = [f for f in os.listdir(results_path) if f.endswith('NA.pkl')]

def plot_section(df, suffix, lims=False):

    if lims:
        lo, hi = text[suffix][1]
        postsuffix = '_wlims'
    else:
        lo, hi = min(df['estimate']), max(df['estimate'])
        postsuffix = ''

    if suffix in ('sinkdiv_S', 'sinkdiv_L', 'dvm'):
        if suffix == 'dvm':
            scheme = plt.cm.PRGn_r
        else:
            scheme = plt.cm.PRGn
        normfac = TwoSlopeNorm(0, lo, hi)
    else:
        scheme = plt.cm.viridis
        normfac = Normalize(lo, hi)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.invert_xaxis()
    ax.invert_yaxis()
    axcb = colorbar.make_axes(ax)[0]
    colorbar.ColorbarBase(axcb, cmap=scheme, norm=normfac)
    ax.scatter(df['latitude'], df['depth'], c=df['estimate'], norm=normfac, cmap=scheme, zorder=10)

    for i, lat in enumerate(latitudes):
        ax.text(lat, -30, stations[i], ha='center')
        ax.scatter(lat*np.ones(len(grids[i])), grids[i], c=black, zorder=1, s=1)
    ax.plot(latitudes, mlds, c=black, zorder=1, ls='--')
    ax.plot(latitudes, zgs, c=black, zorder=1)
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Latitude (°N)', fontsize=14)
    fig.suptitle(text[suffix][0], fontsize=16)
    plt.savefig(f'../../results/geotraces/section_{suffix}{postsuffix}.pdf')
    plt.close()

def get_path(filename):

    station, priors_from = filename.split('_')
    s = station[3:]
    pf = priors_from[:2]
    file_path = os.path.join(results_path, filename)

    return file_path, s, pf

def get_station_latitude(station):

    station_poc = poc_data[int(station)]
    latitude = round(station_poc.iloc[0]['latitude'], 1)

    return latitude

###################
#SOME METADATA
###################
latitudes = []
mlds = []
zgs = []
stations = []
grids = []

for f in pickled_files:

    file_path, s, _ = get_path(f)

    with open(file_path, 'rb') as file:
        unpickled = pickle.load(file)
        _, _, _, grid, zg, mld, *_ = unpickled
    
    latitudes.append(get_station_latitude(s))
    mlds.append(mld)
    zgs.append(zg)
    stations.append(s)
    grids.append(grid)

stations = [s for _, s in sorted(zip(latitudes, stations))]
mlds = [mld for _, mld in sorted(zip(latitudes, mlds))]
zgs = [zg for _, zg in sorted(zip(latitudes, zgs))]
grids = [grid for _, grid in sorted(zip(latitudes, grids))]
latitudes.sort()

###################
#PARAMS
###################
file_path = os.path.join(results_path, pickled_files[0])
with open(file_path, 'rb') as file:
    unpickled = pickle.load(file)
    _, params, *_ = unpickled
dv_params = [p for p in params if params[p]['dv']]

for p in dv_params:

    df = pd.DataFrame(columns=('depth', 'latitude', 'estimate'))

    for f in pickled_files:

        file_path, s, pf = get_path(f)
        latitude = get_station_latitude(s)

        with open(file_path, 'rb') as file:
            unpickled = pickle.load(file)
            _, params, _, grid, _, _, layers, *_ = unpickled

        for l in layers:
            if 'w' in p:
                depth = grid[l]
            else:
                depth = np.mean(get_layer_bounds(l, grid))
            df.loc[df.shape[0]] = [depth, latitude, params[p]['posterior'][l]]
    
    plot_section(df, p)

# ###################
# #INTEGRATED FLUXES
# ###################
# file_path = os.path.join(results_path, pickled_files[0])
# with open(file_path, 'rb') as file:
#     unpickled = pickle.load(file)
#     _, _, _, _, int_fluxes, *_ = unpickled

# for i in int_fluxes:

#     df = pd.DataFrame(columns=('depth', 'latitude', 'estimate'))

#     for f in pickled_files:

#         file_path, s, pf = get_path(f)
#         latitude = get_station_latitude(s)

#         with open(file_path, 'rb') as file:
#             unpickled = pickle.load(file)
#             _, _, _, _, int_fluxes, _, _, _, grid, zg, _, layers, *_ = unpickled

#         for l in layers:
#             zi, zim1 = get_layer_bounds(l, grid)
#             depth = np.mean((zi, zim1))
#             if zi <= zg and i == 'dvm':
#                 df.loc[df.shape[0]] = [depth, latitude, -int_fluxes[i][l][0]]
#             else:
#                 df.loc[df.shape[0]] = [depth, latitude, int_fluxes[i][l][0]]

#     plot_section(df, i)

#     if i in ('remin_S', 'aggregation', 'sinkdiv_S', 'sinkdiv_L', 'dvm'):
#         plot_section(df, i, lims=True)

# ###################
# #SINKING FLUXES
# ###################
# file_path = os.path.join(results_path, pickled_files[0])
# with open(file_path, 'rb') as file:
#     unpickled = pickle.load(file)
#     _, _, _, _, _, sink_fluxes, *_ = unpickled

# for sf in sink_fluxes:

#     df = pd.DataFrame(columns=('depth', 'latitude', 'estimate'))

#     for f in pickled_files:

#         file_path, s, pf = get_path(f)
#         latitude = get_station_latitude(s)

#         with open(file_path, 'rb') as file:
#             unpickled = pickle.load(file)
#             _, _, _, _, _, sink_fluxes, _, _, grid, zg, _, layers, *_ = unpickled

#         for l in layers:
#             depth = grid[l]
#             df.loc[df.shape[0]] = [depth, latitude, sink_fluxes[sf][l][0]]

#     plot_section(df, f'sink_{sf}')
#     plot_section(df, f'sink_{sf}', lims=True)
#     # df.to_csv(f'../../results/geotraces/sink_{sf}.csv', index=False)
