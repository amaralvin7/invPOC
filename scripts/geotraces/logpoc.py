#!/usr/bin/env python3
# plot log(POC) profiles (after data have been cleaned)

import matplotlib.pyplot as plt
import numpy as np

import src.geotraces.data as data
from src.colors import *


poc_data = data.load_poc_data()

for s in poc_data['station'].unique():

    station_poc = data.get_station_poc(poc_data, s)

    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)

    ax1.set_xlabel('log($P_{S}$) (mmol m$^{-3}$)', fontsize=14)
    ax2.set_xlabel('log($P_{L}$) (mmol m$^{-3}$)', fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    ax1.scatter(np.log10(station_poc['POCS']), station_poc['depth'], c=blue)
    ax2.scatter(np.log10(station_poc['POCL']), station_poc['depth'], c=blue)

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=610)
        ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(labelleft=False)

    plt.savefig(f'../../results/geotraces/logpoc_stn{int(s)}')
    plt.close()
    
