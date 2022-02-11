#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt
import numpy as np

import src.geotraces.data as data

start_time = time.time()

poc_data = data.load_poc_data()

lp = data.get_lp_by_station(poc_data)

lp_vals = []
fig, ax = plt.subplots()
ax.set_xlabel('Latitude')
ax.set_ylabel('Lp (m)')
ax.invert_xaxis()

for s in lp:
    lp_vals.append(lp[s][1])
    ax.scatter(lp[s][0], lp[s][1], c='b')

lp_mean = np.mean(lp_vals)
lp_sd = np.std(lp_vals, ddof=1)
fig.suptitle(f'mean: {lp_mean:.2f}, SD: {lp_sd:.2f}')

plt.show()