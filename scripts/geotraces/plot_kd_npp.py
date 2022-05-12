import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import pandas as pd

from src.geotraces.data import extract_nc_data
from src.colors import *

npp_data = extract_nc_data('cbpm')
kd_data = extract_nc_data('modis')

fig = plt.figure()
plt.subplots_adjust(left = 0.15, right=0.85)
host1 = host_subplot(111, axes_class=AA.Axes, figure=fig)
par1 = host1.twinx()
par1.axis['right'].toggle(all=True)

host1.plot(npp_data['nc_lat'], npp_data['npp'], c=green, marker='o', ls='none')
par1.plot(kd_data['nc_lat'], kd_data['Kd'], c=orange, marker='o', ls='none')

host1.invert_xaxis()
par1.invert_xaxis()

host1.axis['left'].label.set_color(green)
par1.axis['right'].label.set_color(orange)

host1.axis['bottom', 'left'].label.set_fontsize(14)
par1.axis['right'].label.set_fontsize(14)

host1.axis['bottom','left'].major_ticklabels.set_fontsize(11)
par1.axis['right'].major_ticklabels.set_fontsize(11)

host1.axis['bottom', 'left'].major_ticks.set_ticksize(6)
host1.axis['bottom', 'left'].major_ticks.set_tick_out('out')
par1.axis['right'].major_ticks.set_ticksize(6)
par1.axis['right'].major_ticks.set_tick_out('out')

host1.set_xlabel('Latitude (°N)')
host1.set_ylabel('NPP (mg m$^{-2}$ d$^{-1}$)')
par1.set_ylabel('$K_d$ (m$^{-1}$)')

plt.show()