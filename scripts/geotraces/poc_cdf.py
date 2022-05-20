import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
import numpy as np

from src.geotraces.data import poc_by_station

poc = poc_by_station()

data = pd.DataFrame(columns=['depth', 'POCS'])

for s, df in poc.items():
    data = pd.concat([data, df], join='inner', ignore_index=True)

data.sort_values('POCS', inplace=True)
data.reset_index(inplace=True, drop=True)

y = data['POCS']
z = data['depth']
x = [sum(j <= i for j in y)/len(y) for i in y]

fig, ax = plt.subplots(1, 1)
ax.set_ylabel('$P_S$ (mmol m$^{-3}$)', fontsize=14)
axcb = colorbar.make_axes(ax)[0]
normfac = colors.Normalize(z.min(), z.max())
scheme = plt.cm.viridis_r
colorbar.ColorbarBase(axcb, cmap=scheme, norm=normfac)

ax.scatter(x, y, c=z, norm=normfac, cmap=scheme)
ax.set_yscale('log')
# ax.set_ylabel(title, fontsize=16)
ax.set_xticks(np.arange(0, 1.1, 0.1))

mean = np.mean(y)
median = np.median(y)

ax.axhline(mean, ls='--', c='k')
ax.axhline(median, ls='-', c='k')

print(median)
print(mean)

plt.show()