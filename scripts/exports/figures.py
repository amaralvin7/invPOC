import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from src.colors import *

grid = (30, 50, 100, 150, 200, 330, 500)

############################
#FIGURE 10
############################
with open('../../results/exports/NA_0.5_0.5.pkl', 'rb') as pickled:
            _, _, _, _, _, NA_fluxes, *_ = pickle.load(pickled)
with open('../../results/exports/SP_0.5_0.5.pkl', 'rb') as pickled:
            _, _, _, _, _, SP_fluxes, *_ = pickle.load(pickled)
flux_dict = {'NA': NA_fluxes, 'SP': SP_fluxes}

datapath = '../../data/exports.xlsx'
th_fluxes = pd.read_excel(datapath, sheet_name='POC_fluxes_thorium')
th_depths = th_fluxes['depth']
th_flux = th_fluxes['flux']
th_flux_u = th_fluxes['flux_u']
st_fluxes = pd.read_excel(datapath, sheet_name='POC_fluxes_traps')
st_depths = st_fluxes['depth']
st_flux = st_fluxes['flux']
st_flux_u = st_fluxes['flux_u']

fig, (na_axs, sp_axs) = plt.subplots(2, 2, figsize=(6, 6))
fig.subplots_adjust(left=0.16, right=0.92, top=0.95, bottom=0.11, wspace=0.15, hspace=0.1)
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center', rotation='vertical')
fig.text(0.54, 0.03, 'POC flux (mmol m$^{-2}$ d$^{-1}$)',fontsize=14, ha='center', va='center')

for inversion in flux_dict:
    if inversion == 'NA':
        axs = na_axs
        [ax.axes.xaxis.set_ticklabels([]) for ax in axs]
    else:
        axs = sp_axs
    ylabel = f'{inversion} inversion'
    for ax in axs:
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=530)
        ax.axhline(100, ls=':', c=black, zorder=1)
        ax.tick_params(axis='both', which='major', labelsize=12)

    axs[1].set_ylabel(ylabel, fontsize=14, rotation=270,
                        labelpad=20)
    axs[1].yaxis.set_label_position('right')

    axs[0].errorbar(
        [x[0] for x in flux_dict[inversion]['S']],
        np.array(grid) + 2,
        fmt='o', xerr=[x[1] for x in flux_dict[inversion]['S']],
        ecolor=blue, c=blue, capsize=4,
        label='$w_SP_S$', fillstyle='none',
        elinewidth=1.5, capthick=1.5)

    axs[0].errorbar(
        [x[0] for x in flux_dict[inversion]['L']],
        np.array(grid) - 2,
        fmt='o', xerr=[x[1] for x in flux_dict[inversion]['L']],
        ecolor=orange, c=orange, capsize=4,
        label='$w_LP_L$', fillstyle='none',
        elinewidth=1.5, capthick=1.5)

    axs[1].tick_params(labelleft=False)
    axs[1].errorbar(
        [x[0] for x in flux_dict[inversion]['T']], grid, fmt='o',
        xerr=[x[1] for x in flux_dict[inversion]['T']],
        ecolor=vermillion, c=vermillion, capsize=4, zorder=3,
        label='$w_TP_T$', elinewidth=1.5, capthick=1.5,
        fillstyle='none')
    axs[1].errorbar(
        th_flux, th_depths + 4, fmt='^', xerr=th_flux_u,
        ecolor=green, c=green, capsize=4,
        label='$^{234}$Th-based', elinewidth=1.5, capthick=1.5)
    axs[1].errorbar(
        st_flux, st_depths - 4, fmt='d', xerr=st_flux_u, c=black,
        ecolor=black, capsize=4, label='Sed. Traps',
        elinewidth=1.5, capthick=1.5)

    axs[0].set_xlim([0, 6])
    axs[1].set_xlim([0, 10])

    if ylabel == 'SP inversion':
        axs[0].legend(loc='lower right', fontsize=12,
                        handletextpad=0.01)
        axs[1].legend(loc='lower right', fontsize=12,
                        handletextpad=0.01)

fig.savefig('../../results/exports/figures/Figure10.pdf')
plt.close()