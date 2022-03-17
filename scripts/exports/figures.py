import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import sys
from src.colors import *

grid = (30, 50, 100, 150, 200, 330, 500)
thick = np.diff((0,) + grid)

with open('../../results/exports/NA_0.5_0.5.pkl', 'rb') as pickled:
            NA_results = pickle.load(pickled)
with open('../../results/exports/SP_0.5_0.5.pkl', 'rb') as pickled:
            SP_results = pickle.load(pickled)

############################
#FIGURE 10, sinking fluxes
############################
flux_dict = {'NA': NA_results['sink_fluxes'], 'SP': SP_results['sink_fluxes']}

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

############################
#FIGURE 12, DVM comparisons
############################
flux_dict = {'NA': NA_results['int_fluxes'], 'SP': SP_results['int_fluxes']}

fig = plt.figure()
fig.text(0.025, 0.5, 'Depth (m)', fontsize=14, ha='center',
            va='center', rotation='vertical')
fig.subplots_adjust(wspace=0.3, hspace=0.1)

for inversion in flux_dict:
    if inversion == 'NA':
        i = 0
    else:
        i = 1

    hostL = host_subplot(2, 2, 1+2*i, axes_class=AA.Axes, figure=fig)
    parL = hostL.twiny()
    parL.axis['top'].toggle(all=True)
    hostL.set_xlim(0, 0.3)
    parL.set_xlim(0, 0.3)

    hostR = host_subplot(2, 2, 2*(1+i), axes_class=AA.Axes, figure=fig)
    hostR.yaxis.set_ticklabels([])
    parR = hostR.twiny()
    parR.axis['top'].toggle(all=True)
    hostR.set_xlim(-0.02, 0.02)
    parR.set_xlim(0.02, -0.02)
    hostR.axvline(c=black, alpha=0.3)

    if inversion == 'SP':
        hostL.set_xlabel('Ingestion flux (mmol m$^{-3}$ d$^{-1}$)')
        hostR.set_xlabel('Egestion flux (mmol m$^{-3}$ d$^{-1}$)')
        hostR.text(1.05, 0.2, 'SP inversion' , fontsize=14,
                    rotation=270, transform=hostR.transAxes)
        parR.xaxis.set_ticklabels([])
        parL.xaxis.set_ticklabels([])
    else:
        parL.set_xlabel('$P_S$ remin. flux (mmol m$^{-3}$ d$^{-1}$)')
        parR.set_xlabel('$P_L$ SFD (mmol m$^{-3}$ d$^{-1}$)')
        hostR.xaxis.set_ticklabels([])
        hostL.xaxis.set_ticklabels([])
        hostR.text(1.05, 0.2, 'NA inversion' , fontsize=14,
                    rotation=270, transform=hostR.transAxes)

    for host, par in ((hostL, parL), (hostR, parR)):
        host.axis['right'].toggle(all=False)
        host.axis['left', 'top', 'bottom'].major_ticks.set_tick_out(
            'out')
        par.axis['left', 'top', 'bottom'].major_ticks.set_tick_out(
            'out')
        host.axis['bottom'].label.set_color(blue)
        par.axis['top'].label.set_color(orange)
        host.axis['left'].label.set_fontsize(14)
        host.axis['bottom'].label.set_fontsize(12)
        host.axis['bottom', 'left'].major_ticklabels.set_size(12)
        par.axis['top'].label.set_fontsize(12)
        par.axis['top'].major_ticklabels.set_size(12)

    for j, _ in enumerate(grid):
        if j < 3:
            host = hostL
            par = parL
            par_flux = 'remin_S'
        else:
            host = hostR
            par = parR
            par_flux = 'sinkdiv_L'
        if j == 0:
            depths = 0, grid[j]
        else:
            depths = grid[j-1], grid[j]

        host.scatter(
            flux_dict[inversion]['dvm'][j][0]/thick[j], np.mean(depths),
            marker='o', c=blue, s=14, zorder=3, lw=0.7)
        host.fill_betweenx(
            depths,
            (flux_dict[inversion]['dvm'][j][0] - flux_dict[inversion]['dvm'][j][1])/thick[j],
            (flux_dict[inversion]['dvm'][j][0] + flux_dict[inversion]['dvm'][j][1])/thick[j],
            color=blue, alpha=0.25)
        par.scatter(
            flux_dict[inversion][par_flux][j][0]/thick[j], np.mean(depths),
            marker='o', c=orange, s=14, zorder=3, lw=0.7)
        par.fill_betweenx(
            depths,
            (flux_dict[inversion][par_flux][j][0] - flux_dict[inversion][par_flux][j][1])/thick[j],
            (flux_dict[inversion][par_flux][j][0] + flux_dict[inversion][par_flux][j][1])/thick[j],
            color=orange, alpha=0.25)
    for ax in (hostL, hostR):
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=510)
        ax.axhline(100, ls=':', c=black)

fig.savefig('../../results/exports/figures/Figure12.pdf')
plt.close()