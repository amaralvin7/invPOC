"""Plot all code-generated figures in Amaral et al., 2022."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist import Axes

from src.colors import *
from src.constants import DPY, MMC
from src.exports.constants import *
from src.exports.data import load_data

all_data = load_data()

with open('../../results/exports/NA_0.5_0.5.pkl', 'rb') as pickled:
    NA_results = pickle.load(pickled)
with open('../../results/exports/SP_0.5_0.5.pkl', 'rb') as pickled:
    SP_results = pickle.load(pickled)

results_dict = {'NA': NA_results, 'SP': SP_results}
Po_prior = NA_results['params']['Po']['prior']

######################################################
# FIGURE 1, hydrography data
######################################################
hydro_df = all_data['hydrography']

fig = plt.figure(figsize=(7, 5))
host1 = host_subplot(121, axes_class=Axes, figure=fig)
host1.axis['right'].toggle(all=False)
plt.subplots_adjust(top=0.75, right=0.7, bottom=0.1)
par1 = host1.twiny()
par2 = host1.twiny()

par1.axis['top'].toggle(all=True)
offset = 40
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis['top'] = new_fixed_axis(loc='top', axes=par2, offset=(0, offset))
par2.axis['top'].toggle(all=True)

host1.set_ylim(0, 520)
host1.invert_yaxis()
host1.set_xlim(23.7, 27.2)
par1.set_xlim(3, 15)
par2.set_xlim(31.5, 34.5)

host1.set_ylabel('Depth (m)')
host1.set_xlabel('$\\sigma_{\\theta}$ (kg m$^{-3}$)')
par1.set_xlabel('Temperature (°C)')
par2.set_xlabel('Salinity')

host1.plot(hydro_df['sigtheta_mean'], hydro_df['depth'], c=orange, marker='o',
           ms=2, ls='none')
par1.plot(hydro_df['temp_mean'], hydro_df['depth'], c=vermillion, marker='o',
          ms=2, ls='none')
par2.plot(hydro_df['sal_mean'], hydro_df['depth'], c=blue, marker='o', ms=2,
          ls='none')

host1.fill_betweenx(hydro_df['depth'],
                    hydro_df['sigtheta_mean'] - hydro_df['sigtheta_sd'],
                    hydro_df['sigtheta_mean'] + hydro_df['sigtheta_sd'],
                    alpha=0.3, color=orange)
par1.fill_betweenx(hydro_df['depth'],
                   hydro_df['temp_mean'] - hydro_df['temp_sd'],
                   hydro_df['temp_mean'] + hydro_df['temp_sd'],
                   alpha=0.3, color=vermillion)
par2.fill_betweenx(hydro_df['depth'],
                   hydro_df['sal_mean'] - hydro_df['sal_sd'],
                   hydro_df['sal_mean'] + hydro_df['sal_sd'],
                   alpha=0.3, color=blue)

host1.axis['bottom'].label.set_color(orange)
par1.axis['top'].label.set_color(vermillion)
par2.axis['top'].label.set_color(blue)

host1.axis['bottom', 'left'].label.set_fontsize(14)
par1.axis['top'].label.set_fontsize(14)
par2.axis['top'].label.set_fontsize(14)

host1.axis['bottom', 'left'].major_ticklabels.set_fontsize(11)
par1.axis['top'].major_ticklabels.set_fontsize(11)
par2.axis['top'].major_ticklabels.set_fontsize(11)

host1.axis['bottom', 'left'].major_ticks.set_ticksize(6)
host1.axis['left'].major_ticks.set_tick_out('out')
par1.axis['top'].major_ticks.set_ticksize(6)
par2.axis['top'].major_ticks.set_ticksize(6)

host2 = host_subplot(122, axes_class=Axes, figure=fig)
host2.axis['right'].toggle(all=False)
plt.subplots_adjust(right=0.95)
par3 = host2.twiny()
par4 = host2.twiny()

par3.axis['top'].toggle(all=True)
offset = 40
new_fixed_axis = par4.get_grid_helper().new_fixed_axis
par4.axis['top'] = new_fixed_axis(loc='top', axes=par4, offset=(0, offset))
par4.axis['top'].toggle(all=True)

host2.set_ylim(0, 520)
host2.invert_yaxis()
host2.yaxis.set_ticklabels([])
par3.set_xlim(-0.02, 0.4)
par4.set_xlim(0, 320)

host2.set_xlabel('$c_P$ (m$^{-1}$)')
par3.set_xlabel('Chlorophyll (mg m$^{-3}$)')
par4.set_xlabel('Dissolved O$_2$ (µmol kg$^{-1}$)')

host2.plot(hydro_df['cp_mean'], hydro_df['depth'], c=radish, marker='o', ms=2,
           ls='none', zorder=10)
par3.plot(hydro_df['chl_mean'], hydro_df['depth'], c=green, marker='o', ms=2,
          ls='none')
par4.plot(hydro_df['o2_mean'], hydro_df['depth'], c=sky, marker='o', ms=2,
          ls='none')

host2.fill_betweenx(hydro_df['depth'],
                    hydro_df['cp_mean'] - hydro_df['cp_sd'],
                    hydro_df['cp_mean'] + hydro_df['cp_sd'],
                    alpha=0.3, color=radish, zorder=10)
par3.fill_betweenx(hydro_df['depth'],
                   hydro_df['chl_mean'] - hydro_df['chl_sd'],
                   hydro_df['chl_mean'] + hydro_df['chl_sd'],
                   alpha=0.3, color=green)
par4.fill_betweenx(hydro_df['depth'],
                   hydro_df['o2_mean'] - hydro_df['o2_sd'],
                   hydro_df['o2_mean'] + hydro_df['o2_sd'],
                   alpha=0.3, color=blue)

host2.axis['bottom'].label.set_color(radish)
par3.axis['top'].label.set_color(green)
par4.axis['top'].label.set_color(sky)

host2.axis['bottom', 'left'].label.set_fontsize(14)
par3.axis['top'].label.set_fontsize(14)
par4.axis['top'].label.set_fontsize(14)

host2.axis['bottom', 'left'].major_ticklabels.set_fontsize(11)
par3.axis['top'].major_ticklabels.set_fontsize(11)
par4.axis['top'].major_ticklabels.set_fontsize(11)

host2.axis['bottom', 'left'].major_ticks.set_ticksize(6)
host2.axis['left'].major_ticks.set_tick_out('out')
par3.axis['top'].major_ticks.set_ticksize(6)
par4.axis['top'].major_ticks.set_ticksize(6)

for ax in (host1, host2):
    for d in [30, 50, 100, 150, 200, 330, 500]:
        ax.axhline(d, c=black, ls=':', zorder=1)

fig.savefig('../../results/exports/figures/Figure1.pdf')
plt.close()

######################################################
# FIGURE 3, theoretical dvm
######################################################

zm = 500
zg = 100
a = 3
depths = np.arange(zg, zm)
co = np.pi / (2 * (zm - zg)) * a * zg
flux = co * (np.sin(np.pi * (depths - zg) / (zm - zg)))

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))
ax.set_ylim([0, zm])
ax.invert_yaxis()
ax.plot([0, 0], [0, zg], lw=2, c=black)
ax.plot(flux, depths, lw=2, c=black)
ax.set_yticks([0, zg, (zm + zg) / 2, zm])
ax.set_yticklabels(['0', '$z_g$', '$\\frac{z_g + z_m}{2}$', '$z_m$'],
                   fontsize=14)
ax.set_xticks([0, co])
ax.set_xticklabels(['0', '$E_{max}$'], fontsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('$E(z)$', fontsize=14, labelpad=10)
ax.set_ylabel('Depth, $z$', fontsize=14, labelpad=10)
ax.xaxis.set_label_position('top')

ax.axvline(0, lw=0.5, ls='--', c=black)
ax.axvline(co, lw=0.5, ls='--', c=black)
ax.axhline(0, lw=0.5, ls='--', c=black)
ax.axhline(zg, lw=0.5, ls='--', c=black)
ax.axhline(zm, lw=0.5, ls='--', c=black)
ax.axhline((zm + zg) / 2, lw=0.5, ls='--', c=black)

fig.savefig('../../results/exports/figures/Figure3.pdf')
plt.close()

######################################################
# FIGURE 4, POC data
######################################################

poc_data = all_data['POC']

fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
fig.subplots_adjust(wspace=0.5)

ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
ax1.set_ylabel('Depth (m)', fontsize=14)

ax1.errorbar(NA_results['tracers']['POCS']['prior'], GRID, fmt='^',
             xerr=NA_results['tracers']['POCS']['prior_e'], ecolor=blue,
             elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
ax1.scatter(poc_data['POCS'], poc_data['mod_depth'], c=blue, alpha=0.4)

ax2.errorbar(NA_results['tracers']['POCL']['prior'], GRID, fmt='^',
             xerr=NA_results['tracers']['POCL']['prior_e'], ecolor=blue,
             elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
ax2.scatter(poc_data['POCL'], poc_data['mod_depth'], c=blue, alpha=0.4)

ax1.set_xticks([0, 1, 2, 3])
ax1.set_xlim([0, 3.4])
ax2.set_xticks([0, 0.05, 0.1, 0.15])
ax2.set_xticklabels(['0', '0.05', '0.1', '0.15'])
ax2.tick_params(labelleft=False)

for ax in (ax1, ax2):
    ax.invert_yaxis()
    ax.set_ylim(top=0, bottom=530)
    ax.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(labelleft=False)

fig.savefig('../../results/exports/figures/Figure4.pdf')
plt.close()

######################################################
# FIGURE 5, POC estimates
######################################################

fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
fig.subplots_adjust(wspace=0.5)

ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
ax1.set_ylabel('Depth (m)', fontsize=14)

ngrid = [d - 5 for d in GRID]
sgrid = [d + 5 for d in GRID]

ax1.errorbar(NA_results['tracers']['POCS']['prior'], GRID, fmt='^',
             xerr=NA_results['tracers']['POCS']['prior_e'], ecolor=blue,
             elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
ax1.errorbar(NA_results['tracers']['POCS']['posterior'], ngrid, fmt='o',
             xerr=NA_results['tracers']['POCS']['posterior_e'], ecolor=orange,
             elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
             zorder=3, markeredgewidth=1)
ax1.errorbar(SP_results['tracers']['POCS']['posterior'], sgrid, fmt='s',
             xerr=SP_results['tracers']['POCS']['posterior_e'], ecolor=green,
             elinewidth=1, c=green, ms=8, capsize=5, fillstyle='none',
             zorder=3, markeredgewidth=1)

ax2.errorbar(NA_results['tracers']['POCL']['prior'], GRID, fmt='^',
             xerr=NA_results['tracers']['POCL']['prior_e'], ecolor=blue,
             elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full',
             label='Data')
ax2.errorbar(NA_results['tracers']['POCL']['posterior'], ngrid, fmt='o',
             xerr=NA_results['tracers']['POCL']['posterior_e'], ecolor=orange,
             elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
             zorder=3, markeredgewidth=1, label='NA')
ax2.errorbar(SP_results['tracers']['POCL']['posterior'], sgrid, fmt='s',
             xerr=SP_results['tracers']['POCL']['posterior_e'], ecolor=green,
             elinewidth=1, c=green, ms=8, capsize=5, fillstyle='none',
             zorder=3, markeredgewidth=1, label='SP')

ax1.set_xticks([0, 1, 2, 3])
ax1.set_xlim([0, 3.4])
ax2.set_xticks([0, 0.05, 0.1, 0.15])
ax2.set_xticklabels(['0', '0.05', '0.1', '0.15'])
ax2.tick_params(labelleft=False)

for ax in (ax1, ax2):
    ax.invert_yaxis()
    ax.set_ylim(top=0, bottom=530)
    ax.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(labelleft=False)
ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
          loc='lower right')

fig.savefig('../../results/exports/figures/Figure5.pdf')
plt.close()

######################################################
# FIGURE 6, non-uniform params
######################################################

param_text = {'ws': ('$w_S$', 'm d$^{-1}$'), 'wl': ('$w_L$', 'm d$^{-1}$'),
              'B2p': ('$\\beta^,_2$', 'm$^3$ mmol$^{-1}$ d$^{-1}$'),
              'Bm2': ('$\\beta_{-2}$', 'd$^{-1}$'),
              'Bm1s': ('$\\beta_{-1,S}$', 'd$^{-1}$'),
              'Bm1l': ('$\\beta_{-1,L}$', 'd$^{-1}$'),
              'B2': ('$\\beta_2$', 'd$^{-1}$'),
              'Po': ('$\\.P_{S,ML}$', 'mmol m$^{-3}$ d$^{-1}$'),
              'Lp': ('$L_P$', 'm'), 'B3': ('$\\beta_3$', 'd$^{-1}$'),
              'a': ('$\\alpha$', None), 'zm': ('$z_m$', 'm')}

fig, (na_axs, sp_axs) = plt.subplots(2, 4, figsize=(6.5, 4))
fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.15)
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center',
         rotation='vertical')

xlims = {'ws': (0.5, 3.2), 'wl': (9, 31), 'Bm1s': (0, 0.16), 'Bm2': (-1, 3)}

for inversion, inv_result in results_dict.items():

    if inversion == 'NA':
        axs = na_axs
        [ax.axes.xaxis.set_ticklabels([]) for ax in axs]
    else:
        axs = sp_axs
    results = inv_result['params']
    ylabel = f'{inversion} inversion'
    dv_params = [p for p in results if results[p]['dv'] and p in xlims]

    for i, p in enumerate(dv_params):

        if p not in xlims.keys():
            continue
        ax = axs[i]
        if i:
            ax.tick_params(labelleft=False)
        if i == 3:
            ax.set_ylabel(ylabel, fontsize=14, rotation=270, labelpad=20)
            ax.yaxis.set_label_position('right')
        if ylabel == 'SP inversion':
            ax.set_xlabel(
                f'{param_text[p][0]} ({param_text[p][1]})', fontsize=12)
        else:
            ax.axes.xaxis.set_ticklabels([])
        ax.invert_yaxis()
        ax.set_xlim(xlims[p])
        ax.set_ylim(top=0, bottom=530)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axvline(results[p]['prior'], c=blue, lw=1.5, ls=':')
        ax.axvline(results[p]['prior'] - results[p]['prior_e'], c=blue, lw=1.5,
                   ls='--')
        ax.axvline(results[p]['prior'] + results[p]['prior_e'], c=blue, lw=1.5,
                   ls='--')

        for j, _ in enumerate(GRID):

            if j == 0:
                depths = 0, GRID[j]
            else:
                depths = GRID[j - 1], GRID[j]

            if 'w' in p:
                ax.errorbar(results[p]['posterior'][j], depths[1], fmt='o',
                            xerr=results[p]['posterior_e'][j], ms=8,
                            ecolor=orange, elinewidth=1, c=orange,
                            capsize=6, fillstyle='none', zorder=3,
                            markeredgewidth=1)
            else:
                depth = np.mean(depths)
                ax.scatter(results[p]['posterior'][j], depth, marker='o',
                           c=orange, s=14, zorder=3)
                ax.fill_betweenx(depths,
                                 (results[p]['posterior'][j] -
                                  results[p]['posterior_e'][j]),
                                 (results[p]['posterior'][j]
                                  + results[p]['posterior_e'][j]),
                                 color=orange, alpha=0.25)

fig.savefig('../../results/exports/figures/Figure6.pdf')
plt.close()

######################################################
# FIGURE 7, uniform params
######################################################

dc_params = [p for p in results if not results[p]['dv']]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    2, 3, tight_layout=True)
dc_axs = ax1, ax2, ax3, ax4, ax5
ax6.axis('off')

for i, p in enumerate(dc_params):
    ax = dc_axs[i]
    xlabel = param_text[p][0]
    if param_text[p][1]:
        xlabel += f' ({param_text[p][1]})'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.errorbar(1, NA_results['params'][p]['prior'],
                yerr=NA_results['params'][p]['prior_e'], fmt='^',
                c=blue, elinewidth=1.5, ecolor=blue, ms=9,
                capsize=6, label='Prior', markeredgewidth=1.5)
    ax.errorbar(2, NA_results['params'][p]['posterior'], fmt='o',
                yerr=NA_results['params'][p]['posterior_e'], c=orange, ms=9,
                ecolor=orange, elinewidth=1.5, capsize=6,
                label='Estimate (NA)', markeredgewidth=1.5)
    ax.errorbar(3, SP_results['params'][p]['posterior'], fmt='s',
                yerr=SP_results['params'][p]['posterior_e'], c=green, ms=9,
                ecolor=green, elinewidth=1.5, capsize=6,
                label='Estimate (SP)', markeredgewidth=1.5)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_xticks(np.arange(5))

handles, labels = ax5.get_legend_handles_labels()
handles = [h[0] for h in handles]
unique = [(h, l) for i, (h, l) in enumerate(
    zip(handles, labels)) if l not in labels[:i]]
ax6.legend(*zip(*unique), fontsize=12, loc='center', frameon=False,
           ncol=1, labelspacing=2, bbox_to_anchor=(0.35, 0.5))

fig.savefig('../../results/exports/figures/Figure7.pdf')
plt.close()

######################################################
# FIGURE 8, residual profiles
######################################################

fig, (na_axs, sp_axs) = plt.subplots(2, 2)
fig.subplots_adjust(left=0.14, right=0.92, top=0.95, bottom=0.15, wspace=0.1,
                    hspace=0.1)
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center',
         rotation='vertical')

for inversion, inv_result in results_dict.items():

    if inversion == 'NA':
        axs = na_axs
        [ax.axes.xaxis.set_ticklabels([]) for ax in axs]
    else:
        axs = sp_axs
        axs[0].set_xlabel(
            '$\\overline{\\varepsilon_{S}}h$ (mmol m$^{-2}$ d$^{-1}$)',
            fontsize=14)
        axs[1].set_xlabel(
            '$\\overline{\\varepsilon_{L}}h$ (mmol m$^{-2}$ d$^{-1}$)',
            fontsize=14)
    ylabel = f'{inversion} inversion'
    results = inv_result['residuals']

    for i, t in enumerate(results):

        ax = axs[i]
        ax.invert_yaxis()
        ax.set_xlim([-4, 6])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_yticks([0, 100, 200, 300, 400, 500])

        if i == 1:
            ax.set_ylabel(ylabel, fontsize=14, rotation=270,
                          labelpad=20)
            ax.yaxis.set_label_position('right')

        for j in range(len(GRID)):

            if j == 0:
                depths = 0, GRID[j]
            else:
                depths = GRID[j - 1], GRID[j]

            ax.scatter(results[t][j][0], np.mean(depths), marker='o',
                       c=orange, s=100, zorder=3, lw=0.7)
            ax.fill_betweenx(depths,
                             (results[t][j][0] - results[t][j][1]),
                             (results[t][j][0] + results[t][j][1]),
                             color=orange, alpha=0.25)
        prior_err = 0.5 * Po_prior * 30
        ax.axvline(prior_err, ls='--', c=blue)
        ax.axvline(-prior_err, ls='--', c=blue)
        ax.axvline(0, ls=':', c=blue)
        axs[1].tick_params(labelleft=False)

fig.savefig('../../results/exports/figures/Figure8.pdf')
plt.close()

######################################################
# FIGURE 9, param comparisons
######################################################

data1 = {'MNA': {'B2': (2 / DPY, 0.2 / DPY), 'Bm2': (156 / DPY, 17 / DPY),
                 'Bm1s': (13 / DPY, 1 / DPY)},
         'MNWA': {0: {'depth': 25.5,
                      'THICK': 50.9,
                      'Bm1s': (70 / DPY, 137 / DPY),
                      'B2': (9 / DPY, 24 / DPY),
                      'Bm2': (2690 / DPY, 10000 / DPY)},
                  1: {'depth': 85.1,
                      'THICK': 68.4,
                      'Bm1s': (798 / DPY, 7940 / DPY),
                      'B2': (11 / DPY, 30 / DPY),
                      'Bm2': (2280 / DPY, 10000 / DPY)},
                  2: {'depth': 169.5,
                      'THICK': 100.4,
                      'Bm1s': (378 / DPY, 3520 / DPY),
                      'B2': (13 / DPY, 50 / DPY),
                      'Bm2': (1880 / DPY, 10000 / DPY)},
                  3: {'depth': 295.3,
                      'THICK': 151.1,
                      'Bm1s': (1766 / DPY, 10000000 / DPY),
                      'B2': (18 / DPY, 89 / DPY),
                      'Bm2': (950 / DPY, 5700 / DPY)},
                  4: {'depth': 482.8,
                      'THICK': 224,
                      'Bm1s': (113 / DPY, 10000 / DPY),
                      'B2': (17 / DPY, 77 / DPY),
                      'Bm2': (870 / DPY, 5000 / DPY)}},
         'BRIG': {'depth': np.arange(250, 555, 50),
                  'Bm2': (0.27 * np.exp(-0.0024 * np.arange(250, 555, 50)),
                          0.03 * np.exp(-0.00027 * np.arange(250, 555, 50)))}}

data2 = {'B2': {'EP': all_data['C91_agg_EP'], 'SP': all_data['C91_agg_SP']},
         'Bm1s': {'EP': all_data['C91_remin_EP'],
                  'SP': all_data['C91_remin_SP']},
         'Bm2': {'EP': all_data['C91_disagg_EP']}}

fig, (na_axs, sp_axs) = plt.subplots(2, 3, figsize=(7, 6))
fig.subplots_adjust(bottom=0.12, top=0.85, hspace=0.1)
capsize = 4
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center',
         rotation='vertical')

for inversion, inv_result in results_dict.items():

    if inversion == 'NA':
        axs = na_axs
        [ax.axes.xaxis.set_ticklabels([]) for ax in axs]
    else:
        axs = sp_axs
    results = inv_result['params']
    ylabel = f'{inversion} inversion'

    for i, ax in enumerate(axs):
        if i == 0:
            p = 'Bm1s'
        elif i == 1:
            p = 'Bm2'
            ax.tick_params(labelleft=False)
        else:
            p = 'B2'
            ax.tick_params(labelleft=False)
            ax.set_ylabel(ylabel, fontsize=14, rotation=270, labelpad=20)
            ax.yaxis.set_label_position('right')
        if ax in na_axs:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(
                f'{param_text[p][0]} ({param_text[p][1]})', fontsize=14)

        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_ylim([600, -50])

        for j in range(len(GRID)):

            if j == 0:
                depths = 0, GRID[j]
            else:
                depths = GRID[j - 1], GRID[j]

            d_av = np.mean(depths)
            d_err = THICK[j] / 2

            ax.errorbar(results[p]['posterior'][j], d_av, fmt='o', yerr=d_err,
                        c=vermillion, capsize=capsize, zorder=9)
            ax.scatter(results[p]['posterior_e'][j], d_av, marker='o',
                       facecolors='none', edgecolors=black, zorder=10)

        for z in data1['MNWA']:
            d_av = data1['MNWA'][z]['depth']
            d_err = data1['MNWA'][z]['THICK'] / 2
            ax.errorbar(data1['MNWA'][z][p][0], d_av, fmt='s', yerr=d_err,
                        c=radish, capsize=4)
            ax.scatter(data1['MNWA'][z][p][1], d_av, marker='s',
                       facecolors='none', edgecolors=black, zorder=10)

        ax.errorbar(data1['MNA'][p][0], 225, fmt='d', yerr=75, c=green,
                    capsize=capsize, zorder=4)
        ax.scatter(data1['MNA'][p][1], 225, marker='d', zorder=10,
                   edgecolors=black, facecolors='none')

        if p == 'Bm2':
            ax.scatter(data1['BRIG'][p][0], data1['BRIG']['depth'], marker='*',
                       c=orange, s=60)
            ax.scatter(data1['BRIG'][p][1], data1['BRIG']['depth'], marker='*',
                       zorder=10, edgecolors=black, facecolors='none', s=60)

        for s in data2[p]:
            if s == 'EP':
                m = '^'
                c = sky
            else:
                m = 'v'
                c = blue
            ax.scatter(data2[p][s]['mean'], data2[p][s]['depth'], marker=m,
                       c=c)
            if 'error' in data2[p][s].columns:
                ax.scatter(data2[p][s]['error'], data2[p][s]['depth'],
                           marker=m, zorder=10, edgecolors=black,
                           facecolors='none')

    axs[0].set_xlim([0.001, 100000])
    axs[0].set_xticks([0.001, 0.1, 10, 1000, 10**5])
    axs[1].set_xlim([0.01, 100])
    axs[1].set_xticks([0.01, 0.1, 1, 10, 100])
    axs[2].set_xlim([0.00001, 1])
    axs[2].set_xticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    axs[2].yaxis.set_label_position('right')

leg_elements = [Line2D([0], [0], marker='o', mec=black, c=white,
                       label='This study \nStation P',
                       markerfacecolor=vermillion, ms=9),
                Line2D([0], [0], marker='s', mec=black, c=white,
                       label='Murnane et al. (1994)\nNWAO',
                       markerfacecolor=radish, ms=9),
                Line2D([0], [0], marker='d', mec=black, c=white,
                       label='Murnane et al. (1996)\nNABE',
                       markerfacecolor=green, ms=9),
                Line2D([0], [0], marker='*', mec=black, c=white,
                       label='Briggs et al. (2020)\nSNAO, SO',
                       markerfacecolor=orange, ms=12),
                Line2D([0], [0], marker='^', mec=black, c=white,
                       label='Clegg et al. (1991)\nEPO',
                       markerfacecolor=sky, ms=9),
                Line2D([0], [0], marker='v', mec=black, c=white,
                       label='Clegg et al. (1991)\nStation P',
                       markerfacecolor=blue, ms=9)]

na_axs[1].legend(handles=leg_elements, fontsize=10, ncol=3,
                 bbox_to_anchor=(0.44, 1.02), loc='lower center',
                 handletextpad=0.01, frameon=False)

fig.savefig('../../results/exports/figures/Figure9.pdf')
plt.close()

######################################################
# FIGURE 10, sinking fluxes
######################################################

th_fluxes = all_data['POC_fluxes_thorium']
th_depths = th_fluxes['depth']
th_flux = th_fluxes['flux']
th_flux_u = th_fluxes['flux_u']
st_fluxes = all_data['POC_fluxes_traps']
st_depths = st_fluxes['depth']
st_flux = st_fluxes['flux']
st_flux_u = st_fluxes['flux_u']

fig, (na_axs, sp_axs) = plt.subplots(2, 2, figsize=(6, 6))
fig.subplots_adjust(left=0.16, right=0.92, top=0.95, bottom=0.11, wspace=0.15,
                    hspace=0.1)
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center',
         rotation='vertical')
fig.text(0.54, 0.03, 'POC flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14,
         ha='center', va='center')

for inversion, inv_result in results_dict.items():
    if inversion == 'NA':
        axs = na_axs
        [ax.axes.xaxis.set_ticklabels([]) for ax in axs]
    else:
        axs = sp_axs
    ylabel = f'{inversion} inversion'
    result = inv_result['sink_fluxes']

    for ax in axs:
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=530)
        ax.axhline(100, ls=':', c=black, zorder=1)
        ax.tick_params(axis='both', which='major', labelsize=12)

    axs[1].set_ylabel(ylabel, fontsize=14, rotation=270,
                      labelpad=20)
    axs[1].yaxis.set_label_position('right')

    axs[0].errorbar(
        [x[0] for x in result['S']],
        np.array(GRID) + 2,
        fmt='o', xerr=[x[1] for x in result['S']],
        ecolor=blue, c=blue, capsize=4,
        label='$w_SP_S$', fillstyle='none',
        elinewidth=1.5, capthick=1.5)

    axs[0].errorbar(
        [x[0] for x in result['L']],
        np.array(GRID) - 2,
        fmt='o', xerr=[x[1] for x in result['L']],
        ecolor=orange, c=orange, capsize=4,
        label='$w_LP_L$', fillstyle='none',
        elinewidth=1.5, capthick=1.5)

    axs[1].tick_params(labelleft=False)
    axs[1].errorbar(
        [x[0] for x in result['T']], GRID, fmt='o',
        xerr=[x[1] for x in result['T']],
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

######################################################
# FIGURE 11, volumetric fluxes
######################################################

flux_text = {'sinkdiv_S': '$\\frac{d}{dz}w_SP_S$',
             'sinkdiv_L': '$\\frac{d}{dz}w_LP_L$',
             'remin_S': '$\\beta_{-1,S}P_S$',
             'remin_L': '$\\beta_{-1,L}P_L$',
             'aggregation': '$\\beta^,_2P^2_S$',
             'disaggregation': '$\\beta_{-2}P_L$',
             'production': '${\\.P_S}$'}

fig, (na_axs, sp_axs) = plt.subplots(2, 4, figsize=(7, 6))
fig.subplots_adjust(left=0.14, right=0.95, top=0.85, bottom=0.17, wspace=0.1)
fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center', va='center',
         rotation='vertical')
fig.text(0.55, 0.05, 'POC flux (mmol m$^{-3}$ d$^{-1}$)',
         fontsize=14, ha='center', va='center')

pairs = (('sinkdiv_S', 'sinkdiv_L'), ('remin_S', 'aggregation'),
         ('remin_L', 'disaggregation'), ('production',))

xlims = {'sinkdiv_S': (-0.2, 0.2), 'remin_S': (-0.05, 0.3),
         'production': (-0.01, 0.25)}


for inversion, inv_result in results_dict.items():

    if inversion == 'NA':
        axs = na_axs
    else:
        axs = sp_axs
    ylabel = f'{inversion} inversion'

    for i, pr in enumerate(pairs):

        ax = axs[i]
        if i:
            ax.tick_params(labelleft=False)
        if i == 3:
            ax.set_ylabel(ylabel, fontsize=14, rotation=270, labelpad=20)
            ax.yaxis.set_label_position('right')
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        if pr[0] != 'remin_L':
            ax.set_xlim(xlims[pr[0]])

        if pr[0] != 'production':

            result = inv_result['int_fluxes']

            for j in range(len(GRID)):

                if j == 0:
                    depths = 0, GRID[j]
                else:
                    depths = GRID[j - 1], GRID[j]

                ax.scatter(
                    result[pr[0]][j][0] / THICK[j], np.mean(depths),
                    marker='o', c=blue, s=14, zorder=3, lw=0.7,
                    label=flux_text[pr[0]])
                ax.fill_betweenx(
                    depths,
                    (result[pr[0]][j][0] - result[pr[0]][j][1]) / THICK[j],
                    (result[pr[0]][j][0] + result[pr[0]][j][1]) / THICK[j],
                    color=blue, alpha=0.25)
                ax.scatter(
                    result[pr[1]][j][0] / THICK[j], np.mean(depths),
                    marker='o', c=orange, s=14, zorder=3, lw=0.7,
                    label=flux_text[pr[1]])
                ax.fill_betweenx(
                    depths,
                    (result[pr[1]][j][0] - result[pr[1]][j][1]) / THICK[j],
                    (result[pr[1]][j][0] + result[pr[1]][j][1]) / THICK[j],
                    color=orange, alpha=0.25)
            if i == 0:
                ax.axvline(0, ls=':', c=black, zorder=1)

        else:
            result = inv_result['production_profile']
            depths = GRID
            df = all_data['NPP']
            H = 30
            npp = df.loc[df['target_depth'] >= H]['NPP']
            depth = df.loc[df['target_depth'] >= H]['target_depth']
            ax.scatter(
                npp / MMC,
                depth,
                c=orange,
                alpha=0.5,
                label='NPP',
                s=10)
            ax.scatter(
                [x[0] for x in result], depths, marker='o', c=blue, s=14,
                label=flux_text[pr[0]], zorder=3, lw=0.7)
            ax.errorbar(
                [x[0] for x in result], GRID, fmt='o',
                xerr=[x[1] for x in result], ecolor=blue, elinewidth=0.5,
                c=blue, ms=1.5, capsize=2, fillstyle='full',
                markeredgewidth=0.5)

        if ylabel == 'NA inversion':
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l) for i, (h, l) in enumerate(
                    zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique), loc='center', fontsize=12,
                      handletextpad=0.01, bbox_to_anchor=(0.45, 1.2),
                      frameon=False)
        else:
            ax.set_xlabel(('A', 'B', 'C', 'D')[i], fontsize=14)

fig.savefig('../../results/exports/figures/Figure11.pdf')
plt.close()

######################################################
# FIGURE 12, DVM comparisons
######################################################

fig = plt.figure()
fig.text(0.025, 0.5, 'Depth (m)', fontsize=14, ha='center',
         va='center', rotation='vertical')
fig.subplots_adjust(wspace=0.3, hspace=0.1)

for inversion, inv_result in results_dict.items():
    if inversion == 'NA':
        i = 0
    else:
        i = 1
    result = inv_result['int_fluxes']

    hostL = host_subplot(2, 2, 1 + 2 * i, axes_class=Axes, figure=fig)
    parL = hostL.twiny()
    parL.axis['top'].toggle(all=True)
    hostL.set_xlim(0, 0.3)
    parL.set_xlim(0, 0.3)

    hostR = host_subplot(2, 2, 2 * (1 + i), axes_class=Axes, figure=fig)
    hostR.yaxis.set_ticklabels([])
    parR = hostR.twiny()
    parR.axis['top'].toggle(all=True)
    hostR.set_xlim(-0.02, 0.02)
    parR.set_xlim(0.02, -0.02)
    hostR.axvline(c=black, alpha=0.3)

    if inversion == 'SP':
        hostL.set_xlabel('Ingestion flux (mmol m$^{-3}$ d$^{-1}$)')
        hostR.set_xlabel('Egestion flux (mmol m$^{-3}$ d$^{-1}$)')
        hostR.text(1.05, 0.2, 'SP inversion', fontsize=14, rotation=270,
                   transform=hostR.transAxes)
        parR.xaxis.set_ticklabels([])
        parL.xaxis.set_ticklabels([])
    else:
        parL.set_xlabel('$P_S$ remin. flux (mmol m$^{-3}$ d$^{-1}$)')
        parR.set_xlabel('$P_L$ SFD (mmol m$^{-3}$ d$^{-1}$)')
        hostR.xaxis.set_ticklabels([])
        hostL.xaxis.set_ticklabels([])
        hostR.text(1.05, 0.2, 'NA inversion', fontsize=14, rotation=270,
                   transform=hostR.transAxes)

    for host, par in ((hostL, parL), (hostR, parR)):
        host.axis['right'].toggle(all=False)
        host.axis['left', 'top', 'bottom'].major_ticks.set_tick_out('out')
        par.axis['left', 'top', 'bottom'].major_ticks.set_tick_out('out')
        host.axis['bottom'].label.set_color(blue)
        par.axis['top'].label.set_color(orange)
        host.axis['left'].label.set_fontsize(14)
        host.axis['bottom'].label.set_fontsize(12)
        host.axis['bottom', 'left'].major_ticklabels.set_size(12)
        par.axis['top'].label.set_fontsize(12)
        par.axis['top'].major_ticklabels.set_size(12)

    for j, _ in enumerate(GRID):
        if j < 3:
            host = hostL
            par = parL
            par_flux = 'remin_S'
        else:
            host = hostR
            par = parR
            par_flux = 'sinkdiv_L'
        if j == 0:
            depths = 0, GRID[j]
        else:
            depths = GRID[j - 1], GRID[j]

        host.scatter(
            result['dvm'][j][0] / THICK[j], np.mean(depths),
            marker='o', c=blue, s=14, zorder=3, lw=0.7)
        host.fill_betweenx(
            depths,
            (result['dvm'][j][0] - result['dvm'][j][1]) / THICK[j],
            (result['dvm'][j][0] + result['dvm'][j][1]) / THICK[j],
            color=blue, alpha=0.25)
        par.scatter(
            result[par_flux][j][0] / THICK[j], np.mean(depths),
            marker='o', c=orange, s=14, zorder=3, lw=0.7)
        par.fill_betweenx(
            depths,
            (result[par_flux][j][0] - result[par_flux][j][1]) / THICK[j],
            (result[par_flux][j][0] + result[par_flux][j][1]) / THICK[j],
            color=orange, alpha=0.25)
    for ax in (hostL, hostR):
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=510)
        ax.axhline(100, ls=':', c=black)

fig.savefig('../../results/exports/figures/Figure12.pdf')
plt.close()

######################################################
# FIGURES 13 & 14, budgets
######################################################

for z, fig_label in (('EZ', 13), ('UMZ', 14)):

    fig, (na_axs, sp_axs) = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.02, wspace=0.1, bottom=0.2, top=0.9)
    sp_axs[0].set_ylabel('Integrated flux (mmol m$^{-2}$ d$^{-1}$)',
                         fontsize=14)
    sp_axs[0].yaxis.set_label_coords(-0.2, 1)

    for inversion, inv_result in results_dict.items():

        if inversion == 'NA':
            axs = na_axs
            [ax.axes.xaxis.set_visible(False) for ax in axs]
        else:
            axs = sp_axs
        ylabel = f'{inversion} inversion'

        int_fluxes = inv_result['int_fluxes']
        residuals = inv_result['residuals']

        ax1, ax2 = axs
        ax2.set_ylabel(ylabel, fontsize=14, rotation=270, labelpad=20)
        ax2.yaxis.set_label_position('right')

        for group in ((ax1, 'S', -1, 1), (ax2, 'L', 1, -1)):
            ax, sf, agg_sign, dagg_sign = group
            ax.axhline(0, c=black, lw=1)
            ax.set_ylim([-16, 15])
            ax.set_xlabel(f'$P_{sf}$ fluxes', fontsize=14)
            labels = ['SFD', 'Remin.', 'Agg.', 'Disagg.', 'Resid.']
            fluxes = [-int_fluxes[f'sinkdiv_{sf}'][z][0],
                      -int_fluxes[f'remin_{sf}'][z][0],
                      agg_sign * int_fluxes['aggregation'][z][0],
                      dagg_sign * int_fluxes['disaggregation'][z][0],
                      residuals[f'POC{sf}'][z][0]]
            flux_errs = [int_fluxes[f'sinkdiv_{sf}'][z][1],
                         int_fluxes[f'remin_{sf}'][z][1],
                         int_fluxes['aggregation'][z][1],
                         int_fluxes['disaggregation'][z][1],
                         residuals[f'POC{sf}'][z][1]]

            if sf == 'S':
                labels.insert(-1, 'Prod.')
                fluxes.insert(-1, int_fluxes['production'][z][0])
                flux_errs.insert(-1, int_fluxes['production'][z][1])
            else:
                ax.tick_params(labelleft=False)

            if sf == 'S' and z == 'EZ':
                labels.insert(-1, 'DVM')
                fluxes.insert(-1, -int_fluxes['dvm'][z][0])
                flux_errs.insert(-1, int_fluxes['dvm'][z][1])
            elif sf == 'L' and z == 'UMZ':
                labels.insert(-1, 'DVM')
                fluxes.insert(-1, int_fluxes['dvm'][z][0])
                flux_errs.insert(-1, int_fluxes['dvm'][z][1])

            ax.bar(list(range(len(fluxes))), fluxes, yerr=flux_errs,
                   tick_label=labels, color=blue)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()

######################################################
# FIGURES S1 & S2, state element residuals
######################################################

state_elements = NA_results['state_elements']
eq_resids = []


def normalized_state_residual_plots(x_resids, fig_label):

    eq_resids = []

    j = 0
    for x in state_elements:
        if 'R' in x:
            eq_resids.append(x_resids.pop(j))
        else:
            j += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.set_ylabel('Probability density', fontsize=16)
    ax1.set_xlabel(r'$\frac{\^x_{i}-x_{o,i}}{\sigma_{o,i}}$', fontsize=14)
    ax2.set_xlabel(r'$\frac{\^{\overline{\varepsilon}h}}'
                   r'{\sigma_{\overline{\varepsilon}h}}$', fontsize=14)

    ax1.hist(x_resids, density=True, color=blue)
    ax2.hist(eq_resids, density=True, color=blue)

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()


normalized_state_residual_plots(NA_results['x_resids'], 'S1')
normalized_state_residual_plots(SP_results['x_resids'], 'S2')

######################################################
# FIGURES S3 & S4, sensitivity tests
######################################################

width = 0.2
# first (0.5, 0.5) is for prior_fluxes, second is for inverted fluxes
combos = ((0.5, 0.5), (0.5, 0.5), (0.5, 1), (1, 0.5), (1, 1))
run_colors = {0: blue, 1: orange, 2: green, 3: vermillion, 4: radish}
basepath = '../../results/exports'

for z, fig_label in (('EZ', 'S3'), ('UMZ', 'S4')):

    fig, (na_axs, sp_axs) = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.05, bottom=0.2, top=0.9)
    sp_axs[0].set_ylabel('Integrated flux (mmol m$^{-2}$ d$^{-1}$)',
                         fontsize=14)
    sp_axs[0].yaxis.set_label_coords(-0.2, 1)

    for inversion in ('NA', 'SP'):
        runs = []
        gammas = []
        for (g, re) in combos:
            with open(f'{basepath}/{inversion}_{re}_{g}.pkl', 'rb') as pickled:
                inv_result = pickle.load(pickled)
                runs.append(inv_result)
                gammas.append(g)
        if inversion == 'NA':
            axs = na_axs
            [ax.axes.xaxis.set_visible(False) for ax in axs]
        else:
            axs = sp_axs
            [ax.set_ylim([-20, 20]) for ax in axs]

        ax1, ax2 = axs
        ylabel = f'{inversion} inversions'
        ax2.set_ylabel(ylabel, fontsize=14, rotation=270, labelpad=20)
        ax2.yaxis.set_label_position('right')

        for i, run in enumerate(runs):
            if i > 0:
                rfi = {**run['int_fluxes'], **run['residuals']}
            else:
                rfi = run['prior_fluxes']
            color = run_colors[i]
            for group in ((ax1, 'S', -1, 1), (ax2, 'L', 1, -1)):
                ax, sf, agg_sign, dagg_sign = group
                ax.axhline(0, c=black, lw=0.5)
                ax.set_xlabel(f'$P_{sf}$ fluxes', fontsize=14)
                ax_labels = ['SFD', 'Remin.', 'Agg.', 'Disagg.', 'Resid.']
                fluxes = [-rfi[f'sinkdiv_{sf}'][z][0],
                          -rfi[f'remin_{sf}'][z][0],
                          agg_sign * rfi['aggregation'][z][0],
                          dagg_sign * rfi['disaggregation'][z][0]]
                flux_errs = [rfi[f'sinkdiv_{sf}'][z][1],
                             rfi[f'remin_{sf}'][z][1],
                             rfi['aggregation'][z][1],
                             rfi['disaggregation'][z][1]]
                if i > 0:
                    fluxes.append(rfi[f'POC{sf}'][z][0])
                    flux_errs.append(rfi[f'POC{sf}'][z][1])
                else:
                    fluxes.append(0)
                    flux_errs.append(gammas[i] * Po_prior * 30)
                if sf == 'S':
                    ax_labels.insert(-1, 'Prod.')
                    fluxes.insert(-1, rfi['production'][z][0])
                    flux_errs.insert(-1, rfi['production'][z][1])
                    if z == 'EZ':
                        ax_labels.insert(-1, 'DVM')
                        fluxes.insert(-1, -rfi['dvm'][z][0])
                        flux_errs.insert(-1, rfi['dvm'][z][1])
                elif sf == 'L' and z == 'UMZ':
                    ax_labels.insert(-1, 'DVM')
                    fluxes.insert(-1, rfi['dvm'][z][0])
                    flux_errs.insert(-1, rfi['dvm'][z][1])
                x = np.arange(len(ax_labels))
                if i == 0:
                    positions = x - width * 2
                elif i == 1:
                    positions = x - width
                elif i == 2:
                    positions = x
                elif i == 3:
                    positions = x + width
                else:
                    positions = x + width * 2
                ax.bar(positions, fluxes, width=width, yerr=flux_errs,
                       tick_label=ax_labels, color=color,
                       error_kw={'elinewidth': 1})
                ax.set_xticks(x)
                ax.set_xticklabels(ax_labels)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
    leg_elements = [
        Line2D([0], [0], c=blue, marker='s', ls='none', ms=6,
               label='prior ($\\gamma = 0.5, RE = 0.5$)'),
        Line2D([0], [0], c=orange, marker='s', ls='none', ms=6,
               label='$\\gamma = 0.5, RE = 0.5$'),
        Line2D([0], [0], c=green, marker='s', ls='none', ms=6,
               label='$\\gamma = 0.5, RE = 1$'),
        Line2D([0], [0], c=vermillion, marker='s', ls='none', ms=6,
               label='$\\gamma = 1, RE = 0.5$'),
        Line2D([0], [0], c=radish, marker='s', ls='none', ms=6,
               label='$\\gamma = 1, RE = 1$')]
    na_axs[1].legend(handles=leg_elements, fontsize=9, frameon=False,
                     handletextpad=-0.5, loc=(-0.04, 0.54), labelspacing=0)

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()

######################################################
# FIGURES S5 & S6, twin experiment POC profiles
######################################################

with open('../../results/exports/NA_0.5_0.5_TE.pkl', 'rb') as pickled:
    NA_results = pickle.load(pickled)
with open('../../results/exports/SP_0.5_0.5_TE.pkl', 'rb') as pickled:
    SP_results = pickle.load(pickled)

for inv, fig_label in ((NA_results, 'S5'), (SP_results, 'S6')):

    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)

    ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
    ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    ax1.errorbar(
        inv['tracers']['POCS']['prior'], GRID, fmt='^',
        xerr=inv['tracers']['POCS']['prior_e'], ecolor=blue,
        elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
    ax1.errorbar(
        inv['tracers']['POCS']['posterior'], GRID, fmt='o',
        xerr=inv['tracers']['POCS']['posterior_e'], ecolor=orange,
        elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
        zorder=3, markeredgewidth=1)

    ax2.errorbar(
        inv['tracers']['POCL']['prior'], GRID,
        fmt='^', xerr=inv['tracers']['POCL']['prior_e'], ecolor=blue,
        elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full',
        label='Data')
    ax2.errorbar(
        inv['tracers']['POCL']['posterior'], GRID, fmt='o',
        xerr=inv['tracers']['POCL']['posterior_e'], ecolor=orange,
        elinewidth=1, c=orange, ms=8, capsize=5,
        label='Estimate', fillstyle='none',
        zorder=3, markeredgewidth=1)

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=530)
        ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(labelleft=False)
    ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
              loc='lower right')

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()

######################################################
# FIGURES S7 & S8, twin experiments non-uniform params
######################################################
NA_targets = NA_results['targets']['params']
dv_params = [p for p in NA_targets if NA_targets[p]['dv'] and p != 'B2']

for inv, fig_label in ((NA_results, 'S7'), (SP_results, 'S8')):

    results = inv['params']
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    dv_axs = ax1, ax2, ax3, ax4, ax5, ax6

    fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center',
             va='center', rotation='vertical')
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.15,
                        hspace=0.5)

    for i, p in enumerate(dv_params):
        ax = dv_axs[i]
        xlabel = param_text[p][0]
        if param_text[p][1]:
            xlabel += f' ({param_text[p][1]})'
        ax.set_xlabel(xlabel, fontsize=12)
        if i not in (0, 3):
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=530)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axvline(results[p]['prior'], c=blue, lw=1.5, ls=':')
        ax.axvline(results[p]['prior'] - results[p]['prior_e'],
                   c=blue, lw=1.5, ls='--')
        ax.axvline(results[p]['prior'] + results[p]['prior_e'],
                   c=blue, lw=1.5, ls='--')

        for j, _ in enumerate(GRID):
            if j == 0:
                depths = 0, GRID[j]
            else:
                depths = GRID[j - 1], GRID[j]

            if 'w' in p:
                depth = depths[1]
                ax.errorbar(results[p]['posterior'][j], depth, fmt='o',
                            xerr=results[p]['posterior_e'][j], ms=8,
                            ecolor=orange, elinewidth=1, c=orange,
                            capsize=6, fillstyle='none', zorder=3,
                            markeredgewidth=1)
            else:
                depth = np.mean(depths)
                ax.scatter(results[p]['posterior'][j], depth, marker='o',
                           c=orange, s=14, zorder=3)
                ax.fill_betweenx(depths,
                                 (results[p]['posterior'][j] -
                                  results[p]['posterior_e'][j]),
                                 (results[p]['posterior'][j]
                                  + results[p]['posterior_e'][j]),
                                 color=orange, alpha=0.25)
            ax.scatter(inv['targets']['params'][p]['posterior'][j], depth,
                       marker='x', s=90, c=green)
            if p == 'Bm2' and fig_label == 'S8':
                ax.set_xlim([-1, 3])

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()

######################################################
# FIGURES S9 & S10, twin experiments uniform params
######################################################
dc_params = [p for p in NA_targets if not NA_targets[p]['dv']]

for inv, fig_label in ((NA_results, 'S9'), (SP_results, 'S10')):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, tight_layout=True)
    results = inv['params']
    dc_axs = ax1, ax2, ax3, ax4, ax5
    ax6.axis('off')

    for i, p in enumerate(dc_params):
        ax = dc_axs[i]
        xlabel = param_text[p][0]
        if param_text[p][1]:
            xlabel += f' ({param_text[p][1]})'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.errorbar(1, results[p]['prior'], yerr=results[p]['prior_e'],
                    fmt='^', c=blue, elinewidth=1.5, ecolor=blue, ms=9,
                    capsize=6, label='Prior', markeredgewidth=1.5)
        ax.errorbar(3, results[p]['posterior'], fmt='o',
                    yerr=results[p]['posterior_e'], c=orange, ms=9,
                    ecolor=orange, elinewidth=1.5, capsize=6, label='Estimate',
                    markeredgewidth=1.5)
        ax.scatter(2, inv['targets']['params'][p]['posterior'], marker='x',
                   s=90, c=green, label='Target')

        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_xticks(np.arange(5))

    handles, labels = ax5.get_legend_handles_labels()
    handles[-2:] = [h[0] for h in handles[-2:]]
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax6.legend(*zip(*unique), fontsize=12, loc='center', frameon=False,
               ncol=1, labelspacing=2, bbox_to_anchor=(0.35, 0.5))

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()

######################################################
# FIGURES S11 & S12, twin experiments residual profiles
######################################################

for inv, fig_label in ((NA_results, 'S11'), (SP_results, 'S12')):

    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)
    axs = (ax1, ax2)

    ax1.set_xlabel('$\\overline{\\varepsilon_{S}}h$ (mmol m$^{-2}$ d$^{-1}$)',
                   fontsize=14)
    ax2.set_xlabel('$\\overline{\\varepsilon_{L}}h$ (mmol m$^{-2}$ d$^{-1}$)',
                   fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    results = inv['residuals']

    for i, t in enumerate(results):

        ax = axs[i]
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_yticks([0, 100, 200, 300, 400, 500])

        for j in range(len(GRID)):

            if j == 0:
                depths = 0, GRID[j]
            else:
                depths = GRID[j - 1], GRID[j]

            ax.scatter(results[t][j][0], np.mean(depths), marker='o',
                       c=orange, s=100, zorder=3, lw=0.7)
            ax.fill_betweenx(depths,
                             (results[t][j][0] - results[t][j][1]),
                             (results[t][j][0] + results[t][j][1]),
                             color=orange, alpha=0.25)
            ax.scatter(inv['targets']['residuals'][t][j][0], np.mean(depths),
                       marker='x', c=green, s=250)
        prior_err = 0.5 * Po_prior * 30

        ax.axvline(prior_err, ls='--', c=blue)
        ax.axvline(-prior_err, ls='--', c=blue)
        ax.axvline(0, ls=':', c=blue)
        axs[1].tick_params(labelleft=False)

        for ax in (ax1, ax2):
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(labelleft=False)

    fig.savefig(f'../../results/exports/figures/Figure{fig_label}.pdf')
    plt.close()
