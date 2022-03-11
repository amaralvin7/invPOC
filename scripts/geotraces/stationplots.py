#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from src.colors import *

results_path = f'../../results/geotraces/'
pickled_files = [f for f in os.listdir(results_path) if f.endswith('.pkl')]

for f in pickled_files:
    
    station, priors_from = f.split('_')
    s = station[3:]
    pf = priors_from[:2]
    file_path = os.path.join(results_path, f)

    with open(file_path, 'rb') as file:
        unpickled = pickle.load(file)
        tracers, params, residuals, inventories, int_fluxes, residence_times, turnover_times, grid, zg, mld, layers, convergence_evolution, cost_evolution = unpickled
    
    #####################
    #POC
    #####################

    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)

    ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
    ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    ax1.errorbar(
        tracers['POCS']['prior'], grid,
        fmt='^', xerr=tracers['POCS']['prior_e'], ecolor=blue,
        elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
    ax1.errorbar(
        tracers['POCS']['posterior'], grid, fmt='o',
        xerr=tracers['POCS']['posterior_e'], ecolor=orange,
        elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
        zorder=3, markeredgewidth=1)

    ax2.errorbar(
        tracers['POCL']['prior'], grid,
        fmt='^', xerr=tracers['POCL']['prior_e'], ecolor=blue,
        elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full', label='Data')
    ax2.errorbar(
        tracers['POCL']['posterior'], grid, fmt='o',
        xerr=tracers['POCL']['posterior_e'], ecolor=orange,
        elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
        zorder=3, markeredgewidth=1, label='Estimate')

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=610)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axhline(zg, c=green)
        ax.axhline(mld, c=black)
    ax2.tick_params(labelleft=False)
    ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                loc='lower right')
    plt.savefig(f'../../results/geotraces/stn{s}_POC_{pf}')
    plt.close()

    #####################
    #RESIDUALS
    #####################

    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)
    axs = (ax1, ax2)

    ax1.set_xlabel('$\\varepsilon_{S}$ (mmol m$^{-2}$ d$^{-1}$)',
                    fontsize=14)
    ax2.set_xlabel('$\\varepsilon_{L}$ (mmol m$^{-2}$ d$^{-1}$)',
                    fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    for i, t in enumerate(residuals):
        ax = axs[i]
        for l in layers:
            zi = grid[l]
            zim1 = grid[grid.index(zi) - 1] if l > 0 else 0
            depths = (zi, zim1)
            ax.scatter(residuals[t][l][0], np.mean(depths),
                        marker='o', c=orange, s=100, zorder=3, lw=0.7)
            ax.fill_betweenx(depths, (residuals[t][l][0] - residuals[t][l][1]),
                             (residuals[t][l][0] + residuals[t][l][1]),
                             color=orange, alpha=0.25)
        ax.axvline(residuals[t]['prior_e'], ls='--', c=blue)
        ax.axvline(-residuals[t]['prior_e'], ls='--', c=blue)
        ax.axvline(residuals[t]['prior'], ls=':', c=blue)

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(labelleft=False)
    plt.savefig(f'../../results/geotraces/stn{s}_residuals_{pf}')
    plt.close()

    #####################
    #DV PARAMS
    #####################
    
    param_info = {'ws': ('$w_S$', 'm d$^{-1}$'), 'wl': ('$w_L$', 'm d$^{-1}$'),
                  'B2p': ('$\\beta^,_2$', 'm$^{3}$ mmol$^{-1}$ d$^{-1}$'),
                  'Bm2': ('$\\beta_{-2}$', 'd$^{-1}$'),
                  'Bm1s': ('$\\beta_{-1,S}$', 'd$^{-1}$'),
                  'Bm1l': ('$\\beta_{-1,L}$', 'd$^{-1}$'),
                  'Po': ('$\.P_{S,ML}$', 'mmol m$^{-3}$ d$^{-1}$'),
                  'Lp': ('$L_P$', 'm'), 'B3': ('$\\beta_3$', 'd$^{-1}$'),
                  'a': ('$\\alpha$', None), 'zm': ('$z_m$', 'm')}

    dv, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    dv_axs = ax1, ax2, ax3, ax4, ax5, ax6

    dc, ((ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(
        2, 3, tight_layout=True)
    dc_axs = ax7, ax8, ax9, ax10, ax11
    ax12.axis('off')

    dv.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center',
            va='center', rotation='vertical')
    dv.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.15,
                       hspace=0.5)

    dv_params = [p for p in params if params[p]['dv']]
    dc_params = [p for p in params if not params[p]['dv']]

    for i, p in enumerate(dv_params):
        ax = dv_axs[i]
        ax.set_xlabel(f'{param_info[p][0]} ({param_info[p][1]})', fontsize=12)
        if i not in (0,3):
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=610)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axvline(params[p]['prior'], c=blue, lw=1.5, ls=':')
        ax.axvline(
            params[p]['prior'] - params[p]['prior_e'], c=blue, lw=1.5, ls='--')
        ax.axvline(
            params[p]['prior'] + params[p]['prior_e'], c=blue, lw=1.5, ls='--')
        for l in layers:
            if params[p]['posterior'][l] < 0:
                print(s, pf, p, l, params[p]['posterior'][l])
            if 'w' in p:
                depth = grid[l]
                ax.errorbar(
                    params[p]['posterior'][l], depth, fmt='o', ms=8,
                    xerr=params[p]['posterior_e'][l],
                    ecolor=orange, elinewidth=1, c=orange,
                    capsize=6, fillstyle='none', zorder=3,
                    markeredgewidth=1)
            else:
                zi = grid[l]
                zim1 = grid[grid.index(zi) - 1] if l > 0 else 0
                depths = (zi, zim1)
                depth = np.mean(depths)
                ax.scatter(
                    params[p]['posterior'][l], depth, marker='o',
                    c=orange, s=14, zorder=3)
                ax.fill_betweenx(
                    depths,
                    (params[p]['posterior'][l] - params[p]['posterior_e'][l]),
                    (params[p]['posterior'][l] + params[p]['posterior_e'][l]),
                    color=orange, alpha=0.25)

    for i, p in enumerate(dc_params):
        ax = dc_axs[i]
        xlabel = param_info[p][0]
        if param_info[p][1]:
            xlabel += f' ({param_info[p][1]})'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.errorbar(1, params[p]['prior'], yerr=params[p]['prior_e'], fmt='^',
                    ms=9, c=blue, elinewidth=1.5, ecolor=blue,
                    capsize=6, label='Prior', markeredgewidth=1.5)
        ax.errorbar(3, params[p]['posterior'],
                    yerr=params[p]['posterior_e'], fmt='o',
                    c=orange, ms=9, elinewidth=1.5, label='Estimate',
                    ecolor=orange, capsize=6, markeredgewidth=1.5)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_xticks(np.arange(5))

    handles, labels = ax11.get_legend_handles_labels()
    handles[-2:] = [h[0] for h in handles[-2:]]
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax12.legend(*zip(*unique), fontsize=12, loc='center', frameon=False,
                ncol=1, labelspacing=2, bbox_to_anchor=(0.35, 0.5))

    dv.savefig(f'../../results/geotraces/stn{s}_DVparams_{pf}')
    dc.savefig(f'../../results/geotraces/stn{s}_DCparams_{pf}')
    plt.close(dc)
    plt.close(dv)
    
    #####################
    #CONVERGENCE AND COST EVOLUTION
    #####################   
    
    k = len(cost_evolution)

    fig, ax = plt.subplots(1, tight_layout=True)
    ax.plot(np.arange(2, k+1), convergence_evolution, marker='o', ms=3, c=blue)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration, $k$', fontsize=16)
    ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',
                  fontsize=16)
    plt.savefig(f'../../results/geotraces/stn{s}_conv_{pf}')
    plt.close()
    
    fig, ax = plt.subplots(1, tight_layout=True)
    ax.plot(np.arange(1, k+1), cost_evolution, marker='o', ms=3, c=blue)
    ax.set_xlabel('Iteration, $k$', fontsize=16)
    ax.set_ylabel('Cost, $J$', fontsize=16)
    plt.savefig(f'../../results/geotraces/stn{s}_cost_{pf}')
    plt.close()
    
