import os
import sys
from itertools import product
from time import time

import cartopy.crs
import gsw
import h5py
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

import src.geotraces.data as data
from src.colors import *
from src.modelequations import get_layer_bounds


def compile_param_estimates_dv():
    
    dv_params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')

    dv_rows = []

    with h5py.File(output_fp, 'r') as f:
        stations = list(f.keys())
        for stn in stations:
            inversions = list(f[stn].keys())
            for inv in inversions:
                dv_dict = {p: f[stn][inv][p][()] for p in dv_params}
                dv_dict['depth'] = station_data[int(stn)]['grid']
                dv_dict['avg_depth'] = [np.mean(get_layer_bounds(l, station_data[int(stn)]['grid'])) for l in station_data[int(stn)]['layers']]
                dv_dict['latitude'] = station_data[int(stn)]['latitude'] * np.ones(len(station_data[int(stn)]['grid']))
                dv_dict['station'] = int(stn) * np.ones(len(station_data[int(stn)]['grid']))
                avg_ps = []
                ps_post = f[stn][inv]['POCS']
                for i, ps in enumerate(ps_post):
                    if i == 0:
                        avg_ps.append(ps)
                    else:
                        avg_ps.append(np.mean([ps, ps_post[i-1]]))
                dv_dict['B2'] = dv_dict['B2p']*np.array(avg_ps)
                dv_dict['aggratio'] = dv_dict['Bm2']/dv_dict['B2']
                dv_rows.append(pd.DataFrame(dv_dict))
                
    dv_df = pd.concat(dv_rows, ignore_index=True)

    return dv_df


def compile_param_estimates_dc():
    
    dc_params = ('Po', 'Lp', 'zm', 'a', 'B3')

    dc_rows = []

    with h5py.File(output_fp, 'r') as f:
        stations = list(f.keys())
        for stn in stations:
            inversions = list(f[stn].keys())
            for inv in inversions:
                dc_dict = {p: [f[stn][inv][p][()]] for p in dc_params}
                dc_dict['latitude'] = [station_data[int(stn)]['latitude']]
                dc_dict['station'] = [int(stn)]
                dc_rows.append(pd.DataFrame(dc_dict))
    dc_df = pd.concat(dc_rows, ignore_index=True)
    
    return dc_df


def param_section_compilation_dv():
    
    param_text = get_param_text()

    df = compile_param_estimates_dv()

    params = ('B2p', 'B2', 'Bm2', 'aggratio', 'Bm1s', 'Bm1l', 'ws', 'wl')
    lims = {'B2p': (0, 0.2), 'B2': (0, 0.2), 'Bm2': (0, 1),
            'aggratio': (0, 20), 'Bm1s': (0, 0.1), 'Bm1l': (0, 0.25),
            'ws': (0, 1), 'wl': (0, 40)}
    scheme = plt.cm.viridis
    lats = [station_data[s]['latitude'] for s in station_data]
    mlds_unsorted = [station_data[s]['mld'] for s in station_data]
    zgs_unsorted = [station_data[s]['zg'] for s in station_data]
    mlds = [mld for _, mld in sorted(zip(lats, mlds_unsorted))]
    zgs = [zg for _, zg in sorted(zip(lats, zgs_unsorted))]
    lats.sort()
    
    fig, axs = plt.subplots(len(params), 1, figsize=(6, 12), tight_layout=True)
    fig.subplots_adjust(right=0.8)
    
    for i, p in enumerate(params):
        p_df = df[['depth', 'avg_depth', 'latitude', p]]
        mean = p_df.groupby(['depth', 'avg_depth', 'latitude']).mean().reset_index()
        
        ax = axs[i]
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=630)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.plot(lats, mlds, c='k', zorder=1, ls='--')
        ax.plot(lats, zgs, c='k', zorder=1)
        if param_text[p][1]:
            units = f'\n({param_text[p][1]})'
        else:
            units = ''
        cbar_label  = f'{param_text[p][0]}{units}'
        to_plot = mean[p]
        for s, d in station_data.items():
            ax.text(d['latitude'], -30, s, ha='center', size=6)
        if i < len(params) - 1:
            ax.tick_params(axis='x', label1On=False)
        else:
            ax.set_xlabel('Latitude (°N)', fontsize=12)
        norm = Normalize(*lims[p])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)
        cbar.set_label(cbar_label, rotation=270, labelpad=40, fontsize=12)
        if 'w' not in p:
            depth_str = 'avg_depth'
            for s in station_data: # plot sampling depths
                depths = station_data[s]['grid']
                ax.scatter(np.ones(len(depths))*station_data[s]['latitude'], depths, c='k', zorder=1, s=1)
        else:
            depth_str = 'depth'
        ax.scatter(mean['latitude'], mean[depth_str], c=to_plot, norm=norm, cmap=scheme, zorder=10)
    fig.savefig('../../results/geotraces/figs/FigureS2.pdf', bbox_inches='tight')
    plt.close()


def param_section_compilation_dc():
    
    param_text = get_param_text()

    df = compile_param_estimates_dc()

    npp_data = data.extract_nc_data(poc_data, 'cbpm')
    Lp_priors = data.get_Lp_priors(poc_data)
    Po_priors = data.get_Po_priors(poc_data, Lp_priors, npp_data)
    B3_priors = data.get_B3_priors(npp_data)

    params = ('Po', 'Lp', 'zm', 'a', 'B3')
    lats = [station_data[s]['latitude'] for s in station_data]
    
    # get priors
    priors = {p: {} for p in params}
    for s in station_data.keys():
        priors['Lp'][s] = (Lp_priors[s], Lp_priors[s]*0.5)
        priors['Po'][s] = (Po_priors[s], Po_priors[s]*0.5)
        priors['B3'][s] = (B3_priors[s], B3_priors[s]*0.5)
        priors['a'][s] = (0.3, 0.3*0.5)
        priors['zm'][s] = (500, 500*0.5)

            
    fig, axs = plt.subplots(len(params), 1, figsize=(6, 10), tight_layout=True)
    fig.subplots_adjust(right=0.8, hspace=0.2)
    
    for i, p in enumerate(params):
        p_df = df[['latitude', p]]
        mean = p_df.groupby(['latitude']).mean().reset_index()
        sd = p_df.groupby(['latitude']).sem().reset_index()
        merged = mean.merge(sd, suffixes=(None, '_se'), on=['latitude'])
        ax = axs[i]
        if param_text[p][1]:
            units = f'\n({param_text[p][1]})'
        else:
            units = ''
        ax.set_ylabel(f'{param_text[p][0]}{units}', fontsize=14)

        ax.invert_xaxis()
        ax.errorbar(lats, [priors[p][s][0] for s in station_data], yerr=[priors[p][s][1] for s in station_data],
                    c=gray, fmt='d', zorder=2, elinewidth=1, ecolor=gray, ms=4,
                    capsize=2)
        if i < len(axs) - 1:
            ax.tick_params(axis='x',label1On=False)
        else:
            ax.set_xlabel('Latitude (°N)')
        for s in station_data:  # station labels and faint gridlines
            s_df = merged.loc[merged['latitude'] == station_data[s]['latitude']]
            station_color = get_station_color(s)
            ax.errorbar(s_df['latitude'], s_df[p],
                            yerr=s_df[f'{p}_se'], fmt='o',
                            c=station_color, elinewidth=1, ecolor=station_color, ms=4,
                            capsize=2, zorder=10)
        
    for s, ax in product(station_data, axs):  # station labels and faint gridlines
        ax.text(station_data[s]['latitude'], 1.02, s, ha='center', size=6, transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

    fig.savefig('../../results/geotraces/figs/Figure6.pdf', bbox_inches='tight')
    plt.close()


def poc_section():
    
    lims = {'POCS': (0.06, 6), 'POCL': (0.004, 2)}
    cbar_labels = {'POCS': '$P_{S}$ (mmol m$^{-3}$)',
                   'POCL': '$P_{L}$ (mmol m$^{-3}$)'}
    scheme = plt.cm.viridis
    lats = [station_data[s]['latitude'] for s in station_data]
    mlds_unsorted = [station_data[s]['mld'] for s in station_data]
    zgs_unsorted = [station_data[s]['zg'] for s in station_data]
    ezds_unsorted = [station_data[s]['ezd'] for s in station_data]
    mlds = [mld for _, mld in sorted(zip(lats, mlds_unsorted))]
    zgs = [zg for _, zg in sorted(zip(lats, zgs_unsorted))]
    ezds = [zg for _, zg in sorted(zip(lats, ezds_unsorted))]
    lats.sort()
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    fig.subplots_adjust(right=0.8)
    
    for ax in axs:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=630)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.plot(lats, mlds, c='k', zorder=1, ls='--')
        ax.plot(lats, ezds, c='k', zorder=1)
        ax.plot(lats, zgs, c='k', zorder=1, ls=':')
    
    axs[0].tick_params(axis='x', label1On=False)
    axs[1].set_xlabel('Latitude (°N)', fontsize=12)
    
    for (ax, tracer) in ((axs[0], 'POCS'), (axs[1], 'POCL')):
        norm = LogNorm(*lims[tracer])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)
        cbar.set_label(cbar_labels[tracer], rotation=270, labelpad=20, fontsize=12)
        for s in station_data:
            ax.text(station_data[s]['latitude'], 1.02, s, ha='center', size=6, transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
            ax.scatter(poc_data[s]['latitude'], poc_data[s]['depth'], c=poc_data[s][tracer], norm=norm, cmap=scheme, zorder=10)

    fig.savefig('../../results/geotraces/figs/Figure2.pdf', bbox_inches='tight')
    plt.close()
    

def get_station_color_legend(all_regimes=False):

    lines = [Line2D([0], [0], color=green, lw=4),
             Line2D([0], [0], color=orange, lw=4),
             Line2D([0], [0], color=vermillion, lw=4),
             Line2D([0], [0], color=blue, lw=4)]
    
    labels = ['Subarctic', 'N. Pac', 'Equator', 'S. Pac']
    
    if all_regimes:
        lines.append(Line2D([0], [0], color=gray, lw=4))
        labels = ['Subarctic', 'N. Pac', 'Eq.', 'S. Pac', 'All']
    
    
    line_length = 1
    
    return lines, labels, line_length

        
def spaghetti_params():

    param_text = get_param_text()
    
    df = compile_param_estimates_dv()   

    params = ('B2p', 'B2', 'Bm2', 'aggratio', 'Bm1s', 'Bm1l', 'ws', 'wl')
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(7, 12))
    axs = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
    fig.subplots_adjust(wspace=0.1, hspace=0.5, left=0.15)
    
    fig.text(0.04, 0.5, 'Depth (m)', fontsize=14, va='center', rotation='vertical')

    for i, p in enumerate(params):
        depth_str = 'depth' if 'w' in p else 'avg_depth'
        p_df = df[[depth_str, 'station', 'latitude', p]]
        mean = p_df.groupby([depth_str, 'station', 'latitude']).mean().reset_index()
        
        if i % 2:
            axs[i].yaxis.set_ticklabels([])
        axs[i].set_ylim(0, 600)
        axs[i].invert_yaxis()
        if param_text[p][1]:
            units = f' ({param_text[p][1]})'
        else:
            units = ''
        axs[i].set_xlabel(f'{param_text[p][0]}{units}', fontsize=14)
        
        for s in station_data:
            c = get_station_color(s)
            s_df = mean.loc[mean['station'] == s]
            axs[i].plot(s_df[p], s_df[depth_str], c=c)
        
        if p == 'aggratio':
            axs[i].axvline(1, c=black, alpha=0.2, zorder=1)

    lines, labels, line_length = get_station_color_legend()
    axs[0].legend(lines, labels, frameon=False, handlelength=line_length)

    fig.savefig('../../results/geotraces/figs/Figure7.pdf', bbox_inches='tight')
    plt.close()


def spaghetti_ctd():

    station_fname = ctd_files_by_station()
    fig, axs = plt.subplots(1, 2, figsize=(6, 5), tight_layout=True)

    # profiles of T, O2, N2, params
    for s in station_fname:
        color = get_station_color(s)
        ctd_df = pd.read_csv(os.path.join('../../data/geotraces/ctd', station_fname[s]), header=12)
        ctd_df.drop([0, len(ctd_df) - 1], inplace=True)  # don't want first and last rows (non-numerical)
        for c in ['CTDPRS', 'CTDOXY', 'CTDTMP']:
            ctd_df[c] = pd.to_numeric(ctd_df[c])
        ctd_df = ctd_df.loc[ctd_df['CTDPRS'] <= 600]
        ctd_df = ctd_df[['CTDPRS', 'CTDOXY', 'CTDTMP']]

        lat = station_data[s]['latitude']
        depth = -gsw.z_from_p(ctd_df['CTDPRS'].values, lat)

        axs[0].set_ylabel('Depth (m)', fontsize=14, labelpad=10)
        axs[0].set_xlabel('Temperature\n(°C)', fontsize=14)
        axs[0].plot(ctd_df['CTDTMP'], depth, c=color)
        
        axs[1].yaxis.set_ticklabels([])
        axs[1].set_xlabel('Dissolved O$_2$\n(µmol kg$^{-1}$)', fontsize=14)
        axs[1].plot(ctd_df['CTDOXY'], depth, c=color)
            
        for ax in axs:
            ax.set_ylim(0, 600)
            ax.invert_yaxis()

    lines, labels, line_length = get_station_color_legend()
    axs[0].legend(lines, labels, frameon=False, handlelength=line_length)

    fig.savefig('../../results/geotraces/figs/Figure5.pdf', bbox_inches='tight')
    plt.close()


def spaghetti_poc():
    
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    
    for ax in (ax1, ax2):
        ax.set_ylim(0, 610)
        ax.invert_yaxis()
        ax.set_xscale('log')
    
    ax1.set_ylabel('Depth (m)', fontsize=14)
    ax2.yaxis.set_ticklabels([])
    ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
    ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
    
    for s in poc_data:
        station_color = get_station_color(s)
        ax1.plot(poc_data[s]['POCS'], poc_data[s]['depth'], c=station_color)
        ax2.plot(poc_data[s]['POCL'], poc_data[s]['depth'], c=station_color)

    lines, labels, line_length = get_station_color_legend()
    ax1.legend(lines, labels, frameon=False, handlelength=line_length)

    fig.savefig('../../results/geotraces/figs/FigureS1.pdf', bbox_inches='tight')
    plt.close()


def ctd_files_by_station():

    # pigrath (ODF) casts from Jen's POC flux table for all stations except
    # 8, 14, 29, and 39, which are from GTC
    station_cast = {4: 5, 5: 5, 6: 5, 8: 6, 10: 5, 12: 6, 14: 6, 16: 5, 
                    18: 5, 19: 4, 21: 5, 23: 4, 25: 5, 27: 5, 29: 6, 31: 5,
                    33: 5, 35: 5, 37: 5, 39: 6}
    
    station_fname = {}
    fnames = [f for f in os.listdir('../../data/geotraces/ctd') if '.csv' in f]  # get filenames for each station
    for f in fnames:
        prefix  = f.split('_')[0]
        station = int(prefix[:3])
        cast = int(prefix[3:])
        if station in station_cast and station_cast[station] == cast:
            station_fname[station] = f
    
    return station_fname


def ctd_plots_remin():
        
    # get mean param df across all stations
    df = compile_param_estimates_dv()
    param_means = df.groupby(['depth', 'station']).mean().reset_index()

    station_fname = ctd_files_by_station()
    
    params = ('Bm1s', 'Bm1l')
    param_text = get_param_text()
    
    fig, axs = plt.subplots(2, 2, figsize=(6, 5), tight_layout=True)
    t_axs = [axs.flatten()[i] for i in [0, 2]]
    o_axs = [axs.flatten()[i] for i in [1, 3]]
    
    t_axs[1].set_xlabel('Temperature (°C)')
    o_axs[1].set_xlabel('Dissolved O$_2$ (µmol kg$^{-1}$)')

    for (s, (i, p)) in product(station_fname, enumerate(params)):
        
        color = get_station_color(s)
        
        s_p_df = param_means.loc[param_means['station'] == s][['depth', p]]
        ctd_df = pd.read_csv(os.path.join('../../data/geotraces/ctd', station_fname[s]), header=12)
        ctd_df.drop([0, len(ctd_df) - 1], inplace=True)  # don't want first and last rows (non-numerical)
        for c in ['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']:
            ctd_df[c] = pd.to_numeric(ctd_df[c])
        ctd_df = ctd_df.loc[ctd_df['CTDPRS'] <= 600]
        ctd_df = ctd_df[['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']]

        lat = station_data[s]['latitude']
        ctd_df['depth'] = -gsw.z_from_p(ctd_df['CTDPRS'].values, lat)
        
        for j, (_, r) in enumerate(s_p_df.iterrows()):
            
            ydeep, yshal = get_layer_bounds(j, s_p_df['depth'].values)
            ctd_in_layer = ctd_df.loc[(ctd_df['depth'] < ydeep) & (ctd_df['depth'] > yshal)]
            avg_T = ctd_in_layer['CTDTMP'].mean()
            avg_O = ctd_in_layer['CTDOXY'].mean()

            t_axs[i].scatter(avg_T, r[p], s=7, color=color)
            o_axs[i].scatter(avg_O, r[p], s=7, color=color)

    for ax in t_axs:
        ax.set_xlim(0, 32)
    for ax in o_axs:
        ax.set_xlim(0, 330)

    for i, p in enumerate(params):
        if param_text[p][1]:
            units = f' ({param_text[p][1]})'
        else:
            units = ''
        t_axs[i].set_ylabel(f'{param_text[p][0]}{units}', fontsize=14)
        o_axs[i].yaxis.set_ticklabels([])
        if i < 1:
            o_axs[i].xaxis.set_ticklabels([])
            t_axs[i].xaxis.set_ticklabels([])
            
    lines, labels, line_length = get_station_color_legend()
    o_axs[0].legend(lines, labels, frameon=False, handlelength=line_length)
    
    fig.savefig('../../results/geotraces/figs/FigureS4.pdf', bbox_inches='tight')
    plt.close()


def ctd_plots_sink():
        
    # get mean param df across all stations
    df = compile_param_estimates_dv()
    param_means = df.groupby(['depth', 'station']).mean().reset_index()

    station_fname = ctd_files_by_station()
    
    params = ('ws', 'wl')
    param_text = get_param_text()
    
    fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True)
    
    axs[1].set_xlabel('Viscosity (g cm$^{-1}$ s$^{-1}$)')

    for (s, (i, p)) in product(station_fname, enumerate(params)):
        
        color = get_station_color(s)
        
        s_p_df = param_means.loc[param_means['station'] == s][['depth', p]]
        ctd_df = pd.read_csv(os.path.join('../../data/geotraces/ctd', station_fname[s]), header=12)
        ctd_df.drop([0, len(ctd_df) - 1], inplace=True)  # don't want first and last rows (non-numerical)
        for c in ['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']:
            ctd_df[c] = pd.to_numeric(ctd_df[c])
        ctd_df = ctd_df.loc[ctd_df['CTDPRS'] <= 600]
        ctd_df = ctd_df[['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']]
        ctd_df['CTDVIS'] = (2.97e-9 * (ctd_df['CTDTMP']**4)
                            -2.92e-7 * (ctd_df['CTDTMP']**3)
                            +1.5e-5 * (ctd_df['CTDTMP']**2)
                            -6e-4 * ctd_df['CTDTMP']
                            + 0.0188
                            + 2.43e-5 * (ctd_df['CTDSAL'] - 35))
        lat = station_data[s]['latitude']
        ctd_df['depth'] = -gsw.z_from_p(ctd_df['CTDPRS'].values, lat)
        
        closest_ctd = pd.merge_asof(s_p_df, ctd_df, on='depth', direction='nearest')
        axs[i].scatter(closest_ctd['CTDVIS'], s_p_df[p], s=7, color=color)

    for ax in axs:
        ax.set_xlim(0.008, 0.018)

    for i, p in enumerate(params):
        if param_text[p][1]:
            units = f' ({param_text[p][1]})'
        else:
            units = ''
        axs[i].set_ylabel(f'{param_text[p][0]}{units}', fontsize=14)
        if i < 1:
            axs[i].xaxis.set_ticklabels([])
            
    lines, labels, line_length = get_station_color_legend()
    axs[1].legend(lines, labels, frameon=False, handlelength=line_length)
    
    fig.savefig('../../results/geotraces/figs/FigureS3.pdf', bbox_inches='tight')
    plt.close() 


def get_param_text():

    param_text = {'ws': ('$w_S$', 'm d$^{-1}$'), 'wl': ('$w_L$', 'm d$^{-1}$'),
                'B2p': ('$\\beta^,_2$', 'm$^3$ mmol$^{-1}$ d$^{-1}$'),
                'Bm2': ('$\\beta_{-2}$', 'd$^{-1}$'),
                'Bm1s': ('$\\beta_{-1,S}$', 'd$^{-1}$'),
                'Bm1l': ('$\\beta_{-1,L}$', 'd$^{-1}$'),
                'B2': ('$\\beta_2$', 'd$^{-1}$'),
                'Po': ('$\\.P_{S,0}$', 'mmol m$^{-3}$ d$^{-1}$'),
                'Lp': ('$L_P$', 'm'), 'B3': ('$\\beta_3$', 'd$^{-1}$'),
                'a': ('$\\alpha$', None), 'zm': ('$z_m$', 'm'),
                'aggratio': ('$\\beta_{-2}$/$\\beta_2$', None)}
    
    return param_text


def get_avg_pigs():
    
    names = ('but', 'hex', 'allo', 'chla', 'chlb', 'fuco', 'peri', 'zea')
    avg_pigs = {s: {n: {} for n in names} for s in station_data}

    pig_data = pd.read_csv('../../data/geotraces/pigments.csv',
                           usecols=['station', 'depth', 'but', 'hex', 'allo',
                                    'chla', 'chlb', 'fuco', 'peri', 'zea'])
    
    for s in station_data:
        depth = station_data[s]['zg']
        s_df = pig_data.loc[(pig_data['station'] == s) & (pig_data['depth'] <= depth)]
        for n in names:
            avg_pigs[s][n] = s_df[n].mean()
    
    return avg_pigs


def get_ml_nuts():
    
    names = ('nitrate', 'phosphate', 'silicate')
    ml_nuts = {s: {n: {} for n in names} for s in station_data}

    # ODF data
    nut_data = pd.read_csv('../../data/geotraces/bottledata.csv',
                           usecols=['STNNBR',
                                    'CTDPRS',
                                    'NITRATE_D_CONC_BOTTLE_bugat8',
                                    'Flag_NITRATE_D_CONC_BOTTLE_bugat8',
                                    'PHOSPHATE_D_CONC_BOTTLE_d0rgav',
                                    'Flag_PHOSPHATE_D_CONC_BOTTLE_d0rgav',
                                    'SILICATE_D_CONC_BOTTLE_3fot83',
                                    'Flag_SILICATE_D_CONC_BOTTLE_3fot83',])
    nut_data = nut_data.rename({'STNNBR': 'station',
                                'CTDPRS': 'depth',
                                'NITRATE_D_CONC_BOTTLE_bugat8': 'nitrate',
                                'Flag_NITRATE_D_CONC_BOTTLE_bugat8': 'nitrate_flag',
                                'PHOSPHATE_D_CONC_BOTTLE_d0rgav': 'phosphate',
                                'Flag_PHOSPHATE_D_CONC_BOTTLE_d0rgav': 'phosphate_flag',
                                'SILICATE_D_CONC_BOTTLE_3fot83': 'silicate',
                                'Flag_SILICATE_D_CONC_BOTTLE_3fot83': 'silicate_flag',}, axis='columns')
    nut_data.replace('nd', np.nan, inplace=True)
    nut_data = nut_data.apply(pd.to_numeric)
    
    for s in station_data:
        ml = station_data[s]['mld']
        s_df = nut_data.loc[(nut_data['station'] == s) & (nut_data['depth'] <= ml)]
        for n in names:
            n_df = s_df.loc[s_df[f'{n}_flag'] < 3]
            ml_nuts[s][n] = n_df[n].mean()
    
    return ml_nuts


def get_station_color(station):
    
    if station < 9:
        c = green
    elif station < 28:
        c = orange
    elif station < 34:
        c = vermillion
    else:
        c = blue
    
    return c


def flux_pigs_scatter():

    pig_data = get_avg_pigs()
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    
    axs[0][0].set_ylabel('EZ flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[1][0].set_ylabel('Export efficiency', fontsize=14)
    
    for i, ax in enumerate(axs.flatten()):
        if i % 4:
            ax.yaxis.set_ticklabels([])
        if i < 4:
            ax.xaxis.set_ticklabels([])
    
    axs[1][0].set_xlabel('Chl. a (ng L$^{-1}$)', fontsize=14)
    axs[1][1].set_xlabel('Frac. pico', fontsize=14)
    axs[1][2].set_xlabel('Frac. nano', fontsize=14)
    axs[1][3].set_xlabel('Frac. micro', fontsize=14)
    
    stations = list(station_data.keys())
    chla_bs = []  # by station
    pico_bs = []
    nano_bs = []
    micro_bs = []
    zg_fluxes_bs = []
    xport_effs_bs = []

    for s in stations:
        c = get_station_color(s)
        grid = np.array(station_data[s]['grid'])
        zg = station_data[s]['zg']
        zgi = list(grid).index(zg)
        zg_fluxes = []
        xport_effs = []
        with h5py.File(output_fp, 'r') as f:
            inversions = list(f[str(s)].keys())
            for i in inversions:
                ws = f[str(s)][i]['ws'][()]   
                ps = f[str(s)][i]['POCS'][()]
                wl = f[str(s)][i]['wl'][()]
                pl = f[str(s)][i]['POCL'][()]
                fluxes = ws*ps + wl*pl  # sinkflux_T
                zg_fluxes.append(fluxes[zgi])
                Lp = f[str(s)][i]['Lp'][()]
                Po = f[str(s)][i]['Po'][()]
                npp = 0
                for layer in range(zgi + 1):
                    zi, zim1 = get_layer_bounds(layer, grid)
                    npp += Lp * Po * (np.exp(-zim1 / Lp) - np.exp(-zi / Lp))
                xport_effs.append(fluxes[zgi] / npp)
                
        zg_flux = np.mean(zg_fluxes)
        xport_eff = np.mean(xport_effs)
        
        pico, nano, micro = phyto_size_index(pig_data[s])

        for i, ydata in enumerate((zg_flux, xport_eff)):
            axs[i][0].scatter(pig_data[s]['chla'], ydata, s=16, color=c, zorder=2)
            axs[i][1].scatter(pico, ydata, s=16, color=c, zorder=2)  
            axs[i][2].scatter(nano, ydata, s=16, color=c, zorder=2)  
            axs[i][3].scatter(micro, ydata, s=16, color=c, zorder=2)
        
        zg_fluxes_bs.append(zg_flux)
        xport_effs_bs.append(xport_eff)
        
        chla_bs.append(pig_data[s]['chla'])
        pico_bs.append(pico)
        nano_bs.append(nano)
        micro_bs.append(micro)

    lines, labels, line_length = get_station_color_legend()
    axs[0][0].legend(lines, labels, frameon=False, handlelength=line_length)

    xdatas = (chla_bs, pico_bs, nano_bs, micro_bs)
    for j, xdata0 in enumerate(xdatas):
        ydata0 = zg_fluxes_bs
        
        xdata1 = [x for i, x in enumerate(xdata0) if stations[i] > 9]
        ydata1 = [y for i, y in enumerate(ydata0) if stations[i] > 9]
        
        reg0 = sm.OLS(ydata0, sm.add_constant(xdata0)).fit()
        y_fit0 = reg0.predict()

        reg1 = sm.OLS(ydata1, sm.add_constant(xdata1)).fit()
        y_fit1 = reg1.predict()
        
        if j != 1:
            axs[0][j].plot(np.sort(xdata0), np.sort(y_fit0), c=gray, zorder=1)
            axs[0][j].plot(np.sort(xdata1), np.sort(y_fit1), c=gray, ls=':', zorder=1)
            axs[0][j].text(0.68, 0.02, f'{reg0.rsquared:.2f} ({reg0.f_pvalue:.2f})\n{reg1.rsquared:.2f} ({reg1.f_pvalue:.2f})',
                            transform=transforms.blended_transform_factory(axs[0][j].transAxes, axs[0][j].transAxes))
        else:
            axs[0][j].plot(np.sort(xdata0), np.sort(y_fit0)[::-1], c=gray, zorder=1)
            axs[0][j].plot(np.sort(xdata1), np.sort(y_fit1)[::-1], c=gray, ls=':', zorder=1)
            axs[0][j].text(0.02, 0.02, f'{reg0.rsquared:.2f} ({reg0.f_pvalue:.2f})\n{reg1.rsquared:.2f} ({reg1.f_pvalue:.2f})',
                            transform=transforms.blended_transform_factory(axs[0][j].transAxes, axs[0][j].transAxes))

    for ax in axs.flatten()[4:]:
        ax.set_ylim(0)

    fig.savefig(f'../../results/geotraces/figs/Figure9.pdf', bbox_inches='tight')
    plt.close()


def agg_pigs_scatter():

    dv_df = compile_param_estimates_dv()

    params_df = dv_df[['depth', 'station', 'aggratio', 'Bm2', 'B2', 'B2p']].copy()
    mean_params = params_df.groupby(['depth', 'station']).mean().reset_index()

    dc_df = compile_param_estimates_dc()    
    
    npp_df = dc_df[['station', 'Po', 'Lp']]
    mean_npp = npp_df.groupby(['station']).mean().reset_index()

    param_text = get_param_text()
    pig_data = get_avg_pigs()
    fig, axs = plt.subplots(4, 5, figsize=(16, 12))
    
    for i, ax in enumerate(axs.flatten()):
        if i % 5:
            ax.yaxis.set_ticklabels([])
        if i < 15:
            ax.xaxis.set_ticklabels([])
    
    axs[3][0].set_xlabel('Integrated NPP (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[3][1].set_xlabel('Chl. a (ng L$^{-1}$)', fontsize=14)
    axs[3][2].set_xlabel('Frac. pico', fontsize=14)
    axs[3][3].set_xlabel('Frac. nano', fontsize=14)
    axs[3][4].set_xlabel('Frac. micro', fontsize=14)

    axs[0][0].set_ylabel(f"{param_text['B2p'][0]} ({param_text['B2p'][1]})", fontsize=14)
    axs[1][0].set_ylabel(f"{param_text['B2'][0]} ({param_text['B2'][1]})", fontsize=14)
    axs[2][0].set_ylabel(f"{param_text['Bm2'][0]} ({param_text['Bm2'][1]})", fontsize=14)
    axs[3][0].set_ylabel(param_text['aggratio'][0], fontsize=14)
    
    stations = list(station_data.keys())
    npp_bs = []  # by station
    chla_bs = []
    pico_bs = []
    nano_bs = []
    micro_bs = []
    Bm2_bs = []
    Bm2_e_bs = []
    B2_bs = []
    B2_e_bs = []
    B2p_bs = []
    B2p_e_bs = []
    ratio_bs = []
    ratio_e_bs = []

    for s in stations:
        
        depth = station_data[s]['zg']
        c = get_station_color(s)
        s_df_params = mean_params[mean_params['station'] == s]
        s_df_npp = mean_npp[mean_npp['station'] == s].iloc[0]
        Lp = s_df_npp['Lp']
        Po = s_df_npp['Po']
        
        if len(s_df_params.loc[s_df_params['depth'] <= depth]) == 0:
            continue
        
        Bm2 = s_df_params.loc[s_df_params['depth'] <= depth]['Bm2'].mean()
        Bm2_e = s_df_params.loc[s_df_params['depth'] <= depth]['Bm2'].std(ddof=1)
        B2 = s_df_params.loc[s_df_params['depth'] <= depth]['B2'].mean()
        B2_e = s_df_params.loc[s_df_params['depth'] <= depth]['B2'].std(ddof=1)
        B2p = s_df_params.loc[s_df_params['depth'] <= depth]['B2p'].mean()
        B2p_e = s_df_params.loc[s_df_params['depth'] <= depth]['B2p'].std(ddof=1)
        ratio = s_df_params.loc[s_df_params['depth'] <= depth]['aggratio'].mean()
        ratio_e = s_df_params.loc[s_df_params['depth'] <= depth]['aggratio'].std(ddof=1)
        
        npp = Lp * Po * (1 - np.exp(-depth / Lp))

        Bm2_bs.append(Bm2)
        Bm2_e_bs.append(Bm2_e)
        B2_bs.append(B2)
        B2_e_bs.append(B2_e)
        B2p_bs.append(B2p)
        B2p_e_bs.append(B2p_e)
        ratio_bs.append(ratio)
        ratio_e_bs.append(ratio_e)
        
        pico, nano, micro = phyto_size_index(pig_data[s])
        

        for i, (ydata, yerr) in enumerate(((B2p, B2p_e), (B2, B2_e), (Bm2, Bm2_e), (ratio, ratio_e))):
            axs[i][0].errorbar(npp, ydata, yerr=yerr, c=c, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[i][1].errorbar(pig_data[s]['chla'], ydata, yerr=yerr, c=c, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[i][2].errorbar(pico, ydata, yerr=yerr, c=c, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[i][3].errorbar(nano, ydata, yerr=yerr, c=c, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[i][4].errorbar(micro, ydata, yerr=yerr, c=c, fmt='o', elinewidth=1, ms=4, capsize=2)
        
        npp_bs.append(npp)
        chla_bs.append(pig_data[s]['chla'])
        pico_bs.append(pico)
        nano_bs.append(nano)
        micro_bs.append(micro)

    lines, labels, line_length = get_station_color_legend()
    axs[0][0].legend(lines, labels, frameon=False, handlelength=line_length, loc='lower left')

    ydatas = ((B2p_bs, B2_bs, Bm2_bs, ratio_bs))
    xdatas = (npp_bs, chla_bs, pico_bs, nano_bs, micro_bs)
    
    for (i, ydata0), (j, xdata0) in product(enumerate(ydatas), enumerate(xdatas)):
        
        xdata1 = [x for i, x in enumerate(xdata0) if stations[i] > 9]
        ydata1 = [y for i, y in enumerate(ydata0) if stations[i] > 9]
        
        reg0 = sm.OLS(ydata0, sm.add_constant(xdata0)).fit()
        yfit0 = reg0.predict()
        xdata0_sort = np.sort(xdata0)
        yfit0_sort = np.sort(yfit0)
        if reg0.params[1] < 0:  #if slope is negative
            yfit0_sort = yfit0_sort[::-1]
        axs[i][j].plot(xdata0_sort, yfit0_sort, c=gray, ls='-')

        reg1 = sm.OLS(ydata1, sm.add_constant(xdata1)).fit()
        yfit1 = reg1.predict()
        xdata1_sort = np.sort(xdata1)
        yfit1_sort = np.sort(yfit1)
        if reg1.params[1] < 0:  #if slope is negative
            yfit1_sort = yfit1_sort[::-1]
        axs[i][j].plot(xdata1_sort, yfit1_sort, c=gray, ls=':')

        axs[i][j].text(0.5, 0.94, f'{reg0.rsquared:.2f} ({reg0.f_pvalue:.2f})',
                            transform=transforms.blended_transform_factory(axs[i][j].transAxes, axs[i][j].transAxes))
        axs[i][j].text(0.5, 0.94 - 0.07, f'{reg1.rsquared:.2f} ({reg1.f_pvalue:.2f})',
                            transform=transforms.blended_transform_factory(axs[i][j].transAxes, axs[i][j].transAxes))

        
    fig.savefig(f'../../results/geotraces/figs/Figure10.pdf', bbox_inches='tight')
    plt.close()


def phyto_size_index(d):
    '''r is a row of the pig data, formulas from Bricaud et al 2004'''
    dp = (0.86 * d['zea'] + 1.01 * d['chlb'] + 0.6 * d['allo']
            + 0.35 * d['but'] + 1.27 * d['hex'] + 1.41 * d['fuco']
            + 1.41 * d['peri'])
    pico = (0.86 * d['zea'] + 1.01 * d['chlb']) / dp
    nano = (0.6 * d['allo'] + 0.35 * d['but'] + 1.27 * d['hex']) / dp
    micro = (1.41 * d['fuco'] + 1.41 * d['peri']) / dp

    return pico, nano, micro


def multipanel_context():
    
    pig_data = get_avg_pigs()
    nut_data = get_ml_nuts()
    
    ylabels = {0: 'Nitrate\n(µmol kg$^{-1}$)',
               1: 'Silicate\n(µmol kg$^{-1}$)',
               2: 'Phosphate\n(µmol kg$^{-1}$)',
               3: 'Chl. a\n(ng L$^{-1}$)',
               4: 'Frac. pico', 5: 'Frac. nano', 6: 'Frac. micro'}

    fig, axs = plt.subplots(7, 1, figsize=(5, 10), tight_layout=True)
    fig.subplots_adjust(left=0.2)

    for s in station_data:

        lat = station_data[s]['latitude']
        station_color = get_station_color(s)
        
        pico, nano, micro = phyto_size_index(pig_data[s])
        
        # calculate flux data
        grid = np.array(station_data[s]['grid'])
        zg = station_data[s]['zg']
        zgi = list(grid).index(zg)
        zgp100 = zg + 100
        interp_depths = grid[grid < zgp100].max(), grid[grid > zgp100].min()  # pump depths that surround zgp100
        zg_fluxes = []
        xfer_effs = []
        xport_effs = []

        with h5py.File(output_fp, 'r') as f:
            inversions = list(f[str(s)].keys())
            for i in inversions:
                ws = f[str(s)][i]['ws'][()]   
                ps = f[str(s)][i]['POCS'][()]
                wl = f[str(s)][i]['wl'][()]
                pl = f[str(s)][i]['POCL'][()]
                fluxes = ws*ps + wl*pl  # sinkflux_T
                zg_fluxes.append(fluxes[zgi])
                interp_fluxes = [fluxes[list(grid).index(i)] for i in interp_depths]
                interped_flux = interp1d(interp_depths, interp_fluxes)(zgp100)
                xfer_effs.append(interped_flux[()] / fluxes[zgi])
                Lp = f[str(s)][i]['Lp'][()]
                Po = f[str(s)][i]['Po'][()]
                npp = 0
                for layer in range(zgi + 1):
                    zi, zim1 = get_layer_bounds(layer, grid)
                    npp += Lp * Po * (np.exp(-zim1 / Lp) - np.exp(-zi / Lp))
                xport_effs.append(fluxes[zgi] / npp)
        
        axs[0].scatter(lat, nut_data[s]['nitrate'], c=station_color, s=16, zorder=2)
        axs[1].scatter(lat, nut_data[s]['silicate'], c=station_color, s=16, zorder=2)
        axs[2].scatter(lat, nut_data[s]['phosphate'], c=station_color, s=16, zorder=2)
        axs[3].scatter(lat, pig_data[s]['chla'], c=station_color, s=16, zorder=2)
        axs[4].scatter(lat, pico, c=station_color, s=16, zorder=2)
        axs[5].scatter(lat, nano, c=station_color, s=16, zorder=2)
        axs[6].scatter(lat, micro, c=station_color, s=16, zorder=2)

        for ax in axs:  # faint gridlines
            ax.axvline(lat, c=black, alpha=0.2, zorder=1)
        
    for i, ax in enumerate(axs):
        ax.set_ylabel(ylabels[i])
        ax.invert_xaxis()
        if i < len(ylabels) - 1:
            ax.tick_params(axis='x',label1On=False)
        else:
            ax.set_xlabel('Latitude (°N)')
        for s in station_data:
            ax.text(station_data[s]['latitude'], 1.02, s, ha='center', size=6, transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
    
    axs[3].set_ylim(0, 500)
    axs[4].set_ylim(0, 0.7)
    axs[5].set_ylim(0, 0.6)
    axs[6].set_ylim(0, 0.4)

    fig.savefig(f'../../results/geotraces/figs/Figure4.pdf', bbox_inches='tight')
    plt.close()


def section_map():

    fig = plt.figure(figsize=(5, 7))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), color=black, zorder=2)
    ax.set_extent([-162, -142, -25, 62], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=['left'], zorder=1)
    gl.xlines = False

    for s in station_data:
        c = get_station_color(s)
        ax.scatter(station_data[s]['longitude'], station_data[s]['latitude'], color=c, s=20, zorder=3)
        ax.annotate(s, (station_data[s]['longitude'], station_data[s]['latitude']), fontsize=6, xytext=(station_data[s]['longitude']+0.75, station_data[s]['latitude']))
        
    plt.savefig('../../results/geotraces/figs/Figure1.pdf', bbox_inches='tight')
    plt.close()


def param_barplots():
    
    df = compile_param_estimates_dv()  

    params = ('B2p', 'B2', 'Bm2', 'aggratio', 'Bm1s', 'Bm1l', 'ws', 'wl')
    
    param_text = get_param_text()
    
    stations = np.sort(list(station_data.keys()))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(7, 12), tight_layout=True)
    axs = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
    fig.subplots_adjust(wspace=0.1, hspace=0.5, left=0.15)

    with open('../../results/geotraces/figs/param_barplots.txt', 'w') as sys.stdout:
        for i, p in enumerate(params):
            print(f'***********{p}***********')
            d = {'EZ': {'subarctic': [], 'npac': [], 'eq': [], 'spac': [], 'total': []},
                'UMZ': {'subarctic': [], 'npac': [], 'eq': [], 'spac': [], 'total': []}}
            p_df = df[['depth', 'station', p]]
            mean = p_df.groupby(['depth', 'station']).mean().reset_index()
            for s in stations:
                zg = station_data[s]['zg']
                s_df = mean.loc[mean['station'] == s]
                ez = s_df.loc[s_df['depth'] <= zg][p]
                uz = s_df.loc[s_df['depth'] > zg][p]
                d['EZ']['total'].extend(ez.values)
                d['UMZ']['total'].extend(uz.values)
                if s < 9:
                    k = 'subarctic'
                elif s < 28:
                    k = 'npac'
                elif s < 34:
                    k = 'eq'
                else:
                    k = 'spac'
                d['EZ'][k].extend(ez.values)
                d['UMZ'][k].extend(uz.values)

            if param_text[p][1]:
                units = f' ({param_text[p][1]})'
            else:
                units = ''
            axs[i].set_ylabel(f'{param_text[p][0]}{units}', fontsize=14)
            
            bar_means = []
            bar_stds = []
            bar_colors = []
            bar_hatches = []
            width = 0.6

            for zone, regime in product(('EZ', 'UMZ'), ('subarctic', 'npac', 'eq', 'spac', 'total')):
                print(f'----{zone, regime}----')
                z_r_n = len(d[zone][regime])
                z_r_mean = np.mean(d[zone][regime])
                z_r_std = np.std(d[zone][regime], ddof=1)/np.sqrt(z_r_n)
                print(f'N = {z_r_n}')
                print(f'mean = {z_r_mean}')
                print(f'std = {z_r_std}')
                bar_means.append(z_r_mean)
                bar_stds.append(z_r_std)
                if regime == 'subarctic':
                    bar_colors.append(green)
                elif regime == 'npac':
                    bar_colors.append(orange)
                elif regime == 'eq':
                    bar_colors.append(vermillion)
                elif regime == 'spac':
                    bar_colors.append(blue)
                else:
                    bar_colors.append(gray)
                if zone == 'EZ':
                    bar_hatches.append(None)
                else:
                    bar_hatches.append('.')
                
            axs[i].bar(np.arange(len(bar_means)), bar_means, width, yerr=bar_stds, color=bar_colors, error_kw={'elinewidth': 1}, zorder=10, hatch=bar_hatches)
            axs[i].xaxis.set_ticks([])
            
    sys.stdout = sys.__stdout__
    
    lines, labels, line_length = get_station_color_legend(all_regimes=True)
    axs[1].legend(lines, labels, frameon=False, handlelength=line_length)
        
    fig.savefig('../../results/geotraces/figs/Figure8.pdf', bbox_inches='tight')
    plt.close()
            

if __name__ == '__main__':

    start_time = time()

    poc_data = data.poc_by_station()
    param_uniformity = data.define_param_uniformity()
    Lp_priors = data.get_Lp_priors(poc_data)
    ez_depths = data.get_ez_depths(Lp_priors)
    station_data = data.get_station_data(poc_data, param_uniformity, ez_depths,
                                         flux_constraint=True)

    output_fp = '../../results/geotraces/output.h5'

    multipanel_context()
    flux_pigs_scatter()
    agg_pigs_scatter()
    param_section_compilation_dc()
    param_section_compilation_dv()
    ctd_plots_remin()
    ctd_plots_sink()
    spaghetti_params()
    spaghetti_ctd()
    spaghetti_poc()
    poc_section()
    section_map()
    param_barplots()
    
    print(f'--- {(time() - start_time)/60} minutes ---')
