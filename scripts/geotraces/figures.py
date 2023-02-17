import os
import pickle
from itertools import product
from time import time
import sys

import cartopy.crs
import gsw
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist import Axes
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import src.geotraces.data as data
from src.colors import *
from src.modelequations import get_layer_bounds

def load_data(path):
    """Loads data and returns dfs with all sets and successful sets."""
    with open(os.path.join(path, 'table.pkl'), 'rb') as f:
        sets = pickle.load(f)
    good = sets.loc[sets['set_success'] == True].copy()

    return sets, good


def plot_histograms(path, df, params, file_prefix):
    """Plot histogram of values for all parameters in a df."""
    for p in params:
        data = df[p]
        plt.subplots(tight_layout=True)
        plt.hist(data, bins=30)
        plt.savefig(os.path.join(path, f'{file_prefix}_{p}'))
        plt.close()

def stacked_histograms(path, df, params):
    """Stacked histograms for after param sets have been clustered."""
    n_clusters = len(df['label'].unique())
    for p in params:
        df.pivot(columns='label')[p].plot(kind='hist', stacked=True, bins=30)
        plt.savefig(os.path.join(path, f'stackedhist_{n_clusters}_{p}'))
        plt.close()


def pairplot(path, df):

    df = df[['B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl']]
    sns.pairplot(df)
    plt.savefig(os.path.join(path, 'pairplot'))
    plt.close()

def hist_success(path, filenames):
    """Plot # of succesful inversions for each station."""
    stations = list(data.poc_by_station().keys())
    stations.sort()
    d = {s: len([i for i in filenames if f'stn{s}.pkl' in i]) for s in stations}
    plt.bar(range(len(d)), list(d.values()), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.savefig(os.path.join(path, 'figs/hist_success'))
    plt.close()

def hist_stats(path, filenames, suffix=''):
    """boxplots of param posteriors"""
    stations = list(data.poc_by_station().keys())
    stations.sort()
    
    dv_params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    dc_params = ('Po', 'Lp', 'zm', 'a', 'B3')
    
    d = {p: {s: {e: [] for e in ('prior', 'posterior')} for s in stations} for p in dv_params + dc_params}
    
    for f in filenames:
        s = int(f.split('.')[0].split('_')[1][3:])
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
        for p in dv_params:
            d[p][s]['posterior'].extend(results['params'][p]['posterior'])
            d[p][s]['prior'].append(results['params'][p]['prior'])
        for p in dc_params:
            d[p][s]['posterior'].append(results['params'][p]['posterior'])
            d[p][s]['prior'].append(results['params'][p]['prior'])
    
    for p in dc_params + dv_params:
        fig, ax = plt.subplots(tight_layout=True)
        ax.boxplot([d[p][s]['posterior'] for s in stations], positions=range(len(stations)))
        ax.set_xticks(range(len(stations)), d[p].keys())
        if p in dc_params:
            ax.plot(range(len(stations)), [d[p][s]['prior'][0] for s in stations], marker='*', c='b', ls='None')
        else:
            ax.plot(range(len(stations)), [min(d[p][s]['prior']) for s in stations], marker='*', c='b', ls='None')
            ax.plot(range(len(stations)), [max(d[p][s]['prior']) for s in stations], marker='*', c='b', ls='None')
            
        fig.savefig(os.path.join(path, f'figs/hist_{p}{suffix}'))
        plt.close()


def stationparam_hists(path, params, filenames):
    
    dv_params = params
    dc_params = ('Po', 'Lp', 'B3', 'a', 'zm')
    all_params = dv_params + dc_params
    stations = data.poc_by_station().keys()
    data = {s: {p: {'priors': [], 'posteriors': []} for p in all_params} for s in stations}

    for f in filenames:
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
            _, stn = f.split('.')[0].split('_')
            s = int(stn[3:])
            for p in all_params:
                data[s][p]['priors'].append(results['params'][p]['prior'])
                if p in dv_params:
                    data[s][p]['posteriors'].extend(results['params'][p]['posterior'])
                else:
                    data[s][p]['posteriors'].append(results['params'][p]['posterior'])
    
    for (s, p) in product(stations, dv_params):
        _, axs = plt.subplots(1, 2, tight_layout=True)
        axs[0].hist(data[s][p]['priors'], bins=30)
        axs[1].hist(data[s][p]['posteriors'], bins=30)
        x_lo = min([a.get_xlim()[0] for a in axs])
        x_hi = max([a.get_xlim()[1] for a in axs])
        for a in axs:
            a.set_xlim(x_lo, x_hi)
        plt.savefig(os.path.join(path, f'figs/sp_hist_{s}_{p}'))
        plt.close()

    for (s, p) in product(stations, dc_params):
        _, ax = plt.subplots(tight_layout=True)
        ax.axvline(data[s][p]['priors'][0], c='k')
        ax.hist(data[s][p]['posteriors'], bins=30)
        plt.savefig(os.path.join(path, f'figs/sp_hist_{s}_{p}'))
        plt.close()

def get_filenames(path, successful_sets=False):

    pickled_files = [f for f in os.listdir(path) if 'stn' in f]
    pickled_files.sort(key = lambda x: int(x.split('_')[0][2:]))

    if successful_sets == True:
        stations = data.poc_by_station().keys()
        filenames = []
        set_number = 0
        set_counter = 0
        for i, f in enumerate(pickled_files):
            f_set = int(f.split('_')[0][2:])
            if f_set == set_number:
                set_counter += 1
                if set_counter == len(stations):
                    filenames.extend(pickled_files[i+1-len(stations):i+1])
            else:
                set_number += 1
                set_counter = 1
        print(f'N successful sets: {len(filenames) / len(stations)}')
        print(f'N successful inversions: {len(pickled_files)}')
    else:
        filenames = pickled_files

    return filenames

def xresids(path, station_data):
    
    df = pd.DataFrame(columns=['resid', 'element'])

    for stn in poc_data.keys():
        resids = []
        elements = []
        pickled_files = [f for f in os.listdir(path) if f'stn{stn}.pkl' in f]
        for f in pickled_files:
            with open(os.path.join(path, f), 'rb') as file:
                results = pickle.load(file)
                _, stn = f.split('.')[0].split('_')
                s = int(stn[3:])
                resids.extend(results['x_resids'])
                elements.extend([s.split('_')[0] for s in station_data[s]['s_elements']])

        _, axs = plt.subplots(1, 2, tight_layout=True)
        
        df = pd.DataFrame({'resid': resids, 'element': elements})
        dv_elements = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl', 'POCS', 'POCL')
        df1 = df[df['element'].isin(dv_elements)]
        df1_piv = df1.pivot(columns='element')['resid'].astype(float)
        df2 = df[~df['element'].isin(dv_elements)]
        df2_piv = df2.pivot(columns='element')['resid'].astype(float)
        df1_piv.plot(kind='hist', stacked=True, bins=30, ax=axs[0])
        df2_piv.plot(kind='hist', stacked=True, bins=30, ax=axs[1])
        plt.savefig(os.path.join(path, f'figs/xresids_{stn}'))
        plt.close()

def compile_param_estimates(filenames):
    
    dv_params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    dc_params = ('Po', 'Lp', 'zm', 'a', 'B3')

    dv_rows = []
    dc_rows = []

    for f in tqdm(filenames):
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
            stn = int(f.split('.')[0].split('_')[1][3:])
            dv_dict = {p: results['params'][p]['posterior'] for p in dv_params}
            dc_dict = {p: [results['params'][p]['posterior']] for p in dc_params}
            dv_dict['depth'] = station_data[stn]['grid']
            dv_dict['avg_depth'] = [np.mean(get_layer_bounds(l, station_data[stn]['grid'])) for l in station_data[stn]['layers']]
            dv_dict['latitude'] = station_data[stn]['latitude'] * np.ones(len(station_data[stn]['grid']))
            dc_dict['latitude'] = [station_data[stn]['latitude']]
            dv_dict['station'] = stn * np.ones(len(station_data[stn]['grid']))
            dc_dict['station'] = [stn]
            avg_ps = []
            ps_post = results['tracers']['POCS']['posterior']
            for i, ps in enumerate(ps_post):
                if i == 0:
                    avg_ps.append(ps)
                else:
                    avg_ps.append(np.mean([ps, ps_post[i-1]]))
            dv_dict['B2'] = dv_dict['B2p']*np.array(avg_ps)
            dv_dict['aggratio'] = dv_dict['Bm2']/dv_dict['B2']
            dv_rows.append(pd.DataFrame(dv_dict))
            dc_rows.append(pd.DataFrame(dc_dict))
    dv_df = pd.concat(dv_rows, ignore_index=True)
    dc_df = pd.concat(dc_rows, ignore_index=True)
    
    with open(os.path.join(path, 'saved_params_dv.pkl'), 'wb') as f:
        pickle.dump(dv_df, f)
    with open(os.path.join(path, 'saved_params_dc.pkl'), 'wb') as f:
        pickle.dump(dc_df, f)


def cluster_means(path, saved_params):

    def elbowplot(df):
        
        inertia = []
        k_vals = range(1, 21)
        for i in k_vals:
            kmeans = KMeans(n_clusters=i, random_state=0)
            result = kmeans.fit(df)
            inertia.append(result.inertia_)

        plt.figure()
        plt.plot(k_vals, inertia, marker='o', ls='--')
        plt.xlabel('Number of Clusters')
        plt.xticks(k_vals)
        plt.ylabel('Inertia')
        plt.savefig(os.path.join(path, 'figs/elbowplot'))
        plt.close()
        
    with open(os.path.join(path, saved_params), 'rb') as f:
        df = pickle.load(f)
        merge_on = ['depth', 'latitude', 'station']
        means = df.groupby(merge_on).mean().reset_index()
        means_dropped = means.drop(merge_on, axis=1)
        means_scaled = StandardScaler().fit_transform(means_dropped)
        
    # labels = hdbscan.HDBSCAN(  # doesn't work very well, N is too low?
    #     min_samples=1,
    #     min_cluster_size=4
    #     ).fit_predict(means_scaled)
    # means['cluster'] = labels
    
    kmeans = KMeans(n_clusters=4, random_state=0)  # elbowplot suggests 5-7
    labels = kmeans.fit(means_scaled).labels_

    means['cluster'] = labels
    scheme = plt.cm.tab10
    lats = [station_data[s]['latitude'] for s in station_data]
    mlds_unsorted = [station_data[s]['mld'] for s in station_data]
    zgs_unsorted = [station_data[s]['zg'] for s in station_data]
    mlds = [mld for _, mld in sorted(zip(lats, mlds_unsorted))]
    zgs = [zg for _, zg in sorted(zip(lats, zgs_unsorted))]
    lats.sort()
    
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Latitude (°N)', fontsize=14)
    ax.plot(lats, mlds, c='k', zorder=1, ls='--')
    ax.plot(lats, zgs, c='k', zorder=1)
    cbar_label = 'Cluster'
    for s, d in station_data.items():
        ax.text(d['latitude'], -30, s, ha='center', size=6)

    bounds = np.arange(min(labels), max(labels) + 2, 1)
    bounds_mid = (bounds[1:] + bounds[:-1]) / 2
    norm = BoundaryNorm(bounds, scheme.N)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01, ticks=bounds_mid)
    cbar.ax.set_yticklabels(bounds[:-1])
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=14)
    ax.scatter(means['latitude'], means['depth'], c=means['cluster'],norm=norm, cmap=scheme, zorder=10)
    for s in station_data: # plot sampling depths
        depths = []
        for l in station_data[s]['layers']:
            depths.append(np.mean(get_layer_bounds(l, station_data[s]['grid'])))
        ax.scatter(np.ones(len(depths))*station_data[s]['latitude'], depths, c='k', zorder=1, s=1)
    fig.savefig(os.path.join(path, f'figs/clusteredsection.pdf'))
        

def param_sections_dv(path, station_data):
    
    param_text = get_param_text()
    
    cbar_limits= {'Mean': {'B2p': (0, 0.2), 'Bm1l': (0, 0.3),
                           'Bm1s': (0, 0.4), 'Bm2': (0, 3),
                           'wl': (0, 80), 'ws': (0, 1)},
                  'CoV': {'B2p': (0, 0.8), 'Bm1l': (0, 1.6),
                          'Bm1s': (0, 1.6), 'Bm2': (0, 1.2),
                          'wl': (0, 1), 'ws': (0, 1)}}

    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)    

    params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    scheme = plt.cm.viridis
    lats = [station_data[s]['latitude'] for s in station_data]
    mlds_unsorted = [station_data[s]['mld'] for s in station_data]
    zgs_unsorted = [station_data[s]['zg'] for s in station_data]
    mlds = [mld for _, mld in sorted(zip(lats, mlds_unsorted))]
    zgs = [zg for _, zg in sorted(zip(lats, zgs_unsorted))]
    lats.sort()
    
    for p in params:
        p_df = df[['depth', 'avg_depth', 'latitude', p]]
        mean = p_df.groupby(['depth', 'avg_depth', 'latitude']).mean().reset_index()
        sd = p_df.groupby(['depth', 'avg_depth', 'latitude']).std().reset_index()
        merged = mean.merge(sd, suffixes=(None, '_sd'), on=['depth', 'avg_depth', 'latitude'])
        merged[f'{p}_cv'] = merged[f'{p}_sd'] / merged[p]
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)
        for i, ax in enumerate(axs):
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=610)
            ax.set_ylabel('Depth (m)', fontsize=14)
            ax.plot(lats, mlds, c='k', zorder=1, ls='--')
            ax.plot(lats, zgs, c='k', zorder=1)
            if i == 0:
                cbar_label = f'{param_text[p][0]} ({param_text[p][1]})'
                cbar_lims = cbar_limits['Mean'][p]
                to_plot = merged[p]
                for s, d in station_data.items():
                    ax.text(d['latitude'], -30, s, ha='center', size=6)
                ax.get_xaxis().set_visible(False)
            else:
                cbar_label = 'CoV'
                cbar_lims = cbar_limits['CoV'][p]
                to_plot = merged[f'{p}_cv']
                ax.set_xlabel('Latitude (°N)', fontsize=14)
            # norm = Normalize(to_plot.min(), to_plot.max())
            norm = Normalize(*cbar_lims)
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)
            cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=14)
            if 'w' not in p:
                depth_str = 'avg_depth'
                for s in station_data: # plot sampling depths
                    depths = station_data[s]['grid']
                    ax.scatter(np.ones(len(depths))*station_data[s]['latitude'], depths, c='k', zorder=1, s=1)
            else:
                depth_str = 'depth'
            ax.scatter(merged['latitude'], merged[depth_str], c=to_plot, norm=norm, cmap=scheme, zorder=10)
        fig.savefig(os.path.join(path, f'figs/section_{p}.pdf'))
        plt.close()


def param_section_compilation_dv(path, station_data):
    
    param_text = get_param_text()

    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)    

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
    fig.savefig(os.path.join(path, f'figs/param_section_compilation_dv.pdf'), bbox_inches='tight')
    plt.close()


def param_section_compilation_dc(path, station_data, filenames):
    
    param_text = get_param_text()

    with open(os.path.join(path, 'saved_params_dc.pkl'), 'rb') as f:
        df = pickle.load(f)    

    params = ('Po', 'Lp', 'zm', 'a', 'B3')
    lats = [station_data[s]['latitude'] for s in station_data]
    
    # get priors
    stations_checked = []
    priors = {p: {} for p in params}
    for f in filenames:
        s = int(f.split('.')[0].split('_')[1][3:])
        if s not in stations_checked:
            with open(os.path.join(path, f), 'rb') as file:
                results = pickle.load(file)
                for p in params:
                    priors[p][s] = (results['params'][p]['prior'], results['params'][p]['prior_e'])
            
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
                    c='silver', fmt='d', zorder=2, elinewidth=1, ecolor='silver', ms=4,
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

    fig.savefig(os.path.join(path, f'figs/param_section_compilation_dc.pdf'), bbox_inches='tight')
    plt.close()


def poc_section(path, poc_data, station_data):
    
    lims = {'POCS': (0.06, 6), 'POCL': (0.004, 2)}
    cbar_labels = {'POCS': '$P_{S}$ (mmol m$^{-3}$)',
                   'POCL': '$P_{L}$ (mmol m$^{-3}$)'}
    scheme = plt.cm.viridis
    lats = [station_data[s]['latitude'] for s in station_data]
    mlds_unsorted = [station_data[s]['mld'] for s in station_data]
    zgs_unsorted = [station_data[s]['zg'] for s in station_data]
    mlds = [mld for _, mld in sorted(zip(lats, mlds_unsorted))]
    zgs = [zg for _, zg in sorted(zip(lats, zgs_unsorted))]
    lats.sort()
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    fig.subplots_adjust(right=0.8)
    
    for ax in axs:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=630)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.plot(lats, mlds, c='k', zorder=1, ls='--')
        ax.plot(lats, zgs, c='k', zorder=1)
    
    axs[0].tick_params(axis='x', label1On=False)
    axs[1].set_xlabel('Latitude (°N)', fontsize=12)
    
    for (ax, tracer) in ((axs[0], 'POCS'), (axs[1], 'POCL')):
        norm = LogNorm(*lims[tracer])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)
        cbar.set_label(cbar_labels[tracer], rotation=270, labelpad=20, fontsize=12)
        for s in station_data:
            ax.text(station_data[s]['latitude'], 1.02, s, ha='center', size=6, transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
            ax.scatter(poc_data[s]['latitude'], poc_data[s]['depth'], c=poc_data[s][tracer], norm=norm, cmap=scheme, zorder=10)

    fig.savefig(os.path.join(path, f'figs/section_poc.pdf'), bbox_inches='tight')
    plt.close()
    

def get_station_color_legend():

    lines = [Line2D([0], [0], color=green, lw=4),
             Line2D([0], [0], color=orange, lw=4),
             Line2D([0], [0], color=vermillion, lw=4),
             Line2D([0], [0], color=blue, lw=4)]
    
    labels = ['Shelf', 'N. Pac', 'Eq.', 'S. Pac']
    
    line_length = 1
    
    return lines, labels, line_length

        
def spaghetti_params(path, station_data):

    param_text = get_param_text()
    
    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)    

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

    lines, labels, line_length = get_station_color_legend()
    axs[0].legend(lines, labels, frameon=False, handlelength=line_length)

    fig.savefig(os.path.join(path, f'figs/spaghetti_params.pdf'), bbox_inches='tight')
    plt.close()


def spaghetti_ctd(path, station_data):

    station_fname = ctd_files_by_station()
    fig, axs = plt.subplots(1, 2, figsize=(6, 5), tight_layout=True)

    # profiles of T, O2, N2, params
    for s in station_fname:
        color = get_station_color(s)
        ctd_df = pd.read_csv(os.path.join('../../../geotraces/ctd', station_fname[s]), header=12)
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

    fig.savefig(os.path.join(path, f'figs/spaghetti_ctd.pdf'), bbox_inches='tight')
    plt.close()


def spaghetti_poc(path, poc_data):
    
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

    fig.savefig(os.path.join(path, f'figs/spaghetti_poc.pdf'), bbox_inches='tight')
    plt.close()


def flux_profiles(path, filenames, station_data):

    df_rows = []

    for f in tqdm(filenames):
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
            stn = int(f.split('.')[0].split('_')[1][3:])
            file_dict = {'depth': station_data[stn]['grid'],
                         'station': stn * np.ones(len(station_data[stn]['grid'])),
                         'sflux': np.array([i[0] for i in results['sink_fluxes']['S']]),
                         'lflux': np.array([i[0] for i in results['sink_fluxes']['L']]),
                         'tflux': np.array([i[0] for i in results['sink_fluxes']['T']]),}
            df_rows.append(pd.DataFrame(file_dict))
    df = pd.concat(df_rows, ignore_index=True)

    mean = df.groupby(['depth', 'station']).mean().reset_index()
    sd = df.groupby(['depth', 'station']).std().reset_index()
    pump_fluxes = mean.merge(sd, suffixes=(None, '_sd'), on=['depth', 'station'])
    
    for s in pump_fluxes['station'].unique():
        pf_s = pump_fluxes[pump_fluxes['station'] == s]
        Th_fluxes = station_data[s]['Th_fluxes']
        zg = station_data[s]['zg']
        mld = station_data[s]['mld']
        _, ax = plt.subplots(tight_layout=True)

        ax.errorbar(pf_s['tflux'], pf_s['depth'], fmt='o', xerr=pf_s['tflux_sd'],
            ecolor=vermillion, c=vermillion, capsize=4, zorder=3,
            label='$w_TP_T$', elinewidth=1.5, capthick=1.5,
            fillstyle='none')
        ax.errorbar(pf_s['sflux'], pf_s['depth'] + 5, fmt='o', xerr=pf_s['sflux_sd'],
            ecolor=blue, c=blue, capsize=4, zorder=3,
            label='$w_SP_S$', elinewidth=1.5, capthick=1.5,
            fillstyle='none')
        ax.errorbar(pf_s['lflux'], pf_s['depth'] - 5, fmt='o', xerr=pf_s['lflux_sd'],
            ecolor=orange, c=orange, capsize=4, zorder=3,
            label='$w_LP_L$', elinewidth=1.5, capthick=1.5,
            fillstyle='none')
        ax.errorbar(Th_fluxes['flux'], Th_fluxes['depth'], fmt='o',
                    xerr=Th_fluxes['flux'], ecolor=green,
                    c=green, capsize=4, zorder=3,
                    label='$^{234}$Th-based', elinewidth=1.5, capthick=1.5,
                    fillstyle='none') 

        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.set_xlabel('Flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=610)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axhline(zg, c='k', ls=':')
        ax.axhline(mld, c='k', ls='--')
        ax.legend(loc='lower right', fontsize=12, handletextpad=0.01)
        
        plt.savefig(os.path.join(path, f'figs/sinkfluxes_stn{int(s)}'))
        plt.close()


def aou_scatter(path, params):
    
    # def calculate_aou(T, S):
    #     """From Weiss 70, p. 726"""
    #     T_K = T + 273.15
    #     A1 = -58.3877
    #     A2 = 85.8079
    #     A3 = 23.8439
    #     B1 = -0.034892
    #     B2 = 0.015568
    #     B3 = -0.0019387
    #     B = np.exp(A1
    #                + A2 * (100 / T_K)
    #                + A3 * np.log(T_K / 100)
    #                + S * (B1 + B2 * (T_K / 100) + B3 * (T_K / 100)**2))
    #     print(B*100)
    
    def calculate_aou(row):
        
        T = row['T']
        P = row['P']
        lat = row['lat']
        lon = row['lon']
        SP = row['SP']  # practical salinity
        
        SA = gsw.conversions.SA_from_SP(SP, P, lon, lat)  # absolute salinity
        CT = gsw.conversions.CT_from_t(SA, T, P)  # conservative temperature
        O2_sol = gsw.O2sol(SA, CT, P, lon, lat)
        AOU = O2_sol - row['O2']
        
        return AOU      

    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)

    odf_data = pd.read_csv('../../../geotraces/ODFpump.csv',
                           usecols=['Station',
                                    'CorrectedMeanDepthm',
                                    'Longitudedegrees_east',
                                    'Latitudedegrees_north',
                                    'CTDPRS_T_VALUE_SENSORdbar',
                                    'CTDTMP_T_VALUE_SENSORdegC',
                                    'CTDOXY_D_CONC_SENSORumolkg',
                                    'CTDSAL_D_CONC_SENSORpss78'])
    odf_data = odf_data.rename({'Station': 'station',
                                'CorrectedMeanDepthm': 'depth',
                                'Longitudedegrees_east': 'lon',
                                'Latitudedegrees_north': 'lat',
                                'CTDPRS_T_VALUE_SENSORdbar': 'P',
                                'CTDTMP_T_VALUE_SENSORdegC': 'T',
                                'CTDOXY_D_CONC_SENSORumolkg': 'O2',
                                'CTDSAL_D_CONC_SENSORpss78': 'SP'}, axis='columns')

    param_means = df.groupby(['depth', 'station']).mean().reset_index()
    merged = param_means.merge(odf_data)

    merged['AOU'] = merged.apply(calculate_aou, axis=1)

    for p in params:
        sns.scatterplot(x=p, y='AOU', data=merged, hue='depth')
        plt.savefig(os.path.join(path, f'figs/aouscatter_{p}'))
        plt.close()
        

def ctd_files_by_station():

    # pigrath (ODF) casts from Jen's POC flux table for all staitons except
    # 8, 14, 29, and 39, which are from GTC
    station_cast = {4: 5, 5: 5, 6: 5, 8: 6, 10: 5, 12: 6, 14: 6, 16: 5,  # 3: 4, 
                    18: 5, 19: 4, 21: 5, 23: 4, 25: 5, 27: 5, 29: 6, 31: 5,
                    33: 5, 35: 5, 37: 5, 39: 6}
    
    station_fname = {}
    fnames = [f for f in os.listdir('../../../geotraces/ctd') if '.csv' in f]  # get filenames for each station
    for f in fnames:
        prefix  = f.split('_')[0]
        station = int(prefix[:3])
        cast = int(prefix[3:])
        if station in station_cast and station_cast[station] == cast:
            station_fname[station] = f
    
    return station_fname



def ctd_plots(path, station_data):
        
    # get mean param df across all stations
    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)
    param_means = df.groupby(['depth', 'avg_depth', 'station']).mean().reset_index()

    station_fname = ctd_files_by_station()
    
    params = ('B2p', 'B2', 'Bm2', 'aggratio')
    param_text = get_param_text()
    
    fig, axs = plt.subplots(4, 2, figsize=(6, 10), tight_layout=True)
    t_axs = [axs.flatten()[i] for i in [0, 2, 4, 6]]
    o_axs = [axs.flatten()[i] for i in [1, 3, 5, 7]]
    
    t_axs[3].set_xlabel('Temperature (°C)')
    o_axs[3].set_xlabel('Dissolved O$_2$ (µmol kg$^{-1}$)')

    for (s, (i, p)) in product(station_fname, enumerate(params)):
        
        color = get_station_color(s)
        
        s_p_df = param_means.loc[param_means['station'] == s][['depth', 'avg_depth', p]]
        ctd_df = pd.read_csv(os.path.join('../../../geotraces/ctd', station_fname[s]), header=12)
        ctd_df.drop([0, len(ctd_df) - 1], inplace=True)  # don't want first and last rows (non-numerical)
        for c in ['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']:
            ctd_df[c] = pd.to_numeric(ctd_df[c])
        ctd_df = ctd_df.loc[ctd_df['CTDPRS'] <= 600]
        ctd_df = ctd_df[['CTDPRS', 'CTDTMP', 'CTDOXY', 'CTDSAL']]

        lat = station_data[s]['latitude']
        ctd_df['depth'] = -gsw.z_from_p(ctd_df['CTDPRS'].values, lat)
        
        if 'w' not in p:
            for j, (_, r) in enumerate(s_p_df.iterrows()):
                
                ydeep, yshal = get_layer_bounds(j, s_p_df['depth'].values)
                ctd_in_layer = ctd_df.loc[(ctd_df['depth'] < ydeep) & (ctd_df['depth'] > yshal)]
                avg_T = ctd_in_layer['CTDTMP'].mean()
                avg_O = ctd_in_layer['CTDOXY'].mean()

                t_axs[i].scatter(avg_T, r[p], s=7, color=color)
                o_axs[i].scatter(avg_O, r[p], s=7, color=color)
        else:
            closest_ctd = pd.merge_asof(s_p_df, ctd_df, on='depth', direction='nearest')
            t_axs[i].scatter(closest_ctd['CTDTMP'], s_p_df[p], s=7, color=color)
            o_axs[i].scatter(closest_ctd['CTDOXY'], s_p_df[p], s=7, color=color)

    for ax in t_axs:
        ax.set_xlim(0, 32)
    for ax in o_axs:
        ax.set_xlim(0, 330)

    for i, p in enumerate(params):
        if param_text[p][1]:
            units = f'\n({param_text[p][1]})'
        else:
            units = ''
        t_axs[i].set_ylabel(f'{param_text[p][0]}{units}', fontsize=14)
        o_axs[i].yaxis.set_ticklabels([])
        if i < 3:
            o_axs[i].xaxis.set_ticklabels([])
            t_axs[i].xaxis.set_ticklabels([])
            
    lines, labels, line_length = get_station_color_legend()
    o_axs[0].legend(lines, labels, frameon=False, handlelength=line_length)
    
    fig.savefig(os.path.join(path, f'figs/ctd_scatterplots.pdf'), bbox_inches='tight')
    plt.close() 

            

def param_profile_distribution(path, param):

    with open(os.path.join(path, 'saved_params_sv.pkl'), 'rb') as f:
        df = pickle.load(f)
    
    for s in df['station'].unique():
        sdf = df[df['station'] == s]
        depths = sdf['depth'].unique()
        _, axs = plt.subplots(len(depths), 1, tight_layout=True, figsize=(5,10))
        for i, d in enumerate(depths):
            ddf = sdf[sdf['depth'] == d]
            axs[i].hist(ddf[param])
            axs[i].set_ylabel(f'{d:.0f} m')
            axs[i].axvline(ddf[param].mean(), c=black, ls='--')
            axs[i].axvline(ddf[param].median(), c=black, ls=':')
        plt.savefig(os.path.join(path, f'figs/ppd_{param}_stn{int(s)}'))
        plt.close()


def sinkflux_zg_boxplots(path, filenames, station_data):

    stations = list(station_data.keys())
    stations.sort()
    
    th234_data = pd.read_csv('../../../geotraces/pocfluxes_from_th234.csv')
    th230_data = pd.read_csv('../../../geotraces/hayes_fluxes.csv')
    
    # th234_data.sort_values(by=['station'], inplace=True)
    # th230_data.sort_values(by=['station'], inplace=True)
    
    d = {s: [] for s in stations}
    
    for f in tqdm(filenames, total=len(filenames)):
        s = int(f.split('.')[0].split('_')[1][3:])
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
            zg_index = station_data[s]['grid'].index(station_data[s]['zg'])
            ws = results['params']['ws']['posterior'][zg_index]
            wl =  results['params']['wl']['posterior'][zg_index]
            ps = results['tracers']['POCS']['posterior'][zg_index]
            pl = results['tracers']['POCL']['posterior'][zg_index]
            d[s].append(ws*ps + wl*pl)
    
    fig, ax = plt.subplots(tight_layout=True)
    ax.boxplot([d[s] for s in stations], positions=stations)
    ax.plot(th234_data['station'], th234_data['ppz'], marker='*', c='b', ls='None', label='Th234')
    ax.plot(th230_data['station'], th230_data['flux'], marker='^', c='r', ls='None', label='Th230')
    ax.set_xticks(stations)
    ax.legend()

    fig.savefig(os.path.join(path, 'figs/sinkflux_zg_boxplots'))
    plt.close()


def compare_ppz_zg_ez():

    stations = list(station_data.keys())
    ppz_data = pd.read_csv('../../../geotraces/PPZstats.csv')
    ppz_dict = dict(zip(ppz_data['Station'], ppz_data['PPZmean']))

    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    fig.subplots_adjust(wspace=0.5)
    scheme = plt.cm.viridis
    norm = Normalize(min(stations), max(stations))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=axs[2], pad=0.01)
    
    axs[0].set_xlabel('PPZ (m)', fontsize=14)
    axs[0].set_ylabel('zg (m)', fontsize=14)

    axs[1].set_xlabel('PPZ (m)', fontsize=14)
    axs[1].set_ylabel('EZ (m)', fontsize=14)

    axs[2].set_xlabel('EZ (m)', fontsize=14)
    axs[2].set_ylabel('zg (m)', fontsize=14)

    for s in stations:  
        axs[0].scatter(ppz_dict[s], station_data[s]['zg'], c=s, norm=norm, cmap=scheme)
        axs[1].scatter(ppz_dict[s], ez_depths[s], c=s, norm=norm, cmap=scheme)
        axs[2].scatter(ez_depths[s], station_data[s]['zg'], c=s, norm=norm, cmap=scheme)
    
    for ax in axs:
        ax.plot(np.linspace(20, 300), np.linspace(20, 300), c='k')

    plt.savefig(f'../../results/geotraces/ppz_zg_ez_compare')
    plt.close()
    
    
def compare_zg_ez():

    stations = list(station_data.keys())

    fig, ax = plt.subplots()
    scheme = plt.cm.viridis
    norm = Normalize(min(stations), max(stations))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), ax=ax, pad=0.01)

    ax.set_xlabel('EZ (m)', fontsize=14)
    ax.set_ylabel('zg (m)', fontsize=14)

    for s in stations:  
        ax.scatter(ez_depths[s], station_data[s]['zg'], c=s, norm=norm, cmap=scheme)
    
    ax.plot(np.linspace(20, 300), np.linspace(20, 300), c='k')

    plt.savefig(os.path.join(path, f'figs/ppz_zg_ez_compare'))
    plt.close()


def total_sinking_flux_check(path, filenames, station_data):
    
    differences = []

    for f in filenames:
        s = int(f.split('.')[0].split('_')[1][3:])
        Th_fluxes = station_data[s]['Th_fluxes']
        with open(os.path.join(path, f), 'rb') as file:
            results = pickle.load(file)
            for i, l in enumerate(Th_fluxes['layer'].values):
                tsf = results['total_sinking_flux']['posterior'][i]
                sum_of_fluxes = results['sink_fluxes']['T'][l][0]
                difference = abs(sum_of_fluxes - tsf)
                differences.append(difference)
    print(max(differences))
    

def plot_Th_flux_data():
    
    flux_data = pd.read_excel('../../../geotraces/gp15_flux.xlsx',
                              usecols=('station_number', 'depth', 'POCFlux1d'))

    diffs = []
    
    for s in poc_data:
        
        fig, ax = plt.subplots(tight_layout=True)
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.set_xlabel('Flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
        ax.axvline(0, c=black, ls='--')
        
        depths = poc_data[s]['depth']
        s_df = flux_data.loc[(flux_data['station_number'] == s) & (flux_data['depth'] < 620)]
        ax.scatter(s_df['POCFlux1d'], s_df['depth'], c=blue)
        
        for d in depths:
            ax.axhline(d, c=black, ls=':')
            nearby = s_df.iloc[(s_df['depth']-d).abs().argsort()[:1]].iloc[0]
            ax.scatter(nearby['POCFlux1d'], nearby['depth'], c=orange)
            diffs.append(abs(d - nearby['depth']))
            # print(s, abs(d - nearby['depth']))
        fig.savefig(f'../../fluxes_stn{s}.png')
        plt.close(fig)
    # plt.hist(diffs)
    # plt.show()


def plot_ctd_data():
    
    # pigrath (ODF) casts from Jen's POC flux table for all staitons except
    # 8, 14, 29, and 39, which are from GTC
    station_cast = {4: 5, 5: 5, 6: 5, 8: 6, 10: 5, 12: 6, 14: 6, 16: 5,
                    18: 5, 19: 4, 21: 5, 23: 4, 25: 5, 27: 5, 29: 6, 31: 5,
                    33: 5, 35: 5, 37: 5, 39: 6}
    
    station_fname = {}

    path = '../../../geotraces/ctd'
    
    # get filenames for each station
    fnames = [f for f in os.listdir(path) if '.csv' in f]
    for f in fnames:
        prefix  = f.split('_')[0]
        station = int(prefix[:3])
        cast = int(prefix[3:])
        if station in station_cast and station_cast[station] == cast:
            station_fname[station] = f

    for s in station_fname:
        
        df = pd.read_csv(os.path.join(path, station_fname[s]), header=12)
        
        df.drop([0, len(df) - 1], inplace=True)  # don't want first and last rows (non-numerical)
        for c in ['CTDPRS', 'CTDTMP', 'CTDOXY']:
            df[c] = pd.to_numeric(df[c])
        df = df.loc[df['CTDPRS'] <= 600]
         
        for c in ['CTDTMP', 'CTDOXY']:
            _, ax = plt.subplots(tight_layout=True)
            ax.set_ylabel('Depth (dbar)', fontsize=14)
            ax.invert_yaxis()
            ax.plot(df[c], df['CTDPRS'], 'b')
            plt.savefig(os.path.join(path, f'figs/{c}_stn{int(s)}'))
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


def get_ml_pigs(station_data):
    
    names = ('but', 'hex', 'allo', 'chla', 'chlb', 'fuco', 'peri', 'zea')
    ml_pigs = {s: {n: {} for n in names} for s in station_data}

    pig_data = pd.read_csv('../../../geotraces/pigments.csv',
                           usecols=['station', 'depth', 'but', 'hex', 'allo',
                                    'chla', 'chlb', 'fuco', 'peri', 'zea'])
    
    for s in station_data:
        ml = station_data[s]['mld']
        s_df = pig_data.loc[(pig_data['station'] == s) & (pig_data['depth'] <= ml)]
        for n in names:
            ml_pigs[s][n] = s_df[n].mean()
    
    return ml_pigs

def get_ml_nuts(station_data):
    
    names = ('nitrate', 'phosphate', 'silicate')
    ml_nuts = {s: {n: {} for n in names} for s in station_data}

    # ODF data
    nut_data = pd.read_csv('../../../geotraces/bottledata.csv',
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

def zg_phyto_scatter(station_data):

    pig_data = get_ml_pigs(station_data)
    fig, axs = plt.subplots(1, 4, figsize=(12, 4), tight_layout=True)
    
    axs[0].set_ylabel('EZ flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    
    for ax in axs.flatten()[1:]:
        ax.yaxis.set_ticklabels([])
    
    axs[0].set_xlabel('Chl. a (ng L$^{-1}$)', fontsize=14)
    axs[1].set_xlabel('Frac. pico', fontsize=14)
    axs[2].set_xlabel('Frac. nano', fontsize=14)
    axs[3].set_xlabel('Frac. micro', fontsize=14)
    
    a0_data0 = {'x': [], 'y': []}
    a1_data0 = {'x': [], 'y': []}
    a2_data0 = {'x': [], 'y': []}
    a3_data0 = {'x': [], 'y': []}

    a0_data1 = {'x': [], 'y': []}
    a1_data1 = {'x': [], 'y': []}
    a2_data1 = {'x': [], 'y': []}
    a3_data1 = {'x': [], 'y': []}
    
    for s in station_data:
        
        c = get_station_color(s)
        
        grid = np.array(station_data[s]['grid'])
        zg = station_data[s]['zg']
        zgi = list(grid).index(zg)
        t_fluxes = []
        pickled_files = [f for f in os.listdir(path) if f'stn{s}.pkl' in f]
        for f in pickled_files:
            with open(os.path.join(path, f), 'rb') as file:
                fluxes = pickle.load(file)['sink_fluxes']
                t_fluxes.append(fluxes['T'][zgi][0])
        t_flux = np.mean(t_fluxes)

        pico, nano, micro = phyto_size_index(pig_data[s])

        axs[0].scatter(pig_data[s]['chla'], t_flux, s=16, color=c)
        axs[1].scatter(pico, t_flux, s=16, color=c)  
        axs[2].scatter(nano, t_flux, s=16, color=c)  
        axs[3].scatter(micro, t_flux, s=16, color=c)
        
        a0_data0['x'].append(pig_data[s]['chla'])
        a1_data0['x'].append(pico)
        a2_data0['x'].append(nano)
        a3_data0['x'].append(micro)
        
        for a in (a0_data0, a1_data0, a2_data0, a3_data0):
            a['y'].append(t_flux)
        
        if s > 9:
    
            a0_data1['x'].append(pig_data[s]['chla'])
            a1_data1['x'].append(pico)
            a2_data1['x'].append(nano)
            a3_data1['x'].append(micro)
            
            for a in (a0_data1, a1_data1, a2_data1, a3_data1):
                a['y'].append(t_flux)       
        

    lines, labels, line_length = get_station_color_legend()
    axs[0].legend(lines, labels, frameon=False, handlelength=line_length)

    for i, a in enumerate(((a0_data0, a0_data1), (a1_data0, a1_data1), (a2_data0, a2_data1), (a3_data0, a3_data1))):
        
        x0 = np.array(a[0]['x']).reshape(-1, 1)
        y0 = np.array(a[0]['y']).reshape(-1, 1)
        reg0 = LinearRegression().fit(x0, y0)
        y_fit0 = reg0.predict(x0)
        axs[i].plot(x0, y_fit0, c=black)

        x1 = np.array(a[1]['x']).reshape(-1, 1)
        y1 = np.array(a[1]['y']).reshape(-1, 1)
        reg1 = LinearRegression().fit(x1, y1)
        y_fit1 = reg1.predict(x1)
        axs[i].plot(x1, y_fit1, c=black, ls=':')
        
        axs[i].set_title(f'$R^2$ = {reg0.score(x0, y0):.2f}, {reg1.score(x1, y1):.2f}')

    fig.savefig(os.path.join(path, f'figs/zg_phyto_scatter.pdf'), bbox_inches='tight')
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


def multipanel_context(path, station_data):
    
    pig_data = get_ml_pigs(station_data)
    nut_data = get_ml_nuts(station_data)
    
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
        pickled_files = [f for f in os.listdir(path) if f'stn{s}.pkl' in f]
        for f in pickled_files:
            with open(os.path.join(path, f), 'rb') as file:
                results = pickle.load(file)
                fluxes = results['sink_fluxes']['T']
                zg_fluxes.append(fluxes[zgi][0])
                interp_fluxes = [fluxes[list(grid).index(i)][0] for i in interp_depths]
                interped_flux = interp1d(interp_depths, interp_fluxes)(zgp100)
                xfer_effs.append(interped_flux[()] / fluxes[zgi][0])
                Lp = results['params']['Lp']['posterior']
                Po = results['params']['Po']['posterior']
                npp = 0
                for layer in range(zgi + 1):
                    zi, zim1 = get_layer_bounds(layer, grid)
                    npp += Lp * Po * (np.exp(-zim1 / Lp) - np.exp(-zi / Lp))
                xport_effs.append(fluxes[zgi][0] / npp)
        
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

    fig.savefig(os.path.join(path, f'figs/multipanel_context.pdf'), bbox_inches='tight')
    plt.close()
    

def poc_profiles(path, station_data):
    
    for s in station_data:
        
        tracers = station_data[s]['tracers']
        grid = station_data[s]['grid']
        zg = station_data[s]['zg']
        mld = station_data[s]['mld']

        fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        ax1.errorbar(
            tracers['POCS']['prior'], grid,
            fmt='^', xerr=tracers['POCS']['prior_e'], ecolor=blue,
            elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full')
        # ax1.errorbar(
        #     tracers['POCS']['posterior'], grid, fmt='o',
        #     xerr=tracers['POCS']['posterior_e'], ecolor=orange,
        #     elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
        #     zorder=3, markeredgewidth=1)

        ax2.errorbar(
            tracers['POCL']['prior'], grid,
            fmt='^', xerr=tracers['POCL']['prior_e'], ecolor=blue,
            elinewidth=1, c=blue, ms=10, capsize=5, fillstyle='full', label='Data')
        # ax2.errorbar(
        #     tracers['POCL']['posterior'], grid, fmt='o',
        #     xerr=tracers['POCL']['posterior_e'], ecolor=orange,
        #     elinewidth=1, c=orange, ms=8, capsize=5, fillstyle='none',
        #     zorder=3, markeredgewidth=1, label='Estimate')

        for ax in (ax1, ax2):
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=610)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.axhline(zg, c=black, ls=':')
            ax.axhline(mld, c=black, ls='--')
        ax2.tick_params(labelleft=False)
        ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                    loc='lower right')
        plt.savefig(os.path.join(path, f'figs/pocprof_stn{s}'))
        plt.close()
        

def ballast_scatterplots(path, station_data):
    
    scheme = plt.cm.plasma
    param_text = get_param_text()
    
    pump_data = pd.read_csv('../../../geotraces/GP15merge.csv',
                            usecols=['GTStn',
                                    'CorrectedMeanDepthm',
                                    'Latitudedegrees_north',
                                    'PIC_SPT_nM', 'fCaCO3_SPT', 'PIC_LPT_nM', 'fCaCO3_LPT',
                                    'bSi_SPT_nM', 'fopal_SPT', 'bSi_LPT_nM', 'fopal_LPT'])
    pump_data = pump_data.rename({'GTStn': 'station',
                                  'CorrectedMeanDepthm': 'depth',
                                  'Latitudedegrees_north': 'latitude'}, axis='columns')

    # get mean param df across all stations
    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        df = pickle.load(f)
    df = df[['depth', 'station', 'ws', 'wl']]
    param_means = df.groupby(['depth', 'station']).mean().reset_index()

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    norm = Normalize(-20, 60)  # min and max latitudes
    
    for s in station_data:
        s_params = param_means.loc[(param_means['station'] == s) & (param_means['depth'] >= station_data[s]['zg'])]
        merged = s_params.merge(pump_data, on=['depth', 'station'])
        param_strings = (('ws', 'SPT'), ('wl', 'LPT'))
        for i, (p, sf) in enumerate(param_strings):
            axs[i][0].set_ylabel(f'{param_text[p][0]} ({param_text[p][1]})', fontsize=14, labelpad=10)
            axs[i][0].scatter(merged[f'PIC_{sf}_nM'], merged[p], c=scheme(norm(merged['latitude'])), s=3)
            axs[i][1].scatter(merged[f'fCaCO3_{sf}'], merged[p], c=scheme(norm(merged['latitude'])), s=3)
            axs[i][2].scatter(merged[f'bSi_{sf}_nM'], merged[p], c=scheme(norm(merged['latitude'])), s=3)
            axs[i][3].scatter(merged[f'fopal_{sf}'], merged[p], c=scheme(norm(merged['latitude'])), s=3)
            # axs[i][0].set_xscale('log')
            # axs[i][2].set_xscale('log')
            for a in axs[i][1:]:
                a.yaxis.set_ticklabels([])
            axs[1][0].set_xlabel('PIC (nM)', fontsize=14)
            axs[1][1].set_xlabel('f_CaCO3', fontsize=14)
            axs[1][2].set_xlabel('bSi (nM)', fontsize=14)
            axs[1][3].set_xlabel('f_opal', fontsize=14)

    fig.subplots_adjust(right=0.8, wspace=0.05, hspace=0.3, top=0.98, bottom=0.2)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.78])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=scheme), cax=cbar_ax)
    cbar.set_label('Latitude (°N)', rotation=270, labelpad=10, fontsize=14)

    fig.savefig(os.path.join(path, f'figs/ballast_scatterplots.pdf'))
    plt.close()
    

def section_map(path, station_data):

    fig = plt.figure(figsize=(5, 7))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), color=black, zorder=2)
    ax.set_extent([-162, -142, -25, 62], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=['left'], zorder=1)
    gl.xlines = False
    import math
    for s in station_data:
        c = get_station_color(s)
        ax.scatter(station_data[s]['longitude'], station_data[s]['latitude'], color=c, s=20, zorder=3)
        
    plt.savefig(os.path.join(path, f'figs/section_map.pdf'), bbox_inches='tight')
    plt.close()


def aggratio_scatter(path, station_data):
    
    nuts = get_ml_nuts(station_data)

    with open(os.path.join(path, 'saved_params_dv.pkl'), 'rb') as f:
        dv_df = pickle.load(f)    

    params_df = dv_df[['depth', 'station', 'aggratio', 'Bm2', 'B2']].copy()
    mean_params = params_df.groupby(['depth', 'station']).mean().reset_index()

    with open(os.path.join(path, 'saved_params_dc.pkl'), 'rb') as f:
        dc_df = pickle.load(f)    
    
    npp_df = dc_df[['station', 'Po', 'Lp']]
    mean_npp = npp_df.groupby(['station']).mean().reset_index()
    
    fig1, axs = plt.subplots(3, 2, tight_layout=True, figsize=(7,10))
    fig2, ax = plt.subplots(1, 1, tight_layout=True)
    param_text = get_param_text()
    
    axs[0][0].set_ylabel(f"{param_text['Bm2'][0]} ({param_text['Bm2'][1]})")
    axs[1][0].set_ylabel(f"{param_text['B2'][0]} ({param_text['B2'][1]})")
    axs[2][0].set_ylabel(param_text['aggratio'][0])
    
    axs[2][0].set_xlabel('Integrated NPP (mmol m$^{-2}$ d$^{-1}$)')
    axs[2][1].set_xlabel('Surface nitrate (µmol kg$^{-1}$)')

    ax.set_ylabel('Integrated NPP (mmol m$^{-2}$ d$^{-1}$)')
    ax.set_xlabel('Surface nitrate (µmol kg$^{-1}$)')
    
    for i in (0, 1, 2):
        axs[i][1].yaxis.set_ticklabels([])
    
    for s in station_data:
        
        c = get_station_color(s)
        s_df_params = mean_params[mean_params['station'] == s]
        s_df_npp = mean_npp[mean_npp['station'] == s].iloc[0]
        Lp = s_df_npp['Lp']
        Po = s_df_npp['Po']
        
        mld = station_data[s]['mld']
        
        Bm2 = s_df_params.loc[s_df_params['depth'] <= mld]['Bm2'].mean()
        B2 = s_df_params.loc[s_df_params['depth'] <= mld]['B2'].mean()
        ratio = s_df_params.loc[s_df_params['depth'] <= mld]['aggratio'].mean()
        
        npp = Lp * Po * (1 - np.exp(-mld / Lp))
        
        axs[0][0].scatter(npp, Bm2, color=c)
        axs[0][1].scatter(nuts[s]['nitrate'], Bm2, color=c)
        axs[1][0].scatter(npp, B2, color=c)
        axs[1][1].scatter(nuts[s]['nitrate'], B2, color=c)
        axs[2][0].scatter(npp, ratio, color=c)
        axs[2][1].scatter(nuts[s]['nitrate'], ratio, color=c)
        
        ax.scatter(nuts[s]['nitrate'], npp, color=c)

    fig1.savefig(os.path.join(path, f'figs/aggratio_scatter.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(path, f'figs/npp_nitrate_scatter.pdf'), bbox_inches='tight')
    plt.close()
        
    
if __name__ == '__main__':
    
    start_time = time()

    poc_data = data.poc_by_station()
    param_uniformity = data.define_param_uniformity()
    Lp_priors = data.get_Lp_priors(poc_data)
    ez_depths = data.get_ez_depths(Lp_priors)
    station_data = data.get_station_data(poc_data, param_uniformity, ez_depths,
                                         flux_constraint=True)

    n_sets = 100000
    path = f'../../results/geotraces/mc_{n_sets}'
    all_files = get_filenames(path)
    compile_param_estimates(all_files)%
    # multipanel_context(path, station_data)
    # zg_phyto_scatter(station_data)
    # param_section_compilation_dc(path, station_data, all_files)
    param_section_compilation_dv(path, station_data)
    ctd_plots(path, station_data)
    spaghetti_params(path, station_data)
    # spaghetti_ctd(path, station_data)
    # spaghetti_poc(path, poc_data)
    # poc_section(path, poc_data, station_data)
    # section_map(path, station_data)
    aggratio_scatter(path, station_data)

    print(f'--- {(time() - start_time)/60} minutes ---')

    