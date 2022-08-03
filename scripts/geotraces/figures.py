import os
import pickle
from itertools import product
from time import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import src.geotraces.data as data

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

def elbow_plot(path, df):
    
    df_scaled = StandardScaler().fit_transform(df)
    inertia = []
    
    k_vals = range(1, 21)
    for i in k_vals:
        kmeans = KMeans(n_clusters=i, random_state=0)
        result = kmeans.fit(df_scaled)
        inertia.append(result.inertia_)

    plt.figure()
    plt.plot(k_vals, inertia, marker='o', ls='--')
    plt.xlabel('Number of Clusters')
    plt.xticks(k_vals)
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(path, 'elbowplot'))
    plt.close()

def cluster(df, n_clusters):

    df_scaled = StandardScaler().fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    result = kmeans.fit(df_scaled)
    df['label'] = result.labels_
    
    unique_labels = df['label'].unique()
    unique_labels.sort()
    for l in unique_labels:
        label_df = df[df['label'] == l]
        print(f'Fraction of successful sets in cluster {l}: {len(label_df)/len(df):.2f}')
    
    return df

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
        # print(f'N successful sets: {len(filenames) / len(stations)}')
        # print(f'N successful inversions: {len(pickled_files)}')
    else:
        filenames = pickled_files

    return filenames

def xresids(path, params):
    
    df = pd.DataFrame(columns=['resid', 'element'])
    poc_data = data.poc_by_station()
    param_uniformity = data.define_param_uniformity()
    Lp_priors = data.get_Lp_priors(poc_data)
    ez_depths = data.get_ez_depths(Lp_priors)
    station_data = data.get_station_data(poc_data, param_uniformity, ez_depths)

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
        dv_elements = params + ('POCS', 'POCL')
        df1 = df[df['element'].isin(dv_elements)]
        df1_piv = df1.pivot(columns='element')['resid'].astype(float)
        df2 = df[~df['element'].isin(dv_elements)]
        df2_piv = df2.pivot(columns='element')['resid'].astype(float)
        df1_piv.plot(kind='hist', stacked=True, bins=30, ax=axs[0])
        df2_piv.plot(kind='hist', stacked=True, bins=30, ax=axs[1])
        plt.savefig(os.path.join(path, f'figs/xresids_{stn}'))
        plt.close()

if __name__ == '__main__':
    
    start_time = time()
    
    # n_sets = 200
    # path = f'../../results/geotraces/mc_{n_sets}'
    # params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    # all_files = get_filenames(path)
    # success_files = get_filenames(path, True)
    # hist_success(path)
    # xresids(path, params)
    # get_successful_sets(path)
    # hist_stats(path, all_files)
    # hist_stats(path, success_files, '_succcess')
            
    print(f'--- {(time() - start_time)/60} minutes ---')

    