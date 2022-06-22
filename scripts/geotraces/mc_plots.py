import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn

def load_data(path):
    
    sets = pd.read_csv(os.path.join(path, 'table.csv'))
    good = sets.loc[sets['success'] == True].copy()
    bad = sets.loc[sets['success'] == False].copy()
    
    return sets, good, bad


def plot_histograms(path, df, file_prefix):
    
    for p in df.columns:
        data = df[p]
        plt.subplots(tight_layout=True)
        plt.hist(data, bins=30)
        plt.savefig(os.path.join(path, f'{file_prefix}_{p}'))
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

def cluster(path, df, n_clusters):

    df_scaled = StandardScaler().fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    result = kmeans.fit(df_scaled)
    df['label'] = result.labels_
    
    unique_labels = df['label'].unique()
    unique_labels.sort()
    for l in unique_labels:
        label_df = df[df['label'] == l]
        print(f'Fraction of successful sets in cluster {l}: {len(label_df)/len(df):.2f}')
    #     plot_histograms(path, label_df, f'cluster{l}_hist')
    
    return df

def labeled_hist(path, df, params, column):
    
    for p in params:
        df.pivot(columns=column)[p].plot(kind='hist', stacked=True, colormap='Spectral', bins=30)
        plt.savefig(os.path.join(path, f'labeled_hist_{p}_{column}'))
        plt.close()

if __name__ == '__main__':
    
    path = '../../results/geotraces/mc_0p1_5k'
    params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    sets, good, bad = load_data(path)
    good_params = good[[*params]].copy()
    # plot_histograms(path, good_params, 'success_hist')
    elbow_plot(path, good_params)
    df_labeled = cluster(path, good_params, 7)
    labeled_hist(path, df_labeled, params, 'label')
    labeled_hist(path, sets, params, 'succes_int')
    