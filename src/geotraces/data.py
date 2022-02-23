import pandas as pd
import numpy as np
from os import path
from itertools import product
from scipy.interpolate import interp1d

import netCDF4 as nc

from src.constants import MMC

def get_src_parent_path():
    
    module_path = path.abspath(__file__)
    src_parent_path = module_path.split('src')[0]
    
    return src_parent_path

def load_poc_data():
    
    src_parent_path = get_src_parent_path()
    
    metadata = pd.read_csv(path.join(src_parent_path,'data/values_v9.csv'),
                           usecols=('GTNum', 'GTStn', 'CorrectedMeanDepthm',
                                    'Latitudedegrees_north', 
                                    'Longitudedegrees_east'))

    # Sr_SPT_pM has NaN for intercal samples, useful for dropping later
    cols = ('SPM_SPT_ugL', 'POC_SPT_uM', 'POC_LPT_uM')
    
    values = pd.read_csv(path.join(src_parent_path, 'data/values_v9.csv'),
                         usecols=cols)
    errors = pd.read_csv(path.join(src_parent_path, 'data/error_v9.csv'),
                         usecols=cols)
    flags = pd.read_csv(path.join(src_parent_path, 'data/flag_v9.csv'),
                        usecols=cols)
    
    merged = merge_poc_data(metadata, values, errors, flags)
    merged.dropna(inplace=True)
    merged.drop('SPM_SPT_ugL', axis=1, inplace=True)

    # station 1 only has upper 100m, 18.3 excludes upper 500m
    merged = merged[~merged['station'].isin([1., 18.3])]

    depth_cutoff = 600
    merged = merged.loc[merged['depth'] < depth_cutoff]
    
    return merged


def merge_poc_data(metadata, values, errors, flags):

    rename_cols = {'GTStn': 'station', 'CorrectedMeanDepthm': 'depth',
                   'POC_SPT_uM': 'POCS', 'POC_LPT_uM': 'POCL',
                   'Latitudedegrees_north': 'latitude',
                   'Longitudedegrees_east': 'longitude'}
    
    
    for df in (metadata, values, errors, flags):
        df.rename(columns=rename_cols, inplace=True)

    data = pd.merge(metadata, values, left_index=True, right_index=True)
    data = pd.merge(data, errors, left_index=True, right_index=True,
                    suffixes=(None, '_unc'))
    data = pd.merge(data, flags, left_index=True, right_index=True,
                    suffixes=(None, '_flag'))
    
    return data

def get_station_poc(data, station):

    raw_station_data = data[data['station'] == station].copy()
    raw_station_data.sort_values('depth', inplace=True, ignore_index=True)

    clean_station_data = clean_by_flags(raw_station_data)
    
    return clean_station_data

def clean_by_flags(raw):
    
    cleaned = raw.copy()
    flags_to_clean = (3, 4)

    tracers = ('POCS', 'POCL')
    for ((i, row), t) in product(cleaned.iterrows(), tracers):
        if row[f'{t}_flag'] in flags_to_clean:
            poc = cleaned.at[i - 1, t], cleaned.at[i + 1, t]
            depth = cleaned.at[i - 1, 'depth'], cleaned.at[i + 1, 'depth']
            interp = interp1d(depth, poc)
            cleaned.at[i, t] = interp(row['depth'])
            cleaned.at[i, f'{t}_unc'] = cleaned.at[i, t]

    return cleaned

def load_modis_data():
    
    src_parent_path = get_src_parent_path()
    # 8-day averages, 4km resolution, averaged over 20180926-20181122
    modis_data = nc.Dataset(path.join(src_parent_path,'data/modis_kd.nc'))
    
    return modis_data

def get_Lp_priors(poc_data):

    modis_data = load_modis_data()
    modis_lat = modis_data.variables['lat'][:]
    modis_lon = modis_data.variables['lon'][:]
    kd = modis_data.variables['MODISA_L3m_KD_8d_4km_2018_Kd_490'][:,:]

    Lp_priors = {}
    
    for s in poc_data['station'].unique():
        
        raw_station_data = poc_data.loc[poc_data['station'] == s]
        pump_lat = round(raw_station_data.iloc[0]['latitude'], 1)
        pump_lon = round(raw_station_data.iloc[0]['longitude'], 1)
        
        modis_lat_index = min(
            range(len(modis_lat)), key=lambda i: abs(modis_lat[i] - pump_lat))
        modis_lon_index = min(
            range(len(modis_lon)), key=lambda i: abs(modis_lon[i] - pump_lon))
     
        Lp_priors[s] = 1/kd[modis_lat_index, modis_lon_index]
        
    return Lp_priors

def load_npp_data():
    
    src_parent_path = get_src_parent_path()
    
    npp_df = pd.read_csv(path.join(src_parent_path,'data/npp.csv'))
    npp_df['npp'] = npp_df['mgC_m2_d']/MMC
    npp_df.drop('mgC_m2_d', axis=1, inplace=True)
    
    npp_dict = dict(zip(npp_df['station'], npp_df['npp']))
    # npp_std = np.std(list(npp_dict.values()), ddof=1)

    return npp_dict

def load_mixed_layer_depths():
    
    src_parent_path = get_src_parent_path()
    mld_df = pd.read_excel(path.join(src_parent_path,'data/gp15_mld.xlsx'))
    mld_dict = dict(zip(mld_df['Station No'], mld_df['MLD JAK']))
    # npp_std = np.std(list(npp_dict.values()), ddof=1)

    return mld_dict

def load_ppz_data():
    
    src_parent_path = get_src_parent_path()
    ppz_df = pd.read_excel(path.join(src_parent_path,'data/gp15_ppz.xlsx'))
    ppz_dict = {}
    
    for s in ppz_df['Station'].unique():
        ppz_dict[s] = ppz_df[ppz_df['Station'] == s]['PPZ Depth'].mean()

    return ppz_dict

def get_Po_priors(Lp_priors):
    
    npp = load_npp_data()
    Po_priors = {}
    
    for s in Lp_priors:
        Po_priors[s] = npp[s] / Lp_priors[s]
    
    return Po_priors

def get_residual_prior_error(Po_priors, mixed_layer_depths):
    
    products = []

    for s in Po_priors:
        if s not in mixed_layer_depths:
            continue
        products.append(Po_priors[s]*mixed_layer_depths[s])
    
    return np.mean(products)
