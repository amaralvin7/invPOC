import pandas as pd
import numpy as np
from os import path

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

    cols = ('POC_SPT_uM', 'POC_LPT_uM')
    
    values = pd.read_csv(path.join(src_parent_path, 'data/values_v9.csv'),
                         usecols=cols)
    errors = pd.read_csv(path.join(src_parent_path, 'data/error_v9.csv'),
                         usecols=cols)
    flags = pd.read_csv(path.join(src_parent_path, 'data/flag_v9.csv'),
                        usecols=cols)
    
    merged = merge_poc_data(metadata, values, errors, flags)
    merged.dropna(inplace=True)

    depth_cutoff = 1000
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

def get_super_station_data(data):

    supers = (8., 14., 23., 29., 35., 39.)
    super_data = data.loc[data['station'].isin(supers)].copy()

    # super_data.to_csv('poc_data_to_invert.csv', index=False)
    return super_data

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
        
        station_data = poc_data.loc[poc_data['station'] == s]
        pump_lat = round(station_data.iloc[0]['latitude'], 1)
        pump_lon = round(station_data.iloc[0]['longitude'], 1)
        
        modis_lat_index = min(
            range(len(modis_lat)), key=lambda i: abs(modis_lat[i] - pump_lat))
        modis_lon_index = min(
            range(len(modis_lon)), key=lambda i: abs(modis_lon[i] - pump_lon))
     
        Lp_priors[s] = 1/kd[modis_lat_index, modis_lon_index]

    Lp_prior_error = np.std(list(Lp_priors.values()), ddof=1)
        
    return Lp_priors, Lp_prior_error

def load_npp_data():
    
    src_parent_path = get_src_parent_path()
    
    npp_df = pd.read_csv(path.join(src_parent_path,'data/npp.csv'))
    npp_df['npp'] = npp_df['mgC_m2_ d']/MMC
    npp_df.drop('mgC_m2_ d', axis=1, inplace=True)
    
    npp_dict = dict(zip(npp_df['station'], npp_df['npp']))
    npp_std = np.std(list(npp_dict.values()), ddof=1)

    return npp_dict, npp_std

def get_Po_priors(Lp_priors, Lp_error):
    
    npp, npp_error = load_npp_data()

    Po_priors = {}
    
    for s in Lp_priors:
        Po_estimate = npp[s] / Lp_priors[s]
        Po_error = np.sqrt((npp_error / Lp_priors[s])**2
                           + (-npp[s] / Lp_priors[s]**2 * Lp_error)**2)
        Po_priors[s] = (Po_estimate, Po_error)
    
    return Po_priors
    



