import pandas as pd
import numpy as np
import os
from itertools import product
from scipy.interpolate import interp1d
from datetime import datetime
import sys
import netCDF4 as nc

from src.constants import MMC

def get_src_parent_path():
    
    module_path = os.path.abspath(__file__)
    src_parent_path = module_path.split('src')[0]
    
    return src_parent_path

def load_poc_data():
    
    src_parent_path = get_src_parent_path()
    
    metadata = pd.read_csv(os.path.join(src_parent_path,'data/values_v9.csv'),
                           usecols=('GTNum', 'GTStn', 'CorrectedMeanDepthm',
                                    'Latitudedegrees_north', 
                                    'Longitudedegrees_east',
                                    'DateatMidcastGMTyyyymmdd'))

    # SPM_SPT_pM has NaN for intercal samples, useful for dropping later
    cols = ('SPM_SPT_ugL', 'POC_SPT_uM', 'POC_LPT_uM')
    
    values = pd.read_csv(os.path.join(src_parent_path, 'data/values_v9.csv'),
                         usecols=cols)
    errors = pd.read_csv(os.path.join(src_parent_path, 'data/error_v9.csv'),
                         usecols=cols)
    flags = pd.read_csv(os.path.join(src_parent_path, 'data/flag_v9.csv'),
                        usecols=cols)

    merged = merge_poc_data(metadata, values, errors, flags)
    merged.dropna(inplace=True)
    merged = merged.loc[:, ~merged.columns.str.startswith('SPM_SPT_ugL')]

    # station 18.3 excludes upper 500m
    merged = merged[merged['station'] != 18.3]
    merged = merged[merged['depth'] < 1000]

    return merged


def merge_poc_data(metadata, values, errors, flags):

    rename_cols = {'GTStn': 'station', 'CorrectedMeanDepthm': 'depth',
                   'POC_SPT_uM': 'POCS', 'POC_LPT_uM': 'POCL',
                   'Latitudedegrees_north': 'latitude',
                   'Longitudedegrees_east': 'longitude',
                   'DateatMidcastGMTyyyymmdd': 'datetime'}
    
    
    for df in (metadata, values, errors, flags):
        df.rename(columns=rename_cols, inplace=True)

    data = pd.merge(metadata, values, left_index=True, right_index=True)
    data = pd.merge(data, errors, left_index=True, right_index=True,
                    suffixes=(None, '_unc'))
    data = pd.merge(data, flags, left_index=True, right_index=True,
                    suffixes=(None, '_flag'))
    
    return data

def get_station_poc(data, station, maxdepth):

    raw_station_data = data[data['station'] == station].copy()
    raw_station_data.sort_values('depth', inplace=True, ignore_index=True)

    clean_station_data = clean_by_flags(raw_station_data)
    cleaned = clean_station_data.loc[clean_station_data['depth'] < maxdepth]
    
    return cleaned

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
    modis_path = os.path.join(src_parent_path,'data/modis')
    filenames = [f for f in os.listdir(modis_path) if '.nc' in f]

    modis_data = {}

    for f in filenames:
        date = f.split('.')[3]
        modis_data[date] = nc.Dataset(os.path.join(modis_path, f))
  
    return modis_data

def get_Lp_priors(poc_data):

    Lp_priors = {}
    df = poc_data.copy()
    df = df[df['depth'] < 50]
    df = df[['station', 'latitude', 'longitude', 'datetime']]
    df.drop_duplicates(subset=['station'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    modis_data = load_modis_data()
    modis_dates = [datetime.strptime(d,'%Y%m%d') for d in modis_data]
    
    for i, row in df.iterrows(): 
        
        date = datetime.strptime(row['datetime'], '%m/%d/%y %H:%M')
        prev_modis_dates = [d for d in modis_dates if d <= date]
        df.at[i, 'modis_date'] = min(
            prev_modis_dates, key=lambda x: abs(x - date))
        modis_8day = modis_data[df.at[i, 'modis_date'].strftime('%Y%m%d')]
        kd_8day = modis_8day.variables['MODISA_L3m_KD_8d_4km_2018_Kd_490'][0]

        station_coord = np.array((row['latitude'], row['longitude']))       
        modis_lats = list(modis_8day.variables['lat'][:])
        modis_lons = list(modis_8day.variables['lon'][:])
        modis_coords = list(product(modis_lats, modis_lons))
        distances = np.linalg.norm(modis_coords - station_coord, axis=1)
        modis_coords_sorted = [
            x for _, x in sorted(zip(distances, modis_coords))]
        
        j = 0
        while True:
            modis_lat_index = modis_lats.index(modis_coords_sorted[j][0])
            modis_lon_index = modis_lons.index(modis_coords_sorted[j][1])
            station_kd = kd_8day[modis_lat_index, modis_lon_index]
            if station_kd:
                break
            j += 1

        df.at[i, 'modis_lat'] = modis_coords_sorted[j][0]
        df.at[i, 'modis_lon'] = modis_coords_sorted[j][1]
        df.at[i, 'Lp'] = 1/station_kd
        Lp_priors[row['station']] = 1/station_kd
    
    return Lp_priors

def load_npp_data():
    
    src_parent_path = get_src_parent_path()
    
    df = pd.read_csv(os.path.join(src_parent_path,'data/npp.csv'))
    dates = [datetime.strptime(d, '%d-%b-%y') for d in df.columns[2:]]

    npp_df = df[['Station', 'Sampling Date']].copy()
    npp_df['mgC_m2_d'] = 0.0
    for i, r in npp_df.iterrows():
        date = datetime.strptime(r['Sampling Date'], '%m/%d/%y')
        prior_dates = [d for d in dates if d <= date]
        closest = min(prior_dates, key=lambda x: abs(x - date))
        npp_df.at[i, 'mgC_m2_d'] = df.at[i, closest.strftime('%-d-%b-%y')]

    npp_df['npp'] = npp_df['mgC_m2_d']/MMC
    npp_df.drop('mgC_m2_d', axis=1, inplace=True)
    npp_df.drop('Sampling Date', axis=1, inplace=True)

    npp_dict = dict(zip(npp_df['Station'], npp_df['npp']))

    return npp_dict

def load_mixed_layer_depths():
    
    src_parent_path = get_src_parent_path()
    mld_df = pd.read_excel(os.path.join(src_parent_path,'data/gp15_mld.xlsx'))
    mld_dict = dict(zip(mld_df['Station No'], mld_df['MLD']))
    # npp_std = np.std(list(npp_dict.values()), ddof=1)

    return mld_dict

def load_ppz_data():
    
    src_parent_path = get_src_parent_path()
    ppz_df = pd.read_excel(os.path.join(src_parent_path,'data/gp15_ppz.xlsx'))
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
