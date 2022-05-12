import pandas as pd
import numpy as np
import os
from itertools import product
from scipy.interpolate import interp1d
from datetime import datetime
import sys
import netCDF4 as nc
from geopy.distance import distance

from src.constants import MMC

def load_poc_data():
    
    metadata = pd.read_csv('../../data/values_v9.csv',
                           usecols=('GTNum', 'GTStn', 'CastType',
                                    'CorrectedMeanDepthm',
                                    'Latitudedegrees_north', 
                                    'Longitudedegrees_east',
                                    'DateatMidcastGMTyyyymmdd'))

    # SPM_SPT_pM has NaN for intercal samples, useful for dropping later
    cols = ('SPM_SPT_ugL', 'POC_SPT_uM', 'POC_LPT_uM')
    
    values = pd.read_csv('../../data/values_v9.csv', usecols=cols)
    errors = pd.read_csv('../../data/error_v9.csv', usecols=cols)
    flags = pd.read_csv('../../data/flag_v9.csv', usecols=cols)

    merged = merge_poc_data(metadata, values, errors, flags)
    merged.dropna(inplace=True)
    merged = merged.loc[:, ~merged.columns.str.startswith('SPM_SPT_ugL')]

    # station 18.3 excludes upper 500m
    merged = merged[merged['station'] != 18.3]
    merged = merged[merged['depth'] < 1000]

    return merged


def merge_poc_data(metadata, values, errors, flags):

    rename_cols = {'GTStn': 'station', 'CastType': 'cast',
                   'CorrectedMeanDepthm': 'depth',
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

def load_nc_data(dir):
    
    datainfo = {'modis': {'ext': '.nc', 'dateidx': 3},
                'cbpm': {'ext': '.hdf', 'dateidx': 1}}
    
    path = f'../../data/{dir}'
    filenames = [f for f in os.listdir(path) if datainfo[dir]['ext'] in f]
    data = {}

    for f in filenames:
        date = f.split('.')[datainfo[dir]['dateidx']]
        if dir == 'cbpm':
            date = datetime.strptime(date, '%Y%j').strftime('%Y%m%d')
        data[date] = nc.Dataset(os.path.join(path, f))
    
    return data

def extract_nc_data(poc_data, dir):
    
    var_by_station = {}
    
    df = poc_data.copy()
    df = df[df['cast'] == 'S']  
    df = df[['station', 'latitude', 'longitude', 'datetime']]
    df.drop_duplicates(subset=['station'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    nc_data = load_nc_data(dir)
    nc_dates = [datetime.strptime(d,'%Y%m%d') for d in nc_data]
    
    if dir == 'cbpm':
        nc_lats = [90 - x*(1/12) - 1/24 for x in range(2160)]
        nc_lons = [x*(1/12) - 180 + 1/24 for x in range(4320)]

    for i, row in df.iterrows(): 

        date = datetime.strptime(row['datetime'], '%m/%d/%y %H:%M')
        station_coord = np.array((row['latitude'], row['longitude']))
        prev_nc_dates = [d for d in nc_dates if d <= date]
        df.at[i, 'nc_date'] = min(prev_nc_dates, key=lambda x: abs(x - date))
        nc_8day = nc_data[df.at[i, 'nc_date'].strftime('%Y%m%d')]
        
        if dir == 'cbpm':
            var_name = 'npp'
            var_8day = nc_8day.variables[var_name]
        if dir == 'modis':
            var_name = 'Kd'
            var_8day = nc_8day.variables['MODISA_L3m_KD_8d_4km_2018_Kd_490'][0]
            nc_lats = list(nc_8day.variables['lat'][:])
            nc_lons = list(nc_8day.variables['lon'][:])

        close_nc_lats = [
            l for l in nc_lats if abs(station_coord[0] - l) < 1]
        close_nc_lons = [
            l for l in nc_lons if abs(station_coord[1] - l) < 1]
        nc_coords = list(product(close_nc_lats, close_nc_lons))
        distances = [distance(ncc, station_coord) for ncc in nc_coords]
        nc_coords_sorted = [
            x for _, x in sorted(zip(distances, nc_coords))]
        
        j = 0
        while True:
            nc_lat_index = nc_lats.index(nc_coords_sorted[j][0])
            nc_lon_index = nc_lons.index(nc_coords_sorted[j][1])
            station_var = var_8day[nc_lat_index, nc_lon_index]
            if station_var > -9999:
                break
            j += 1

        # df.at[i, 'nc_lat'] = nc_coords_sorted[j][0]
        # df.at[i, 'nc_lon'] = nc_coords_sorted[j][1]
        # df.at[i, var_name] = station_var
        var_by_station[row['station']] = station_var
    
    return var_by_station

def load_mixed_layer_depths():
    
    mld_df = pd.read_excel('../../data/gp15_mld.xlsx')
    mld_dict = dict(zip(mld_df['Station No'], mld_df['MLD']))

    return mld_dict

def load_ppz_data():
    
    ppz_df = pd.read_excel('../../data/gp15_ppz.xlsx')
    ppz_dict = {}
    
    for s in ppz_df['Station'].unique():
        ppz_dict[s] = ppz_df[ppz_df['Station'] == s]['PPZ Depth'].mean()

    return ppz_dict