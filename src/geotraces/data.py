import pandas as pd
import numpy as np
import os
from itertools import product
from scipy.interpolate import interp1d
from datetime import datetime

import netCDF4 as nc
from geopy.distance import distance

from src.constants import MMC
from src.framework import define_equation_elements, define_state_elements

def load_poc_data():
    
    metadata = pd.read_csv('../../../data/geotraces/poc_conc.csv',
                           usecols=('GTNum', 'GTStn', 'CastType',
                                    'CorrectedMeanDepthm',
                                    'Latitudedegrees_north', 
                                    'Longitudedegrees_east',
                                    'DateatMidcastGMTyyyymmdd'))

    # SPM_SPT_pM has NaN for intercal samples, useful for dropping later
    cols = ('SPM_SPT_ugL', 'POC_SPT_uM', 'POC_LPT_uM')
    
    values = pd.read_csv('../../../data/geotraces/poc_conc.csv', usecols=cols)
    errors = pd.read_csv('../../../data/geotraces/poc_sd.csv', usecols=cols)
    flags = pd.read_csv('../../../data/geotraces/poc_flag.csv', usecols=cols)

    data = merge_poc_data(metadata, values, errors, flags)
    data.dropna(inplace=True)
    data = data.loc[:, ~data.columns.str.startswith('SPM_SPT_ugL')]

    data = data[~data['station'].isin((1, 3, 18.3))]  # exclude stations 1, 18.3
    data = data[data['depth'] < 1000]  # don't need data below 1km

    return data


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


def poc_by_station():
    
    df = load_poc_data()
    data = {}
    maxdepth = 600

    for s in df['station'].unique():
        raw = df[df['station'] == s].copy()
        raw.sort_values('depth', inplace=True, ignore_index=True)
        cleaned = clean_by_flags(raw)
        data[int(s)] = cleaned.loc[cleaned['depth'] < maxdepth]

    return data


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
    
    path = f'../../../data/geotraces/{dir}'
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

    nc_data = load_nc_data(dir)
    nc_dates = [datetime.strptime(d,'%Y%m%d') for d in nc_data]
    
    if dir == 'cbpm':
        nc_lats = [90 - x*(1/12) - 1/24 for x in range(2160)]
        nc_lons = [x*(1/12) - 180 + 1/24 for x in range(4320)]

    for s in poc_data:

        df = poc_data[s].copy()
        row = df[df['cast'] == 'S'].iloc[0]

        date = datetime.strptime(row['datetime'], '%m/%d/%y %H:%M')
        station_coord = np.array((row['latitude'], row['longitude']))
        prev_nc_dates = [d for d in nc_dates if d <= date]
        nc_date = min(prev_nc_dates, key=lambda x: abs(x - date))
        nc_8day = nc_data[nc_date.strftime('%Y%m%d')]
        
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

        var_by_station[row['station']] = station_var
    
    return var_by_station


def load_mixed_layer_depths():

    mld_df = pd.read_csv('../../../data/geotraces/mld.csv')
    mld_dict = dict(zip(mld_df['station'], mld_df['depth']))

    return mld_dict

def load_Th_fluxes():

    df = pd.read_csv('../../../data/geotraces/sinkingflux_Th.csv',
                     usecols=('station', 'depth', 'flux'))

    return df


def get_median_POCS():
    
    poc = poc_by_station()
    data = pd.DataFrame(columns=['depth', 'POCS'])
    for  df in poc.values():
        data = pd.concat([data, df], join='inner', ignore_index=True)

    median = np.median(data['POCS'])
    
    return median


def get_station_Th_fluxes(grid, max_depth, station, flux_df):

    flux_layers = []
    flux_depths = []
    flux_vals = []
    s_df = flux_df.loc[(flux_df['station'] == station) & (flux_df['depth'] < max_depth)]
    for i, depth in enumerate(grid):
        nearby = s_df.iloc[(s_df['depth'] - depth).abs().argsort()[:1]].iloc[0]
        if nearby['flux'] > 0:
            flux_layers.append(i)
            flux_depths.append(nearby['depth'])
            flux_vals.append(nearby['flux'])
    fluxes = pd.DataFrame(list(zip(flux_layers, flux_depths, flux_vals)), columns=['layer', 'depth', 'flux'])
    
    return fluxes  

def get_station_data(poc_data, params, ez_depths, flux_constraint=False):
    
    d = {s: {} for s in poc_data}
    mixed_layer_depths = load_mixed_layer_depths()
    if flux_constraint:
        max_depth = 620
        flux_df = load_Th_fluxes()
    
    for s in poc_data.keys():
        grid = tuple(poc_data[s]['depth'].values)
        layers = tuple(range(len(grid)))
        zg = min(grid, key=lambda x:abs(x - ez_depths[s]))  # grazing depth
        tracers = define_tracers(poc_data[s])
        d[s]['mld'] = mixed_layer_depths[s]
        d[s]['grid'] = grid
        d[s]['latitude'] = poc_data[s].iloc[0]['latitude']
        d[s]['longitude'] = poc_data[s].iloc[0]['longitude']
        d[s]['layers'] = layers
        d[s]['zg'] = zg
        d[s]['umz_start'] = grid.index(zg) + 1
        d[s]['tracers'] = tracers
        if flux_constraint:
            d[s]['Th_fluxes'] = get_station_Th_fluxes(grid, max_depth, s, flux_df)
        else:
            d[s]['Th_fluxes'] = None
        d[s]['e_elements'] = define_equation_elements(tracers, layers, Th_fluxes=d[s]['Th_fluxes'])
        d[s]['s_elements'] = define_state_elements(tracers, params, layers, Th_fluxes=d[s]['Th_fluxes'])
        
    return d


def define_tracers(data):
    
    tracers = {'POCS': {}, 'POCL': {}}
    
    for t in tracers:
        tracers[t]['prior'] = data[t]
        tracers[t]['prior_e'] = data[f'{t}_unc']

    return tracers


def define_residuals(prior_error, gamma):
    
    residuals = {'POCS': {}, 'POCL': {}}
    
    for tracer in residuals:
        residuals[tracer]['prior'] = 0
        residuals[tracer]['prior_e'] = gamma * prior_error
    
    return residuals


def set_param_priors(params, Lp_prior, Po_prior, B3_prior, mc_params):

    def set_prior(param_name, prior, error):
        
        params[param_name]['prior'] = prior
        params[param_name]['prior_e'] = error
    
    set_prior('B2p', mc_params['B2p'], mc_params['B2p'])
    set_prior('Bm2', mc_params['Bm2'], mc_params['Bm2'])
    set_prior('Bm1s', mc_params['Bm1s'], mc_params['Bm1s'])
    set_prior('Bm1l', mc_params['Bm1l'], mc_params['Bm1l'])
    set_prior('ws', mc_params['ws'], mc_params['ws'])
    set_prior('wl', mc_params['wl'], mc_params['wl'])
    set_prior('Po', Po_prior, Po_prior*0.5)
    set_prior('Lp', Lp_prior, Lp_prior*0.5)
    set_prior('B3', B3_prior, B3_prior*0.5)
    set_prior('a', 0.3, 0.3*0.5)
    set_prior('zm', 500, 500*0.5)


def define_param_uniformity():

    param_uniformity = {}

    nonuniform_params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    uniform_params = ('Po', 'Lp', 'B3', 'a', 'zm')
    
    for p in nonuniform_params:
        param_uniformity[p] = {'dv': True}
    
    for p in uniform_params:
        param_uniformity[p] = {'dv': False}

    return param_uniformity


def get_Lp_priors(poc_data):

    Kd = extract_nc_data(poc_data, 'modis')
    Lp_priors = {station: 1/k for station, k in Kd.items()}
    
    return Lp_priors


def get_ez_depths(Lp_priors):

    depths = {station: l*np.log(100) for station, l in Lp_priors.items()}
    
    return depths


def get_Po_priors(poc_data, Lp_priors, npp_data):
    """Calculate Po priors at each station.

    Args:
        poc_data (_type_): _description_
        Lp_priors (_type_): _description_
        npp_data (_type_): _description_

    Returns:
        _type_: _description_
    
    Full equation for Po is Po = npp / [Lp * (1 - exp[-ez_depth / Lp])],
    but ez_depth = Lp * ln(100), so (1-exp[]) simplifies to 0.99.
    """
    Po_priors = {s: ((npp_data[s] / MMC)
                     / (Lp_priors[s] * 0.99)) for s in poc_data}
    
    return Po_priors


def get_B3_priors(npp_data):
    
    B3_priors = {}
    
    for s in npp_data:
        B3_priors[s] = 10**(-2.42 + 0.53*np.log10(npp_data[s]))

    return B3_priors
