#!/usr/bin/env python3
import numpy as np

from src.constants import MMC
from src.geotraces.data import extract_nc_data

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

def define_params(Lp_prior, Po_prior, B3_prior, priors_from, rel_err):
    
    params = {}
    
    # B2p_prior, B2p_error, Bm2_prior, Bm2_error = contextual_priors(
    #     priors_from, rel_err)
    
    # Bm1l = 0.03
    # Bm1s = 0.01
    # Bm2 = 0.21
    # wl = 47
    # ws = 0.6
    # B2 = 0.004

    # Bm1l = 0.15
    # Bm1s = 0.04
    # Bm2 = 0.73
    # wl = 62
    # ws = 0.8
    # B2 = 0.01
    
    Bm1l = 0.68
    Bm1s = 0.23
    Bm2 = 2.38
    wl = 78
    ws = 1.1
    B2 = 0.04

    params['ws'] = set_prior(ws, ws)
    params['wl'] = set_prior(wl, wl)
    params['B2p'] = set_prior(B2/1.57, B2/1.57)
    params['Bm2'] = set_prior(Bm2, Bm2)
    params['Bm1s'] = set_prior(Bm1s, Bm1s)
    params['Bm1l'] = set_prior(Bm1l, Bm1l)
    params['Po'] = set_prior(Po_prior, Po_prior*0.25, depth_varying=False)
    params['Lp']= set_prior(Lp_prior, Lp_prior*0.25, depth_varying=False)
    params['B3'] = set_prior(B3_prior, B3_prior*0.25, depth_varying=False)
    params['a'] = set_prior(0.3, 0.3, depth_varying=False)
    params['zm'] = set_prior(500, 500, depth_varying=False) 
    
    return params

def set_prior(prior, error, depth_varying=True):
    
    data = {}
    
    data['prior'] = prior
    data['prior_e'] = error
    data['dv'] = depth_varying
    
    return data

def contextual_priors(priors_from, rel_err):

    if priors_from == 'NA':  # Murnane et al. 1996, DSR
        B2p_prior = (2/21) # m^3 mg^-1 y^-1
        B2p_error = B2p_prior*rel_err
        Bm2_prior = 156  # y^-1
        Bm2_error = Bm2_prior*rel_err
    else:  # Murnane 1994, JGR
        B2p_prior = (0.8/1.57) # m^3 mg^-1 y^-1
        B2p_error = B2p_prior*rel_err
        Bm2_prior = 400  # y^-1
        Bm2_error = Bm2_prior*rel_err
    
    return B2p_prior, B2p_error, Bm2_prior, Bm2_error

def get_Lp_priors(poc_data):

    Kd = extract_nc_data(poc_data, 'modis')
    Lp_priors = {station: 1/k for station, k in Kd.items()}
    
    return Lp_priors

def get_ez_depths(Lp_priors):

    depths = {station: l*np.log(100) for station, l in Lp_priors.items()}
    
    return depths

def get_Po_priors(poc_data, Lp_priors, npp_data, ez_depths):

    Po_priors = {calculate_surface_npp(
        poc_data[s], Lp_priors[s], npp_data[s], ez_depths[s], volumetric=True
        ) for s in poc_data}
    
    return Po_priors

def get_B3_priors(npp_data):
    
    B3_priors = {}
    
    for s in npp_data:
        B3_priors[s] = 10**(-2.42 + 0.53*np.log10(npp_data[s]))

    return B3_priors

def calculate_surface_npp(poc, Lp, npp, ez_depth, volumetric=False):
    
    z0 = poc.iloc[0]['depth']
    ratio = ((1 - np.exp(-z0/Lp))/(1 - np.exp(-ez_depth/Lp)))
    surface_npp = npp/MMC * ratio
    if volumetric:
        return surface_npp/z0
    
    return surface_npp

def get_residual_prior_error(poc_data, Lp_priors, npp_data, ez_depths):
    
    products = [calculate_surface_npp(
        poc_data[s], Lp_priors[s], npp_data[s], ez_depths[s]
        ) for s in poc_data]
    
    return np.mean(products)