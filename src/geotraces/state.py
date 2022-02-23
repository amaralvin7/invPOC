#!/usr/bin/env python3
import numpy as np

import statsmodels.formula.api as smf

from src.constants import MMC, DPY

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

def define_params(Lp_prior, Po_prior, priors_from, rel_err):
    
    params = {}
    
    B2p_prior, B2p_error, Bm2_prior, Bm2_error = contextual_priors(
        priors_from, rel_err)

    params['ws'] = set_prior(2, 2*rel_err)
    params['wl'] = set_prior(20, 20*rel_err)
    params['B2p'] = set_prior(B2p_prior*MMC/DPY, B2p_error*MMC/DPY)
    params['Bm2'] = set_prior(Bm2_prior/DPY, Bm2_error/DPY)
    params['Bm1s'] = set_prior(0.1, 0.1*rel_err)
    params['Bm1l'] = set_prior(0.15, 0.15*rel_err)
    params['Po'] = set_prior(Po_prior, Po_prior*0.25, depth_varying=False)
    params['Lp']= set_prior(Lp_prior, Lp_prior*0.25, depth_varying=False)
    params['B3'] = set_prior(0.06, 0.06*rel_err, depth_varying=False)
    params['a'] = set_prior(0.3, 0.3*rel_err, depth_varying=False)
    params['zm'] = set_prior(500, 500*rel_err, depth_varying=False)        
    
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