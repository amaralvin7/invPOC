#!/usr/bin/env python3
from itertools import product
from budgets import eval_sym_expression
from constants import ZONE_LAYERS

def calculate_residence_times(
    inventories_sym, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
    tracers, params, state_elements, Ckp1):
    
    res_times = {tracer:{} for tracer in inventories_sym}

    for (tracer, z) in product(res_times, ZONE_LAYERS):
        inventory = inventories_sym[tracer][z]
        fluxes = sum_of_fluxes(
            tracer, z, int_fluxes_sym, int_fluxes, residuals_sym, residuals)
        res_times[tracer][z] = eval_sym_expression(
            inventory / fluxes, state_elements, Ckp1, tracers=tracers,
            residuals=residuals, params=params)
    
    return res_times

def sum_of_fluxes(
    tracer, z, int_fluxes_sym, int_fluxes, residuals_sym, residuals):
    
    fluxes_in = {'POCS': ['production', 'disaggregation'],
                 'POCL': ['aggregation']}
    
    sum_of_fluxes = 0
    
    for f in fluxes_in[tracer]:
        sum_of_fluxes += int_fluxes_sym[f][z]
    if int_fluxes[f'sinkdiv_{tracer[-1]}'][z][0] < 0:
        sum_of_fluxes += -int_fluxes_sym[f'sinkdiv_{tracer[-1]}'][z]
    if residuals[tracer][z][0] > 0:
        sum_of_fluxes += residuals_sym[tracer][z]
    if tracer == 'POCL' and z in ('UMZ', 3, 4, 5, 6):
        sum_of_fluxes += int_fluxes_sym['dvm'][z]
    
    return sum_of_fluxes

def calculate_turnover_times(
    inventories_sym, int_fluxes_sym, int_fluxes, tracers, params,
    state_elements, Ckp1):
    
    flux_tracers = {'sinkdiv_S': ('POCS',), 'sinkdiv_L': ('POCL',),
                    'remin_S': ('POCS',), 'remin_L': ('POCL',),
                    'aggregation': ('POCS', 'POCL'),
                    'disaggregation': ('POCS', 'POCL'),
                    'production': ('POCS',), 'dvm': ('POCS', 'POCL')}
    
    turnover = {t: {z: {} for z in ZONE_LAYERS} for t in tracers}
    
    for (t, z, f) in product(tracers, ZONE_LAYERS, int_fluxes):
        if t in flux_tracers[f]:
            inventory = inventories_sym[t][z]
            flux = int_fluxes_sym[f][z]
            turnover[t][z][f] = eval_sym_expression(
                inventory / flux, state_elements, Ckp1, tracers=tracers,
                params=params)
    
    return turnover