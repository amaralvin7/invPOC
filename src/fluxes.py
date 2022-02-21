#!/usr/bin/env python3
import sympy as sym
import numpy as np

from src.constants import LAYERS, GRID, MLD, ZG, THICK
"""
todo: combine for loop statements in get_symbolic_int_fluxes
        (it is the way it is to preserve order in original output file)
"""

def get_symbolic_int_fluxes():

    int_fluxes_sym = {}
    
    for size in ('S', 'L'):
        int_fluxes_sym[f'sinkdiv_{size}'] = get_symbolic_sinkdiv(size)
        
    for size in ('S', 'L'):    
        int_fluxes_sym[f'remin_{size}'] = get_symbolic_remin(size)
    
    int_fluxes_sym['aggregation'] = get_symbolic_agg()
    int_fluxes_sym['disaggregation'] = get_symbolic_disagg()
    int_fluxes_sym['production'] = get_symbolic_production()
    int_fluxes_sym['dvm'] = get_symbolic_dvm()

    return int_fluxes_sym

def profile_to_dict(profile):
    
    dictionary = {layer: value for layer, value in enumerate(profile)}
    dictionary['EZ'] = np.sum(profile[:3])
    dictionary['UMZ'] = np.sum(profile[3:])

    return dictionary

def avg_tracer_concentration_sym(tracer, layer):
    
    if layer == 0:
        return sym.symbols(f'{tracer}_{layer}')
    
    ti, tim1 = sym.symbols(f'{tracer}_{layer} {tracer}_{layer - 1}')
    
    return (ti + tim1)/2
        
def get_symbolic_sinkdiv(size):
    
    profile = []
    
    for l in LAYERS:
        wi, ti = sym.symbols(f'w{size.lower()}_{l} POC{size}_{l}')
        if l == 0:
            profile.append(wi*ti)
        else:
            wim1, tim1 = sym.symbols(
                f'w{size.lower()}_{l - 1} POC{size}_{l - 1}')
            profile.append(wi*ti - wim1*tim1)
    
    return profile_to_dict(profile)

def get_symbolic_remin(size):
    
    profile = []
    
    for l in LAYERS:
        Bm1i = sym.symbols(f'Bm1{size.lower()}_{l}')
        ti = avg_tracer_concentration_sym(f'POC{size}', l)
        profile.append(Bm1i*ti*THICK[l])

    return profile_to_dict(profile)

def get_symbolic_agg():
    
    profile = []
    
    for l in LAYERS:
        B2pi = sym.symbols(f'B2p_{l}')
        psi = avg_tracer_concentration_sym(f'POCS', l)
        profile.append((B2pi*psi**2)*THICK[l])
        
    return profile_to_dict(profile)

def get_symbolic_disagg():
    
    profile = []
    
    for l in LAYERS:
        Bm2i = sym.symbols(f'Bm2_{l}')
        pli = avg_tracer_concentration_sym(f'POCL', l)
        profile.append(Bm2i*pli*THICK[l])
        
    return profile_to_dict(profile)

def get_symbolic_production():
    
    profile = []
    
    for l in LAYERS:
        Po, Lp = sym.symbols('Po Lp')
        if l == 0:
            profile.append(Po*MLD)
        else:
            zi = GRID[l]
            zim1 = GRID[l - 1]
            profile.append(Lp*Po*(sym.exp(-(zim1 - MLD)/Lp)
                                   - sym.exp(-(zi - MLD)/Lp)))
    
    return profile_to_dict(profile)

def get_symbolic_dvm():
    
    profile = []
    
    B3, a, zm = sym.symbols('B3 a zm')
    
    for l in LAYERS[:3]:
        psi = sym.symbols(f'POCS_{l}')
        if l == 0:
            profile.append(B3*psi*THICK[l])
        else:
            psim1 = sym.symbols(f'POCS_{l - 1}')
            ps_av = (psi + psim1)/2
            profile.append(B3*ps_av*THICK[l])
    
    for l in LAYERS[3:]:
        zi = GRID[l]
        zim1 = GRID[l - 1]
        ps_0, ps_1, ps_2 = sym.symbols('POCS_0 POCS_1 POCS_2')
        B3Ps_av = (B3/ZG)*(ps_0*THICK[0] + (ps_0 + ps_1)/2*THICK[1]
                           + (ps_1 + ps_2)/2*THICK[2])
        co = np.pi/(2*(zm - ZG))*a*ZG
        profile.append(B3Ps_av*co*((zm - ZG)/np.pi*(
            sym.cos(np.pi*(zim1 - ZG)/(zm - ZG))
            - sym.cos(np.pi*(zi - ZG)/(zm - ZG)))))
    
    return profile_to_dict(profile)
        
        