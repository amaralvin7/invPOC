import sympy as sym
import numpy as np

from src.budgets import eval_sym_expression
"""
todo: combine for loop statements in get_symbolic_int_fluxes
        (it is the way it is to preserve order in original output file)
"""

def get_symbolic_int_fluxes(umz_start, layers, thick, grid, mld, zg):

    int_fluxes_sym = {}
    
    for size in ('S', 'L'):
        int_fluxes_sym[f'sinkdiv_{size}'] = get_symbolic_sinkdiv(
            size, umz_start, layers)
        
    for size in ('S', 'L'):    
        int_fluxes_sym[f'remin_{size}'] = get_symbolic_remin(
            size, umz_start, layers, thick)
    
    int_fluxes_sym['aggregation'] = get_symbolic_agg(umz_start, layers, thick)
    int_fluxes_sym['disaggregation'] = get_symbolic_disagg(
        umz_start, layers, thick)
    int_fluxes_sym['production'] = get_symbolic_production(
        umz_start, layers, grid, mld)
    int_fluxes_sym['dvm'] = get_symbolic_dvm(
        umz_start, layers, thick, grid, zg)

    return int_fluxes_sym

def profile_to_dict(profile, umz_start):
    
    dictionary = {layer: value for layer, value in enumerate(profile)}
    dictionary['EZ'] = np.sum(profile[:umz_start])
    dictionary['UMZ'] = np.sum(profile[umz_start:])

    return dictionary

def avg_tracer_concentration_sym(tracer, layer):
    
    if layer == 0:
        return sym.symbols(f'{tracer}_{layer}')
    
    ti, tim1 = sym.symbols(f'{tracer}_{layer} {tracer}_{layer - 1}')
    
    return (ti + tim1)/2
        
def get_symbolic_sinkdiv(size, umz_start, layers):
    
    profile = []
    
    for l in layers:
        wi, ti = sym.symbols(f'w{size.lower()}_{l} POC{size}_{l}')
        if l == 0:
            profile.append(wi*ti)
        else:
            wim1, tim1 = sym.symbols(
                f'w{size.lower()}_{l - 1} POC{size}_{l - 1}')
            profile.append(wi*ti - wim1*tim1)
    
    return profile_to_dict(profile, umz_start)

def get_symbolic_remin(size, umz_start, layers, thick):
    
    profile = []
    
    for l in layers:
        Bm1i = sym.symbols(f'Bm1{size.lower()}_{l}')
        ti = avg_tracer_concentration_sym(f'POC{size}', l)
        profile.append(Bm1i*ti*thick[l])

    return profile_to_dict(profile, umz_start)

def get_symbolic_agg(umz_start, layers, thick):
    
    profile = []
    
    for l in layers:
        B2pi = sym.symbols(f'B2p_{l}')
        psi = avg_tracer_concentration_sym(f'POCS', l)
        profile.append((B2pi*psi**2)*thick[l])
        
    return profile_to_dict(profile, umz_start)

def get_symbolic_disagg(umz_start, layers, thick):
    
    profile = []
    
    for l in layers:
        Bm2i = sym.symbols(f'Bm2_{l}')
        pli = avg_tracer_concentration_sym(f'POCL', l)
        profile.append(Bm2i*pli*thick[l])
        
    return profile_to_dict(profile, umz_start)

def get_symbolic_production(umz_start, layers, grid, mld):
    
    profile = []
    
    for l in layers:
        Po, Lp = sym.symbols('Po Lp')
        if l == 0:
            profile.append(Po*mld)
        else:
            zi = grid[l]
            zim1 = grid[l - 1]
            profile.append(Lp*Po*(sym.exp(-(zim1 - mld)/Lp)
                                   - sym.exp(-(zi - mld)/Lp)))
    
    return profile_to_dict(profile, umz_start)

def get_symbolic_dvm(umz_start, layers, thick, grid, zg):
    
    profile = []
    
    B3, a, zm = sym.symbols('B3 a zm')
    
    for l in layers[:umz_start]:
        psi = sym.symbols(f'POCS_{l}')
        if l == 0:
            profile.append(B3*psi*thick[l])
        else:
            psim1 = sym.symbols(f'POCS_{l - 1}')
            ps_av = (psi + psim1)/2
            profile.append(B3*ps_av*thick[l])
    
    for l in layers[umz_start:]:
        zi = grid[l]
        zim1 = grid[l - 1]
        ps_0, ps_1, ps_2 = sym.symbols('POCS_0 POCS_1 POCS_2')
        B3Ps_av = (B3/zg)*(ps_0*thick[0] + (ps_0 + ps_1)/2*thick[1]
                           + (ps_1 + ps_2)/2*thick[2])
        co = np.pi/(2*(zm - zg))*a*zg
        profile.append(B3Ps_av*co*((zm - zg)/np.pi*(
            sym.cos(np.pi*(zim1 - zg)/(zm - zg))
            - sym.cos(np.pi*(zi - zg)/(zm - zg)))))
    
    return profile_to_dict(profile, umz_start)

def calculate_sinking_fluxes(layers, state_elements, Ckp1, tracers, params):
    
    sink_fluxes = {'S': [], 'L': [], 'T': []}

    for l in layers:
        ws, ps, wl, pl = sym.symbols(f'ws_{l} POCS_{l} wl_{l} POCL_{l}')
        sink_fluxes['S'].append(
            eval_sym_expression(ws*ps, state_elements, Ckp1, tracers=tracers,
            params=params))
        sink_fluxes['L'].append(
            eval_sym_expression(wl*pl, state_elements, Ckp1, tracers=tracers,
            params=params))
        sink_fluxes['T'].append(
            eval_sym_expression(ws*ps + wl*pl, state_elements, Ckp1,
            tracers=tracers, params=params))
    
    return sink_fluxes

        
        