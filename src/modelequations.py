import numpy as np
import sympy as sym

from src.constants import MLD

def evaluate_model_equations(
    tracers, state_elements, equation_elements, xk, grid, zg, productionbool):
    
    n_tracer_elements = len(tracers) * len(grid)
    n_state_elements = len(state_elements)

    f = np.zeros(n_tracer_elements)
    F = np.zeros((n_tracer_elements, n_state_elements))

    for i, element in enumerate(equation_elements):
        tracer, layer = element.split('_')
        y = equation_builder(tracer, int(layer), grid, zg, productionbool)
        x_sym, x_num, x_ind = extract_equation_variables(state_elements, y, xk)
        f[i] = sym.lambdify(x_sym, y)(*x_num)
        for j, x in enumerate(x_sym):
            dy = y.diff(x)
            dx_sym, dx_num, _ = extract_equation_variables(state_elements,
                                                           dy, xk)
            F[i, x_ind[j]] = sym.lambdify(dx_sym, dy)(*dx_num)

    return f, F

def extract_equation_variables(state_elements, y, xk):

    x_symbolic = list(y.free_symbols)
    x_numerical = []
    x_indices = []

    for x in x_symbolic:
        element_index = state_elements.index(x.name)
        x_indices.append(element_index)
        x_numerical.append(xk[element_index])

    return x_symbolic, x_numerical, x_indices

def equation_builder(tracer, layer, grid, zg, productionbool):

    zi = grid[layer]
    zim1 = grid[grid.index(zi) - 1] if layer > 0 else 0
    h = zi - zim1
    in_EZ = zi <= zg
    
    t_syms = get_tracer_symbols(layer)
    p_syms = get_param_symbols(layer)
    RPsi, RPli = get_residual_symbols(layer)
    
    Psi, Pli = t_syms[:2]
    Bm2, B2p, Bm1s, Bm1l, ws, wl, Po, Lp, B3, a, zm = p_syms[:11]
    if layer != 0:
        Psim1, Plim1, Psa, Pla = t_syms[2:]
        wsm1, wlm1 = p_syms[11:]        

    if tracer == 'POCS':
        eq = production(productionbool, layer, Po, Lp, zi, zim1)
        if layer == 0:
            eq += (-ws*Psi + Bm2*Pli*h - (B2p*Psi + Bm1s)*Psi*h + RPsi
                   - B3*Psi*h)
        else:
            eq += (-ws*Psi + wsm1*Psim1 + Bm2*Pla*h - (B2p*Psa + Bm1s)*Psa*h 
                   + RPsi)
            if in_EZ:
                eq += -B3*Psa*h
    else:
        if layer == 0:
            eq = -wl*Pli + B2p*Psi**2*h - (Bm2 + Bm1l)*Pli*h + RPli
        else:
            eq = (-wl*Pli + wlm1*Plim1 + B2p*Psa**2*h - (Bm2 + Bm1l)*Pla*h
                  + RPli)
            if not in_EZ:
                eq += dvm_excretion(B3, a, zm, zg, zi, zim1, grid)

    return eq

def production(productionbool, layer, Po, Lp, zi, zim1):
    
    if productionbool:
        if layer == 0:
            return Po*MLD
        else:
            return Lp*Po*(sym.exp(-(zim1 - MLD)/Lp)- sym.exp(-(zi - MLD)/Lp))
        
    return Lp*Po*(sym.exp(-zim1/Lp) - sym.exp(-zi/Lp))

def dvm_excretion(B3, a, zm, zg, zi, zim1, grid):
    
    EZ_layers = list(range(grid.index(zg) + 1))
    thick_EZ_layers = np.diff((0,) + grid[:len(EZ_layers)])
    ps_syms = [sym.symbols(f'POCS_{l}') for l in EZ_layers]

    Ps_avg = ps_syms[0] * thick_EZ_layers[0]
    for i, thick in enumerate(thick_EZ_layers[1:]):
        Ps_avg += (ps_syms[i+1] + ps_syms[i])/2 * thick

    B3Ps_a = (B3/zg)*Ps_avg
    co = np.pi/(2*(zm - zg))*a*zg
    result = B3Ps_a*co*((zm - zg)/np.pi*(sym.cos(np.pi*(zim1 - zg)/(zm - zg))
                                         - sym.cos(np.pi*(zi - zg)/(zm - zg))))
    
    return result

def get_tracer_symbols(layer):
    
    if layer == 0:
        Psi = sym.symbols('POCS_0')
        Pli = sym.symbols('POCL_0')
        return Psi, Pli
    else:
        Psi, Psim1 = sym.symbols(f'POCS_{layer} POCS_{layer - 1}')
        Pli, Plim1 = sym.symbols(f'POCL_{layer} POCL_{layer - 1}')
        Psa = (Psi + Psim1)/2
        Pla = (Pli + Plim1)/2
        return Psi, Pli, Psim1, Plim1, Psa, Pla

def get_param_symbols(layer):
    
    Bm2 = sym.symbols(f'Bm2_{layer}')
    B2p = sym.symbols(f'B2p_{layer}')
    Bm1s = sym.symbols(f'Bm1s_{layer}')
    Bm1l = sym.symbols(f'Bm1l_{layer}')
    ws = sym.symbols(f'ws_{layer}')
    wl = sym.symbols(f'wl_{layer}')
    Po = sym.symbols('Po')
    Lp = sym.symbols('Lp')
    B3 = sym.symbols('B3')
    a = sym.symbols('a')
    zm = sym.symbols('zm')
    
    params = [Bm2, B2p, Bm1s, Bm1l, ws, wl, Po, Lp, B3, a, zm]
    
    if layer != 0:
        wsm1 = sym.symbols(f'ws_{layer - 1}')
        wlm1 = sym.symbols(f'wl_{layer - 1}')
        params.extend([wsm1, wlm1])
    
    return params

def get_residual_symbols(layer):
    
    RPsi = sym.symbols(f'RPOCS_{layer}')
    RPli = sym.symbols(f'RPOCL_{layer}')
    
    return RPsi, RPli
    