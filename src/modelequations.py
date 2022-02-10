import numpy as np
import sympy as sym

from src.constants import LAYERS, THICK, GRID, MLD, ZG

def evaluate_model_equations(tracers, state_elements, equation_elements, xk):
    
    n_tracer_elements = len(tracers) * len(LAYERS)
    n_state_elements = len(state_elements)

    f = np.zeros(n_tracer_elements)
    F = np.zeros((n_tracer_elements, n_state_elements))

    for i, element in enumerate(equation_elements):
        tracer, layer = element.split('_')
        y = equation_builder(tracer, int(layer))
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

def equation_builder(tracer, layer):

    zi = GRID[layer]
    zim1 = zi - THICK[layer]
    h = zi - zim1
    
    t_syms = get_tracer_symbols(layer)
    p_syms = get_param_symbols(layer)
    RPsi, RPli = get_residual_symbols(layer)
    
    Psi, Pli = t_syms[:2]
    Bm2, B2p, Bm1s, Bm1l, ws, wl, P30, Lp, B3, a, zm = p_syms[:11]
    if layer != 0:
        Psim1, Plim1, Psa, Pla = t_syms[2:]
        wsm1, wlm1 = p_syms[11:]        

    if tracer == 'POCS':
        if layer == 0:
            eq = (-ws*Psi + Bm2*Pli*h - (B2p*Psi + Bm1s)*Psi*h + RPsi
                  - B3*Psi*h) + P30*MLD
        else:
            eq = (-ws*Psi + wsm1*Psim1 + Bm2*Pla*h - (B2p*Psa + Bm1s)*Psa*h 
                  + RPsi + Lp*P30*(sym.exp(-(zim1 - MLD)/Lp) 
                                   - sym.exp(-(zi - MLD)/Lp)))
            if layer in (1, 2):
                eq += -B3*Psa*h
    else:
        if layer == 0:
            eq = -wl*Pli + B2p*Psi**2*h - (Bm2 + Bm1l)*Pli*h + RPli
        else:
            eq = -wl*Pli + wlm1*Plim1 + B2p*Psa**2*h - (Bm2 + Bm1l)*Pla*h + RPli
            if layer in (3, 4, 5, 6):
                Ps_0, Ps_1, Ps_2 = sym.symbols('POCS_0 POCS_1 POCS_2')
                B3Ps_av = (B3/ZG)*(Ps_0*30 + (Ps_0 + Ps_1)/2*20
                                   + (Ps_1 + Ps_2)/2*50)
                co = np.pi/(2*(zm - ZG))*a*ZG
                eq += B3Ps_av*co*((zm - ZG)/np.pi*(
                        sym.cos(np.pi*(zim1 - ZG)/(zm - ZG))
                        - sym.cos(np.pi*(zi - ZG)/(zm - ZG))))
    return eq

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
    P30 = sym.symbols('P30')
    Lp = sym.symbols('Lp')
    B3 = sym.symbols('B3')
    a = sym.symbols('a')
    zm = sym.symbols('zm')
    
    params = [Bm2, B2p, Bm1s, Bm1l, ws, wl, P30, Lp, B3, a, zm]
    
    if layer != 0:
        wsm1 = sym.symbols(f'ws_{layer - 1}')
        wlm1 = sym.symbols(f'wl_{layer - 1}')
        params.extend([wsm1, wlm1])
    
    return params

def get_residual_symbols(layer):
    
    RPsi = sym.symbols(f'RPOCS_{layer}')
    RPli = sym.symbols(f'RPOCL_{layer}')
    
    return RPsi, RPli
    