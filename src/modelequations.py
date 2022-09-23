"""Construct and evaluate model equations."""
import numpy as np
import sympy as sym

from src.budgets import evaluate_symbolic_expression


def evaluate_model_equations(equation_elements, xk, grid, zg, umz_start, mld,
                             state_elements=None, targets=None,
                             soft_constraint=False):
    """Evaluate the model equations.

    Args:
        state_elements (list[str]): Names of state elements.
        equation_elements (list[str]): Names of state elements that have
        associated model equations (tracers only).
        xk (np.ndarray): State vector of estimates at the beginning of an
        iteration, k.
        grid (list[float]): The model grid depths.
        zg (float): The maximum grazing depth, also the base of the euphotic
        zone.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.
        mld (float): Mixed layer depth.
        targets (dict): Known model parameters (only provided in twin
        experiment). Defaults to None.

    Returns:
        f (np.ndarray): Vector of functions containing the model equations.
        F (np.ndarray): The Jacobian matrix.
    """
    f_sym = []
    if targets:
        x_sym = [sym.symbols(v) for v  in equation_elements]
    else:
        x_sym = [sym.symbols(v) for v  in state_elements]

    for element in equation_elements:
        tracer, layer = element.split('_')
        y = equation_builder(tracer, int(layer), grid, zg, umz_start, mld,
                             targets=targets, soft_constraint=soft_constraint)
        f_sym.append(y)
    
    f_sym = sym.Matrix(f_sym)
    f = np.squeeze(sym.lambdify(x_sym, f_sym, 'numpy')(*xk))
    F = sym.lambdify(x_sym, f_sym.jacobian(x_sym), 'numpy')(*xk)

    return f, F


def equation_builder(tracer, layer, grid, zg, umz_start, mld, targets=None,
                     soft_constraint=False):
    """Build a symbolic model equation for a given tracer at a given layer.

    Args:
        tracer (str): The tracer for which an equation is being built. Can be
        POCS or POCL.
        layer (int): Integer index of the model layer corresponding to the
        equation.
        grid (list[float]): The model grid depths.
        zg (float): The maximum grazing depth, also the base of the euphotic
        zone.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.
        mld (float): Mixed layer depth.
        targets (dict): Known model parameters (only provided in twin
        experiment). Defaults to None.

    Returns:
        eq (sympy.core): A symbolic representation of the model equation
        being built.
    """
    zi, zim1 = get_layer_bounds(layer, grid)
    h = zi - zim1
    in_EZ = zi <= zg

    tracers = get_tracer_symbols(layer)
    if targets:
        params = get_param_targets(layer, targets['params'])
        RPsi, RPli = get_residual_targets(layer, targets['residuals'])
    else:
        params = get_param_symbols(layer)
        if soft_constraint:
            RPsi, RPli = get_residual_symbols(layer)

    Psi, Pli = tracers[:2]
    Bm2, B2p, Bm1s, Bm1l, ws, wl, Po, Lp, B3, a, zm = params[:11]
    if layer != 0:
        Psim1, Plim1, Psa, Pla = tracers[2:]
        wsm1, wlm1 = params[11:]

    if tracer == 'POCS':
        if layer == 0:
            eq = (-ws * Psi + Bm2 * Pli * h - (B2p * Psi + Bm1s) * Psi * h
                  - B3 * Psi * h)
        else:
            eq = (-ws * Psi + wsm1 * Psim1 + Bm2 * Pla * h
                  - (B2p * Psa + Bm1s) * Psa * h)
            if in_EZ:
                eq += -B3 * Psa * h
        if targets is None:
            eq += production(layer, Po, Lp, zi, zim1, mld)
        if soft_constraint:
            eq += RPsi
    elif tracer == 'POCL':
        if layer == 0:
            eq = -wl * Pli + B2p * Psi**2 * h - (Bm2 + Bm1l) * Pli * h
        else:
            eq = (-wl * Pli + wlm1 * Plim1 + B2p * (Psa ** 2) * h
                  - (Bm2 + Bm1l) * Pla * h)
            if not in_EZ:
                eq += dvm_egestion(B3, a, zm, zg, zi, zim1, grid, umz_start)
        if soft_constraint:
            eq += RPli
    else:
        ppzf = sym.symbols('ppzf')
        eq = (ws * Psi + wl * Pli) - ppzf

    return eq


def production(layer, Po, Lp, zi, zim1, mld):
    """Build the production term. mld==None for GEOTRACES inversions."""
    if mld:
        if layer == 0:
            return Po * mld
        else:
            return Lp * Po * (sym.exp(-(zim1 - mld) / Lp) -
                              sym.exp(-(zi - mld) / Lp))

    return Lp * Po * (sym.exp(-zim1 / Lp) - sym.exp(-zi / Lp))


def dvm_egestion(B3, a, zm, zg, zi, zim1, grid, umz_start):
    """Build the DVM egestion term."""
    thick_EZ_layers = np.diff((0,) + grid[:umz_start])
    Ps_syms = [sym.symbols(f'POCS_{l}') for l in list(range(umz_start))]

    Ps_avg = Ps_syms[0] * thick_EZ_layers[0]
    for i, thick in enumerate(thick_EZ_layers[1:]):
        Ps_avg += (Ps_syms[i + 1] + Ps_syms[i]) / 2 * thick

    B3Ps_a = (B3 / zg) * Ps_avg
    co = np.pi / (2 * (zm - zg)) * a * zg
    result = B3Ps_a * co * ((zm - zg) / np.pi
                            * (sym.cos(np.pi * (zim1 - zg) / (zm - zg))
                               - sym.cos(np.pi * (zi - zg) / (zm - zg))))

    return result


def get_tracer_symbols(layer):
    """Get symbolic tracers for a given layer."""
    if layer == 0:
        Psi = sym.symbols('POCS_0')
        Pli = sym.symbols('POCL_0')
        return Psi, Pli
    else:
        Psi, Psim1 = sym.symbols(f'POCS_{layer} POCS_{layer - 1}')
        Pli, Plim1 = sym.symbols(f'POCL_{layer} POCL_{layer - 1}')
        Psa = (Psi + Psim1) / 2
        Pla = (Pli + Plim1) / 2
        return Psi, Pli, Psim1, Plim1, Psa, Pla


def get_param_symbols(layer):
    """Get symbolic model parameters for a given layer."""
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
    """Get symbolic residuals for a given layer."""
    RPsi = sym.symbols(f'RPOCS_{layer}')
    RPli = sym.symbols(f'RPOCL_{layer}')

    return RPsi, RPli


def get_param_targets(layer, targets):
    """Get known parameter values for a given layer (twin experiment only)."""
    Bm2 = targets['Bm2']['posterior'][layer]
    B2p = targets['B2p']['posterior'][layer]
    Bm1s = targets['Bm1s']['posterior'][layer]
    Bm1l = targets['Bm1l']['posterior'][layer]
    ws = targets['ws']['posterior'][layer]
    wl = targets['wl']['posterior'][layer]
    Po = None
    Lp = None
    B3 = targets['B3']['posterior']
    a = targets['a']['posterior']
    zm = targets['zm']['posterior']

    params = [Bm2, B2p, Bm1s, Bm1l, ws, wl, Po, Lp, B3, a, zm]

    if layer != 0:
        wsm1 = targets['ws']['posterior'][layer - 1]
        wlm1 = targets['wl']['posterior'][layer - 1]
        params.extend([wsm1, wlm1])

    return params


def get_residual_targets(layer, targets):
    """Get known residual values for a given layer (twin experiment only)."""
    RPsi = targets['POCS'][layer][0]
    RPli = targets['POCL'][layer][0]

    return RPsi, RPli


def get_layer_bounds(layer, grid):
    """Get depths of layer boundaries."""
    zi = grid[layer]
    zim1 = grid[layer - 1] if layer > 0 else 0

    return zi, zim1


def calculate_B2(grid, state_elements, Ckp1, tracers, params):
    """Calculate estimates of a first-order aggregation rate constant."""
    params['B2'] = {'dv': True, 'posterior': [], 'posterior_e': []}

    for i in range(len(grid)):
        B2p, Psi = sym.symbols(f'B2p_{i} POCS_{i}')
        if i == 0:
            Psa = Psi
        else:
            Psim1 = sym.symbols(f'POCS_{i-1}')
            Psa = (Psi + Psim1) / 2
        y = B2p * Psa
        estimate, error = evaluate_symbolic_expression(
            y, state_elements, Ckp1, tracers=tracers, params=params)

        params['B2']['posterior'].append(estimate)
        params['B2']['posterior_e'].append(error)
