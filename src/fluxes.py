"""Calculate POC fluxes."""
import numpy as np
import sympy as sym

from src.budgets import evaluate_symbolic_expression
from src.modelequations import dvm_egestion, production, get_layer_bounds


def get_symbolic_int_fluxes(umz_start, layers, thick, grid, zg, mld):
    """Get symbolic expressions for fluxes integrated by model layer."""
    int_fluxes_sym = {}

    for size in ('S', 'L'):
        int_fluxes_sym[f'sinkdiv_{size}'] = get_symbolic_sinkdiv(
            size, umz_start, layers)
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
    """Create a dict (with model layers and zones as keys) from profile."""
    dictionary = {layer: value for layer, value in enumerate(profile)}
    dictionary['EZ'] = np.sum(profile[:umz_start])
    dictionary['UMZ'] = np.sum(profile[umz_start:])

    return dictionary


def avg_tracer_concentration_sym(tracer, layer):
    """Get symbolic expression for average tracer concentration in a layer."""
    if layer == 0:
        return sym.symbols(f'{tracer}_{layer}')

    ti, tim1 = sym.symbols(f'{tracer}_{layer} {tracer}_{layer - 1}')

    return (ti + tim1) / 2


def get_symbolic_sinkdiv(size, umz_start, layers):
    """Get symbolic expressions for sinking flux divergences in all layers."""
    profile = []

    for l in layers:
        wi, ti = sym.symbols(f'w{size.lower()}_{l} POC{size}_{l}')
        if l == 0:
            profile.append(wi * ti)
        else:
            wim1, tim1 = sym.symbols(
                f'w{size.lower()}_{l - 1} POC{size}_{l - 1}')
            profile.append(wi * ti - wim1 * tim1)

    return profile_to_dict(profile, umz_start)


def get_symbolic_remin(size, umz_start, layers, thick):
    """Get symbolic expressions for remineralization in all layers."""
    profile = []

    for l in layers:
        Bm1i = sym.symbols(f'Bm1{size.lower()}_{l}')
        ti = avg_tracer_concentration_sym(f'POC{size}', l)
        profile.append(Bm1i * ti * thick[l])

    return profile_to_dict(profile, umz_start)


def get_symbolic_agg(umz_start, layers, thick):
    """Get symbolic expressions for aggregation in all layers."""
    profile = []

    for l in layers:
        B2pi = sym.symbols(f'B2p_{l}')
        psi = avg_tracer_concentration_sym(f'POCS', l)
        profile.append((B2pi * psi**2) * thick[l])

    return profile_to_dict(profile, umz_start)


def get_symbolic_disagg(umz_start, layers, thick):
    """Get symbolic expressions for disaggregation in all layers."""
    profile = []

    for l in layers:
        Bm2i = sym.symbols(f'Bm2_{l}')
        pli = avg_tracer_concentration_sym(f'POCL', l)
        profile.append(Bm2i * pli * thick[l])

    return profile_to_dict(profile, umz_start)


def get_symbolic_production(umz_start, layers, grid, mld):
    """Get symbolic expressions for integrated particle in all layers."""
    profile = []

    for l in layers:
        Po, Lp = sym.symbols('Po Lp')
        zi, zim1 = get_layer_bounds(l, grid)
        profile.append(production(l, Po, Lp, zi, zim1, mld))

    return profile_to_dict(profile, umz_start)


def get_symbolic_dvm(umz_start, layers, thick, grid, zg):
    """Get symbolic expressions for DVM fluxes in all layers."""
    profile = []

    B3, a, zm = sym.symbols('B3 a zm')

    for l in layers[:umz_start]:
        psi = sym.symbols(f'POCS_{l}')
        if l == 0:
            profile.append(B3 * psi * thick[l])
        else:
            psim1 = sym.symbols(f'POCS_{l - 1}')
            ps_av = (psi + psim1) / 2
            profile.append(B3 * ps_av * thick[l])

    for l in layers[umz_start:]:
        zi, zim1 = get_layer_bounds(l, grid)
        profile.append(dvm_egestion(B3, a, zm, zg, zi, zim1, grid, umz_start))

    return profile_to_dict(profile, umz_start)


def sinking_fluxes(layers, state_elements, Ckp1, tracers, params):
    """Calculate sinking fluxes at all grid depths."""
    sink_fluxes = {'S': [], 'L': [], 'T': []}

    for l in layers:
        ws, ps, wl, pl = sym.symbols(f'ws_{l} POCS_{l} wl_{l} POCL_{l}')
        sink_fluxes['S'].append(evaluate_symbolic_expression(
            ws * ps, state_elements, Ckp1, tracers=tracers, params=params))
        sink_fluxes['L'].append(evaluate_symbolic_expression(
            wl * pl, state_elements, Ckp1, tracers=tracers, params=params))
        sink_fluxes['T'].append(evaluate_symbolic_expression(
            ws * ps + wl * pl, state_elements, Ckp1, tracers=tracers,
            params=params))

    return sink_fluxes


def production_profile(layers, state_elements, Ckp1, tracers, params, grid,
                       mld=None):
    """Calculate production profile. If mld, start decaying at base of ML."""
    profile = []
    Po, Lp = sym.symbols('Po Lp')

    for l in layers:
        z = grid[l]
        if mld:
            y = Po * sym.exp(-(z - mld) / Lp)
        else:
            y = Po * sym.exp(-(z) / Lp)
        profile.append(evaluate_symbolic_expression(
            y, state_elements, Ckp1, tracers=tracers, params=params))

    return profile
