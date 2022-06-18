"""Calculate residence and turnover timescales."""
from itertools import product

from src.budgets import evaluate_symbolic_expression


def calculate_residence_times(inventories_sym, int_fluxes_sym, int_fluxes,
                              residuals_sym, residuals, tracers, params,
                              state_elements, Ckp1, zone_layers, umz_start):
    """Calculate residence times for tracers in all model layers and zones.

    Args:
        inventories_sym (dict): Symbolic expressions for tracer inventories.
        int_fluxes_sym (dict): Symbolic expressions for integrated fluxes.
        int_fluxes (dict): Numerical integrated fluxes.
        residuals_sym (dict): Symbolic expressions for integrated residuals.
        residuals (dict): Numerical residual values.
        tracers (dict): Data container for tracers.
        params (dict): Data container for model parameters.
        state_elements (list[str]): Names of state elements.
        Ckp1 (np.ndarray): Error covariance matrix of posterior estimates.
        zone_layers (tuple): Model zones (EZ and UMZ) and layers.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.

    Returns:
        res_times (dict): Data container for residence times.
    """
    res_times = {tracer: {} for tracer in inventories_sym}

    for (tracer, z) in product(res_times, zone_layers):
        inventory = inventories_sym[tracer][z]
        fluxes = sum_of_fluxes(
            tracer, z, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
            umz_start)
        res_times[tracer][z] = evaluate_symbolic_expression(
            inventory / fluxes, state_elements, Ckp1, tracers=tracers,
            residuals=residuals, params=params)

    return res_times


def sum_of_fluxes(tracer, z, int_fluxes_sym, int_fluxes, residuals_sym,
                  residuals, umz_start):
    """Get the (symbolic) sum of fluxes for a tracer in a zone or layer."""
    fluxes_in = {'POCS': ['production', 'disaggregation'],
                 'POCL': ['aggregation']}
    sum_of_fluxes = 0
    in_umz = (z == 'UMZ' or (isinstance(z, int) and z >= umz_start))

    for f in fluxes_in[tracer]:
        sum_of_fluxes += int_fluxes_sym[f][z]
    if int_fluxes[f'sinkdiv_{tracer[-1]}'][z][0] < 0:
        sum_of_fluxes += -int_fluxes_sym[f'sinkdiv_{tracer[-1]}'][z]
    if residuals[tracer][z][0] > 0:
        sum_of_fluxes += residuals_sym[tracer][z]
    if tracer == 'POCL' and in_umz:
        sum_of_fluxes += int_fluxes_sym['dvm'][z]

    return sum_of_fluxes


def calculate_turnover_times(inventories_sym, int_fluxes_sym, int_fluxes,
                             tracers, params, state_elements, Ckp1,
                             zone_layers):
    """Calculate turnover times for tracers in all model layers and zones."""
    flux_tracers = {'sinkdiv_S': ('POCS',), 'sinkdiv_L': ('POCL',),
                    'remin_S': ('POCS',), 'remin_L': ('POCL',),
                    'aggregation': ('POCS', 'POCL'),
                    'disaggregation': ('POCS', 'POCL'),
                    'production': ('POCS',), 'dvm': ('POCS', 'POCL')}

    turnover = {t: {z: {} for z in zone_layers} for t in tracers}

    for (t, z, f) in product(tracers, zone_layers, int_fluxes):
        if t in flux_tracers[f]:
            inventory = inventories_sym[t][z]
            flux = int_fluxes_sym[f][z]
            turnover[t][z][f] = evaluate_symbolic_expression(
                inventory / flux, state_elements, Ckp1, tracers=tracers,
                params=params)

    return turnover
