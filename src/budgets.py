"""Calculate budgets and inventories."""
from itertools import product

import numpy as np
import sympy as sym


def evaluate_symbolic_expression(y, state_elements, C, tracers=[],
                                 residuals=[], params=[]):
    """Evaluate a symbolic expression with its corresponding numerical values.

    For elements in y that require numerical values, dictionaries in which
    those values can be found should be passed as keyword arguments. Otherwise,
    the keyword arguments are initialized as empty lists so that numerical
    values are only searchd for in the dictionaries provided. For example, if
    y is thought to contain tracers, then a dicitonary that contains estimates
    of tracers should be passed to the 'tracers' keyword argument.

    Args:
        y (sympy.core): Symbolic expression to be evaluated.
        state_elements (list[str]): Names of state elements.
        C (np.ndarray): Error covariance matrix.
        tracers (dict): Data container for tracers.
        residuals (dict): Data container for residuals.
        params (dict): Data container for model parameters.

    Returns:
        result (float): Result of the expression.
        error (float): Propagated error of the result.
    """
    x_symbolic = list(y.free_symbols)
    x_numerical = []
    x_indices = []

    for x in x_symbolic:
        x_indices.append(state_elements.index(x.name))
        if '_' in x.name:  # if it varies with depth
            element, layer = x.name.split('_')
            layer = int(layer)
            if element in tracers:
                x_numerical.append(tracers[element]['posterior'][layer])
            elif element[1:] in residuals:
                x_numerical.append(residuals[element[1:]][layer][0])
            else:  # if it's a depth-varying parameter
                x_numerical.append(params[element]['posterior'][layer])
        else:  # if it's a depth-independent parameter
            x_numerical.append(params[x.name]['posterior'])

    variance_sym = 0  # symbolic expression for variance of y
    derivs = [y.diff(x) for x in x_symbolic]

    # sub-CVM corresponding to state elements in y
    cvm = C[np.ix_(x_indices, x_indices)]
    nrows, ncols = cvm.shape

    for (i, j) in product(range(nrows), range(ncols)):
        if i > j:
            continue
        if i == j:
            variance_sym += (derivs[i]**2) * cvm[i, j]
        else:
            variance_sym += 2 * derivs[i] * derivs[j] * cvm[i, j]

    result = sym.lambdify(x_symbolic, y)(*x_numerical)
    variance = sym.lambdify(x_symbolic, variance_sym)(*x_numerical)
    error = np.sqrt(variance)

    return result, error


def get_symbolic_residuals(residuals, umz_start, layers):
    """Get symbolic residual expressions for model layers and zones.

    Args:
        residuals (dict): Data container for residuals.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.
        layers (tuple[int]): Integer indexes of model layers.

    Returns:
        resids_sym (dict): Symbolic residual expressions by model layer and
        zone.
    """
    resids_sym = {r: {} for r in residuals}

    for r in resids_sym:
        profile = [sym.symbols(f'R{r}_{l}') for l in layers]
        resids_sym[r]['EZ'] = np.sum(profile[:umz_start])
        resids_sym[r]['UMZ'] = np.sum(profile[umz_start:])
        for l in layers:
            resids_sym[r][l] = profile[l]

    return resids_sym


def get_symbolic_inventories(tracers, umz_start, layers, thick):
    """Get symbolic tracer inventory expressions for model layers and zones.

    Args:
        tracers (dict): Data container for tracers.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.
        layers (tuple[int]): Integer indexes of model layers.
        thick (np.ndarray): Thicknesses of mode layers.

    Returns:
        inventories_sym (dict): Symbolic tracer inventory expressions by model
        layer and zone.
    """
    inventories_sym = {t: {} for t in tracers}

    for t in tracers:
        concentrations = [sym.symbols(f'{t}_{l}') for l in layers]
        profile = [concentrations[0] * thick[0]]  # mixed layer
        for i, h in enumerate(thick[1:], 1):  # all other layers
            avg_conc = np.mean([concentrations[i], concentrations[i - 1]])
            profile.append(avg_conc * h)
        inventories_sym[t]['EZ'] = np.sum(profile[:umz_start])
        inventories_sym[t]['UMZ'] = np.sum(profile[umz_start:])
        for l in layers:
            inventories_sym[t][l] = profile[l]

    return inventories_sym


def integrate_by_zone(symbolic, state_elements, C, **state_element_types):
    """Integrate symbolic expressions by model zone."""
    integrated = {k: {} for k in symbolic}

    for (k, z) in product(integrated, ('EZ', 'UMZ')):
        y = symbolic[k][z]
        integrated[k][z] = evaluate_symbolic_expression(
            y, state_elements, C, **state_element_types)

    return integrated


def integrate_by_zone_and_layer(symbolic, state_elements, C, layers,
                                **state_element_types):
    """Integrate symbolic expressions by model layer and zone."""
    integrated = integrate_by_zone(
        symbolic, state_elements, C, **state_element_types)

    for (k, l) in product(integrated, layers):
        y = symbolic[k][l]
        integrated[k][l] = evaluate_symbolic_expression(
            y, state_elements, C, **state_element_types)

    return integrated
