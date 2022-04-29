import numpy as np

def unpack_state_estimates(tracers, params, state_elements, x, C, layers):

    x_e = np.sqrt(np.diag(C))
    
    tracer_estimates = unpack_tracers(tracers, state_elements, x, x_e)
    residual_estimates = unpack_resids(tracers, state_elements, x, x_e, layers)
    param_estimates = unpack_params(params, state_elements, x, x_e)
    
    return tracer_estimates, residual_estimates, param_estimates

def slice_by_species(to_slice, species, state_elements):

    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == species]
    
    return sliced

def unpack_tracers(tracers, state_elements, x, x_e):
    
    tracer_estimates = {t: {} for t in tracers}
    
    for t in tracer_estimates:
        tracer_estimates[t]['posterior'] = slice_by_species(
            x, t, state_elements)
        tracer_estimates[t]['posterior_e'] = slice_by_species(
            x_e, t, state_elements)
        
    return tracer_estimates

def unpack_resids(tracers, state_elements, x, x_e, layers):
    
    resid_estimates = {t: {} for t in tracers}
    
    for t in resid_estimates:
        posterior = slice_by_species(x, f'R{t}', state_elements)
        posterior_e = slice_by_species(x_e, f'R{t}', state_elements)
        for l in layers:
            resid_estimates[t][l] = (posterior[l], posterior_e[l])
        
    return resid_estimates

def unpack_params(params, state_elements, x, x_e):
    
    param_estimates = {p: {} for p in params}
    
    for p in params:
        if params[p]['dv']:
            param_estimates[p]['posterior'] = slice_by_species(
                x, p, state_elements)
            param_estimates[p]['posterior_e'] = slice_by_species(
                x_e, p, state_elements)
        else:
            i = state_elements.index(p)
            param_estimates[p]['posterior'] = x[i]
            param_estimates[p]['posterior_e'] = x_e[i]
    
    return param_estimates