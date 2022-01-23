#!/usr/bin/env python3
import numpy as np
from constants import LAYERS

"""
To do:
- integrated residuals
"""

def unpack_state_estimates(tracers, params, state_elements, xhat, Ckp1):

    xhat_e = np.sqrt(np.diag(Ckp1))

    for t in tracers:
        tracers[t]['posterior'] = slice_by_tracer(xhat, t, state_elements)
        tracers[t]['posterior_e'] = slice_by_tracer(xhat_e, t, state_elements)

        # run.integrated_resids[t] = {}
        # for l in LAYERS:
        #     run.integrated_resids[t][l] = (
        #         xhat[state_elements.index(f'R{t}_{l}')],
        #         xhat_e[state_elements.index(f'R{t}_{l}')])

    for p in params:
        if params[p]['dv']:
            for l in LAYERS:
                lone_param = '_'.join([p, l])
                i = state_elements.index(lone_param)
                params[p]['posterior'][l] = xhat[i]
                params[p]['posterior_e'][l] = xhat_e[i]
        else:
            i = state_elements.index(p)
            params[p]['posterior'] = xhat[i]
            params[p]['posterior_e'] = xhat_e[i]

def slice_by_tracer(to_slice, tracer, state_elements):

    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == tracer]

    return sliced