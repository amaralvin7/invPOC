#!/usr/bin/env python3
from constants import LAYERS, ZONE_LAYERS

def merge_by_keys(merge_this, into_this):
    
    for i in into_this:
        for j in merge_this[i]:
            into_this[i][j] = merge_this[i][j]

def write_output(
    params, residuals, inventories, fluxes, residence_times, turnover_times):

    file = f'out/out.txt'
    with open(file, 'w') as f:
        print('#################################', file=f)
        print(f'GAMMA = {0.5}, RE = {0.5}', file=f)
        print('#################################', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Parameter Estimates', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for p in params:
            if params[p]['dv']:
                for l in LAYERS:
                    est = params[p]['posterior'][l]
                    err = params[p]['posterior_e'][l]
                    print(f'{p} ({l}): {est:.8f} ± {err:.8f}', file=f)
            else:
                est = params[p]['posterior']
                err = params[p]['posterior_e']
                print(f'{p}: {est:.3f} ± {err:.3f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Tracer Inventories', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ZONE_LAYERS:
            print(f'--------{z}--------', file=f)
            for i in inventories:
                est, err = inventories[i][z]
                print(f'{i}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Integrated Fluxes', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ZONE_LAYERS:
            print(f'--------{z}--------', file=f)
            for flx in fluxes:
                est, err = fluxes[flx][z]
                print(f'{flx}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Integrated Residuals', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ZONE_LAYERS:
            print(f'--------{z}--------', file=f)
            for r in residuals:
                est, err = residuals[r][z]
                print(f'{r}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Residence Times', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ZONE_LAYERS:
            print(f'--------{z}--------', file=f)
            for t in inventories:
                est, err = residence_times[t][z]
                print(f'{t}: {est:.1f} ± {err:.1f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Turnover Timescales', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ZONE_LAYERS:
            print(f'--------{z}--------', file=f)
            for t in turnover_times:
                print(f'***{t}***', file=f)
                for flx in turnover_times[t][z]:
                    est, err = turnover_times[t][z][flx]
                    print(f'{flx}: {est:.3f} ± {err:.3f}',
                            file=f)