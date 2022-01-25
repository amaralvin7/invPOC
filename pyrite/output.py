#!/usr/bin/env python3
from constants import LAYERS

def write(params, residuals, inventories):

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
        for z in ('EZ', 'UMZ'):
            print(f'--------{z}--------', file=f)
            for i in inventories:
                est = inventories[i][z]
                err = inventories[i][f'{z}_e']
                print(f'{i}: {est:.2f} ± {err:.2f}', file=f)
        for j, l in enumerate(LAYERS):
            print(f'--------{l}--------', file=f)
            for i in inventories:
                est = inventories[i]['posterior'][j]
                err = inventories[i]['posterior_e'][j]
                print(f'{i}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Integrated Residuals', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in ('EZ', 'UMZ'):
            print(f'--------{z}--------', file=f)
            for r in residuals:
                est = residuals[r][z]
                err = residuals[r][f'{z}_e']
                print(f'{r}: {est:.2f} ± {err:.2f}', file=f)
        for i, l in enumerate(LAYERS):
            print(f'--------{l}--------', file=f)
            for r in residuals:
                est = residuals[r]['posterior'][i]
                err = residuals[r]['posterior_e'][i]
                print(f'{r}: {est:.2f} ± {err:.2f}', file=f)