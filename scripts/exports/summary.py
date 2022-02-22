import pickle
from itertools import product

from src.constants import LAYERS, ZONE_LAYERS

priors_from = ('NA', 'SP')
gammas = (0.5, 1, 5, 10)
rel_errs = (0.1, 0.2, 0.5, 1)

output_file = '../../results/exports/out.txt'
with open(output_file, 'w') as f:

    for (pf, g, re) in product(priors_from, gammas, rel_errs):

        save_path = f'../../results/exports/{pf}_{re}_{g}.pkl'
        with open(save_path, 'rb') as pickled:
                    unpickled = pickle.load(pickled)
                    tracers, params, residuals, inventories, int_fluxes, residence_times, turnover_times = unpickled

        print('#################################', file=f)
        print(f'GAMMA = {g}, RE = {re}, {pf}', file=f)
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
            for flx in int_fluxes:
                est, err = int_fluxes[flx][z]
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