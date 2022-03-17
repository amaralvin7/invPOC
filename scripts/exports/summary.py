import pickle
from itertools import product

priors_from = ('NA', 'SP')
gammas = (0.5, 1, 5, 10)
rel_errs = (0.1, 0.2, 0.5, 1)

grid = (30, 50, 100, 150, 200, 330, 500)
layers = tuple(range(len(grid)))
zone_layers = ('EZ', 'UMZ') + layers

output_file = '../../results/exports/out.txt'
with open(output_file, 'w') as f:

    for (pf, g, re) in product(priors_from, gammas, rel_errs):

        save_path = f'../../results/exports/{pf}_{re}_{g}.pkl'
        with open(save_path, 'rb') as pickled:
                    results = pickle.load(pickled)

        print('#################################', file=f)
        print(f'GAMMA = {g}, RE = {re}, {pf}', file=f)
        print('#################################', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Parameter Estimates', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for p in results['params']:
            if results['params'][p]['dv']:
                for l in layers:
                    est = results['params'][p]['posterior'][l]
                    err = results['params'][p]['posterior_e'][l]
                    print(f'{p} ({l}): {est:.8f} ± {err:.8f}', file=f)
            else:
                est = results['params'][p]['posterior']
                err = results['params'][p]['posterior_e']
                print(f'{p}: {est:.3f} ± {err:.3f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Tracer Inventories', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in zone_layers:
            print(f'--------{z}--------', file=f)
            for i in results['inventories']:
                est, err = results['inventories'][i][z]
                print(f'{i}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Integrated Fluxes', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in zone_layers:
            print(f'--------{z}--------', file=f)
            for flx in results['int_fluxes']:
                est, err = results['int_fluxes'][flx][z]
                print(f'{flx}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Integrated Residuals', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in zone_layers:
            print(f'--------{z}--------', file=f)
            for r in results['residuals']:
                est, err = results['residuals'][r][z]
                print(f'{r}: {est:.2f} ± {err:.2f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Residence Times', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in zone_layers:
            print(f'--------{z}--------', file=f)
            for t in results['residence_times']:
                est, err = results['residence_times'][t][z]
                print(f'{t}: {est:.1f} ± {err:.1f}', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        print('Turnover Timescales', file=f)
        print('+++++++++++++++++++++++++++', file=f)
        for z in zone_layers:
            print(f'--------{z}--------', file=f)
            for t in results['turnover_times']:
                print(f'***{t}***', file=f)
                for flx in results['turnover_times'][t][z]:
                    est, err = results['turnover_times'][t][z][flx]
                    print(f'{flx}: {est:.3f} ± {err:.3f}',
                            file=f)