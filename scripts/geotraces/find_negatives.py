# Checking for negative state estmates and errors for each station in each
# param set that had a successful inversion saved out
import pickle
import os

path = '/home/vamaral/pyrite/results/geotraces/mc_hard_25k_uniform_iqr'
pickled_files = [f for f in os.listdir(path) if 'stn' in f]

for p in pickled_files:
    with open(os.path.join(path, p), 'rb') as file:
        results = pickle.load(file)
        params = results['params']
        tracers = results['tracers']
        s1, s2 = p.split('.')[0].split('_')
        ps = s1[2:]
        stn = s2[3:]
        for p in params:
            if params[p]['dv']:
                for i, d in enumerate(params[p]['posterior']):
                    if not d > 0:
                        print(f'{ps}, {stn}: Negative {p} estimate at {i}')
                for i, d in enumerate(params[p]['posterior_e']):
                    if not d > 0:
                        print(f'{ps}, {stn}: Negative {p} error at {i}')
            else:
                if not params[p]['posterior'] > 0:
                    print(f'{ps}, {stn}: Negative {p} estimate')
                if not params[p]['posterior_e'] > 0:
                    print(f'{ps}, {stn}: Negative {p} error')
        for t in tracers:
            for i, d in enumerate(tracers[t]['posterior']):
                if not d > 0:
                    print(f'{ps}, {stn}: Negative {t} estimate at {i}')
            for i, d in enumerate(tracers[t]['posterior_e']):
                if not d > 0:
                    print(f'{ps}, {stn}: Negative {t} error at {i}')
    