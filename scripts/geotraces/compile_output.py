import os

import h5py
from tqdm import tqdm

files = [f for f in os.listdir('../../results/geotraces/output') if '.h5' in f]
stations = list(set([f.split('_')[0] for f in files]))


with h5py.File('../../results/geotraces/output.h5', 'w') as compilation:
    for f in tqdm(files):  # write results from each inversion file
        stn, paramset = f.split('.')[0].split('_')
        paramset_grp = compilation.create_group(f'/{stn}/{paramset}/')
        with h5py.File(f'../../results/geotraces/output/{f}', 'r') as file:
            keys = list(file.keys())
            for k in keys:
                paramset_grp.create_dataset(k, data=file[k])
