#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:05:20 2020

@author: vamaral

!!!WARNING!!!
OVERWRITES ANY EXISTING targets.csv FILE

Generates a given number of sets of target values to be used in twin experiments.

"""

import csv
import numpy as np

def lognormsamp(mux, sigx):
    '''Given the mean and standard deviation of a lognormally distributed
    variable, return a randomly sampled instance from that distribution
    '''   
    muy = np.log(mux**2/np.sqrt(mux**2+sigx**2)) 
    sigy = np.sqrt(np.log(1+sigx**2/mux**2))
    return np.random.lognormal(muy,sigy)

def uniformsamp(mux, sigx):
    lo = max(mux-sigx,0)
    hi = mux+sigx
    return np.random.uniform(lo,hi)

#number of sets to generate
ngroups = 10

#some conversion factors
mm = 12 #molar mass of C (12 g/mol)
dpy = 365.24 #days per year

#param info
layers = ['A','B']
params = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l', 'Gh', 'Lp']
params_dv = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l'] #depth-varying params
params_dc = ['Gh', 'Lp'] #depth-constant params
pdi = {param:{} for param in params}

#add a key for each parameter designating if it is depth-varying or constant
for k in pdi.keys():
    if k in params_dv: pdi[k]['dv'] = 1
    else: pdi[k]['dv'] = 0

#priors
p_o = {'ws':2, #m/d
        'wl':20, #m/d, reported from Murnane 1990.
        'B2p':0.5*mm/dpy, #m3/(mmol*yr), arbritrary
        'Bm2':400/dpy, #from Murnane 1994, converted to d
        'Bm1s':36/dpy, #from Clegg surface value, converted to d
        'Bm1l':0.15, #based on preliminary RESPIRE data from A Santoro
        'Gh':0.28, #prior set to typical NPP shared data value (divided by h), error is 25% of that. mg/m2/d converted to mmol
        'Lp':28} #from NPP data, m
#prior errors
p_oe = {'ws':2, 
        'wl':15, 
        'B2p':0.5*mm/dpy, 
        'Bm2':1000/dpy, 
        'Bm1s':36/dpy, 
        'Bm1l':0.15, 
        'Gh':0.12, 
        'Lp':28*0.5}

#update entries in pd
for p in pdi.keys():
        pdi[p]['o'] = p_o[p]
        pdi[p]['oe'] = p_oe[p]

#write a csv file
with open('targets.csv', 'w') as file:
    fwriter = csv.DictWriter(file, ['group', 'param', 'val'])
    fwriter.writeheader()
    for g in range(1,ngroups+1):
        for p in params:
            if pdi[p]['dv']:
                for l in layers:
                    #val = lognormsamp(pdi[p]['o'], pdi[p]['oe'])
                    val = uniformsamp(pdi[p]['o'], pdi[p]['oe'])
                    fwriter.writerow({
                        'group':g, 'param':'_'.join([p,l]), 'val':val
                        })
            else:
                #val = lognormsamp(pdi[p]['o'], pdi[p]['oe'])
                val = uniformsamp(pdi[p]['o'], pdi[p]['oe'])
                fwriter.writerow({
                    'group':g, 'param':p, 'val':val
                    })            