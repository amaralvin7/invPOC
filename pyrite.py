#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:55:53 2021

@author: Vinicius J. Amaral

PYRITE Model (Particle cYcling Rates from Inversion of Tracers in the ocEan)
"""
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
#from varname import nameof

class PyriteModel:

    def __init__(self, gammas=[0.02]):
        
        self.GAMMAS = gammas
        self.MIXED_LAYER_DEPTH = 30
        self.MAX_DEPTH = 500
        self.GRID_STEP = 5
        self.GRID = np.arange(self.MIXED_LAYER_DEPTH,
                              self.MAX_DEPTH + self.GRID_STEP,
                              self.GRID_STEP)
        self.N_GRID_POINTS = len(self.GRID)
        self.BOUNDARY = 112.5
        self.MOLAR_MASS_C = 12
        self.DAYS_PER_YEAR = 365.24
        
        self.load_data()
        self.unpack_tracers()
        self.define_params()

    def __repr__(self):

        return f'PyriteModel(gammas={self.GAMMAS})'
                    
    def load_data(self):
        
        self.DATA = pd.read_excel('pyrite_data.xlsx',sheet_name=None)
    
    def unpack_tracers(self):
        
        self.POC_S_MEAN = self.DATA['poc_means']['SSF_mean']/self.MOLAR_MASS_C
        self.POC_S_SE = self.DATA['poc_means']['SSF_se']/self.MOLAR_MASS_C
        self.POC_L_MEAN = self.DATA['poc_means']['LSF_mean']/self.MOLAR_MASS_C
        self.POC_L_SE = self.DATA['poc_means']['LSF_se']/self.MOLAR_MASS_C
    
    def define_params(self):
        
        P30_PRIOR, P30_PRIOR_E, LP_PRIOR, LP_PRIOR_E = self.process_npp_data()
        
        self.ws = PyriteParam(2, 2, 'ws', '$w_S$')
        self.wl = PyriteParam(20, 15, 'wl', '$w_L$')
        self.B2p = PyriteParam(0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               'B2p', '$\\beta^,_2$')
        self.Bm2 = PyriteParam(400*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               10000*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               'Bm2', '$\\beta_{-2}$')
        self.Bm1s = PyriteParam(0.1, 0.1, 'Bm1s', '$\\beta_{-1,S}$')
        self.Bm1l = PyriteParam(0.15, 0.15, 'Bm1l', '$\\beta_{-1,L}$')
        self.P30 = PyriteParam(P30_PRIOR, P30_PRIOR_E, 'P30', '$\.P_{S,30}$',
                               depth_vary=False)
        self.Lp = PyriteParam(LP_PRIOR, LP_PRIOR_E, 'Lp', '$L_P$',
                              depth_vary=False)

        self.model_params = (self.ws, self.wl, self.B2p, self.Bm2, self.Bm1s,
                             self.Bm1l, self.P30, self.Lp)

    def process_npp_data(self):
        
        NPP_DATA_RAW = self.DATA['npp']
        NPP_DATA_CLEAN = NPP_DATA_RAW.loc[(NPP_DATA_RAW['npp'] > 0)]
        
        MIXED_LAYER_UPPER_BOUND, MIXED_LAYER_LOWER_BOUND = 28, 35
        
        NPP_MIXED_LAYER = NPP_DATA_CLEAN.loc[
            (NPP_DATA_CLEAN['target_depth'] >= MIXED_LAYER_UPPER_BOUND) & 
            (NPP_DATA_CLEAN['target_depth'] <= MIXED_LAYER_LOWER_BOUND)]
        
        NPP_BELOW_MIXED_LAYER = NPP_DATA_CLEAN.loc[
            NPP_DATA_CLEAN['target_depth'] >=  MIXED_LAYER_UPPER_BOUND]
        
        P30_PRIOR = NPP_MIXED_LAYER['npp'].mean()/self.MOLAR_MASS_C
        P30_PRIOR_E = NPP_MIXED_LAYER['npp'].sem()/self.MOLAR_MASS_C
        
        NPP_REGRESSION = smf.ols(
            formula='np.log(npp/(P30_PRIOR*self.MOLAR_MASS_C)) ~ target_depth',
            data=NPP_BELOW_MIXED_LAYER).fit()

        LP_PRIOR = -1/NPP_REGRESSION.params[1]
        LP_PRIOR_E = NPP_REGRESSION.bse[1]/NPP_REGRESSION.params[1]**2
        
        return P30_PRIOR, P30_PRIOR_E, LP_PRIOR, LP_PRIOR_E

class PyriteParam:

    def __init__(self, prior, prior_error, name, label, depth_vary=True):
        
        self.PRIOR = prior
        self.PRIOR_E = prior_error
        self.NAME = name
        self.LABEL = label
        self.DV = depth_vary

    def __repr__(self):

        return f'PyriteParam({self.NAME})'

if __name__ == '__main__':

    start_time = time.time()
    model = PyriteModel()

    print(f'--- {(time.time() - start_time)/60} minutes ---')