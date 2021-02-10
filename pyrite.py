#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:55:53 2021

@author: Vinicius J. Amaral

Pyrite (Particle cYcling Rates from Inversion of Tracers in the ocEan)
"""
import time
start_time = time.time()

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

class Pyrite:
    
    def __init__(self,gammas=[0.02]):
        
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
                    
    def load_data(self):
        
        self.DATA = pd.read_excel('pyrite_data.xlsx',sheet_name=None)
    
    def unpack_tracers(self):
        
        self.POC_S_MEAN = self.DATA['poc_means']['SSF_mean']/self.MOLAR_MASS_C
        self.POC_S_SE = self.DATA['poc_means']['SSF_se']/self.MOLAR_MASS_C
        self.POC_L_MEAN = self.DATA['poc_means']['LSF_mean']/self.MOLAR_MASS_C
        self.POC_L_SE = self.DATA['poc_means']['LSF_se']/self.MOLAR_MASS_C
    
    def define_params(self):
        
        P30_PRIOR, P30_PRIOR_E, LP_PRIOR, LP_PRIOR_E = self.process_npp_data()
        
        self.PARAM_DICT = {'ws': {'prior': 2,
                                    'prior error': 2,
                                    'typeset': '$w_S$'},
                             'wl': {'prior': 20,
                                    'prior error': 15,
                                    'typeset': '$w_L$'},
                             'B2p': {'prior': (0.5*self.MOLAR_MASS_C/
                                               self.DAYS_PER_YEAR),
                                     'prior error': (0.5*self.MOLAR_MASS_C/
                                                     self.DAYS_PER_YEAR),
                                     'typeset': '$\\beta^,_2$'},
                             'Bm2': {'prior': (400*self.MOLAR_MASS_C/
                                               self.DAYS_PER_YEAR),
                                     'prior error': (10000*self.MOLAR_MASS_C/
                                                     self.DAYS_PER_YEAR),
                                     'typeset': '$\\beta_{-2}$'},
                             'Bm1s': {'prior': 0.1,
                                      'prior error': 0.1,
                                      'typeset': '$\\beta_{-1,S}$'},
                             'Bm1;': {'prior': 0.15,
                                      'prior error': 0.15,
                                      'typeset': '$\\beta_{-1,L}$'},
                             'P30': {'prior': P30_PRIOR,
                                     'prior error': P30_PRIOR_E,
                                     'typeset': '$\.P_{S,30}$'},
                             'Lp': {'prior': LP_PRIOR,
                                     'prior error': LP_PRIOR_E,
                                     'typeset': '$L_P$'}}
    
    def process_npp_data(self):
        
        NPP_DATA_RAW = self.DATA['npp']
        NPP_DATA_CLEAN = NPP_DATA_RAW.loc[(NPP_DATA_RAW['npp'] > 0)]
        
        MIXED_LAYER_UPPER_BOUND, MIXED_LAYER_LOWER_BOUND = 28, 35
        
        NPP_MIXED_LAYER = NPP_DATA_CLEAN.loc[
            (NPP_DATA_CLEAN['target_depth'] >= MIXED_LAYER_UPPER_BOUND) & 
            (NPP_DATA_CLEAN['target_depth'] <= MIXED_LAYER_LOWER_BOUND)]
        
        NPP_BELOW_MIXED_LAYER = NPP_DATA_CLEAN.loc[
            NPP_DATA_CLEAN['target_depth'] >= 28]
        
        P30_PRIOR = NPP_MIXED_LAYER['npp'].mean()/self.MOLAR_MASS_C
        P30_PRIOR_E = NPP_MIXED_LAYER['npp'].sem()/self.MOLAR_MASS_C
        
        NPP_REGRESSION = smf.ols(
            formula='np.log(npp/(P30_PRIOR*self.MOLAR_MASS_C)) ~ target_depth',
            data=NPP_BELOW_MIXED_LAYER).fit()

        LP_PRIOR = -1/NPP_REGRESSION.params[1]
        LP_PRIOR_E = NPP_REGRESSION.bse[1]/NPP_REGRESSION.params[1]**2
        
        return P30_PRIOR, P30_PRIOR_E, LP_PRIOR, LP_PRIOR_E
        
        
model = Pyrite()

print(f'--- {(time.time() - start_time)/60} minutes ---')