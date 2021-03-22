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
import statsmodels.tsa.stattools as smt
import pickle
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as mplc
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import operator as op
import itertools
import scipy.linalg as splinalg
import sympy as sym
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib as mpl
import sys

class PyriteModel:

    def __init__(self, gammas=[0.02, 0.05, 0.1, 0.5, 1],
                 pickle_into='out/pyrite_Amaral21a.pkl'):

        self.gammas = gammas
        self.pickled = pickle_into
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
        self.define_tracers()
        self.define_params()
        self.define_fluxes()
        self.process_cp_data()
        self.define_zones()
        
        self.objective_interpolation()
        xo, xo_log, Co, Co_log = self.define_prior_vector_and_cov_matrix()
        self.define_equation_elements()
        
        self.model_runs = [
            PyriteModelRun(self, xo, xo_log, Co, Co_log, g) for g in gammas]

        self.pickle_model()

    def __repr__(self):

        return 'PyriteModel object'
                    
    def load_data(self):
        
        self.data = pd.read_excel('pyrite_data.xlsx',sheet_name=None)
        self.sample_depths = self.data['poc_means']['depth']
    
    def define_tracers(self):
        
        self.Ps = Tracer('POC', 'S', '$P_S$', self.data['poc_means'][
            ['depth', 'SSF_mean', 'SSF_se']])
        self.Pl = Tracer('POC', 'L', '$P_L$', self.data['poc_means'][
            ['depth', 'LSF_mean', 'LSF_se']])

        self.tracers = (self.Ps, self.Pl)

    def define_params(self):
        
        P30_prior, P30_prior_e, Lp_prior, Lp_prior_e = self.process_npp_data()

        self.ws = Param(2, 2, 'ws', '$w_S$')
        self.wl = Param(20, 15, 'wl', '$w_L$')
        self.B2p = Param(0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               'B2p', '$\\beta^,_2$')
        self.Bm2 = Param(400/self.DAYS_PER_YEAR,
                               10000/self.DAYS_PER_YEAR,
                               'Bm2', '$\\beta_{-2}$')
        self.Bm1s = Param(0.1, 0.1, 'Bm1s', '$\\beta_{-1,S}$')
        self.Bm1l = Param(0.15, 0.15, 'Bm1l', '$\\beta_{-1,L}$')
        self.P30 = Param(P30_prior, P30_prior_e, 'P30', '$\.P_{S,30}$',
                               depth_vary=False)
        self.Lp = Param(Lp_prior, Lp_prior_e, 'Lp', '$L_P$',
                              depth_vary=False)

        self.params = (self.ws, self.wl, self.B2p, self.Bm2, self.Bm1s,
                             self.Bm1l, self.P30, self.Lp)
    
    def define_fluxes(self):
        
        self.sink_S = Flux('sink_S', '$w_SP_S$', 'POCS', 'ws')
        self.sink_L = Flux('sink_L', '$w_LP_L$', 'POCL', 'wl')
        self.sink_T = Flux('sink_T', '$w_TP_T$', 'POCT', 'wt')
        self.sinkdiv_S = Flux(
            'sinkdiv_S', '$\\frac{d}{dz}w_SP_S$', 'POCS', 'ws', wrt=('POCS',))
        self.sinkdiv_L = Flux(
            'sinkdiv_L', '$\\frac{d}{dz}w_LP_L$', 'POCL', 'wl', wrt=('POCL',))
        self.remin_S = Flux(
            'remin_S', '$\\beta_{-1,S}P_S$', 'POCS', 'Bm1s', wrt=('POCS',))
        self.remin_L = Flux(
            'remin_L', '$\\beta_{-1,L}P_L$', 'POCL', 'Bm1l', wrt=('POCL',))
        self.aggregation = Flux(
            'aggregation', '$\\beta^,_2P^2_S$', 'POCS', 'B2p',
            wrt=('POCS', 'POCL'))
        self.disaggregation = Flux(
            'disaggregation', '$\\beta_{-2}P_L$', 'POCL', 'Bm2',
            wrt=('POCS', 'POCL'))
        self.production = Flux(
            'production', '${\.P_S}$', 'POCS', None, wrt=('POCS',))
        
        self.fluxes = (self.sink_S, self.sink_L, self.sink_T, self.sinkdiv_S,
                       self.sinkdiv_L, self.remin_S, self.remin_L,
                       self.aggregation, self.disaggregation, self.production)

    def process_npp_data(self):
        
        npp_data_raw = self.data['npp']
        npp_data_clean = npp_data_raw.loc[(npp_data_raw['npp'] > 0)]
        
        MIXED_LAYER_UPPER_BOUND, MIXED_LAYER_LOWER_BOUND = 28, 35
        
        npp_mixed_layer = npp_data_clean.loc[
            (npp_data_clean['target_depth'] >= MIXED_LAYER_UPPER_BOUND) &
            (npp_data_clean['target_depth'] <= MIXED_LAYER_LOWER_BOUND)]
        
        npp_below_mixed_layer = npp_data_clean.loc[
            npp_data_clean['target_depth'] >=  MIXED_LAYER_UPPER_BOUND]
        
        P30_prior = npp_mixed_layer['npp'].mean()/self.MOLAR_MASS_C
        P30_prior_e = npp_mixed_layer['npp'].sem()/self.MOLAR_MASS_C

        npp_regression = smf.ols(
            formula='np.log(npp/(P30_prior*self.MOLAR_MASS_C)) ~ target_depth',
            data=npp_below_mixed_layer).fit()

        Lp_prior = -1/npp_regression.params[1]
        Lp_prior_e = npp_regression.bse[1]/npp_regression.params[1]**2
        
        return P30_prior, P30_prior_e, Lp_prior, Lp_prior_e
    
    def define_zones(self):
        
        self.LEZ = Zone(self, op.lt, 'LEZ')  # lower euphotic zone
        self.UMZ = Zone(self, op.gt, 'UMZ')  # upper mesopelagic zone
        self.zones = (self.LEZ, self.UMZ)

    def process_cp_data(self):

        cast_match_table = self.data['cast_match']
        cast_match_dict = dict(zip(cast_match_table['pump_cast'],
                                   cast_match_table['ctd_cast']))
        poc_discrete = self.data['poc_discrete']
        cp_bycast = self.data['cp_bycast']
        self.poc_cp_df = poc_discrete.copy()

        self.poc_cp_df['ctd_cast'] = self.poc_cp_df.apply(
            lambda x: cast_match_dict[x['pump_cast']], axis=1)
        self.poc_cp_df['cp'] = self.poc_cp_df.apply(
            lambda x: cp_bycast.at[x['depth']-1, x['ctd_cast']], axis=1)

        self.cp_Pt_regression_nonlinear = smf.ols(
            formula='Pt ~ np.log(cp)', data=self.poc_cp_df).fit()
        self.cp_Pt_regression_linear = smf.ols(
            formula='Pt ~ cp', data=self.poc_cp_df).fit()

        cp_bycast_to_mean = cp_bycast.loc[self.GRID-1,
                                          cast_match_table['ctd_cast']]
        self.cp_mean = cp_bycast_to_mean.mean(axis=1)
        self.Pt_mean_nonlinear = (
            self.cp_Pt_regression_nonlinear.get_prediction(
                exog=dict(cp=self.cp_mean)).predicted_mean)
        self.Pt_mean_linear = self.cp_Pt_regression_linear.get_prediction(
            exog=dict(cp=self.cp_mean)).predicted_mean

    def objective_interpolation(self):

        def R_matrix(list1, list2):

            m = len(list1)
            n = len(list2)
            R = np.zeros((m,n))

            for i in np.arange(0,m):
                for j in np.arange(0,n):
                    R[i,j] = np.exp(-np.abs(list1[i]-list2[j])/L)

            return R

        for tracer in self.tracers:
            
            tracer_data_oi = pd.DataFrame(columns=tracer.data.columns)
            tracer_data_cov_matrices = []
            
            for zone in self.zones:
            
                L = zone.length_scale
                min_depth = zone.depths.min()
                max_depth = zone.depths.max()

                zone_data = tracer.data[
                    tracer.data['depth'].between(min_depth, max_depth)]
                
                sample_depths, conc, conc_e = zone_data.T.values
                
                Rxxmm = R_matrix(sample_depths, sample_depths)
                Rxxnn = R_matrix(zone.depths, zone.depths)
                Rxy = R_matrix(zone.depths, sample_depths)
                
                conc_anom = conc - conc.mean()
                conc_var_discrete = conc_e**2
                conc_var = (np.var(conc, ddof=1)
                            + np.sum(conc_var_discrete)/len(sample_depths))
                
                Rnn = np.diag(conc_var_discrete)
                Rxxmm = Rxxmm*conc_var
                Rxxnn = Rxxnn*conc_var
                Rxy = Rxy*conc_var
                Ryy = Rxxmm + Rnn
                Ryyi = np.linalg.inv(Ryy)
                
                conc_anom_oi = np.matmul(np.matmul(Rxy, Ryyi), conc_anom)
                conc_oi = conc_anom_oi + conc.mean()
                P = Rxxnn - np.matmul(np.matmul(Rxy, Ryyi), Rxy.T)
                conc_e_oi = np.sqrt(np.diag(P))
                tracer_data_cov_matrices.append(P)
                
                tracer_data_oi = tracer_data_oi.append(
                    pd.DataFrame(np.array([zone.depths,conc_oi, conc_e_oi]).T,
                                 columns=tracer_data_oi.columns),
                    ignore_index=True)
            
            tracer.cov_matrices = tracer_data_cov_matrices
            tracer.prior = tracer_data_oi
            
    def define_prior_vector_and_cov_matrix(self):
        
        tracer_priors = []
        self.state_elements = []
        
        for t in self.tracers:
            tracer_priors.append(t.prior['conc'])
            for i in range(0,self.N_GRID_POINTS):
                self.state_elements.append(f'{t.species}{t.sf}_{i}')
                   
        tracer_priors = list(itertools.chain.from_iterable(tracer_priors))
        self.nte = len(tracer_priors)  # number of tracer elements
           
        param_priors = []
        param_priors_var = []
        
        for p in self.params:
            if p.dv:
                for z in self.zones:
                    param_priors.append(p.prior)
                    param_priors_var.append(p.prior_e**2)
                    self.state_elements.append(f'{p.name}_{z.label}')
            else:
                param_priors.append(p.prior)
                param_priors_var.append(p.prior_e**2)
                self.state_elements.append(f'{p.name}')

        xo = np.concatenate((tracer_priors,param_priors))
        xo_log = np.log(xo)  
        self.nse = len(xo)  # number of state elements
        
        Co = np.zeros((self.nse, self.nse))
        Co_log = np.zeros((self.nse, self.nse))
        
        tracer_cov_matrices = [t.cov_matrices for t in self.tracers]
        tracer_cov_matrices = list(
            itertools.chain.from_iterable(tracer_cov_matrices))
        
        Co[:self.nte, :self.nte] = splinalg.block_diag(
            *tracer_cov_matrices)
        np.fill_diagonal(Co[self.nte:, self.nte:], param_priors_var)
        
        for i, row in enumerate(Co):
            for j, val in enumerate(row):
                Co_log[i,j] = np.log(1 + val/(xo[i]*xo[j]))
        
        return xo, xo_log, Co, Co_log
     
    def define_equation_elements(self):
        
        self.equation_elements = self.state_elements[:self.nte]
        
        for i in range(0,self.N_GRID_POINTS):
            self.equation_elements.append(f'POCT_{i}')  
            
    def pickle_model(self):

        with open(self.pickled, 'wb') as file:
            pickle.dump(self,file)

class Tracer:

    def __init__(self, species, size_fraction, label, data):
        self.species = species
        self.sf = size_fraction
        self.label = label
        self.data = pd.DataFrame(data)
        self.data.rename(columns={self.data.columns[1]: 'conc',
                                  self.data.columns[2]: 'conc_e'},
                         inplace=True)

    def __repr__(self):

        return f'Tracer(species={self.species}, size_frac={self.sf})'
        
class Param:

    def __init__(self, prior, prior_error, name, label, depth_vary=True):
        
        self.prior = prior
        self.prior_e = prior_error
        self.name = name
        self.label = label
        self.dv = depth_vary

    def __repr__(self):

        return f'Param({self.name})'

class Zone:

    def __init__(self, model, operator, label):

        self.model = model
        self.indices = np.where(operator(model.GRID, model.BOUNDARY))[0]
        self.depths = model.GRID[self.indices]
        self.label = label
        
        self.calculate_length_scales(0.25)
        self.set_integration_intervals()

    def __repr__(self):

        return f'Zone({self.label})'
    
    def calculate_length_scales(self, fraction):

        Pt = self.model.Pt_mean_nonlinear[self.indices]
        n_lags = int(np.ceil(len(Pt)*fraction))
        self.grid_steps = np.arange(
            0, (n_lags + 1)*self.model.GRID_STEP, self.model.GRID_STEP)
        self.autocorrelation = smt.acf(Pt, nlags=n_lags, fft=False)
        
        acf_regression = smf.ols(
            formula='np.log(ac) ~ gs',
            data = {'ac':self.autocorrelation, 'gs':self.grid_steps}).fit()
        b, m = acf_regression.params
        
        self.length_scale = -1/m
        self.length_scale_fit = b + m*self.grid_steps
        self.fit_rsquared = acf_regression.rsquared
    
    def set_integration_intervals(self):
        
        intervals = np.ones(len(self.depths))*self.model.GRID_STEP
        
        if self.label == 'LEZ':
            intervals[0] = self.model.GRID_STEP/2
        else:
            intervals[-1] = self.model.GRID_STEP/2
        
        self.integration_intervals = intervals

class Flux:
    
    def __init__(self, name, label, tracer, param, wrt=None):
        
        self.name = name
        self.label = label
        self.tracer = tracer
        self.param = param
        self.wrt = wrt
    
    def __repr__(self):

        return f'Flux({self.name})'
        
class PyriteModelRun():
    
    def __init__(self, model, xo, xo_log, Co, Co_log, gamma):
        
        self.model = model
        self.gamma = gamma
        
        Cf = self.define_model_error_matrix()
        xhat = self.ATI(xo_log, Co_log, Cf)
            
        self.calculate_total_POC()
        self.calculate_residuals(xo, Co, xhat, Cf)
        inventories = self.calculate_inventories()
        fluxes_sym = self.calculate_fluxes()
        flux_names, integrated_fluxes = self.integrate_fluxes(fluxes_sym)
        self.calculate_timescales(inventories, flux_names, integrated_fluxes)

    def __repr__(self):
        
        return f'PyriteModelRun(gamma={self.gamma})'

    def which_zone(self, depth):

        if int(depth) in self.model.LEZ.indices:
            return 'LEZ'
        return 'UMZ'

    def define_model_error_matrix(self):
        
        Cf_Ps_Pl = np.zeros((self.model.nte, self.model.nte))
        Cf_Pt = np.zeros((self.model.N_GRID_POINTS, self.model.N_GRID_POINTS))
        
        np.fill_diagonal(Cf_Ps_Pl, (self.model.P30.prior**2)*self.gamma)
        np.fill_diagonal(
            Cf_Pt,self.model.cp_Pt_regression_nonlinear.mse_resid)
        
        Cf = splinalg.block_diag(Cf_Ps_Pl, Cf_Pt)
        
        return Cf
    
    def slice_by_tracer(self, to_slice, tracer):
        
        start_index = [i for i, el in enumerate(self.model.state_elements) 
                       if tracer in el][0]
        sliced = to_slice[
            start_index:start_index + self.model.N_GRID_POINTS]
        
        return sliced

    def equation_builder(self, species, depth):

        Psi, Psip1, Psim1, Psim2 = sym.symbols(
            'POCS_0 POCS_1 POCS_-1 POCS_-2')
        Pli, Plip1, Plim1, Plim2 = sym.symbols(
            'POCL_0 POCL_1 POCL_-1 POCL_-2')
        Bm2i, B2pi, Bm1si, Bm1li, P30i, Lpi, wsi, wli, = sym.symbols(
            'Bm2 B2p Bm1s Bm1l P30 Lp ws wl')
        
        h = self.model.MIXED_LAYER_DEPTH
        depth = int(depth)
        
        if species == 'POCS':
            if depth == 0:
                eq = P30i + Bm2i*Pli - (wsi/h + Bm1si + B2pi*Psi)*Psi
            else:
                if (depth == 1 or depth == 2):
                    multiply_by = Psip1 - Psim1
                else:
                    multiply_by = 3*Psi - 4*Psim1 + Psim2
                eq = (P30i*sym.exp(-(self.model.GRID[depth] - h)/(Lpi))
                      + (Bm2i*Pli) - (Bm1si + B2pi*Psi)*Psi
                      - wsi/(2*self.model.GRID_STEP)*multiply_by)
        elif species == 'POCL':
            if depth == 0:
                eq = B2pi*Psi**2 - (wli/h + Bm2i + Bm1li)*Pli
            else:
                if (depth == 1 or depth == 2):
                    multiply_by = Plip1 - Plim1
                else:
                    multiply_by = 3*Pli - 4*Plim1 + Plim2
                eq = (B2pi*Psi**2 - (Bm2i + Bm1li)*Pli
                      - wli/(2*self.model.GRID_STEP)*multiply_by)
        else: 
            Pti = self.model.Pt_mean_nonlinear[
                (self.model.equation_elements.index(f'POCT_{depth}')
                 - self.model.nte)]
            eq = Pti - (Psi + Pli)
        
        return eq
        
    def extract_equation_variables(self, y, depth, v, in_ATI=False):
        
        x_symbolic = y.free_symbols
        x_numerical = []
        x_indices = []
            
        for x in x_symbolic:
            if '_' in x.name:  # if it's a tracer
                tracer, relative_depth = x.name.split('_')
                real_depth = str(int(depth) + int(relative_depth))
                element = '_'.join([tracer, real_depth])
            else:  # if it's a parameter
                param = eval(f'self.model.{x.name}')
                if param.dv:
                    zone = self.which_zone(depth)
                    element = '_'.join([param.name, zone])
                else:
                    element = param.name
            element_index = self.model.state_elements.index(element)
            x_indices.append(element_index)
            if in_ATI:
                x_numerical.append(np.exp(v[element_index]))
            else:
                x_numerical.append(v[element_index])

        return x_symbolic, x_numerical, x_indices
    
    def evaluate_model_equations(self, v, in_ATI=False):
        
        f = np.zeros(len(self.model.equation_elements))
        if in_ATI:
            F = np.zeros((len(self.model.equation_elements), self.model.nse))
        
        for i, element in enumerate(self.model.equation_elements):
            species, depth = element.split('_')
            y = self.equation_builder(species, depth)
            x_sym, x_num, x_ind = self.extract_equation_variables(
                y, depth, v, in_ATI=in_ATI)
            f[i] = sym.lambdify(x_sym, y)(*x_num)
            if in_ATI:
                for j, x in enumerate(x_sym):
                    dy = y.diff(x)*x  # dy/d(ln(x)) = x*dy/dx
                    dx_sym, dx_num, _ = self.extract_equation_variables(
                        dy, depth, v, in_ATI=True)
                    F[i,x_ind[j]] = sym.lambdify(dx_sym, dy)(*dx_num)      
        if in_ATI:
            return F, f
        else:
            return f
    
    def eval_symbolic_func(self, y, err=True, cov=True):
        
        x_symbolic = y.free_symbols
        x_numerical = []
        x_indices = []

        for x in x_symbolic:
            x_indices.append(self.model.state_elements.index(x.name))
            if '_' in x.name:  # if it varies with depth
                element, depth = x.name.split('_')
                if element in self.tracer_results:  # if it's a tracer
                    x_numerical.append(
                        self.tracer_results[element]['est'][int(depth)])
                else:  #  if it's a depth-varying parameter
                    x_numerical.append(
                        self.param_results[element][depth]['est'])
            else:  # if it's a depth-independent parameter
                x_numerical.append(self.param_results[x.name]['est'])

        result = sym.lambdify(x_symbolic, y)(*x_numerical)
        
        if err==False:
            return result        
        else:
            variance_sym = 0  # symbolic expression for variance of y
            derivs = [y.diff(x) for x in x_symbolic]
            cvm = self.cvm[ # sub-CVM corresponding to state elements in y
                np.ix_(x_indices, x_indices)]  
            for i, row in enumerate(cvm):
                for j, _ in enumerate(row):
                    if i > j:
                        continue
                    elif i == j:
                        variance_sym += (derivs[i]**2)*cvm[i,j]
                    else:
                        if cov:
                            variance_sym += 2*derivs[i]*derivs[j]*cvm[i,j]
            variance = sym.lambdify(x_symbolic, variance_sym)(*x_numerical)
            error = np.sqrt(variance)
            
            return result, error
        
    def ATI(self, xo_log, Co_log, Cf):
        
        def calculate_xkp1(xk, F, f):
            
            CoFT = Co_log @ F.T
            FCoFT = F @ CoFT
            FCoFTpCfi = np.linalg.inv(FCoFT + Cf)
            xkp1 = (xo_log + CoFT @ FCoFTpCfi @ (F @ (xk - xo_log) - f))
                    
            return xkp1, CoFT, FCoFTpCfi
            
        def check_convergence(xk, xkp1):
            
            converged = False
            max_change_limit = 0.01 
            change = np.abs((np.exp(xkp1) - np.exp(xk))/np.exp(xk))
            self.convergence_evolution.append(np.max(change))
            if np.max(change) < max_change_limit:
                converged = True
            
            return converged
        
        def calculate_cost(xk, f):
                      
            cost = ((xk - xo_log).T @ np.linalg.inv(Co_log) @ (xk - xo_log)
                    + f.T @ np.linalg.inv(Cf) @ f)
            
            self.cost_evolution.append(cost)                    
        
        def find_solution():
            
            max_iterations = 25
            self.cost_evolution = []
            self.convergence_evolution = []
            self.converged = False
                                      
            xk = xo_log  # estimate of state vector at iteration k
            xkp1 = np.ones(len(xk))  # at iteration k+1
                        
            for _ in range(0, max_iterations):
                F, f = self.evaluate_model_equations(xk, in_ATI=True)
                xkp1, CoFT, FCoFTpCfi = calculate_xkp1(xk, F, f)
                calculate_cost(xk, f)
                self.converged = check_convergence(xk, xkp1)
                if self.converged:
                    break
                xk = xkp1
            
            return F, xkp1, CoFT, FCoFTpCfi  
         
        def unlog_state_estimates():
            
            F, xkp1, CoFT, FCoFTpCfi = find_solution()
            I = np.identity(Co_log.shape[0])
                
            Ckp1 = ((I - CoFT @ FCoFTpCfi @ F) @ Co_log
                    @ (I - F.T @ FCoFTpCfi @ F @ Co_log))
            
            expected_vals_log = xkp1
            variances_log = np.diag(Ckp1)
            
            xhat = np.exp(expected_vals_log + variances_log/2)
            xhat_e = np.sqrt(
                np.exp(2*expected_vals_log + variances_log)
                *(np.exp(variances_log) - 1))
            
            self.cvm = np.zeros(  # covaraince matrix of posterior estimates
                (len(xhat), len(xhat)))
            
            for i, row in enumerate(self.cvm):
                for j, _ in enumerate(row):
                    ei, ej = expected_vals_log[i], expected_vals_log[j]
                    vi, vj = variances_log[i], variances_log[j]
                    self.cvm[i,j] = (np.exp(ei + ej)*np.exp((vi + vj)/2)
                                      *(np.exp(Ckp1[i,j]) - 1))
        
            return xhat, xhat_e
        
        def unpack_state_estimates():
            
            xhat, xhat_e = unlog_state_estimates()
            
            self.tracer_results = {}
            for tracer in self.model.tracers:
                t = f'{tracer.species}{tracer.sf}'
                self.tracer_results[t] = {
                    'est': self.slice_by_tracer(xhat, t),
                    'err': self.slice_by_tracer(xhat_e, t)}

            self.param_results = {}
            for param in self.model.params:
                p = param.name
                if param.dv:
                    self.param_results[p] = {
                        zone.label: {} for zone in self.model.zones}
                    for zone in self.model.zones:
                        z = zone.label
                        zone_param = '_'.join([p, z])
                        i = self.model.state_elements.index(zone_param)
                        self.param_results[p][z] = {'est': xhat[i],
                                                    'err': xhat_e[i]}
                else:
                    i = self.model.state_elements.index(p)
                    self.param_results[p] = {'est': xhat[i],
                                             'err': xhat_e[i]}

            return xhat

        return unpack_state_estimates()        
    
    def calculate_total_POC(self):
        
        self.Pt = {'est':[], 'err':[]}
        
        for i in range(0, self.model.N_GRID_POINTS):
            Ps_str = f'POCS_{i}'
            Pl_str = f'POCL_{i}'
            Ps, Pl = sym.symbols(f'{Ps_str} {Pl_str}')
            Pt_est, Pt_err = self.eval_symbolic_func(Ps + Pl)
            self.Pt['est'].append(Pt_est)
            self.Pt['err'].append(Pt_err)
        
    def calculate_residuals(self, xo, Co, xhat, Cf):

        x_residuals = xhat - xo
        norm_x_residuals  = x_residuals/np.sqrt(np.diag(Co))
        self.x_resids = norm_x_residuals

        f_residuals = self.evaluate_model_equations(xhat)
        norm_f_residuals = f_residuals/np.sqrt(np.diag(Cf))
        self.f_resids = norm_f_residuals
        
        for t in self.tracer_results.keys():
            self.tracer_results[t]['resids'] = self.slice_by_tracer(
                f_residuals, t)
    
    def calculate_inventories(self):
        
        self.inventories = {}
        self.integrated_resids = {}
        inventory_sym = {}
        
        for zone in self.model.zones:            
            z = zone.label
            dz = zone.integration_intervals
            self.inventories[z] = {} 
            self.integrated_resids[z] = {}
            inventory_sym[z] = {}               
            for t in self.tracer_results.keys():               
                inventory = 0
                int_resids = 0
                for i, di in enumerate(zone.indices):
                    tracer_sym = sym.symbols(f'{t}_{di}')
                    inventory += tracer_sym*dz[i]
                    int_resids += (self.tracer_results[t]['resids'][di]*dz[i])
                self.inventories[z][t] = self.eval_symbolic_func(inventory)
                self.integrated_resids[z][t] = int_resids
                inventory_sym[z][t] = inventory
                
        return inventory_sym
    
    def calculate_fluxes(self):
        
        self.flux_profiles = {}
        MLD = self.model.MIXED_LAYER_DEPTH
        fluxes_sym = {}
        
        for flux in self.model.fluxes:
            f = flux.name
            self.flux_profiles[f]= {'est': [], 'err': []}
            if flux.wrt:
                fluxes_sym[f] = []               
            if 'div' in f:               
                for i in range(0, self.model.N_GRID_POINTS):
                    z = self.which_zone(i)
                    pwi = f'{flux.param}_{z}'
                    twi = f'{flux.tracer}_{i}'
                    w, Pi = sym.symbols(f'{pwi} {twi}')
                    if i == 0:   
                        y = w*Pi/MLD
                    elif (i == 1 or i == 2):
                        twip1 = f'{flux.tracer}_{i+1}'
                        twim1 = f'{flux.tracer}_{i-1}'
                        Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                        y = w*(Pip1 - Pim1)/(2*self.model.GRID_STEP)
                    else:
                        twim1 = f'{flux.tracer}_{i-1}'
                        twim2 = f'{flux.tracer}_{i-2}'
                        Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                        y = w*(3*Pi -4*Pim1 + Pim2)/(2*self.model.GRID_STEP)
                    est, err = self.eval_symbolic_func(y)
                    self.flux_profiles[f]['est'].append(est)
                    self.flux_profiles[f]['err'].append(err)
                    if flux.wrt:
                        fluxes_sym[f].append(y)
            else:
                for i in range(0, self.model.N_GRID_POINTS):
                    if f == 'production':
                        p30, lp = sym.symbols('P30 Lp')
                        y = p30*sym.exp(-(self.model.GRID[i] - MLD)/lp)
                    else:
                        z = self.which_zone(i)
                        if f == 'sink_T':
                            wsi = f'ws_{z}'
                            wli = f'wl_{z}'
                            Psi = f'POCS_{i}'
                            Pli = f'POCL_{i}'
                            ws, wl, Ps, Pl = sym.symbols(
                                f'{wsi} {wli} {Psi} {Pli}')
                            y = ws*Ps + wl*Pl
                        else:
                            if f == 'aggregation':
                                order = 2
                            else:
                                order = 1                            
                            pwi = f'{flux.param}_{z}'
                            twi = f'{flux.tracer}_{i}'
                            p, t = sym.symbols(f'{pwi} {twi}')
                            y = p*t**order
                    if flux.wrt:
                        fluxes_sym[f].append(y)
                    est, err = self.eval_symbolic_func(y)
                    self.flux_profiles[f]['est'].append(est)
                    self.flux_profiles[f]['err'].append(err) 
            
        return fluxes_sym
    
    def integrate_fluxes(self, fluxes_sym):
        
        fluxes = fluxes_sym.keys()
        flux_integrals_sym = {}
        self.flux_integrals = {}
        
        for zone in self.model.zones:
            z = zone.label
            dz = zone.integration_intervals
            flux_integrals_sym[z] = {}
            self.flux_integrals[z] = {}
            for f in fluxes:
                zone_expressions = [fluxes_sym[f][i] for i in zone.indices]
                to_integrate = 0
                for i, ex in enumerate(zone_expressions):
                    to_integrate += ex*dz[i]
                flux_integrals_sym[z][f] = to_integrate
                self.flux_integrals[z][f] = self.eval_symbolic_func(
                    to_integrate)
                
        return fluxes, flux_integrals_sym
    
    def calculate_timescales(self, inventory_sym, fluxes, flux_integrals_sym):
        
        self.timescales = {}
        
        for zone in self.model.zones:
            z = zone.label
            self.timescales[z] = {}
            for tracer in inventory_sym[z].keys():
                self.timescales[z][tracer] = {}
                for flux in fluxes:
                    if tracer in eval(f'self.model.{flux}.wrt'):
                        self.timescales[z][tracer][flux] = (
                            self.eval_symbolic_func(
                                inventory_sym[z][tracer]
                                /flux_integrals_sym[z][flux]))
                                                   
class PyritePlotter:

    def __init__(self, pickled_model):
        
        with open(pickled_model, 'rb') as file:
            self.model = pickle.load(file)

        self.define_colors()
        self.hydrography()
        self.cp_Pt_regression()
        self.Pt_estimate_comparison()
        self.zone_length_scales()
        self.poc_data()
        
        for run in self.model.model_runs:
            self.cost_and_convergence(run)
            self.params(run)
            self.poc_profiles(run)
            self.residual_pdfs(run)
            self.sinking_fluxes(run)
            self.volumetric_fluxes(run)
            if run.gamma == 0.02:
                self.param_comparison(run)
        
        self.integrated_residuals()
        self.param_sensitivity()
        self.param_relative_errors()

        self.write_output()
        
    def define_colors(self):

        self.BLACK = '#000000'
        self.ORANGE = '#E69F00'
        self.SKY = '#56B4E9'
        self.GREEN = '#009E73'
        self.YELLOW = '#F0E442'
        self.BLUE = '#0072B2'
        self.VERMILLION = '#D55E00'
        self.RADISH = '#CC79A7'
        self.WHITE = '#FFFFFF'
        
        self.colors = (
            self.BLACK, self.ORANGE, self.SKY, self.GREEN, self.YELLOW,
            self.BLUE, self.VERMILLION, self.RADISH)

    def hydrography(self):

        hydro_df = self.model.data['hydrography']

        fig = plt.figure()
        host = host_subplot(111, axes_class=AA.Axes, figure=fig)
        plt.subplots_adjust(top=0.75)
        par1 = host.twiny()
        par2 = host.twiny()

        par1.axis['top'].toggle(all=True)
        offset = 40
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis['top'] = new_fixed_axis(loc='top', axes=par2,
                                          offset=(0, offset))
        par2.axis['top'].toggle(all=True)

        host.set_ylim(0, 520)
        host.invert_yaxis(), host.grid(axis='y', alpha=0.5)
        host.set_xlim(24, 27.4)
        par1.set_xlim(3, 14.8)
        par2.set_xlim(32, 34.5)

        host.set_ylabel('Depth (m)',fontsize=14)
        host.set_xlabel('$\sigma_T$ (kg m$^{-3}$)')
        par1.set_xlabel('Temperature (°C)')
        par2.set_xlabel('Salinity (PSU)')

        host.plot(hydro_df['sigT_kgpmc'], hydro_df['depth'], c=self.ORANGE,
                  marker='o')
        par1.plot(hydro_df['t_c'], hydro_df['depth'], c=self.GREEN,
                  marker='o')
        par2.plot(hydro_df['s_psu'], hydro_df['depth'], c=self.BLUE,
                  marker='o')
        host.axhline(self.model.MIXED_LAYER_DEPTH, c=self.BLACK, ls=':',
                     zorder=3)
        host.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', zorder=3)

        host.axis['bottom'].label.set_color(self.ORANGE)
        par1.axis['top'].label.set_color(self.GREEN)
        par2.axis['top'].label.set_color(self.BLUE)

        host.axis['bottom','left'].label.set_fontsize(14)
        par1.axis['top'].label.set_fontsize(14)
        par2.axis['top'].label.set_fontsize(14)

        host.axis['bottom','left'].major_ticklabels.set_fontsize(12)
        par1.axis['top'].major_ticklabels.set_fontsize(12)
        par2.axis['top'].major_ticklabels.set_fontsize(12)

        host.axis['bottom','left'].major_ticks.set_ticksize(6)
        par1.axis['top'].major_ticks.set_ticksize(6)
        par2.axis['top'].major_ticks.set_ticksize(6)

        plt.savefig('out/hydrography.pdf')
        plt.close()

    def cp_Pt_regression(self):

        cp = self.model.poc_cp_df['cp']
        Pt = self.model.poc_cp_df['Pt']
        depths = self.model.poc_cp_df['depth']
        linear_regression = self.model.cp_Pt_regression_linear
        nonlinear_regression = self.model.cp_Pt_regression_nonlinear
        logarithmic = {linear_regression: False, nonlinear_regression: True}

        colormap = plt.cm.viridis_r
        norm = mplc.Normalize(depths.min(), depths.max())

        for fit in (nonlinear_regression, linear_regression):
            fig, ax =  plt.subplots(1,1)
            fig.subplots_adjust(bottom=0.2,left=0.2)
            cbar_ax = colorbar.make_axes(ax)[0]
            cbar = colorbar.ColorbarBase(cbar_ax, norm=norm, cmap=colormap)
            cbar.set_label('Depth (m)\n', rotation=270, labelpad=20,
                           fontsize=14)
            ax.scatter(cp, Pt, norm=norm, edgecolors=self.BLACK, c=depths,
                       s=40, marker='o', cmap=colormap, label='_none')
            ax.set_ylabel('$P_T$ (mmol m$^{-3}$)',fontsize=14)
            ax.set_xlabel('$c_p$ (m$^{-1}$)',fontsize=14)
            x_fit = np.arange(0.01,0.14,0.0001)
            if logarithmic[fit]:
                coefs_log = fit.params
                y_fit_log = [
                    coefs_log[0] + coefs_log[1]*np.log(x) for x in x_fit]
                ax.plot(x_fit, y_fit_log, '--', c=self.BLACK, lw=1,
                    label='non-linear') 
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_xlim(0.0085, 0.15)
                ax.annotate(
                    f'$R^2$ = {fit.rsquared:.2f}\n$N$ = {fit.nobs:.0f}',
                    xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)                
            else:
                coefs_lin = fit.params
                y_fit_linear = [coefs_lin[0] + coefs_lin[1]*x for x in x_fit]
                ax.plot(x_fit, y_fit_linear, '--', c=self.BLACK, lw=1,
                        label='linear')
                ax.plot(x_fit, y_fit_log, ':', c=self.BLACK, lw=1,
                    label='non-linear') 
                ax.legend(fontsize=10, loc='lower right')          
            plt.savefig(f'out/cpptfit_log{logarithmic[fit]}.pdf')
            plt.close()

    def Pt_estimate_comparison(self):

        fig, ax = plt.subplots(1,1)
        fig.subplots_adjust(wspace=0.5)  
        ax.set_xlabel('$P_{T}$ (mmol m$^{-3}$)',fontsize=14)
        ax.set_ylabel('Depth (m)',fontsize=14)        
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)
                  
        ax.scatter(
            self.model.Pt_mean_nonlinear, self.model.GRID, marker='o',
            c=self.BLUE, s=7, label='non-linear', zorder=3, lw=0.7)
        ax.fill_betweenx(
            self.model.GRID,
            (self.model.Pt_mean_nonlinear
             - np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            (self.model.Pt_mean_nonlinear
             + np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            color=self.BLUE, alpha=0.25)
        
        ax.scatter(
            self.model.Pt_mean_linear, self.model.GRID, marker='o',
            c=self.ORANGE, s=7, label='linear', zorder=3, lw=0.7)
        ax.fill_betweenx(
            self.model.GRID,
            (self.model.Pt_mean_linear
             - np.sqrt(self.model.cp_Pt_regression_linear.mse_resid)),
            (self.model.Pt_mean_linear
             + np.sqrt(self.model.cp_Pt_regression_linear.mse_resid)),
            color=self.ORANGE, alpha=0.25)
        
        ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                  loc='lower right')
        ax.set_xticks([0,1,2])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', lw=1)
        
        plt.savefig('out/Pt_estimate_comparison.pdf')
        plt.close()        
        
    def zone_length_scales(self):
        
        zones = {'LEZ': {'color': self.GREEN,
                        'text_coords': (0,-1.7),
                        'marker':'o'},
                 'UMZ': {'color': self.ORANGE,
                        'text_coords': (80,-0.8),
                        'marker':'x'}}
  
        fig, ax = plt.subplots(1,1)
        for z in self.model.zones:
            c = zones[z.label]['color']
            ax.scatter(z.grid_steps, np.log(z.autocorrelation), label=z.label,
                       marker=zones[z.label]['marker'], color=c)
            ax.plot(z.grid_steps, z.length_scale_fit, '--', lw=1, color=c)
            ax.text(
                *zones[z.label]['text_coords'],
                f'$R^2$ = {z.fit_rsquared:.2f}\n$L$ = {z.length_scale:.1f} m',
                fontsize=12, color=c)
        ax.set_xlabel('Vertical spacing (m)',fontsize=14)
        ax.set_ylabel('ln($r_k$)',fontsize=14)
        ax.legend(fontsize=12)
        plt.savefig('out/length_scales.pdf')
        plt.close()
        
    def poc_data(self):
        
        fig,[ax1,ax2,ax3] = plt.subplots(1,3,tight_layout=True)
        fig.subplots_adjust(wspace=0.5)  
        
        ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)',fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)',fontsize=14)
        ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)',fontsize=14)
        ax1.set_ylabel('Depth (m)',fontsize=14)
        
        [ax.invert_yaxis() for ax in (ax1, ax2, ax3)]
        [ax.set_ylim(
            top=0, bottom=self.model.MAX_DEPTH+30) for ax in (ax1, ax2, ax3)]
        
        ax1.errorbar(
            self.model.Ps.data['conc'], self.model.Ps.data['depth'], fmt='^',
            xerr=self.model.Ps.data['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5,
            label='LVISF', fillstyle='full')
        
        ax2.errorbar(
            self.model.Pl.data['conc'], self.model.Pl.data['depth'], fmt='^',
            xerr=self.model.Pl.data['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5,
            label='LVISF', fillstyle='full')
           
        ax3.scatter(
            self.model.Pt_mean_nonlinear, self.model.GRID, marker='o',
            c=self.BLUE, edgecolors=self.WHITE, s=7, label='from $c_p$',
            zorder=3, lw=0.7)
        ax3.fill_betweenx(
            self.model.GRID,
            (self.model.Pt_mean_nonlinear
             - np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            (self.model.Pt_mean_nonlinear
             + np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            color=self.BLUE, alpha=0.25, zorder=2)
        ax3.errorbar(
            self.model.Ps.data['conc'] + self.model.Pl.data['conc'],
            self.model.sample_depths, fmt='^', ms=10, c=self.BLUE,
            xerr=np.sqrt(self.model.Ps.data['conc_e']**2
                         + self.model.Pl.data['conc_e']**2),
            zorder=1, label='LVISF', capsize=5, fillstyle='full',
            elinewidth=1)
        
        ax1.set_xticks([0,1,2,3])
        ax2.set_xticks([0,0.05,0.1,0.15])
        ax2.set_xticklabels(['0','0.05','0.1','0.15'])
        ax3.set_xticks([0,1,2,3])
        ax3.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                  loc='lower right')
    
        for ax in (ax1, ax2, ax3):            
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)

            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--' , lw=1,
                       zorder=10)
            if ax in (ax2, ax3):
                ax.tick_params(labelleft=False)
            if ax in (ax1, ax3):
                ax.set_xlim([-0.2,3.4])

        plt.savefig('out/poc_data.pdf')
        plt.close()
        
    def cost_and_convergence(self, run):
        
        k = len(run.cost_evolution)
        
        fig, ax = plt.subplots(1)
        ax.plot(np.arange(0, k), run.convergence_evolution,
                marker='o', ms=3, c=self.BLUE)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration, $k$',fontsize=16)
        ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',
                      fontsize=16)
        plt.savefig(f'out/conv_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()
        
        fig, ax = plt.subplots(1)
        ax.plot(np.arange(0, k),run.cost_evolution,
                marker='o', ms=3, c=self.BLUE)
        ax.set_xlabel('Iteration, $k$',fontsize=16)
        ax.set_ylabel('Cost, $J$',fontsize=16)
        ax.set_yscale('log')
        plt.savefig(f'out/cost_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()

    def params(self, run):
        
        fig, ([ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]) = plt.subplots(2,4)
        fig.subplots_adjust(wspace=0.8, hspace=0.4)
        axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        for i, param in enumerate(self.model.params):
            p = param.name
            ax = axs[i]
            ax.set_title(eval(f'self.model.{p}.label'),fontsize=14)
            ax.errorbar(
                    1, eval(f'self.model.{p}.prior'),
                    yerr=eval(f'self.model.{p}.prior_e'), fmt='o', ms=9,
                    c=self.BLUE, elinewidth=1.5, ecolor=self.BLUE,
                    capsize=6, label='Prior', markeredgewidth=1.5)
            if param.dv: 
                ax.errorbar(
                    2, run.param_results[p]['LEZ']['est'],
                    yerr=run.param_results[p]['LEZ']['err'], fmt='o',
                    c=self.GREEN, ms=9, elinewidth=1.5, ecolor=self.GREEN,
                    capsize=6, label='LEZ', markeredgewidth=1.5)
                ax.errorbar(
                     3, run.param_results[p]['UMZ']['est'],
                     yerr=run.param_results[p]['UMZ']['err'], fmt='o',
                     c=self.ORANGE, ms=9, elinewidth=1.5,
                     ecolor=self.ORANGE, capsize=6, label='UMZ',
                     markeredgewidth=1.5)
                if i == 5:
                    ax.legend(
                        loc='upper center', bbox_to_anchor=(1.38,-0.07),
                        ncol=3, fontsize=12, frameon=False)
            else:
                ax.errorbar(
                    3, run.param_results[p]['est'],
                    yerr=run.param_results[p]['err'], fmt='o',
                    c=self.RADISH, ms=9, elinewidth=1.5,
                    ecolor=self.RADISH, capsize=6, markeredgewidth=1.5)
            ax.tick_params(bottom=False,labelbottom=False)
            ax.set_xticks(np.arange(0,5))
            if p == 'Bm2': ax.set_ylim(-0.5,2.5)

        plt.savefig(f'out/params_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()
            
    def poc_profiles(self, run):

        fig,[ax1,ax2,ax3] = plt.subplots(1,3,tight_layout=True)
        fig.subplots_adjust(wspace=0.5)  
        
        ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)',fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)',fontsize=14)
        ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)',fontsize=14)
        ax1.set_ylabel('Depth (m)',fontsize=14)
        
        if run.gamma == 1:
             invname = 'I1'
        elif run.gamma == 0.02:
             invname = 'I2'
        else: invname = run.gamma       
        
        ax1.errorbar(
            self.model.Ps.data['conc'], self.model.Ps.data['depth'], fmt='^',
            xerr=self.model.Ps.data['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5,
            label='LVISF', fillstyle='full')
        ax1.errorbar(
            self.model.Ps.prior['conc'], self.model.Ps.prior['depth'],
            fmt='o', xerr=self.model.Ps.prior['conc_e'], ecolor=self.SKY,
            elinewidth=0.5, c=self.SKY, ms=2, capsize=2,
            label='OI', markeredgewidth=0.5)
        ax1.errorbar(
            run.tracer_results['POCS']['est'], self.model.GRID, fmt='o',
            xerr=run.tracer_results['POCS']['err'], ecolor=self.ORANGE,
            elinewidth=0.5, c=self.ORANGE, ms=3, capsize=2, label=invname,
            fillstyle='none', zorder=3, markeredgewidth=0.5)
        
        ax2.errorbar(
            self.model.Pl.data['conc'], self.model.Pl.data['depth'], fmt='^',
            xerr=self.model.Pl.data['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5,
            label='LVISF', fillstyle='full')
        ax2.errorbar(
            self.model.Pl.prior['conc'], self.model.Pl.prior['depth'],
            fmt='o', xerr=self.model.Pl.prior['conc_e'], ecolor=self.SKY,
            elinewidth=0.5, c=self.SKY, ms=2, capsize=2,
            label='OI', markeredgewidth=0.5)
        ax2.errorbar(
            run.tracer_results['POCL']['est'], self.model.GRID, fmt='o',
            xerr=run.tracer_results['POCL']['err'], ecolor=self.ORANGE,
            elinewidth=0.5, c=self.ORANGE, ms=3, capsize=2, label=invname,
            fillstyle='none', zorder=3, markeredgewidth=0.5)
        
        ax3.scatter(
            self.model.Pt_mean_nonlinear, self.model.GRID, marker='o',
            c=self.BLUE, edgecolors=self.BLUE, s=3, label='from $c_p$',
            zorder=3, lw=0.7)
        ax3.fill_betweenx(
            self.model.GRID,
            (self.model.Pt_mean_nonlinear
             - np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            (self.model.Pt_mean_nonlinear 
             + np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            color=self.BLUE,alpha=0.25,zorder=2)
        ax3.errorbar(
            run.Pt['est'], self.model.GRID, fmt='o', xerr=run.Pt['err'],
            ecolor=self.ORANGE, elinewidth=0.5, c=self.ORANGE, ms=3,
            capsize=2, label=invname, fillstyle='none', zorder=3,
            markeredgewidth=0.5)
        
        ax1.set_xticks([0,1,2,3])
        ax2.set_xticks([0,0.05,0.1,0.15])
        ax2.set_xticklabels(['0','0.05','0.1','0.15'])
        ax3.set_xticks([0,1,2])
       
        for ax in (ax1, ax2, ax3):            
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)
            ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                      loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--' , lw=1)
            if ax in (ax2, ax3):
                ax.tick_params(labelleft=False)
        
        plt.savefig(f'out/POCprofs_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()
    
    def residual_pdfs(self, run):
        
        fig, [ax1,ax2] = plt.subplots(1, 2, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)
        ax1.set_ylabel('Probability Density', fontsize=16)
        ax1.set_xlabel(r'$\frac{\^x_{i}-x_{o,i}}{\sigma_{o,i}}$', fontsize=24)
        ax1.hist(run.x_resids, density=True, bins=20, color=self.BLUE)
        ax2.hist(run.f_resids, density=True, bins=20, color=self.BLUE)
        ax2.set_xlabel(r'$\frac{f(\^x)_{i}}{\sigma_{f(\^x)_{i}}}$',
                        fontsize=24)
        for ax in (ax1, ax2):
            ax.set_xlim([-1,1])
        plt.savefig(f'out/pdfs_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()
    
    def sinking_fluxes(self, run):
        
        th_fluxes = pd.read_excel(
            'pyrite_data.xlsx',sheet_name='thorium_fluxes')
        th_depths = th_fluxes['depth']
        th_flux = th_fluxes['flux']
        th_flux_u = th_fluxes['flux_u']
        st_fluxes = pd.read_excel(
            'pyrite_data.xlsx', sheet_name='trap_fluxes')
        st_depths = st_fluxes['depth']
        st_flux = st_fluxes['flux']
        st_flux_u = st_fluxes['flux_u']
        
        fig, (ax1,ax2) = plt.subplots(1, 2)
        ax1.set_ylabel('Depth (m)',fontsize=14)
        fig.text(
            0.5, 0.03, 'POC Flux (mmol m$^{-2}$ d$^{-1}$)',
            fontsize=14, ha='center', va='center')
        for ax in (ax1,ax2):
            ax.invert_yaxis()
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', lw=0.5)
            ax.set_ylim(
                top=0, bottom=self.model.MAX_DEPTH+self.model.GRID_STEP*2)
        
        eb1 = ax1.errorbar(
            run.flux_profiles['sink_S']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_S']['err'], ecolor=self.BLUE,
            elinewidth=0.5, c=self.BLUE, ms=3, capsize=2,
            label=self.model.sink_S.label,fillstyle='none',
            markeredgewidth=0.5)
        eb1[-1][0].set_linestyle('--')
        ax1.axhline(self.model.BOUNDARY, c='k', ls='--', lw=0.5)
        eb2 = ax1.errorbar(
            run.flux_profiles['sink_L']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_L']['err'], ecolor=self.ORANGE,
            elinewidth=0.5, c=self.ORANGE, ms=3, capsize=2,
            label=self.model.sink_L.label,fillstyle='none',
            markeredgewidth=0.5)
        eb2[-1][0].set_linestyle(':')
        ax1.legend(loc='lower right',fontsize=10)
        ax1.annotate(
            'A', xy=(0.91, 0.94), xycoords='axes fraction', fontsize=16)
        
        ax2.tick_params(labelleft=False)
        eb3 = ax2.errorbar(
            run.flux_profiles['sink_T']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_T']['err'], ecolor=self.SKY,
            elinewidth=0.5, c=self.SKY, ms=3, capsize=2,
            label=self.model.sink_T.label,fillstyle='none',
            markeredgewidth=0.5)
        eb3[-1][0].set_linestyle('--')
        eb4 = ax2.errorbar(
            th_flux, th_depths, fmt='^', xerr=th_flux_u,
            ecolor=self.GREEN, elinewidth=1.5, c=self.GREEN, ms=4, capsize=2,
            label='$^{234}$Th-based', markeredgewidth=1.5)
        eb4[-1][0].set_linestyle(':')
        eb5 = ax2.errorbar(
            st_flux, st_depths, fmt='^', xerr=st_flux_u,
            ecolor=self.VERMILLION, elinewidth=1.5, c=self.VERMILLION, ms=4,
            capsize=2, label='Sediment Traps', markeredgewidth=1.5)
        eb5[-1][0].set_linestyle(':')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.annotate(
            'B', xy=(0.91, 0.94), xycoords='axes fraction', fontsize=16)
        
        plt.savefig(f'out/sinkfluxes_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()  

    def volumetric_fluxes(self, run):
        
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        fig.subplots_adjust(left=0.15,bottom=0.15,wspace=0.1)
        c1 = self.BLUE
        c2 = self.ORANGE
        axs = (ax1,ax2,ax3,ax4)
        panels = ('A', 'B', 'C', 'D')
        fig.text(0.5, 0.05, 'Volumetric POC Flux (mmol m$^{-3}$ d$^{-1}$)',
                  fontsize=14, ha='center', va='center')
        fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center',
                  va='center', rotation='vertical')
        
        pairs = (('sinkdiv_S', 'sinkdiv_L'), ('remin_S', 'aggregation'),
                  ('remin_L', 'disaggregation'), ('production',))
        
        
        for i,pr in enumerate(pairs):
            ax = axs[i]
         
            eb1 = ax.errorbar(
                run.flux_profiles[pr[0]]['est'], self.model.GRID, fmt='o',
                xerr=run.flux_profiles[pr[0]]['err'], ecolor=c1,
                elinewidth=0.5, c=c1, ms=1.5, capsize=2,
                label=eval(f'self.model.{pr[0]}.label'), fillstyle='none',
                markeredgewidth=0.5)
            eb1[-1][0].set_linestyle('--')

            
            if len(pr) > 1:
                eb2 = ax.errorbar(
                    run.flux_profiles[pr[1]]['est'], self.model.GRID, fmt='o',
                    xerr=run.flux_profiles[pr[1]]['err'], ecolor=c2,
                    elinewidth=0.5, c=c2, ms=1.5, capsize=2,
                    label=eval(f'self.model.{pr[1]}.label'), fillstyle='none',
                    markeredgewidth=0.5)
                eb2[-1][0].set_linestyle(':')

            if pr[0] == 'production':
                df = self.model.data['npp']
                H = self.model.MIXED_LAYER_DEPTH
                npp = df.loc[df['target_depth'] >= H]['npp']
                depth = df.loc[df['target_depth'] >= H]['target_depth']
                ax.scatter(npp/self.model.MOLAR_MASS_C, depth, c=c2,
                            alpha=0.5, label='NPP', s=10)
                
            ax.legend(loc='lower right',fontsize=12)
            ax.annotate(panels[i], xy=(0.9, 0.8), xycoords='axes fraction',
                        fontsize=12)
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', lw=0.5)
            ax.set_yticks([0,100,200,300,400,500])
            if i % 2:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()
            ax.set_ylim(
                top=0, bottom=self.model.MAX_DEPTH+self.model.GRID_STEP)
        plt.savefig(
            f'out/fluxes_volumetric_gam{str(run.gamma).replace(".","")}.pdf')
        plt.close()
    
    def write_output(self):
                
        file = 'out/invP_out.txt'
        with open (file,'w') as f:
            for run in self.model.model_runs:
                print('#################################', file=f)
                print(f'GAMMA = {run.gamma}', file=f)
                print('#################################', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Parameter Estimates', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for param in self.model.params:
                    p = param.name
                    if param.dv:
                        for z in self.model.zones:
                            est = run.param_results[p][z.label]['est']
                            err = run.param_results[p][z.label]['err']
                            print(f'{p} ({z.label}): {est:.3f} ± {err:.3f}',
                                  file=f)
                    else:
                        est = run.param_results[p]['est']
                        err = run.param_results[p]['err']
                        print(f'{p}: {est:.3f} ± {err:.3f}', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Tracer Inventories', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for z in self.model.zones:
                    print(f'--------{z.label}--------', file=f)
                    for t in run.inventories[z.label].keys():
                        est, err = run.inventories[z.label][t]                       
                        print(f'{t}: {est:.0f} ± {err:.0f}', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Integrated Fluxes', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for z in self.model.zones:
                    print(f'--------{z.label}--------', file=f)
                    for flux in run.flux_integrals[z.label].keys():
                        est, err = run.flux_integrals[z.label][flux]                       
                        print(f'{flux}: {est:.2f} ± {err:.2f}', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Timescales', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for z in self.model.zones:
                    print(f'--------{z.label}--------', file=f)
                    for t in run.integrated_resids[z.label].keys():
                        print(f'***{t}***', file=f)
                        for flux in run.timescales[z.label][t].keys():
                            est, err = run.timescales[z.label][t][flux] 
                            print(f'{flux}: {est:.3f} ± {err:.3f}',
                                  file=f)        
    
    def param_comparison(self, run):
        
        LEZ, UMZ = self.model.zones
        dpy = self.model.DAYS_PER_YEAR
        Ps_LEZ_mean = run.tracer_results['POCS']['est'][LEZ.indices].mean()
        Ps_UMZ_mean = run.tracer_results['POCS']['est'][UMZ.indices].mean()
        B2_EX_LEZ = run.param_results['B2p']['LEZ']['est']*Ps_LEZ_mean
        B2_EX_UMZ = run.param_results['B2p']['UMZ']['est']*Ps_UMZ_mean
        
        data = {'EXP':{'B2':{'EZ':(B2_EX_LEZ,),'MZ':(B2_EX_UMZ,)},
                        'Bm2':{
                            'EZ':(run.param_results['Bm2']['LEZ']['est'],
                                  run.param_results['Bm2']['LEZ']['err']),
                            'MZ':(run.param_results['Bm2']['UMZ']['est'],
                                  run.param_results['Bm2']['UMZ']['err'])},
                        'Bm1s':{
                            'EZ':(run.param_results['Bm1s']['LEZ']['est'],
                                  run.param_results['Bm1s']['LEZ']['err']),
                            'MZ':(run.param_results['Bm1s']['UMZ']['est'],
                                  run.param_results['Bm1s']['UMZ']['err'])}},
                'MOSP':{'B2':{'BZ':(0.8/dpy,0.9/dpy)},
                        'Bm2':{'BZ':(400/dpy,10000/dpy)},
                        'Bm1s':{'BZ':(1.7/dpy,0.9/dpy)}},
                'MNABE':{'B2':{'MZ':{'t1':(2/dpy,0.2/dpy),
                                      't2':(12/dpy,1/dpy),
                                      't3':(76/dpy,9/dpy)}},
                          'Bm2':{'MZ':{'t1':(156/dpy,17/dpy),
                                      't2':(321/dpy,32/dpy),
                                      't3':(524/dpy,74/dpy)}},
                          'Bm1s':{'MZ':{'t1':(13/dpy,1/dpy),
                                        't2':(32/dpy,2/dpy),
                                        't3':(596/dpy,6/dpy)}}},
                'MNWA':{'B2':{'EZ':{'lo':(9/dpy,24/dpy),'hi':(11/dpy,30/dpy)},
                              'MZ':{'lo':(13/dpy,50/dpy),'hi':(18/dpy,89/dpy)}
                                },
                        'Bm2':{
                            'EZ':{'lo':(2280/dpy,10000/dpy),
                                  'hi':(2690/dpy,10000/dpy)},
                            'MZ':{'lo':(870/dpy,5000/dpy),
                                  'hi':(1880/dpy,10000/dpy)}},
                        'Bm1s':{'EZ':{'lo':(70/dpy,137/dpy),
                                      'hi':(798/dpy,7940/dpy)},
                                'MZ':{'lo':(113/dpy,10000/dpy),
                                      'hi':(1766/dpy,10000000/dpy)}}}}
        
        fig, ([ax1,ax2,ax3], [ax4,ax5,ax6]) = plt.subplots(
            2,3,tight_layout=True)
        axs = [ax1,ax2,ax3,ax4,ax5]
        for ax in axs:
            ax.tick_params(bottom=False,labelbottom=False)
            ax.set_yscale('log')
            # next line from https://stackoverflow.com/questions/21920233/
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax1.set_ylabel('Estimate (d$^{-1}$)', fontsize=14)
        ax1.set_title('$\\beta_{-1,S}$', fontsize=14)
        ax2.set_title('$\\beta_{-2}$',fontsize=14)
        ax3.set_title('$\\beta_2$',fontsize=14)
        ax4.set_ylabel('Error (d$^{-1}$)', fontsize=14)
        ax6.axis('off')
        
        study_colors = {'EXP':self.GREEN, 'MOSP':self.BLUE,
                        'MNABE':self.ORANGE, 'MNWA':self.RADISH}
        zone_shapes = {'EZ':'s', 'MZ':'^', 'BZ':'d'}
        axs_dict = {ax1:{'ylim':(0.001, 10), 'panel':'A'},
                    ax2:{'ylim':(0.1, 10), 'panel':'B'},
                    ax3:{'ylim':(0.001, 1), 'panel':'C'},
                    ax4:{'ylim':(0.001, 100000), 'panel':'D'},
                    ax5:{'ylim':(0.01, 100), 'panel':'E'}}
        
        for (ax,p) in ((ax1,'Bm1s'), (ax2,'Bm2'), (ax3,'B2')):
            ct = 0
            for s in data.keys():
                c = study_colors[s]
                for z, vals in data[s][p].items():
                    m = zone_shapes[z]
                    if type(vals) == dict:
                        for k in vals.keys():
                            ax.scatter(
                                ct, data[s][p][z][k][0], s=60, marker=m, c=c,
                                edgecolors=self.BLACK, lw=0.5)
                            if p != 'B2':
                                axs[axs.index(ax)+3].scatter(
                                    ct, data[s][p][z][k][1], s=60, marker=m,
                                    c=c, edgecolors=self.BLACK, lw=0.5)
                            ct += 1
                    else:
                        ax.scatter(
                            ct, data[s][p][z][0], s=60, marker=m, c=c,
                            edgecolors=self.BLACK, lw=0.5)
                        if p != 'B2':
                            axs[axs.index(ax)+3].scatter(
                                ct, data[s][p][z][1], s=60, marker=m, c=c,
                                edgecolors=self.BLACK, lw=0.5)
                        ct += 1
        leg_elements = [
            Line2D([0], [0], marker=zone_shapes['EZ'], c=self.WHITE,
                   label='Euphotic Zone', markerfacecolor=self.WHITE,
                   markeredgecolor=self.BLACK, ms=9, lw=0.5),
            Line2D([0], [0], marker=zone_shapes['MZ'], c=self.WHITE,
                   label='Mesopelagic Zone', markerfacecolor=self.WHITE,
                   markeredgecolor=self.BLACK,ms=9,lw=0.5),
            Line2D([0], [0], marker=zone_shapes['BZ'], c=self.WHITE,
                   label='Bathypelagic Zone', markerfacecolor=self.WHITE,
                   markeredgecolor=self.BLACK, ms=9, lw=0.5),
            Line2D([0], [0], marker='o', c=self.WHITE,
                   label='This study (I2)\nStation P',
                   markerfacecolor=self.GREEN, ms=9),
            Line2D([0], [0], marker='o', c=self.WHITE,
                   label='Murnane (1994)\nStation P',
                   markerfacecolor=self.BLUE, ms=9),
            Line2D([0], [0], marker='o', c=self.WHITE,
                   label='Murnane et al. (1996)\nNABE',
                   markerfacecolor=self.ORANGE, ms=9),
            Line2D([0], [0], marker='o', c=self.WHITE,
                   label='Murnane et al. (1994)\nNWAO',
                   markerfacecolor=self.RADISH, ms=9)]
        ax6.legend(handles=leg_elements, loc='center', fontsize=10,
                   frameon=False)
        for ax in axs:
            ax.set_ylim(axs_dict[ax]['ylim'])
            ax.annotate(axs_dict[ax]['panel'], xy=(0.82, 0.05),
                        xycoords='axes fraction', fontsize=14)
        plt.savefig('out/compare_params.pdf')
        plt.close()
    
    def integrated_residuals(self):
        
        fig, ax = plt.subplots()
        ax.set_xticks([k for k in list(range(0, len(self.model.model_runs)))])
        ax.set_xticklabels(self.model.gammas)
        ax.set_yticks(list(range(-11, 2)))
        ax.grid(axis='y', zorder=1)
        plt.subplots_adjust(bottom=0.1)
        tracerdict = {'POCS':{'marker':'s', 'label': self.model.Ps.label},
                      'POCL':{'marker':'^', 'label': self.model.Pl.label}}
        zone_colors = {'LEZ':self.GREEN, 'UMZ':self.ORANGE}
        for t in tracerdict.keys():
            m = tracerdict[t]['marker']
            l = tracerdict[t]['label']
            for z in zone_colors.keys():
                j=0
                c = zone_colors[z]
                for run in self.model.model_runs:
                    ax.scatter(
                        j, run.integrated_resids[z][t], marker=m, c=c, s=64,
                        label=f'{l}$^{{{z}}}$', zorder=2)
                    j += 1
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.set_ylabel('Integrated Residuals (mmol m$^{-2}$ d$^{-1}$)',
                      fontsize=14)
        ax.set_xlabel('$\gamma$', fontsize=14)
        plt.savefig('out/intresids.pdf')
        plt.close()
        
    def param_sensitivity(self):

        fig, ([ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]) = plt.subplots(
            3, 3, figsize=(8, 8))
        fig.subplots_adjust(wspace=0.3, hspace=0.8)
        axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
        axs[-1].axis('off')
        for i, param in enumerate(self.model.params):
            p = param.name
            ax = axs[i]
            ax.set_title(param.label,fontsize=14)
            if param.dv:
                ax.axvline(9, ls='--', c=self.BLACK, lw=1)
                j = 0
                for zone in self.model.zones:
                    z = zone.label
                    c = self.GREEN if z == 'LEZ' else self.ORANGE
                    for run in self.model.model_runs:
                        ax.errorbar(
                            j*2, run.param_results[p][z]['est'],
                            yerr=run.param_results[p][z]['err'], fmt='o', c=c,
                            ms=8, elinewidth=1.5, ecolor=c, capsize=6,
                            label=z, markeredgewidth=1.5)
                        j += 1
                    ax.set_xticks([k*2 for k in list(
                        range(0, len(self.model.model_runs)*2))])
                    ax.get_xaxis().set_major_formatter(
                        mpl.ticker.ScalarFormatter())
                    ax.set_xticklabels(self.model.gammas + self.model.gammas,
                                       rotation=60)
                if i == 5:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(),
                              loc='upper center', bbox_to_anchor=(0.5,-1),
                              ncol=1, fontsize=12, frameon=False)
            else:
                j = 0
                for run in self.model.model_runs:
                    ax.errorbar(j*3, run.param_results[p]['est'],
                                yerr=run.param_results[p]['err'] ,fmt='o',
                                c=self.RADISH, ms=8, elinewidth=1.5,
                                ecolor=self.RADISH, capsize=6,
                                markeredgewidth=1.5)
                    j += 1
                ax.set_xticks([k*3 for k in list(
                    range(0, len(self.model.model_runs)))])
                ax.get_xaxis().set_major_formatter(
                    mpl.ticker.ScalarFormatter())
                ax.set_xticklabels(self.model.gammas, rotation=60)
        plt.savefig('out/sensitivity_params.pdf')
        plt.close()
        
    def param_relative_errors(self):

        mod = self.model
        fig, ax = plt.subplots(1, 1)
        plt.subplots_adjust(top=0.8)
        tset_list = [param.label for param in mod.params]
        for i, param in enumerate(mod.params):
            p = param.name
            if param.dv:
                for zone in self.model.zones:
                    z = zone.label
                    if z == 'LEZ':
                        m = '^'
                        ls = '--'
                    else:
                        m = 'o'
                        ls = ':'
                    relativeerror = [
                        r.param_results[p][z]['err']
                        /r.param_results[p][z]['est'] for r in mod.model_runs]
                    ax.plot(
                        mod.gammas, relativeerror, m, c=self.colors[i],
                        label=f'{param.label}', fillstyle='none', ls=ls)
            else:
                relativeerror = [
                    r.param_results[p]['err']
                    /r.param_results[p]['est'] for r in mod.model_runs]
                ax.plot(mod.gammas, relativeerror, 'x', c=self.colors[i],
                        label=f'{param.label}', ls='-.')
        ax.set_xscale('log')
        ax.set_xticks(mod.gammas)
        leg_elements = [
            Line2D([0], [0], marker='o', ls='none', color=self.colors[i],
                   label=tset_list[i]) for i,_ in enumerate(tset_list)]
        ax.legend(
            handles=leg_elements, loc='lower center',
            bbox_to_anchor=(0.49, 1), ncol=4, fontsize=12, frameon=False)
        ax.set_xlabel('$\gamma$', fontsize=14)
        ax.set_ylabel('Relative Error', fontsize=14)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.savefig('out/paramrelerror.pdf')
        plt.close()
        
if __name__ == '__main__':

    sys.setrecursionlimit(100000)
    start_time = time.time()
    model = PyriteModel()
    #plotter = PyritePlotter('out/pyrite_Amaral21a.pkl')

    print(f'--- {(time.time() - start_time)/60} minutes ---')

