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
#from varname import nameof

class PyriteModel:

    def __init__(self, gammas=[0.02], pickle_into='out/pyrite_Amaral21a.pkl'):
        
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
        self.process_cp_data()
        
        self.LEZ = PyriteZone(self, op.lt, 'LEZ')  # euphotic zone
        self.UMZ = PyriteZone(self, op.gt, 'UMZ')  # upper mesopelagic zone
        self.zones = (self.LEZ, self.UMZ)
        
        self.objective_interpolation()
        self.define_prior_vector_and_cov_matrix()
        self.define_equation_elements()
        
        self.model_runs = [PyriteModelRun(self, g) for g in gammas]

        self.pickle_model()

    def __repr__(self):

        return 'PyriteModel object'
                    
    def load_data(self):
        
        self.DATA = pd.read_excel('pyrite_data.xlsx',sheet_name=None)
        self.SAMPLE_DEPTHS = self.DATA['poc_means']['depth']
        self.N_SAMPLE_DEPTHS = len(self.SAMPLE_DEPTHS)
    
    def define_tracers(self):
        
        self.Ps = PyriteTracer('POC', 'S', '$P_S$', self.DATA['poc_means'][
            ['depth', 'SSF_mean', 'SSF_se']])
        self.Pl = PyriteTracer('POC', 'L', '$P_L$', self.DATA['poc_means'][
            ['depth', 'LSF_mean', 'LSF_se']])

        self.tracers = (self.Ps, self.Pl)

    def define_params(self):
        
        P30_prior, P30_prior_e, Lp_prior, Lp_prior_e = self.process_npp_data()

        self.ws = PyriteParam(2, 2, 'ws', '$w_S$')
        self.wl = PyriteParam(20, 15, 'wl', '$w_L$')
        self.B2p = PyriteParam(0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               0.5*self.MOLAR_MASS_C/self.DAYS_PER_YEAR,
                               'B2p', '$\\beta^,_2$')
        self.Bm2 = PyriteParam(400/self.DAYS_PER_YEAR,
                               10000/self.DAYS_PER_YEAR,
                               'Bm2', '$\\beta_{-2}$')
        self.Bm1s = PyriteParam(0.1, 0.1, 'Bm1s', '$\\beta_{-1,S}$')
        self.Bm1l = PyriteParam(0.15, 0.15, 'Bm1l', '$\\beta_{-1,L}$')
        self.P30 = PyriteParam(P30_prior, P30_prior_e, 'P30', '$\.P_{S,30}$',
                               depth_vary=False)
        self.Lp = PyriteParam(Lp_prior, Lp_prior_e, 'Lp', '$L_P$',
                              depth_vary=False)

        self.model_params = (self.ws, self.wl, self.B2p, self.Bm2, self.Bm1s,
                             self.Bm1l, self.P30, self.Lp)

    def process_npp_data(self):
        
        npp_data_raw = self.DATA['npp']
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

    def process_cp_data(self):

        cast_match_table = self.DATA['cast_match']
        cast_match_dict = dict(zip(cast_match_table['pump_cast'],
                                   cast_match_table['ctd_cast']))
        poc_discrete = self.DATA['poc_discrete']
        cp_bycast = self.DATA['cp_bycast']
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
        self.Pt_mean_nonlinear = self.cp_Pt_regression_nonlinear.get_prediction(
            exog=dict(cp=self.cp_mean)).predicted_mean
        self.Pt_mean_linear = self.cp_Pt_regression_linear.get_prediction(
            exog=dict(cp=self.cp_mean)).predicted_mean

    def objective_interpolation(self):
        
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
                
                def R_matrix(array1, array2):
                    
                    m = len(array1)
                    n = len(array2)
                    R = np.zeros((m,n))
                    
                    for i in np.arange(0,m):
                        for j in np.arange(0,n):
                            R[i,j] = np.exp(-np.abs(array1[i]-array2[j])/L)
                    
                    return R
                
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
        
        for p in self.model_params:
            if p.dv:
                for z in self.zones:
                    param_priors.append(p.prior)
                    param_priors_var.append(p.prior_e**2)
                    self.state_elements.append(f'{p.name}_{z.label}')
            else:
                param_priors.append(p.prior)
                param_priors_var.append(p.prior_e**2)
                self.state_elements.append(f'{p.name}')
        self.npe = len(param_priors)  # number of parameter elements

        self.xo = np.concatenate((tracer_priors,param_priors))
        self.xo_log = np.log(self.xo)  
        self.nse = len(self.xo)  # number of state elements
        
        self.Co = np.zeros((self.nse, self.nse))
        
        tracer_cov_matrices = [t.cov_matrices for t in self.tracers]
        tracer_cov_matrices = list(
            itertools.chain.from_iterable(tracer_cov_matrices))
        
        self.Co[:self.nte, :self.nte] = splinalg.block_diag(
            *tracer_cov_matrices)
        np.fill_diagonal(self.Co[self.nte:, self.nte:], param_priors_var)
        
        for i, row in enumerate(self.Co):
            for j, val in enumerate(row):
                self.Co[i,j] = np.log(1 + val/(self.xo[i]*self.xo[j]))
     
    def define_equation_elements(self):
        
        self.equation_elements = self.state_elements[:self.nte]
        
        for i in range(0,self.N_GRID_POINTS):
            self.equation_elements.append(f'POCT_{i}')  
            
    def which_zone(self, depth):
        
        if int(depth) in self.LEZ.indices:
            return 'LEZ'
        return 'UMZ'
            
    def pickle_model(self):

        with open(self.pickled, 'wb') as file:
            pickle.dump(self,file)

class PyriteTracer:

    def __init__(self, species, size_fraction, label, data):
        self.species = species
        self.sf = size_fraction
        self.label = label
        self.data = pd.DataFrame(data)
        self.data.rename(columns={self.data.columns[1]: 'conc',
                                  self.data.columns[2]: 'conc_e'},
                         inplace=True)

    def __repr__(self):

        return f'PyriteTracer(species={self.species}, size_frac={self.sf})'
        
class PyriteParam:

    def __init__(self, prior, prior_error, name, label, depth_vary=True):
        
        self.prior = prior
        self.prior_e = prior_error
        self.name = name
        self.label = label
        self.dv = depth_vary

    def __repr__(self):

        return f'PyriteParam({self.name})'

class PyriteZone:

    def __init__(self, model, operator, label):

        self.model = model
        self.indices = np.where(operator(model.GRID, model.BOUNDARY))[0]
        self.depths = model.GRID[self.indices]
        self.label = label
        
        self.calculate_length_scales(0.25)

    def __repr__(self):

        return f'PyriteZone({self.label})'
    
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

class PyriteModelRun():
    
    def __init__(self, model, gamma):
        
        self.model = model
        self.gamma = gamma
        
        self.define_model_error_matrix()
        self.run_model()
        
    def __repr__(self):
        
        return 'PyriteModelRun(gamma={self.gamma})'

    def define_model_error_matrix(self):
        
        Cf_Ps_Pl = np.zeros((self.model.nte, self.model.nte))
        Cf_Pt = np.zeros((self.model.N_GRID_POINTS, self.model.N_GRID_POINTS))
        
        np.fill_diagonal(Cf_Ps_Pl, (self.model.P30.prior**2)*self.gamma)
        np.fill_diagonal(Cf_Pt,
                         self.model.cp_Pt_regression_nonlinear.mse_resid)
        
        self.Cf = splinalg.block_diag(Cf_Ps_Pl, Cf_Pt)
    
    def run_model(self):

        def equation_builder(species, depth):

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
                eq = Psi + Pli - Pti
            
            return eq
            
        def extract_equation_variables(y, depth):
            
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
                        zone = self.model.which_zone(depth)
                        element = '_'.join([param.name, zone])
                    else:
                        element = param.name
                element_index = self.model.state_elements.index(element)
                x_indices.append(element_index)
                x_numerical.append(np.exp(self.xk[element_index]))
            
            return(x_symbolic, x_numerical, x_indices)
        
        def evaluate_jacobian_and_model_equations():
            
            F = np.zeros((len(self.Cf), self.model.nse))  # jacobian matrix
            f = np.zeros(len(self.Cf))  # vector of model equations
            
            for i, element in enumerate(self.model.equation_elements):
                species, depth = element.split('_')
                y = equation_builder(species, depth)
                x_sym, x_num, x_ind = extract_equation_variables(y, depth)
                f[i] = sym.lambdify(x_sym, y)(*x_num)
                for j, x in enumerate(x_sym):
                    dy = y.diff(x)*x  # dy/d(ln(x)) = x*dy/dx
                    dx_sym, dx_num, _ = extract_equation_variables(dy, depth)
                    F[i,x_ind[j]] = sym.lambdify(dx_sym, dy)(*dx_num)   
            
            return(F, f)
        
        def calculate_xkp1(F, f):
            
            CoFT = self.model.Co @ F.T
            FCoFT = F @ CoFT
            FCoFTpCfi = np.linalg.inv(FCoFT + self.Cf)
            return (self.model.xo_log
                    + CoFT @ FCoFTpCfi
                    @ (F @ (self.xk - self.model.xo_log) - f))
            
        def check_convergence(xkp1):
            
            converged = False
            max_change_limit = 0.01 
            change = np.abs((np.exp(xkp1) - np.exp(self.xk))/np.exp(self.xk))
            self.convergence_evolution.append(np.max(change))
            if np.max(change) < max_change_limit:
                converged = True
            
            return converged
        
        def calculate_cost(f):
            
            if self.gamma == 0:
                f = f[:self.model.nte]
                Cf = self.Cf[:self.model.nte,:self.model.nte]
            else:
                Cf = self.Cf           
            cost = ((self.xk - self.model.xo_log).T
                    @ np.linalg.inv(self.model.Co)
                    @ (self.xk - self.model.xo_log)
                    + f.T @ np.linalg.inv(Cf) @ f)            
            self.cost_evolution.append(cost)
                    
            return
                                               
        self.xk = self.model.xo_log  # estimate of state vector at iteration k
        xkp1 = np.ones(len(self.xk))  # at iteration k+1

        max_iterations = 25
        self.cost_evolution = []
        self.convergence_evolution = []
        
        for _ in range(0, max_iterations):
            F, f = evaluate_jacobian_and_model_equations()
            xkp1 = calculate_xkp1(F, f)
            calculate_cost(f)
            if check_convergence(xkp1) == True:
                break
            self.xk = xkp1
        
        self.n_iterations = len(self.cost_evolution) - 1
           
class PyritePlotter:

    def __init__(self, pickled_model):

        with open(pickled_model.pickled, 'rb') as file:
            self.model = pickle.load(file)

        self.define_colors()
        self.plot_hydrography()
        self.plot_cp_Pt_regression()
        self.compare_Pt_estimates()
        self.plot_zone_length_scales()
        self.plot_poc_data()
        
        self.plot_cost_and_convergence()

    def define_colors(self):

        self.BLACK = '#000000'
        self.WHITE = '#FFFFFF'
        self.ORANGE = '#E69F00'
        self.SKY = '#56B4E9'
        self.GREEN = '#009E73'
        self.YELLOW = '#F0E442'
        self.BLUE = '#0072B2'
        self.VERMILLION = '#D55E00'
        self.RADISH = '#CC79A7'

    def plot_hydrography(self):

        hydro_df = self.model.DATA['hydrography']

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

    def plot_cp_Pt_regression(self):

        cp = self.model.poc_cp_df['cp']
        Pt = self.model.poc_cp_df['Pt']
        depths = self.model.poc_cp_df['depth']
        linear_regression = self.model.cp_Pt_regression_linear
        nonlinear_regression = self.model.cp_Pt_regression_nonlinear
        logarithmic = {linear_regression: False, nonlinear_regression: True}

        colormap = plt.cm.viridis_r
        norm = mplc.Normalize(depths.min(), depths.max())

        for fit in (linear_regression, nonlinear_regression):
            fig, ax =  plt.subplots(1,1)
            fig.subplots_adjust(bottom=0.2,left=0.2)
            cbar_ax = colorbar.make_axes(ax)[0]
            cbar = colorbar.ColorbarBase(cbar_ax, norm=norm, cmap=colormap)
            cbar.set_label('Depth (m)\n', rotation=270, labelpad=20,
                           fontsize=14)
            ax.scatter(cp, Pt, norm=norm, edgecolors=self.BLACK, c=depths,
                       s=40, marker='o', cmap=colormap)
            ax.set_ylabel('$P_T$ (mmol m$^{-3}$)',fontsize=14)
            ax.set_xlabel('$c_p$ (m$^{-1}$)',fontsize=14)
            x_fit = np.arange(0.01,0.14,0.0001)
            coefs = fit.params
            if logarithmic[fit]:
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_xlim(0.0085, 0.15)
                y_fit = [coefs[0] + coefs[1]*np.log(x) for x in x_fit]
            else: y_fit = [coefs[0] + coefs[1]*x for x in x_fit]
            ax.annotate(
                f'$R^2$ = {fit.rsquared:.2f}\n$N$ = {fit.nobs:.0f}',
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
            ax.plot(x_fit, y_fit, '--', c=self.BLACK, lw=1)
            plt.savefig(f'out/cpptfit_log{logarithmic[fit]}.pdf')
            plt.close()

    def compare_Pt_estimates(self):

        fig,ax = plt.subplots(1,1)
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
        
    def plot_zone_length_scales(self):
        
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
        
    def plot_poc_data(self):
        
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
            c=self.BLUE, edgecolors=self.WHITE, s=7, label='from $c_P$',
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
            self.model.SAMPLE_DEPTHS, fmt='^', ms=10, c=self.BLUE,
            xerr=np.sqrt(self.model.Ps.data['conc_e']**2
                         + self.model.Pl.data['conc_e']**2),
            zorder=1, label='LVISF', capsize=5, fillstyle='full')
        ax3.legend(fontsize=12, borderpad=0.2, handletextpad=0.4)
        
        ax1.set_xticks([0,1,2,3])
        ax2.set_xticks([0,0.05,0.1,0.15])
        ax2.set_xticklabels(['0','0.05','0.1','0.15'])
        ax3.set_xticks([0,1,2])
        
        [ax.tick_params(labelleft=False) for ax in (ax2,ax3)]
        [ax.set_xlim([-0.2,3.4]) for ax in (ax1,ax3)]
        [ax.tick_params(
            axis='both', which='major', labelsize=12) for ax in (ax1,ax2,ax3)]
        [ax.axhline(
            self.model.BOUNDARY, c=self.BLACK, ls='--', lw=1, zorder=10
            ) for ax in (ax1,ax2,ax3)]
        plt.savefig('out/poc_data.pdf')
        plt.close()
        
        # if with_results:
        #     file_name = 'poc_data_results'
        #     ax1.errorbar(
        #         self.model.Ps.prior['conc'], self.model.Ps.prior['depth'],
        #         fmt='o', xerr=self.model.Ps.prior['conc_e'], ecolor=self.SKY,
        #         elinewidth=0.5, c=self.SKY, ms=2, capsize=2,
        #         label='OI', markeredgewidth=0.5)
        #     ax2.errorbar(
        #         self.model.Pl.prior['conc'], self.model.Pl.prior['depth'],
        #         fmt='o', xerr=self.model.Pl.prior['conc_e'], ecolor=self.SKY,
        #         elinewidth=0.5, c=self.SKY, ms=2, capsize=2,
        #         label='OI', markeredgewidth=0.5)
        #     ax3.errorbar(
        #         self.model.Pt_mean_nonlinear, self.model.GRID, fmt='o',
        #         xerr=np.ones(self.model.N_GRID_POINTS)*np.sqrt(
        #             self.model.cp_Pt_regression_nonlinear.mse_resid),
        #         ecolor=self.BLUE, elinewidth=0.5, c=self.BLUE, ms=2,capsize=2,
        #         label='from $c_P$',markeredgewidth=0.5)
        #     [ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4)
        #      for ax in (ax1,ax2,ax3)]
        
    def plot_cost_and_convergence(self):
        
        for run in self.model.model_runs:
            g = run.gamma
            k = run.n_iterations + 1
            
            fig, ax = plt.subplots(1)
            ax.plot(np.arange(0, k), run.convergence_evolution,
                    marker='o', ms=3, c=self.BLUE)
            ax.set_yscale('log')
            ax.set_xlabel('Iteration, $k$',fontsize=16)
            ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',
                          fontsize=16)
            plt.savefig(f'out/conv_gam{str(g).replace(".","")}.pdf')
            plt.close()
            
            #plot evolution of cost function
            fig, ax = plt.subplots(1)
            ax.plot(np.arange(0, k),run.cost_evolution,
                    marker='o', ms=3, c=self.BLUE)
            ax.set_xlabel('Iteration, $k$',fontsize=16)
            ax.set_ylabel('Cost, $J$',fontsize=16)
            ax.set_yscale('log')
            plt.savefig(f'out/cost_gam{str(g).replace(".","")}.pdf')
            plt.close()

if __name__ == '__main__':

    start_time = time.time()
    model = PyriteModel()
    plotter = PyritePlotter(model)

    print(f'--- {(time.time() - start_time)/60} minutes ---')

