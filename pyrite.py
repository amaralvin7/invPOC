#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:55:53 2021

@author: Vinicius J. Amaral

PYRITE Model (Particle cYcling Rates from Inversion of Tracers in the ocEan)

"""
import pickle
import itertools
import sys
import time
import scipy.linalg as splinalg
import sympy as sym
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as smt
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as mplc
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot


class PyriteModel:
    """A container for attributes and results of model runs.

    As defined, this model produces the results associated with inversions of
    real particulate organic carbon (POC) data as described in Amaral et al.
    (2021).
    """

    def __init__(self, model_id, gammas,
                 pickle_into='out/POC_modelruns_dev.pkl'):
        """Define basic model attributes and run the model.

        Model is run for every value of gamma in gammas.
        gammas -- list of proportionality constants for model runs
        pickle_into -- path for saving model output
        """
        model_ids = {0: ('POC',),
                     1: ('POC', 'Ti')}
        self.species = model_ids[model_id]
        self.gammas = gammas
        self.pickled = pickle_into
        self.MIXED_LAYER_DEPTH = 30
        self.GRID = [0, 30, 50, 100, 150, 200, 330, 500]
        self.MAX_DEPTH = self.GRID[-1]

        self.MOLAR_MASS_C = 12
        self.DAYS_PER_YEAR = 365.24

        self.load_data()
        if str(self) != 'PyriteTwinX object':
            self.define_tracers()
            self.define_params()
            # self.define_fluxes()
            self.process_cp_data()
            self.define_zones()

            xo, xo_log, Co, Co_log = self.define_prior_vector_and_cov_matrix()
            self.define_equation_elements()

            self.model_runs = []
            for g in gammas:
                run = PyriteModelRun(g)
                Cf = self.define_model_error_matrix(g)
                xhat = self.ATI(xo_log, Co_log, Cf, run)
                self.calculate_total_POC(run)
                self.calculate_residuals(xo, Co, xhat, Cf, run)
                # if str(self) != 'PyriteTwinX object':
                #     inventories = self.calculate_inventories(run)
                #     fluxes_sym = self.calculate_fluxes(run)
                #     flux_names, integrated_fluxes = self.integrate_fluxes(
                #         fluxes_sym, run)
                #     self.calculate_timescales(
                #         inventories, flux_names, integrated_fluxes, run)
                self.model_runs.append(run)

        self.pickle_model()

    def __repr__(self):

        return 'PyriteModel object'

    def load_data(self):
        """Load input data (must be from a file called 'pyrite_data.xlsx').

        After loading in data, calculate cruise-averaged POC concentrations.
        """
        self.data = pd.read_excel('pyrite_data.xlsx', sheet_name=None)

        for s in self.species:
            s_all = self.data[s].copy()
            depths = np.sort(s_all['mod_depth'].unique())

            s_means = pd.DataFrame(depths, columns=['depth'])

            s_means['n_casts'] = s_means.apply(
                lambda x: len(s_all.loc[s_all['mod_depth'] == x['depth']]),
                axis=1)

            for t in (f'{s}S', f'{s}L'):
                s_all.loc[s_all[t] < 0, t] = 0
                s_means[t] = s_means.apply(
                    lambda x: s_all.loc[
                        s_all['mod_depth'] == x['depth']][t].mean(), axis=1)
                s_means[f'{t}_sd'] = s_means.apply(
                    lambda x: s_all.loc[
                        s_all['mod_depth'] == x['depth']][t].std(), axis=1)
                if s == 'POC':
                    re_50m = float(s_means.loc[s_means['depth'] == 50, f'{t}_sd']
                                   / s_means.loc[s_means['depth'] == 50, t])
                    s_means.loc[s_means['depth'] == 30, f'{t}_sd'] = (
                        s_means.loc[s_means['depth'] == 30, t]*re_50m)
                s_means[f'{t}_se'] = (s_means[f'{t}_sd']
                                        / np.sqrt(s_means['n_casts']))
            if s == 'Ti':
                s_means = pd.concat(
                    [s_means.iloc[[0]], s_means], ignore_index=True)
                s_means.at[0, 'depth'] = 30

            self.data[f'{s}_means'] = s_means.copy()

    def define_tracers(self):
        """Define tracers to be used in the model."""
        self.POCS = Tracer('POCS', '$P_S$', self.data['POC_means'])
        self.POCL = Tracer('POCL', '$P_L$', self.data['POC_means'])

        self.tracers = [self.POCS, self.POCL]

        if 'Ti' in self.species:
            self.TiS = Tracer('TiS', '$Ti_S$', self.data['Ti_means'])
            self.TiL = Tracer('TiL', '$Ti_L$', self.data['Ti_means'])

            self.tracers.extend([self.TiS, self.TiL])

    def define_params(self):
        """Set prior estimates and errors of model parameters."""
        P30_prior, P30_prior_e, Lp_prior, Lp_prior_e = self.process_npp_data()
        ti_dust = 1.37*0.0042*1000/47.867 #umol m-2 d-1

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
        self.Lp = Param(Lp_prior, Lp_prior_e, 'Lp', '$L_P$', depth_vary=False)

        self.params = [self.ws, self.wl, self.B2p, self.Bm2, self.Bm1s,
                       self.Bm1l, self.P30, self.Lp]

        if 'Ti' in self.species:
            self.Phi = Param(ti_dust, ti_dust, 'Phi', '$\\Phi_D$',
                              depth_vary=False)
            self.params.append(self.Phi)

    # def define_fluxes(self):
    #     """Define fluxes to be calculated."""
    #     self.sink_S = Flux('sink_S', '$w_SP_S$', 'POCS', 'ws')
    #     self.sink_L = Flux('sink_L', '$w_LP_L$', 'POCL', 'wl')
    #     self.sink_T = Flux('sink_T', '$w_TP_T$', 'POCT', 'wt')
    #     self.sinkdiv_S = Flux(
    #         'sinkdiv_S', '$\\frac{d}{dz}w_SP_S$', 'POCS', 'ws', wrt=('POCS',))
    #     self.sinkdiv_L = Flux(
    #         'sinkdiv_L', '$\\frac{d}{dz}w_LP_L$', 'POCL', 'wl', wrt=('POCL',))
    #     self.remin_S = Flux(
    #         'remin_S', '$\\beta_{-1,S}P_S$', 'POCS', 'Bm1s', wrt=('POCS',))
    #     self.remin_L = Flux(
    #         'remin_L', '$\\beta_{-1,L}P_L$', 'POCL', 'Bm1l', wrt=('POCL',))
    #     self.aggregation = Flux(
    #         'aggregation', '$\\beta^,_2P^2_S$', 'POCS', 'B2p',
    #         wrt=('POCS', 'POCL'))
    #     self.disaggregation = Flux(
    #         'disaggregation', '$\\beta_{-2}P_L$', 'POCL', 'Bm2',
    #         wrt=('POCS', 'POCL'))
    #     self.production = Flux(
    #         'production', '${\.P_S}$', 'POCS', None, wrt=('POCS',))

    #     self.fluxes = (self.sink_S, self.sink_L, self.sink_T, self.sinkdiv_S,
    #                    self.sinkdiv_L, self.remin_S, self.remin_L,
    #                    self.aggregation, self.disaggregation, self.production)

    def process_npp_data(self):
        """Obtain prior estiamtes of particle production parameters.

        Lp -- vertical length scale of particle production
        P30 -- production of small POC at the base of the mixed layer
        """
        npp_data_raw = self.data['NPP']
        npp_data_clean = npp_data_raw.loc[(npp_data_raw['NPP'] > 0)]

        MIXED_LAYER_UPPER_BOUND, MIXED_LAYER_LOWER_BOUND = 28, 35

        npp_mixed_layer = npp_data_clean.loc[
            (npp_data_clean['target_depth'] >= MIXED_LAYER_UPPER_BOUND) &
            (npp_data_clean['target_depth'] <= MIXED_LAYER_LOWER_BOUND)]

        npp_below_mixed_layer = npp_data_clean.loc[
            npp_data_clean['target_depth'] >= MIXED_LAYER_UPPER_BOUND]

        P30_prior = npp_mixed_layer['NPP'].mean()/self.MOLAR_MASS_C
        P30_prior_e = npp_mixed_layer['NPP'].sem()/self.MOLAR_MASS_C

        npp_regression = smf.ols(
            formula='np.log(NPP/(P30_prior*self.MOLAR_MASS_C)) ~ target_depth',
            data=npp_below_mixed_layer).fit()

        Lp_prior = -1/npp_regression.params[1]
        Lp_prior_e = npp_regression.bse[1]/npp_regression.params[1]**2

        return P30_prior, P30_prior_e, Lp_prior, Lp_prior_e

    def define_zones(self):
        """Define the grid zones in the model."""
        self.zone_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
        self.zones = [
            GridZone(self, i, z) for i, z in enumerate(self.zone_names)]

    def process_cp_data(self):
        """Obtain estimates of total POC from beam transmissometry data."""
        cast_match_table = self.data['cast_match']
        cast_match_dict = dict(zip(cast_match_table['pump_cast'],
                                   cast_match_table['ctd_cast']))
        poc_concentrations = self.data['POC']
        cp_bycast = self.data['cp_bycast']
        self.poc_cp_df = poc_concentrations.copy()
        self.poc_cp_df['POCT'] = (self.poc_cp_df['POCS']
                                  + self.poc_cp_df['POCL'])

        self.poc_cp_df['ctd_cast'] = self.poc_cp_df.apply(
            lambda x: cast_match_dict[x['pump_cast']], axis=1)
        self.poc_cp_df['cp'] = self.poc_cp_df.apply(
            lambda x: cp_bycast.at[x['depth']-1, x['ctd_cast']], axis=1)

        self.cp_Pt_regression_nonlinear = smf.ols(
            formula='POCT ~ np.log(cp)', data=self.poc_cp_df).fit()
        self.cp_Pt_regression_linear = smf.ols(
            formula='POCT ~ cp', data=self.poc_cp_df).fit()
        cp_bycast_to_mean = cp_bycast.loc[np.array(self.GRID[1:]) -1,
                                          cast_match_table['ctd_cast']]
        cp_mean = cp_bycast_to_mean.mean(axis=1)

        self.Pt_mean_linear = self.cp_Pt_regression_linear.get_prediction(
            exog=dict(cp=cp_mean)).predicted_mean
        self.Pt_mean_nonlinear = (
                self.cp_Pt_regression_nonlinear.get_prediction(
                    exog=dict(cp=cp_mean)).predicted_mean)

        if str(self) != 'PyriteTwinX object':
            self.Pt_constraint = self.Pt_mean_nonlinear

    def define_prior_vector_and_cov_matrix(self):
        """Build the prior vector (xo) and matrix of covariances (Co).

        self.state_elements -- a list of strings corresponding to labels of
        all elements in the state vector
        """

        self.state_elements = []

        tracer_priors = []
        tracer_priors_var = []

        for t in self.tracers:
            tracer_priors.append(t.prior['conc'])
            tracer_priors_var.append(t.prior['conc_e']**2)
            for z in self.zones:
                self.state_elements.append(f'{t.name}_{z.label}')

        tracer_priors = list(itertools.chain.from_iterable(tracer_priors))
        tracer_priors_var = list(
            itertools.chain.from_iterable(tracer_priors_var))
        self.nte = len(tracer_priors)  # number of tracer elements

        param_priors = []
        param_priors_var = []

        for p in self.params:
            if p.dv:
                for z in self.zone_names:
                    param_priors.append(p.prior)
                    param_priors_var.append(p.prior_e**2)
                    self.state_elements.append(f'{p.name}_{z}')
            else:
                param_priors.append(p.prior)
                param_priors_var.append(p.prior_e**2)
                self.state_elements.append(f'{p.name}')

        xo = np.concatenate((tracer_priors, param_priors))
        xo_log = np.log(xo)
        self.nse = len(xo)  # number of state elements

        Co_diag = tracer_priors_var + param_priors_var
        Co = np.diag(Co_diag)
        Co_log_diag = [
            np.log(1 + Co_diag[i]/(xo[i]**2)) for i in range(self.nse)]
        Co_log = np.diag(Co_log_diag)

        return xo, xo_log, Co, Co_log

    def define_equation_elements(self):
        """Define which elements that have an associated model equation.

        Total POC is an element that is not a tracer, but has model equations
        that are used as a constraint. Tracers also have associated model
        equations
        """
        self.equation_elements = self.state_elements[:self.nte]

        for z in self.zones:
            self.equation_elements.append(f'POCT_{z.label}')

        self.nee = len(self.equation_elements)

    # def which_zone(self, depth):
    #     """Given a depth index, return the corresponding grid zone."""
    #     if int(depth) in self.LEZ.indices:
    #         return 'LEZ'
    #     return 'UMZ'

    def define_model_error_matrix(self, g):
        """Return the matrix of model errors (Cf) given a value of gamma."""

        Cf_PsPl = np.diag(np.ones((len(self.GRID) - 1)*2)
                          * ((self.P30.prior*self.MIXED_LAYER_DEPTH)**2)*g)

        Cf_Pt = np.diag(np.ones(len(self.GRID) - 1)
                        * (self.cp_Pt_regression_nonlinear.mse_resid))

        blocks = [Cf_PsPl, Cf_Pt]

        if 'Ti' in self.species:
            n_Ti = len([i for i, el in enumerate(self.state_elements)
                    if 'Ti' in el])
            Cf_Ti = np.diag((np.ones(n_Ti)*(self.Phi.prior**2))*2.5)
            blocks.append(Cf_Ti)

        Cf = splinalg.block_diag(*blocks)

        return Cf

    def slice_by_tracer(self, to_slice, tracer):
        """Return a slice of a list that corresponds to a given tracer.

        to_slice -- list from which to take a slice
        tracer -- return list slice correpsonding to this tracer
        """
        start_index = [i for i, el in enumerate(self.state_elements)
                        if tracer in el][0]
        sliced = to_slice[
            start_index:(start_index + len(self.GRID) - 1)]

        return sliced

    def previous_zone(self, zone_name):

        return self.zone_names[self.zone_names.index(zone_name) - 1]

    def equation_builder(self, species, zone, params_known=None):
        """Return the model equation for a species at a depth index.

        species -- string label for any equation element
        depth -- depth index
        params_known -- a dictionary of parameters from which to draw from,
        should only exist if function is evoked from a TwinX object
        """
        zim1, zi = zone.depths
        h = zi - zim1
        z = zone.label
        if z == 'A':
            Psi = sym.symbols('POCS_A')
            Pli = sym.symbols('POCL_A')
            if 'Ti' in self.species:
                Tsi = sym.symbols('TiS_A')
                Tli = sym.symbols('TiL_A')
        else:
            pz = self.previous_zone(z)
            Psi, Psim1 = sym.symbols(f'POCS_{z} POCS_{pz}')
            Pli, Plim1 = sym.symbols(f'POCL_{z} POCL_{pz}')
            Psa = (Psi + Psim1)/2
            Pla = (Pli + Plim1)/2
            if 'Ti' in self.species:
                Tsi, Tsim1 = sym.symbols(f'TiS_{z} TiS_{pz}')
                Tli, Tlim1 = sym.symbols(f'TiL_{z} TiL_{pz}')
                Tsa = (Tsi + Tsim1)/2
                Tla = (Tli + Tlim1)/2

        if not params_known:
            Bm2 = sym.symbols(f'Bm2_{z}')
            B2p = sym.symbols(f'B2p_{z}')
            Bm1s = sym.symbols(f'Bm1s_{z}')
            Bm1l = sym.symbols(f'Bm1l_{z}')
            ws = sym.symbols(f'ws_{z}')
            wl = sym.symbols(f'wl_{z}')
            P30 = sym.symbols('P30')
            Lp = sym.symbols('Lp')
            if zone.label != 'A':
                wsm1 = sym.symbols(f'ws_{pz}')
                wlm1 = sym.symbols(f'wl_{pz}')
            if 'Ti' in self.species:
                phi = sym.symbols('Phi')
        else:
            Bm2 = params_known['Bm2'][z]['est']
            B2p = params_known['B2p'][z]['est']
            Bm1s = params_known['Bm1s'][z]['est']
            Bm1l = params_known['Bm1l'][z]['est']
            P30 = params_known['P30']['est']
            Lp = params_known['Lp']['est']
            ws = params_known['ws'][z]['est']
            wl = params_known['wl'][z]['est']
            if zone.label != 'A':
                wsm1 = params_known['ws'][pz]['est']
                wlm1 = params_known['wl'][pz]['est']
            if 'Ti' in self.species:
                phi = params_known['Phi']['est']

        if species == 'POCS':
            if zone.label == 'A':
                eq = (-ws*Psi + Bm2*Pli*h - (B2p*Psi + Bm1s)*Psi*h)
                if not params_known:
                    eq += P30*h
            else:
                eq = (-ws*Psi + wsm1*Psim1 + Bm2*Pla*h
                      - (B2p*Psa + Bm1s)*Psa*h)
                if not params_known:
                    eq += Lp*P30*(sym.exp(-zim1/Lp) - sym.exp(-zi/Lp))
        elif species == 'POCL':
            if zone.label == 'A':
                eq = -wl*Pli + B2p*Psi**2*h - (Bm2 + Bm1l)*Pli*h
            else:
                eq = -wl*Pli + wlm1*Plim1 + B2p*Psa**2*h - (Bm2 + Bm1l)*Pla*h
        elif species == 'POCT':
            Pti = self.Pt_constraint[
                (self.equation_elements.index(f'POCT_{z}') - self.nte)]
            eq = Pti - (Psi + Pli)
        elif species == 'TiS':
            if zone.label == 'A':
                eq = -ws*Tsi + (Bm2*Tli - B2p*Psi*Tsi)*h
                if not params_known:
                    eq += phi
            else:
                eq = -ws*Tsi + wsm1*Tsim1 + (Bm2*Tla - B2p*Psa*Tsa)*h
        elif species == 'TiL':
            if zone.label == 'A':
                eq = -wl*Tli + (B2p*Psi*Tsi - Bm2*Tli)*h
            else:
                eq = -wl*Tli + wlm1*Tlim1 + (B2p*Psa*Tsa - Bm2*Tla)*h
        return eq

    def extract_equation_variables(self, y, v, lognormal=False):
        """Return symbolic and numerical values of variables in an equation.

        y -- a symbolic equation
        depth -- depth index at which y exists
        v -- list of values from which to draw numerical values from
        lognormal -- True if numberical values in v are lognormally
        distributed, otherwise False
        """
        x_symbolic = y.free_symbols
        x_numerical = []
        x_indices = []
        for x in x_symbolic:
            element_index = self.state_elements.index(x.name)
            x_indices.append(element_index)
            if lognormal:
                x_numerical.append(np.exp(v[element_index]))
            else:
                x_numerical.append(v[element_index])

        return x_symbolic, x_numerical, x_indices

    def evaluate_model_equations(
            self, v, return_F=False, lognormal=False, params_known=None):
        """Evaluates model equations, and Jacobian matrix (if specified).

        v -- list of values from which to draw numerical values
        return_F -- True if the Jacobian matrix should be returned
        lognormal -- True if numerical variable values in v are lognormally
        distributed, otherwise False (i.e., values are normally distributed)
        params_known -- a dictionary of parameters from which to draw from,
        should only exist if function is evoked from a TwinX object
        """
        if params_known:
            f = np.zeros(self.nte)
            F = np.zeros((self.nte, self.nte))
            eq_elements = self.equation_elements[:self.nte]
        else:
            f = np.zeros(self.nee)
            F = np.zeros((self.nee, self.nse))
            eq_elements = self.equation_elements

        for i, element in enumerate(eq_elements):
            species, zone_name = element.split('_')
            for z in self.zones:
                if z.label == zone_name:
                    zone = z
                    break
            y = self.equation_builder(
                species, zone, params_known=params_known)
            x_sym, x_num, x_ind = self.extract_equation_variables(
                y, v, lognormal=lognormal)
            f[i] = sym.lambdify(x_sym, y)(*x_num)
            if return_F:
                for j, x in enumerate(x_sym):
                    if lognormal:
                        dy = y.diff(x)*x  # dy/d(ln(x)) = x*dy/dx
                    else:
                        dy = y.diff(x)
                    dx_sym, dx_num, _ = self.extract_equation_variables(
                        dy, v, lognormal=lognormal)
                    F[i, x_ind[j]] = sym.lambdify(dx_sym, dy)(*dx_num)

        if return_F:
            return f, F
        return f

    def eval_symbolic_func(self, run, y, err=True, cov=True):
        """Evaluate a symbolic function using results from a given run.

        run -- model run whose results are being calculated
        y -- the symbolic function (i.e., expression)
        err -- True if errors should be propagated (increases runtime)
        cov -- True if covarainces between state variables should be
        considered (increases runtime)
        """
        x_symbolic = y.free_symbols
        x_numerical = []
        x_indices = []

        for x in x_symbolic:
            x_indices.append(self.state_elements.index(x.name))
            if '_' in x.name:  # if it varies with depth
                element, zone = x.name.split('_')
                if element in run.tracer_results:  # if it's a tracer
                    di = self.zone_names.index(zone)
                    x_numerical.append(
                        run.tracer_results[element]['est'][di])
                else:  # if it's a depth-varying parameter
                    x_numerical.append(
                        run.param_results[element][zone]['est'])
            else:  # if it's a depth-independent parameter
                x_numerical.append(run.param_results[x.name]['est'])

        result = sym.lambdify(x_symbolic, y)(*x_numerical)

        if err is False:
            return result

        variance_sym = 0  # symbolic expression for variance of y
        derivs = [y.diff(x) for x in x_symbolic]
        cvm = run.cvm[  # sub-CVM corresponding to state elements in y
            np.ix_(x_indices, x_indices)]
        for i, row in enumerate(cvm):
            for j, _ in enumerate(row):
                if i > j:
                    continue
                if i == j:
                    variance_sym += (derivs[i]**2)*cvm[i, j]
                else:
                    if cov:
                        variance_sym += 2*derivs[i]*derivs[j]*cvm[i, j]
        variance = sym.lambdify(x_symbolic, variance_sym)(*x_numerical)
        error = np.sqrt(variance)

        return result, error

    def ATI(self, xo_log, Co_log, Cf, run):
        """Algorithm of total inversion, returns a vector of state estimates.

        xo_log -- log-transformed prior vector
        Co_log -- log-transformed covariance matrix
        Cf -- model error matrix
        run -- model run whose results are being calculated
        xhat -- Vector that holds estimates of the state elements
        (i.e., the solution vector)

        See: Tarantola A, Valette B. 1982. Generalized nonlinear inverse
        problems solved using the least squares criterion. Reviews of
        Geophysics and Space Physics 20(2): 219–232.
        doi:10.1029/RG020i002p00219.
        """
        def calculate_xkp1(xk, f, F):
            """For iteration k, return a new estimate of the state vector.

            Also returns a couple matrices for future calculations.
            xk -- the state vector estimate at iteration k
            xkp1 -- the state vector estimate at iteration k+1
            f -- vector of model equations
            F -- Jacobian matrix
            """
            CoFT = Co_log @ F.T
            FCoFT = F @ CoFT
            FCoFTpCfi = np.linalg.inv(FCoFT + Cf)
            xkp1 = (xo_log + CoFT @ FCoFTpCfi @ (F @ (xk - xo_log) - f))

            return xkp1, CoFT, FCoFTpCfi

        def check_convergence(xk, xkp1):
            """Return whether or not the ATI has converged after an iteration.

            Convergence is reached if every variable in xkp1 changes by less
            than 1% relative to its estimate at the previous iteration, xk.
            """
            converged = False
            max_change_limit = 0.01
            change = np.abs((np.exp(xkp1) - np.exp(xk))/np.exp(xk))
            run.convergence_evolution.append(np.max(change))
            if np.max(change) < max_change_limit:
                converged = True

            return converged

        def calculate_cost(xk, f):
            """Calculate the cost at a given iteration"""
            cost = ((xk - xo_log).T @ np.linalg.inv(Co_log) @ (xk - xo_log)
                    + f.T @ np.linalg.inv(Cf) @ f)

            run.cost_evolution.append(cost)

        def find_solution():
            """Iteratively finds a solution of the state vector."""
            max_iterations = 100

            xk = xo_log  # estimate of state vector at iteration k
            xkp1 = np.ones(len(xk))  # at iteration k+1
            for _ in range(max_iterations):
                f, F = self.evaluate_model_equations(
                    xk, return_F=True, lognormal=True)
                xkp1, CoFT, FCoFTpCfi = calculate_xkp1(xk, f, F)
                calculate_cost(xk, f)
                run.converged = check_convergence(xk, xkp1)
                if run.converged:
                    break
                xk = xkp1
            print(f'{run.gamma}: {run.converged}')

            return F, xkp1, CoFT, FCoFTpCfi

        def unlog_state_estimates():
            """Convert state estimates from lognormal to normal space."""
            F, xkp1, CoFT, FCoFTpCfi = find_solution()
            Id = np.identity(Co_log.shape[0])

            Ckp1 = ((Id - CoFT @ FCoFTpCfi @ F) @ Co_log
                    @ (Id - F.T @ FCoFTpCfi @ F @ Co_log))

            expected_vals_log = xkp1
            variances_log = np.diag(Ckp1)

            xhat = np.exp(expected_vals_log + variances_log/2)
            xhat_e = np.sqrt(
                np.exp(2*expected_vals_log + variances_log)
                * (np.exp(variances_log) - 1))

            run.cvm = np.zeros(  # covaraince matrix of posterior estimates
                (len(xhat), len(xhat)))

            for i, row in enumerate(run.cvm):
                for j, _ in enumerate(row):
                    ei, ej = expected_vals_log[i], expected_vals_log[j]
                    vi, vj = variances_log[i], variances_log[j]
                    run.cvm[i, j] = (np.exp(ei + ej)*np.exp((vi + vj)/2)
                                     * (np.exp(Ckp1[i, j]) - 1))

            return xhat, xhat_e

        def unpack_state_estimates():
            """Unpack estimates and errors of state elements for later use."""
            xhat, xhat_e = unlog_state_estimates()

            for t in self.tracers:
                run.tracer_results[t.name] = {
                    'est': self.slice_by_tracer(xhat, t.name),
                    'err': self.slice_by_tracer(xhat_e, t.name)}

            for param in self.params:
                p = param.name
                if param.dv:
                    run.param_results[p] = {
                        zone.label: {} for zone in self.zones}
                    for zone in self.zones:
                        z = zone.label
                        zone_param = '_'.join([p, z])
                        i = self.state_elements.index(zone_param)
                        run.param_results[p][z] = {'est': xhat[i],
                                                   'err': xhat_e[i]}
                else:
                    i = self.state_elements.index(p)
                    run.param_results[p] = {'est': xhat[i],
                                            'err': xhat_e[i]}

            return xhat

        return unpack_state_estimates()

    def calculate_total_POC(self, run):
        """Calculate estimates of total POC (with propagated errors)."""
        for z in self.zone_names:
            Ps, Pl = sym.symbols(f'POCS_{z} POCL_{z}')
            Pt_est, Pt_err = self.eval_symbolic_func(run, Ps + Pl)
            run.Pt_results['est'].append(Pt_est)
            run.Pt_results['err'].append(Pt_err)

    def calculate_residuals(self, xo, Co, xhat, Cf, run):
        """Calculate solution and equation residuals."""
        x_residuals = xhat - xo
        norm_x_residuals = x_residuals/np.sqrt(np.diag(Co))
        run.x_resids = norm_x_residuals

        f_residuals = self.evaluate_model_equations(xhat)
        norm_f_residuals = f_residuals/np.sqrt(np.diag(Cf))
        run.f_resids = norm_f_residuals

        for t in run.tracer_results:
            run.tracer_results[t]['resids'] = self.slice_by_tracer(
                f_residuals, t)

        run.tracer_results['POCT'] = {}
        run.tracer_results['POCT']['resids'] = f_residuals[
            -(len(self.GRID) - 1):]

    def calculate_inventories(self, run):
        """Calculate inventories of the model tracers in each grid zone."""
        inventory_sym = {}

        for zone in self.zones:
            z = zone.label
            dz = zone.integration_intervals
            run.inventories[z] = {}
            run.integrated_resids[z] = {}
            inventory_sym[z] = {}
            for t in run.tracer_results:
                inventory = 0
                int_resids = 0
                for i, di in enumerate(zone.indices):
                    tracer_sym = sym.symbols(f'{t}_{di}')
                    inventory += tracer_sym*dz[i]
                    int_resids += (run.tracer_results[t]['resids'][di]*dz[i])
                run.inventories[z][t] = self.eval_symbolic_func(run, inventory)
                run.integrated_resids[z][t] = int_resids
                inventory_sym[z][t] = inventory

        return inventory_sym

    def calculate_fluxes(self, run):
        """Calculate profiles of all model fluxes."""
        MLD = self.MIXED_LAYER_DEPTH
        fluxes_sym = {}

        for flux in self.fluxes:
            f = flux.name
            run.flux_profiles[f] = {'est': [], 'err': []}
            if flux.wrt:
                fluxes_sym[f] = []
            if 'div' in f:
                for i in range(self.GRID):
                    z = self.which_zone(i)
                    pwi = f'{flux.param}_{z}'
                    twi = f'{flux.tracer}_{i}'
                    w, Pi = sym.symbols(f'{pwi} {twi}')
                    if i == 0:
                        y = w*Pi/MLD
                    elif i in (1, 2):
                        twip1 = f'{flux.tracer}_{i+1}'
                        twim1 = f'{flux.tracer}_{i-1}'
                        Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                        y = w*(Pip1 - Pim1)/(2*self.GRID_STEP)
                    else:
                        twim1 = f'{flux.tracer}_{i-1}'
                        twim2 = f'{flux.tracer}_{i-2}'
                        Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                        y = w*(3*Pi - 4*Pim1 + Pim2)/(2*self.GRID_STEP)
                    est, err = self.eval_symbolic_func(run, y)
                    run.flux_profiles[f]['est'].append(est)
                    run.flux_profiles[f]['err'].append(err)
                    if flux.wrt:
                        fluxes_sym[f].append(y)
            else:
                for i in range(self.GRID):
                    if f == 'production':
                        p30, lp = sym.symbols('P30 Lp')
                        y = p30*sym.exp(-(self.GRID[i] - MLD)/lp)
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
                    est, err = self.eval_symbolic_func(run, y)
                    run.flux_profiles[f]['est'].append(est)
                    run.flux_profiles[f]['err'].append(err)

        return fluxes_sym

    def integrate_fluxes(self, fluxes_sym, run):
        """Integrate fluxes within each model grid zone."""
        fluxes = fluxes_sym.keys()
        flux_integrals_sym = {}

        for zone in self.zones:
            z = zone.label
            dz = zone.integration_intervals
            flux_integrals_sym[z] = {}
            run.flux_integrals[z] = {}
            for f in fluxes:
                zone_expressions = [fluxes_sym[f][i] for i in zone.indices]
                to_integrate = 0
                for i, ex in enumerate(zone_expressions):
                    to_integrate += ex*dz[i]
                flux_integrals_sym[z][f] = to_integrate
                run.flux_integrals[z][f] = self.eval_symbolic_func(
                    run, to_integrate)

        return fluxes, flux_integrals_sym

    def calculate_timescales(self, inventory_sym, fluxes, flux_int_sym, run):
        """Calculate turnover timescales associated with each model flux."""
        for zone in self.zones:
            z = zone.label
            run.timescales[z] = {}
            for tracer in inventory_sym[z]:
                run.timescales[z][tracer] = {}
                for flux in fluxes:
                    if tracer in eval(f'self.{flux}.wrt'):
                        run.timescales[z][tracer][flux] = (
                            self.eval_symbolic_func(run,
                                                    inventory_sym[z][tracer]
                                                    / flux_int_sym[z][flux]))

    def pickle_model(self):
        """Pickle (save) the model for future plotting and analysis."""
        with open(self.pickled, 'wb') as file:
            pickle.dump(self, file)


class Tracer:
    """Container for metadata of model tracers."""

    def __init__(self, name, label, data):

        self.name = name
        self.label = label
        self.prior = data[['depth', name, f'{name}_se']].copy()
        self.prior.rename(columns={self.prior.columns[1]: 'conc',
                                   self.prior.columns[2]: 'conc_e'},
                          inplace=True)

    def __repr__(self):

        return f'Tracer({self.name})'


class Param:
    """Container for metadata of model parameters."""

    def __init__(self, prior, prior_error, name, label, depth_vary=True):

        self.prior = prior
        self.prior_e = prior_error
        self.name = name
        self.label = label
        self.dv = depth_vary

    def __repr__(self):

        return f'Param({self.name})'


class GridZone:
    """Container for metadata of model grid zones."""

    def __init__(self, model, index, label):

        self.model = model
        self.label = label
        self.indices = (index, index + 1)
        self.depths = model.GRID[index:index + 2]

        # self.set_integration_intervals()

    def __repr__(self):

        return f'GridZone({self.label})'

    def set_integration_intervals(self):
        """Define integration intervals.

        Required for calculation of inventories, integrated fluxes, and
        timescales.
        """
        intervals = np.ones(len(self.depths))*self.model.GRID_STEP

        if self.label == 'LEZ':
            intervals[0] = self.model.GRID_STEP/2
        else:
            intervals[-1] = self.model.GRID_STEP/2

        self.integration_intervals = intervals


class Flux:
    """Container for metadata of model fluxes."""

    def __init__(self, name, label, tracer, param, wrt=None):

        self.name = name
        self.label = label
        self.tracer = tracer
        self.param = param
        self.wrt = wrt

    def __repr__(self):

        return f'Flux({self.name})'


class PyriteModelRun():
    """Container for storing the results of a model run.

    Each model run has a unique proportionality constant, or gamma value.
    """

    def __init__(self, gamma):
        """Defines model data to be stored."""
        self.gamma = gamma
        self.cost_evolution = []
        self.convergence_evolution = []
        self.converged = False
        self.cvm = None
        self.tracer_results = {}
        self.param_results = {}
        self.Pt_results = {'est': [], 'err': []}
        self.x_resids = None
        self.f_resids = None
        self.inventories = {}
        self.integrated_resids = {}
        self.flux_profiles = {}
        self.flux_integrals = {}
        self.timescales = {}

    def __repr__(self):

        return f'PyriteModelRun(gamma={self.gamma})'


class PyriteTwinX(PyriteModel):
    """Twin experiment class for PyriteModel.

    Verifies that the model is able to produce accurate estiamtes of the state
    elements. Inherits from the PyriteModel class. load_data() is the only
    method that is practically overridden. Other methods that are inherited
    but currently unused are labeled as such in their docstrings.
    """

    def __init__(self, model_id, gammas,
                 pickled_model='out/POC_modelruns_dev.pkl',
                 pickle_into='out/POC_twinX_dev.pkl'):
        """Build a PyriteModel with gamma values to be used for the TwinX.

        gammas -- list of gamma values with which to perform twin experiments.
        self.pickled_model -- the pickled model from which to draw results to
        generate pseudodata.
        """
        self.pickled_model = pickled_model
        super().__init__(model_id, gammas, pickle_into)

    def __repr__(self):

        return 'PyriteTwinX object'

    def load_data(self):
        """Use results from self.pickled_model to generate pseudodata."""
        with open(self.pickled_model, 'rb') as file:
            model = pickle.load(file)

        self.get_target_values(model, self.gammas[0])
        x = self.generate_pseudodata(model)

        self.data = model.data.copy()
        for s in model.species:
            tracer_data = self.data[f'{s}_means'].copy()
            for t in (f'{s}S', f'{s}L'):
                re = tracer_data[f'{t}_se']/tracer_data[t]
                tracer_data[t] = model.slice_by_tracer(x, t)
                tracer_data[f'{t}_se'] = tracer_data[t]*re
                if t == 'POCS':
                    Ps_pseudo = tracer_data[t]
                if t == 'POCL':
                    Pl_pseudo = tracer_data[t]

            self.data[f'{s}_means'] = tracer_data.copy()

        self.Pt_constraint = Ps_pseudo + Pl_pseudo

    def get_target_values(self, model, gamma):
        """Get the target values with which to generate pseudodata.

        The target values are drawn from the model run whose proportionality
        constant (gamma) is specified by the function argument.
        """
        for run in model.model_runs:
            if run.gamma == gamma:
                reference_run = run
                break

        self.target_values = reference_run.param_results.copy()
        # d = {}

        # for param in model.params:
        #     d[param.name] = {}
        #     if param.dv:
        #         for z in model.zone_names:
        #             d[param.name][z] = {}
        #             d[param.name][z]['est'] = param.prior
        #     else:
        #         d[param.name]['est'] = param.prior

        # self.target_values = d.copy()

    def generate_pseudodata(self, model):
        """Generate pseudodata from the model equations."""

        def generate_linear_solution():
            """Obtain estimates of the tracers with a least-squares approach.

            Uses linear formulations of the model equations, which require
            a first-order aggregation term and the assumption of perfectly-
            known particle production.
            """
            A = np.zeros((model.nte, model.nte))
            b = np.zeros(model.nte)
            element_index = model.state_elements

            for i, element in enumerate(element_index[:model.nte]):

                species, z = element.split('_')
                for zo in model.zones:
                    if zo.label == z:
                        zone = zo
                        break
                zim1, zi = zone.depths
                h = zi - zim1

                iPsi = element_index.index(f'POCS_{z}')
                iPli = element_index.index(f'POCL_{z}')
                if 'Ti' in model.species:
                    iTsi = element_index.index(f'TiS_{z}')
                    iTli = element_index.index(f'TiL_{z}')

                if z != 'A':
                    pz = model.previous_zone(z)
                    iPsim1 = element_index.index(f'POCS_{pz}')
                    iPlim1 = element_index.index(f'POCL_{pz}')
                    if 'Ti' in model.species:
                        iTsim1 = element_index.index(f'TiS_{pz}')
                        iTlim1 = element_index.index(f'TiL_{pz}')

                B2 = 0.8/model.DAYS_PER_YEAR
                Bm2 = self.target_values['Bm2'][z]['est']
                Bm1s = self.target_values['Bm1s'][z]['est']
                Bm1l = self.target_values['Bm1l'][z]['est']
                P30 = self.target_values['P30']['est']
                Lp = self.target_values['Lp']['est']
                ws = self.target_values['ws'][z]['est']
                wl = self.target_values['wl'][z]['est']
                if zone.label != 'A':
                    wsm1 = self.target_values['ws'][pz]['est']
                    wlm1 = self.target_values['wl'][pz]['est']
                if 'Ti' in model.species:
                    phi = self.target_values['Phi']['est']

                if species == 'POCS':
                    if z == 'A':
                        A[i, iPsi] = ws + (Bm1s + B2)*h
                        A[i, iPli] = -Bm2*h
                        b[i] = P30*h
                    else:
                        A[i, iPsi] = ws + 0.5*(Bm1s + B2)*h
                        A[i, iPsim1] = -wsm1 + 0.5*(Bm1s + B2)*h
                        A[i, iPli] = -0.5*Bm2*h
                        A[i, iPlim1] = -0.5*Bm2*h
                        b[i] = Lp*P30*(np.exp(-zim1/Lp) - np.exp(-zi/Lp))
                elif species == 'POCL':
                    if z == 'A':
                        A[i, iPli] = wl + (Bm1l + Bm2)*h
                        A[i, iPsi] = -B2*h
                    else:
                        A[i, iPli] = wl + 0.5*(Bm1l + Bm2)*h
                        A[i, iPlim1] = -wlm1 + 0.5*(Bm1l + Bm2)*h
                        A[i, iPsi] = -0.5*B2*h
                        A[i, iPsim1] = -0.5*B2*h
                elif species == 'TiS':
                    if z == 'A':
                        b[i] = phi
                        A[i, iTsi] = ws + B2*h
                        A[i, iTli] = -Bm2*h
                    else:
                        A[i, iTsi] = ws + 0.5*B2*h
                        A[i, iTsim1] = -wsm1 + 0.5*B2*h
                        A[i, iTli] = -0.5*Bm2*h
                        A[i, iTlim1] = -0.5*Bm2*h
                else:
                    if z == 'A':
                        A[i, iTli] = wl + Bm2*h
                        A[i, iTsi] = -B2*h
                    else:
                        A[i, iTli] = wl + 0.5*Bm2*h
                        A[i, iTlim1] = -wlm1 + 0.5*Bm2*h
                        A[i, iTsi] = -0.5*B2*h
                        A[i, iTsim1] = -0.5*B2*h
            x = np.linalg.solve(A, b)

            # Ps = x[:7]
            # Pl = x[7:14]
            # Ts = x[14:21]
            # Tl = x[21:]

            # fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, tight_layout=True)
            # fig1.subplots_adjust(wspace=0.5)

            # ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
            # ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
            # ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)', fontsize=14)
            # ax1.set_ylabel('Depth (m)', fontsize=14)


            # ax1.scatter(Ps, model.GRID[1:],)
            # ax2.scatter(Pl, model.GRID[1:],)
            # ax3.scatter(Ps + Pl, model.GRID[1:],)

            # fig2, [ax4, ax5] = plt.subplots(1, 2, tight_layout=True)
            # fig2.subplots_adjust(wspace=0.5)

            # ax4.set_xlabel('$Ti_{S}$ (mmol m$^{-3}$)', fontsize=14)
            # ax5.set_xlabel('$Ti_{L}$ (mmol m$^{-3}$)', fontsize=14)
            # ax4.set_ylabel('Depth (m)', fontsize=14)


            # ax4.scatter(Ts, model.GRID[1:],)
            # ax5.scatter(Tl, model.GRID[1:],)

            # for ax in (ax1, ax2, ax3, ax4, ax5):
            #     ax.invert_yaxis()

            return x

        def generate_nonlinear_solution():
            """Obtain estimates of the tracers with an iterative approach.

            Takes the previously generated solution to the linear model
            equations and uses it as a prior estimate in an iterative approach
            to obtain estimates of the model tracers from the nonlinear
            model equations that are considered in the real data inversions.
            """
            max_iterations = 20
            max_change_limit = 0.01
            xk = generate_linear_solution()

            P30 = self.target_values['P30']['est']
            Lp = self.target_values['Lp']['est']
            b = np.zeros(model.nte)
            for i, z in enumerate(model.zones):
                zim1, zi = z.depths
                if z.label == 'A':
                    b[i] = -P30*zi
                else:
                    b[i] = -Lp*P30*(np.exp(-zim1/Lp) - np.exp(-zi/Lp))
            if 'Ti' in model.species:
                phi = self.target_values['Phi']['est']
                b[model.state_elements.index('TiS_A')] = -phi

            for count in range(max_iterations):
                f, F = model.evaluate_model_equations(
                    xk, return_F=True, params_known=self.target_values)
                xkp1 = np.linalg.solve(F, (F @ xk - f + b))
                change = np.abs((xkp1 - xk)/xk)
                if np.max(change) < max_change_limit:
                    break
                xk = xkp1

            Ps = xkp1[:7]
            Pl = xkp1[7:14]
            Ts = xkp1[14:21]
            Tl = xkp1[21:]

            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, tight_layout=True)
            fig.subplots_adjust(wspace=0.5)

            ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
            ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
            ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)', fontsize=14)
            ax1.set_ylabel('Depth (m)', fontsize=14)

            ax1.scatter(Ps, model.GRID[1:])
            ax2.scatter(Pl, model.GRID[1:])
            ax3.scatter(Ps + Pl, model.GRID[1:])

            for ax in (ax1, ax2, ax3):
                ax.invert_yaxis()

            fig.savefig('out/POC_fwd.png')
            plt.close()

            if 'Ti' in model.species:

                fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
                fig.subplots_adjust(wspace=0.5)

                ax1.set_xlabel('$Ti_{S}$ (mmol m$^{-3}$)', fontsize=14)
                ax2.set_xlabel('$Ti_{L}$ (mmol m$^{-3}$)', fontsize=14)
                ax1.set_ylabel('Depth (m)', fontsize=14)


                ax1.scatter(Ts, model.GRID[1:])
                ax2.scatter(Tl, model.GRID[1:])


                for ax in (ax1, ax2):
                    ax.invert_yaxis()

                fig.savefig('out/Ti_fwd.png')
                plt.close()

            return xkp1, b

        xkp1, b = generate_nonlinear_solution()

        self.pseudo_check = model.evaluate_model_equations(
            xkp1, params_known=self.target_values) - b

        return xkp1

    def define_fluxes(self):
        """Unused"""
        pass

    def calculate_inventories(self):
        """Unused"""
        pass

    def calculate_fluxes(self):
        """Unused"""
        pass

    def integrate_fluxes(self):
        """Unused"""
        pass

    def calculate_timescales(self):
        """Unused"""
        pass


class PlotterTwinX():
    """Generates all twin experiment plots."""

    def __init__(self, pickled_model):

        with open(pickled_model, 'rb') as file:
            self.model = pickle.load(file)

        if str(self.model) == 'PyriteTwinX object':
            self.is_twinX = True
        else:
            self.is_twinX = False

        self.define_colors()

        for run in self.model.model_runs:
            self.cost_and_convergence(run)
            self.params(run)
            self.poc_profiles(run)
            self.residual_pdfs(run)
            if 'Ti' in self.model.species:
                self.ti_profiles(run)

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

    def cost_and_convergence(self, run):

        k = len(run.cost_evolution)

        fig, ax = plt.subplots(1, tight_layout=True)
        ax.plot(np.arange(1, k+1), run.convergence_evolution,
                marker='o', ms=3, c=self.BLUE)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration, $k$', fontsize=16)
        ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',
                      fontsize=16)

        filename = f'out/conv_gam{str(run.gamma).replace(".","")}'
        if self.is_twinX:
            filename += '_TE'
        fig.savefig(f'{filename}.png')
        plt.close()

        fig, ax = plt.subplots(1, tight_layout=True)
        ax.plot(np.arange(1, k+1), run.cost_evolution, marker='o', ms=3,
                c=self.BLUE)
        ax.set_xlabel('Iteration, $k$', fontsize=16)
        ax.set_ylabel('Cost, $J$', fontsize=16)
        ax.set_yscale('log')

        filename = f'out/cost_gam{str(run.gamma).replace(".","")}'
        if self.is_twinX:
            filename += '_TE'
        fig.savefig(f'{filename}.png')
        plt.close()

    def params(self, run):

        tar = {True: {'LEZ': 2, 'UMZ': 4}, False: 3}
        pri = {True: 2, False: 1}
        est = {True: {True: {'LEZ': 3, 'UMZ': 5}, False: 4},
                False: {True: {'LEZ': 2, 'UMZ': 3}, False: 3}}
        maxtick = {True: 7, False: 5}

        params_out = {'ws':{'LEZ':(1.243,0.384), 'UMZ':(3.660,3.308)},
              'wl':{'LEZ':(19.076,8.833), 'UMZ':(25.320,18.386)},
              'B2p':{'LEZ':(0.021,0.009), 'UMZ':(0.024,0.022)},
              'Bm2':{'LEZ':(0.599,0.329), 'UMZ':(0.127,0.104)},
              'Bm1s':{'LEZ':(0.064,0.008), 'UMZ':(0.029,0.013)},
              'Bm1l':{'LEZ':(0.186,0.183), 'UMZ':(0.105,0.094)},
              'P30':(0.207,0.015), 'Lp':(27.007,2.640)}

        for i, param in enumerate(self.model.params):
            p = param.name
            fig, ax = plt.subplots(tight_layout=True)
            if param.dv:
                ax.set_xlabel(eval(f'self.model.{p}.label'), fontsize=14)
                ax.set_ylabel('Depth (m)', fontsize=14)
                ax.invert_yaxis()
                ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)
                ax.tick_params(axis='both', which='major', labelsize=12)
                # ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                #           loc='lower right')
                ax.fill_betweenx(
                    [0, 112.5],
                    params_out[p]['LEZ'][0] + params_out[p]['LEZ'][1],
                    params_out[p]['LEZ'][0] - params_out[p]['LEZ'][1],
                    color=self.GREEN, alpha=0.25,
                    zorder=2)
                ax.fill_betweenx(
                    [112.5, 500],
                    params_out[p]['UMZ'][0] + params_out[p]['LEZ'][1],
                    params_out[p]['UMZ'][0] - params_out[p]['LEZ'][1],
                    color=self.GREEN, alpha=0.25,
                    zorder=2)
                ax.axvline(param.prior - param.prior_e, c=self.BLUE, lw=2)
                ax.axvline(param.prior + param.prior_e, c=self.BLUE, lw=2)
                for i, z in enumerate(self.model.zone_names):
                    zone = self.model.zones[i]
                    ax.axhline(zone.depths[1], ls=':', c=self.BLACK)
                    if 'w' in p:
                        depth = zone.depths[1]
                    else:
                        depth = np.mean(zone.depths)
                    ax.errorbar(
                        run.param_results[p][z]['est'], depth,
                        fmt='o', xerr=run.param_results[p][z]['err'],
                        ecolor=self.ORANGE, elinewidth=1, c=self.ORANGE, ms=8,
                        capsize=6, fillstyle='none', zorder=3,
                        markeredgewidth=1)
            else:
                ax.set_xlabel(eval(f'self.model.{p}.label'), fontsize=14)
                ax.errorbar(
                    pri[self.is_twinX], eval(f'self.model.{p}.prior'),
                    yerr=eval(f'self.model.{p}.prior_e'), fmt='o', ms=9,
                    c=self.BLUE, elinewidth=1.5, ecolor=self.BLUE,
                    capsize=6, label='Prior', markeredgewidth=1.5)
                ax.errorbar(
                    est[self.is_twinX][param.dv],
                    run.param_results[p]['est'],
                    yerr=run.param_results[p]['err'], fmt='o',
                    c=self.ORANGE, ms=9, elinewidth=1.5,
                    ecolor=self.ORANGE, capsize=6, markeredgewidth=1.5)
                if self.is_twinX:
                    ax.scatter(
                        tar[param.dv], self.model.target_values[p]['est'],
                        marker='+', s=90, c=self.ORANGE)
                if p != 'Phi':
                    ax.fill_between(
                        [0, 4],
                        params_out[p][0] + params_out[p][1],
                        params_out[p][0] - params_out[p][1],
                        color=self.GREEN, alpha=0.25,
                        zorder=2)
                    ax.tick_params(bottom=False, labelbottom=False)
                    ax.set_xticks(np.arange(maxtick[self.is_twinX]))

            filename = f'out/{p}_gam{str(run.gamma).replace(".","")}'
            if self.is_twinX:
                filename += '_TE'
            fig.savefig(f'{filename}.png')
            plt.close()

    def poc_profiles(self, run):

        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
        ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        if run.gamma == 1:
            invname = 'I1'
        elif run.gamma == 0.02:
            invname = 'I2'
        else:
            invname = run.gamma

        art = {True: {'ax1_ticks': [0, 1, 2],
                      'ax2_ticks': [0, 0.05, 0.1],
                      'ax2_labels': ['0', '0.05', '0.1'],
                      'data_label': 'Data', 'inv_label': 'TE',
                      'cp_label': 'Data'},
               False: {'ax1_ticks': [0, 1, 2, 3],
                       'ax2_ticks': [0, 0.05, 0.1, 0.15],
                       'ax2_labels': ['0', '0.05', '0.1', '0.15'],
                       'data_label': 'LVISF', 'inv_label': invname,
                       'cp_label': 'from $c_p$'}}

        ax1.errorbar(
            self.model.POCS.prior['conc'], self.model.POCS.prior['depth'],
            fmt='^', xerr=self.model.POCS.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, fillstyle='full',
            label=art[self.is_twinX]['data_label'])
        ax1.errorbar(
            run.tracer_results['POCS']['est'], self.model.GRID[1:], fmt='o',
            xerr=run.tracer_results['POCS']['err'], ecolor=self.ORANGE,
            elinewidth=1, c=self.ORANGE, ms=8, capsize=5,
            label=art[self.is_twinX]['inv_label'], fillstyle='none',
            zorder=3, markeredgewidth=1)

        ax2.errorbar(
            self.model.POCL.prior['conc'], self.model.POCL.prior['depth'],
            fmt='^', xerr=self.model.POCL.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, fillstyle='full',
            label=art[self.is_twinX]['data_label'])
        ax2.errorbar(
            run.tracer_results['POCL']['est'], self.model.GRID[1:], fmt='o',
            xerr=run.tracer_results['POCL']['err'], ecolor=self.ORANGE,
            elinewidth=1, c=self.ORANGE, ms=8, capsize=5,
            label=art[self.is_twinX]['inv_label'], fillstyle='none',
            zorder=3, markeredgewidth=1)

        ax3.errorbar(
             self.model.Pt_constraint, self.model.GRID[1:], fmt='^',
             xerr=np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid),
             ecolor=self.BLUE, elinewidth=1, c=self.BLUE, ms=10, capsize=5,
             fillstyle='full', label=art[self.is_twinX]['cp_label'])
        ax3.errorbar(
            run.Pt_results['est'], self.model.GRID[1:], fmt='o',
            xerr=run.Pt_results['err'], ecolor=self.ORANGE,
            elinewidth=1, c=self.ORANGE, ms=8, capsize=5,
            label=art[self.is_twinX]['inv_label'], fillstyle='none',
            zorder=3, markeredgewidth=1)

        ax1.set_xticks(art[self.is_twinX]['ax1_ticks'])
        ax2.set_xticks(art[self.is_twinX]['ax2_ticks'])
        ax2.set_xticklabels(art[self.is_twinX]['ax2_labels'])
        ax3.set_xticks([0, 1, 2])

        for ax in (ax1, ax2, ax3):
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)
            ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                      loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax in (ax2, ax3):
                ax.tick_params(labelleft=False)

        filename = f'out/POCprofs_gam{str(run.gamma).replace(".","")}'
        if self.is_twinX:
            filename += '_TE'
        fig.savefig(f'{filename}.png')
        plt.close()

    def ti_profiles(self, run):

        fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$Ti_{S}$ (mmol m$^{-3}$)', fontsize=14)
        ax2.set_xlabel('$Ti_{L}$ (mmol m$^{-3}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        if run.gamma == 1:
            invname = 'I1'
        elif run.gamma == 0.02:
            invname = 'I2'
        else:
            invname = run.gamma

        ax1.errorbar(
            self.model.TiS.prior['conc'], self.model.TiS.prior['depth'],
            fmt='^', xerr=self.model.TiS.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, fillstyle='full',
            label='Data')
        ax1.errorbar(
            run.tracer_results['TiS']['est'], self.model.GRID[1:], fmt='o',
            xerr=run.tracer_results['TiS']['err'], ecolor=self.ORANGE,
            elinewidth=1, c=self.ORANGE, ms=8, capsize=5,
            label=invname, fillstyle='none', zorder=3, markeredgewidth=1)

        ax2.errorbar(
            self.model.TiL.prior['conc'], self.model.TiL.prior['depth'],
            fmt='^', xerr=self.model.TiL.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, fillstyle='full',
            label='Data')
        ax2.errorbar(
            run.tracer_results['TiL']['est'], self.model.GRID[1:], fmt='o',
            xerr=run.tracer_results['TiL']['err'], ecolor=self.ORANGE,
            elinewidth=1, c=self.ORANGE, ms=8, capsize=5,
            label=invname, fillstyle='none', zorder=3, markeredgewidth=1)

        ax2.tick_params(labelleft=False)

        for ax in (ax1, ax2):
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH+30)
            ax.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                      loc='lower right')
            ax.tick_params(axis='both', which='major', labelsize=12)

        filename = f'out/Tiprofs_gam{str(run.gamma).replace(".","")}'
        if self.is_twinX:
            filename += '_TE'
        fig.savefig(f'{filename}.png')
        plt.close()

    def residual_pdfs(self, run):

        fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)
        ax1.set_ylabel('Probability Density', fontsize=16)
        ax1.set_xlabel(r'$\frac{\^x_{i}-x_{o,i}}{\sigma_{o,i}}$', fontsize=24)
        ax1.hist(run.x_resids, density=True, bins=20, color=self.BLUE)
        ax2.hist(run.f_resids, density=True, bins=20, color=self.BLUE)
        ax2.set_xlabel(r'$\frac{f(\^x)_{i}}{\sigma_{f(\^x)_{i}}}$',
                       fontsize=24)
        for ax in (ax1, ax2):
            ax1.set_xlim([-1, 1])

        filename = f'out/pdfs_gam{str(run.gamma).replace(".","")}'
        if self.is_twinX:
            filename += '_TE'
        fig.savefig(f'{filename}.png')
        plt.close()


class PlotterModelRuns(PlotterTwinX):
    """Generates all model run result plots.

    Inherits from the PlotterTwinX class. No methods are overridden or
    extended, new methods are simply added. Writes out some numerical results
    for each model run to a single text file (pyrite_out.txt).
    """

    def __init__(self, pickled_model):
        super().__init__(pickled_model)

        self.cp_Pt_regression()
        self.poc_data()
        if 'Ti' in self.model.species:
            self.ti_data()

        for run in self.model.model_runs:
            self.residual_profiles(run)
            # self.sinking_fluxes(run)
            # self.volumetric_fluxes(run)
            # if run.gamma == 0.02:
            #     self.param_comparison(run)

        # self.integrated_residuals()
        # self.param_sensitivity()
        # self.param_relative_errors()

        # self.write_output()

    def cp_Pt_regression(self):

        cp = self.model.poc_cp_df['cp']
        Pt = self.model.poc_cp_df['POCT']
        depths = self.model.poc_cp_df['depth']
        linear_regression = self.model.cp_Pt_regression_linear
        nonlinear_regression = self.model.cp_Pt_regression_nonlinear
        logarithmic = {linear_regression: False, nonlinear_regression: True}

        colormap = plt.cm.viridis_r
        norm = mplc.Normalize(depths.min(), depths.max())

        for fit in (nonlinear_regression, linear_regression):
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(bottom=0.2, left=0.2)
            cbar_ax = colorbar.make_axes(ax)[0]
            cbar = colorbar.ColorbarBase(cbar_ax, norm=norm, cmap=colormap)
            cbar.set_label('Depth (m)\n', rotation=270, labelpad=20,
                           fontsize=14)
            ax.scatter(cp, Pt, norm=norm, edgecolors=self.BLACK, c=depths,
                       s=40, marker='o', cmap=colormap, label='_none')
            ax.set_ylabel('$P_T$ (mmol m$^{-3}$)', fontsize=14)
            ax.set_xlabel('$c_p$ (m$^{-1}$)', fontsize=14)
            x_fit = np.linspace(0.01, 0.14, 100000)
            if logarithmic[fit]:
                coefs_log = fit.params
                y_fit_log = [
                    coefs_log[0] + coefs_log[1]*np.log(x) for x in x_fit]
                ax.plot(x_fit, y_fit_log, '--', c=self.BLACK, lw=1,
                        label='non-linear')
                ax.set_yscale('log')
                ax.set_xscale('log')
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
            fig.savefig(f'out/cpptfit_log{logarithmic[fit]}.png')
            plt.close()

    def poc_data(self):

        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
        ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        ax1.errorbar(
            self.model.POCS.prior['conc'], self.model.POCS.prior['depth'],
            fmt='^', xerr=self.model.POCS.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, label='LVISF',
            fillstyle='full')

        ax2.errorbar(
            self.model.POCL.prior['conc'], self.model.POCL.prior['depth'],
            fmt='^', xerr=self.model.POCL.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, label='LVISF',
            fillstyle='full')

        ax3.scatter(
            self.model.Pt_mean_nonlinear, self.model.GRID[1:], marker='o',
            c=self.BLUE, edgecolors=self.WHITE, s=7, label='from $c_p$',
            zorder=3, lw=0.7)
        ax3.fill_betweenx(
            self.model.GRID[1:],
            (self.model.Pt_mean_nonlinear
             - np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            (self.model.Pt_mean_nonlinear
             + np.sqrt(self.model.cp_Pt_regression_nonlinear.mse_resid)),
            color=self.BLUE, alpha=0.25, zorder=2)
        ax3.errorbar(
            self.model.POCS.prior['conc'] + self.model.POCL.prior['conc'],
            self.model.GRID[1:], fmt='^', ms=10,
            c=self.BLUE, xerr=np.sqrt(self.model.POCS.prior['conc_e']**2
                                      + self.model.POCL.prior['conc_e']**2),
            zorder=1, label='LVISF', capsize=5, fillstyle='full',
            elinewidth=1)

        ax1.set_xticks([0, 1, 2, 3])
        ax1.set_xlim([-0.2, 3.4])
        ax2.set_xticks([0, 0.05, 0.1, 0.15])
        ax2.set_xticklabels(['0', '0.05', '0.1', '0.15'])
        ax3.set_xticks([0, 1, 2, 3])
        ax3.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                   loc='lower right')
        ax2.tick_params(labelleft=False)

        for ax in (ax1, ax2, ax3):
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH + 30)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax in (ax2, ax3):
                ax.tick_params(labelleft=False)
            if ax in (ax1, ax3):
                ax.set_xlim([-0.2, 3.4])

        fig.savefig('out/poc_data.png')
        plt.close()

    def ti_data(self):

        fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$Ti_{S}$ (nmol m$^{-3}$)', fontsize=14)
        ax2.set_xlabel('$Ti_{L}$ (nmol m$^{-3}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        ax1.errorbar(
            self.model.TiS.prior['conc'], self.model.TiS.prior['depth'],
            fmt='^', xerr=self.model.TiS.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, label='LVISF',
            fillstyle='full')

        ax2.errorbar(
            self.model.TiL.prior['conc'], self.model.TiL.prior['depth'],
            fmt='^', xerr=self.model.TiL.prior['conc_e'], ecolor=self.BLUE,
            elinewidth=1, c=self.BLUE, ms=10, capsize=5, label='LVISF',
            fillstyle='full')

        # ax1.set_xticks([0, 1, 2, 3])
        # ax1.set_xlim([-0.2, 3.4])
        # ax2.set_xticks([0, 0.05, 0.1, 0.15])
        # ax2.set_xticklabels(['0', '0.05', '0.1', '0.15'])
        ax2.legend(fontsize=12, borderpad=0.2, handletextpad=0.4,
                   loc='lower right')
        ax2.tick_params(labelleft=False)

        for ax in (ax1, ax2):
            ax.invert_yaxis()
            ax.set_ylim(top=0, bottom=self.model.MAX_DEPTH + 30)
            ax.tick_params(axis='both', which='major', labelsize=12)

        fig.savefig('out/ti_data.png')
        plt.close()

    def residual_profiles(self, run):

        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, tight_layout=True)
        fig.subplots_adjust(wspace=0.5)

        ax1.set_xlabel('$P_{S}$ (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
        ax2.set_xlabel('$P_{L}$ (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
        ax3.set_xlabel('$P_{T}$ (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
        ax1.set_ylabel('Depth (m)', fontsize=14)

        ax1.scatter(run.tracer_results['POCS']['resids'], self.model.GRID[1:])
        ax2.scatter(run.tracer_results['POCL']['resids'], self.model.GRID[1:])
        ax3.scatter(run.tracer_results['POCT']['resids'], self.model.GRID[1:])

        for ax in (ax1, ax2, ax3):
            ax.invert_yaxis()

        fig.savefig(f'out/POC_resids_gam{str(run.gamma).replace(".","")}.png')
        plt.close()

        if 'Ti' in self.model.species:
            fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
            fig.subplots_adjust(wspace=0.5)

            ax1.set_xlabel('$Ti_{S}$ (nmol m$^{-2}$ d$^{-1}$)', fontsize=14)
            ax2.set_xlabel('$Ti_{L}$ (nmol m$^{-2}$ d$^{-1}$)', fontsize=14)
            ax1.set_ylabel('Depth (m)', fontsize=14)


            ax1.scatter(
                run.tracer_results['TiS']['resids'], self.model.GRID[1:])
            ax2.scatter(
                run.tracer_results['TiL']['resids'], self.model.GRID[1:])

            for ax in (ax1, ax2):
                ax.invert_yaxis()

            fig.savefig(
                f'out/Ti_resids_gam{str(run.gamma).replace(".","")}.png')
            plt.close()

    def sinking_fluxes(self, run):

        th_fluxes = pd.read_excel(
            'pyrite_data.xlsx', sheet_name='POC_fluxes_thorium')
        th_depths = th_fluxes['depth']
        th_flux = th_fluxes['flux']
        th_flux_u = th_fluxes['flux_u']
        st_fluxes = pd.read_excel(
            'pyrite_data.xlsx', sheet_name='POC_fluxes_traps')
        st_depths = st_fluxes['depth']
        st_flux = st_fluxes['flux']
        st_flux_u = st_fluxes['flux_u']

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_ylabel('Depth (m)', fontsize=14)
        fig.text(
            0.5, 0.03, 'POC Flux (mmol m$^{-2}$ d$^{-1}$)',
            fontsize=14, ha='center', va='center')
        for ax in (ax1, ax2):
            ax.invert_yaxis()
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', lw=0.5)
            ax.set_ylim(
                top=0, bottom=self.model.MAX_DEPTH+self.model.GRID_STEP*2)

        eb1 = ax1.errorbar(
            run.flux_profiles['sink_S']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_S']['err'], ecolor=self.BLUE,
            elinewidth=0.5, c=self.BLUE, ms=3, capsize=2,
            label=self.model.sink_S.label, fillstyle='none',
            markeredgewidth=0.5)
        eb1[-1][0].set_linestyle('--')
        ax1.axhline(self.model.BOUNDARY, c='k', ls='--', lw=0.5)
        eb2 = ax1.errorbar(
            run.flux_profiles['sink_L']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_L']['err'], ecolor=self.ORANGE,
            elinewidth=0.5, c=self.ORANGE, ms=3, capsize=2,
            label=self.model.sink_L.label, fillstyle='none',
            markeredgewidth=0.5)
        eb2[-1][0].set_linestyle(':')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.annotate(
            'A', xy=(0.91, 0.94), xycoords='axes fraction', fontsize=16)

        ax2.tick_params(labelleft=False)
        eb3 = ax2.errorbar(
            run.flux_profiles['sink_T']['est'], self.model.GRID, fmt='o',
            xerr=run.flux_profiles['sink_T']['err'], ecolor=self.SKY,
            elinewidth=0.5, c=self.SKY, ms=3, capsize=2,
            label=self.model.sink_T.label, fillstyle='none',
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

        fig.savefig(f'out/sinkfluxes_gam{str(run.gamma).replace(".","")}.png')
        plt.close()

    def volumetric_fluxes(self, run):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.1)
        c1 = self.BLUE
        c2 = self.ORANGE
        axs = (ax1, ax2, ax3, ax4)
        panels = ('A', 'B', 'C', 'D')
        fig.text(0.5, 0.05, 'Volumetric POC Flux (mmol m$^{-3}$ d$^{-1}$)',
                 fontsize=14, ha='center', va='center')
        fig.text(0.05, 0.5, 'Depth (m)', fontsize=14, ha='center',
                 va='center', rotation='vertical')

        pairs = (('sinkdiv_S', 'sinkdiv_L'), ('remin_S', 'aggregation'),
                 ('remin_L', 'disaggregation'), ('production',))

        for i, pr in enumerate(pairs):
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
                df = self.model.data['NPP']
                H = self.model.MIXED_LAYER_DEPTH
                npp = df.loc[df['target_depth'] >= H]['NPP']
                depth = df.loc[df['target_depth'] >= H]['target_depth']
                ax.scatter(npp/self.model.MOLAR_MASS_C, depth, c=c2,
                           alpha=0.5, label='NPP', s=10)

            ax.legend(loc='lower right', fontsize=12)
            ax.annotate(panels[i], xy=(0.9, 0.8), xycoords='axes fraction',
                        fontsize=12)
            ax.axhline(self.model.BOUNDARY, c=self.BLACK, ls='--', lw=0.5)
            ax.set_yticks([0, 100, 200, 300, 400, 500])
            if i % 2:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()
            ax.set_ylim(
                top=0, bottom=self.model.MAX_DEPTH+self.model.GRID_STEP)
        fig.savefig(
            f'out/fluxes_volumetric_gam{str(run.gamma).replace(".","")}.png')
        plt.close()

    def write_output(self):

        file = 'out/pyrite_out.txt'
        with open(file, 'w') as f:
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
                    for t in run.inventories[z.label]:
                        est, err = run.inventories[z.label][t]
                        print(f'{t}: {est:.0f} ± {err:.0f}', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Integrated Fluxes', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for z in self.model.zones:
                    print(f'--------{z.label}--------', file=f)
                    for flux in run.flux_integrals[z.label]:
                        est, err = run.flux_integrals[z.label][flux]
                        print(f'{flux}: {est:.2f} ± {err:.2f}', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                print('Timescales', file=f)
                print('+++++++++++++++++++++++++++', file=f)
                for z in self.model.zones:
                    print(f'--------{z.label}--------', file=f)
                    for t in run.integrated_resids[z.label]:
                        print(f'***{t}***', file=f)
                        for flux in run.timescales[z.label][t]:
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

        data = {'EXP': {'B2': {'EZ': (B2_EX_LEZ,), 'MZ': (B2_EX_UMZ,)},
                        'Bm2': {
                            'EZ': (run.param_results['Bm2']['LEZ']['est'],
                                   run.param_results['Bm2']['LEZ']['err']),
                            'MZ': (run.param_results['Bm2']['UMZ']['est'],
                                   run.param_results['Bm2']['UMZ']['err'])},
                        'Bm1s': {
                            'EZ': (run.param_results['Bm1s']['LEZ']['est'],
                                   run.param_results['Bm1s']['LEZ']['err']),
                            'MZ': (run.param_results['Bm1s']['UMZ']['est'],
                                   run.param_results['Bm1s']['UMZ']['err'])}},
                'MOSP': {'B2': {'BZ': (0.8/dpy, 0.9/dpy)},
                         'Bm2': {'BZ': (400/dpy, 10000/dpy)},
                         'Bm1s': {'BZ': (1.7/dpy, 0.9/dpy)}},
                'MNABE': {'B2': {'MZ': {'t1': (2/dpy, 0.2/dpy),
                                        't2': (12/dpy, 1/dpy),
                                        't3': (76/dpy, 9/dpy)}},
                          'Bm2': {'MZ': {'t1': (156/dpy, 17/dpy),
                                         't2': (321/dpy, 32/dpy),
                                         't3': (524/dpy, 74/dpy)}},
                          'Bm1s': {'MZ': {'t1': (13/dpy, 1/dpy),
                                          't2': (32/dpy, 2/dpy),
                                          't3': (596/dpy, 6/dpy)}}},
                'MNWA': {'B2': {'EZ': {'lo': (9/dpy, 24/dpy),
                                       'hi': (11/dpy, 30/dpy)},
                                'MZ': {'lo': (13/dpy, 50/dpy),
                                       'hi': (18/dpy, 89/dpy)}},
                         'Bm2': {'EZ': {'lo': (2280/dpy, 10000/dpy),
                                        'hi': (2690/dpy, 10000/dpy)},
                                 'MZ': {'lo': (870/dpy, 5000/dpy),
                                        'hi': (1880/dpy, 10000/dpy)}},
                         'Bm1s': {'EZ': {'lo': (70/dpy, 137/dpy),
                                         'hi': (798/dpy, 7940/dpy)},
                                  'MZ': {'lo': (113/dpy, 10000/dpy),
                                         'hi': (1766/dpy, 10000000/dpy)}}}}

        fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(
            2, 3, tight_layout=True)
        axs = [ax1, ax2, ax3, ax4, ax5]
        for ax in axs:
            ax.tick_params(bottom=False, labelbottom=False)
            ax.set_yscale('log')
            # next line from https://stackoverflow.com/questions/21920233/
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax1.set_ylabel('Estimate (d$^{-1}$)', fontsize=14)
        ax1.set_title('$\\beta_{-1,S}$', fontsize=14)
        ax2.set_title('$\\beta_{-2}$', fontsize=14)
        ax3.set_title('$\\beta_2$', fontsize=14)
        ax4.set_ylabel('Error (d$^{-1}$)', fontsize=14)
        ax6.axis('off')

        study_colors = {'EXP': self.GREEN, 'MOSP': self.BLUE,
                        'MNABE': self.ORANGE, 'MNWA': self.RADISH}
        zone_shapes = {'EZ': 's', 'MZ': '^', 'BZ': 'd'}
        axs_dict = {ax1: {'ylim': (0.001, 10), 'panel': 'A'},
                    ax2: {'ylim': (0.1, 10), 'panel': 'B'},
                    ax3: {'ylim': (0.001, 1), 'panel': 'C'},
                    ax4: {'ylim': (0.001, 100000), 'panel': 'D'},
                    ax5: {'ylim': (0.01, 100), 'panel': 'E'}}

        for (ax, p) in ((ax1, 'Bm1s'), (ax2, 'Bm2'), (ax3, 'B2')):
            ct = 0
            for s in data:
                c = study_colors[s]
                for z, vals in data[s][p].items():
                    m = zone_shapes[z]
                    if isinstance(vals, dict):
                        for k in vals:
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
                   markeredgecolor=self.BLACK, ms=9, lw=0.5),
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
        fig.savefig('out/compare_params.png')
        plt.close()

    def integrated_residuals(self):

        fig, ax = plt.subplots()
        ax.set_xticks([k for k in list(range(len(self.model.model_runs)))])
        ax.set_xticklabels(self.model.gammas)
        ax.set_yticks(list(range(-11, 2)))
        ax.grid(axis='y', zorder=1)
        plt.subplots_adjust(bottom=0.1)
        tracerdict = {'POCS': {'marker': 's', 'label': self.model.POCS.label},
                      'POCL': {'marker': '^', 'label': self.model.POCL.label}}
        zone_colors = {'LEZ': self.GREEN, 'UMZ': self.ORANGE}
        for t in tracerdict:
            m = tracerdict[t]['marker']
            lbl = tracerdict[t]['label']
            for z in zone_colors:
                j = 0
                c = zone_colors[z]
                for run in self.model.model_runs:
                    ax.scatter(
                        j, run.integrated_resids[z][t], marker=m, c=c, s=64,
                        label=f'{lbl}$^{{{z}}}$', zorder=2)
                    j += 1
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.set_ylabel('Integrated Residuals (mmol m$^{-2}$ d$^{-1}$)',
                      fontsize=14)
        ax.set_xlabel('$\gamma$', fontsize=14)
        fig.savefig('out/intresids.png')
        plt.close()

    def param_sensitivity(self):

        fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = (
            plt.subplots(3, 3, figsize=(8, 8)))
        fig.subplots_adjust(wspace=0.3, hspace=0.8)
        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        axs[-1].axis('off')
        for i, param in enumerate(self.model.params):
            p = param.name
            ax = axs[i]
            ax.set_title(param.label, fontsize=14)
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
                        range(len(self.model.model_runs)*2))])
                    ax.get_xaxis().set_major_formatter(
                        ticker.ScalarFormatter())
                    ax.set_xticklabels(self.model.gammas + self.model.gammas,
                                       rotation=60)
                if i == 5:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(),
                              loc='upper center', bbox_to_anchor=(0.5, -1),
                              ncol=1, fontsize=12, frameon=False)
            else:
                j = 0
                for run in self.model.model_runs:
                    ax.errorbar(j*3, run.param_results[p]['est'],
                                yerr=run.param_results[p]['err'], fmt='o',
                                c=self.RADISH, ms=8, elinewidth=1.5,
                                ecolor=self.RADISH, capsize=6,
                                markeredgewidth=1.5)
                    j += 1
                ax.set_xticks([k*3 for k in list(
                    range(len(self.model.model_runs)))])
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.set_xticklabels(self.model.gammas, rotation=60)
        fig.savefig('out/sensitivity_params.png')
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
                        / r.param_results[p][z]['est'] for r in mod.model_runs]
                    ax.plot(
                        mod.gammas, relativeerror, m, c=self.colors[i],
                        label=f'{param.label}', fillstyle='none', ls=ls)
            else:
                relativeerror = [
                    r.param_results[p]['err']
                    / r.param_results[p]['est'] for r in mod.model_runs]
                ax.plot(mod.gammas, relativeerror, 'x', c=self.colors[i],
                        label=f'{param.label}', ls='-.')
        ax.set_xscale('log')
        ax.set_xticks(mod.gammas)
        leg_elements = [
            Line2D([0], [0], marker='o', ls='none', color=self.colors[i],
                   label=tset_list[i]) for i, _ in enumerate(tset_list)]
        ax.legend(
            handles=leg_elements, loc='lower center',
            bbox_to_anchor=(0.49, 1), ncol=4, fontsize=12, frameon=False)
        ax.set_xlabel('$\gamma$', fontsize=14)
        ax.set_ylabel('Relative Error', fontsize=14)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        fig.savefig('out/paramrelerror.png')
        plt.close()


if __name__ == '__main__':

    sys.setrecursionlimit(100000)
    start_time = time.time()
    args = (1, [0.01])
    model_1 = PyriteModel(*args)
    PlotterModelRuns('out/POC_modelruns_dev.pkl')
    twinX_1 = PyriteTwinX(*args)
    # PlotterTwinX('out/POC_twinX_dev.pkl')

    print(f'--- {(time.time() - start_time)/60} minutes ---')
