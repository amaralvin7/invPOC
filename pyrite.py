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
#from varname import nameof

class PyriteModel:

    def __init__(self, gammas=[0.02], pickle_into='out/pyrite_Amaral21a.pkl'):
        
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
        self.process_cp_data()
        
        self.LEZ = PyriteZone(self, op.lt, 'LEZ')  # euphotic zone
        self.UMZ = PyriteZone(self, op.gt, 'UMZ')  # upper mesopelagic zone
        self.zones = (self.LEZ, self.UMZ)

        self.pickle_model()

    def __repr__(self):

        return f'PyriteModel(gammas={self.gammas})'
                    
    def load_data(self):
        
        self.DATA = pd.read_excel('pyrite_data.xlsx',sheet_name=None)
        self.SAMPLE_DEPTHS = self.DATA['poc_means']['Depth']
        self.N_SAMPLE_DEPTHS = len(self.SAMPLE_DEPTHS)
    
    def define_tracers(self):
        
        self.Ps = PyriteTracer('POC', 'S', '$P_S$',
                               self.DATA['poc_means']['SSF_mean']/
                               self.MOLAR_MASS_C,
                               self.DATA['poc_means']['SSF_se']/
                               self.MOLAR_MASS_C)
        self.Pl = PyriteTracer('POC', 'L', '$P_L$',
                               self.DATA['poc_means']['LSF_mean']/
                               self.MOLAR_MASS_C,
                               self.DATA['poc_means']['LSF_se']/
                               self.MOLAR_MASS_C)

        self.tracers = (self.Ps, self.Pl)

    def define_params(self):
        
        P30_prior, P30_prior_e, Lp_prior, Lp_prior_e = self.process_npp_data()

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
        self.Pt_mean = self.cp_Pt_regression_nonlinear.get_prediction(
            exog=dict(cp=self.cp_mean)).predicted_mean
     
    def pickle_model(self):

        with open(self.pickled, 'wb') as file:
            pickle.dump(self,file)

class PyriteTracer:

    def __init__(self, species, size_fraction, label, data, data_error):

        self.species = species
        self.sf = size_fraction
        self.label = label
        self.data = data
        self.data_e = data_error

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

        Pt = self.model.Pt_mean[self.indices]
        n_lags = int(np.ceil(len(Pt)*fraction))
        self.grid_steps = np.arange(
            0, (n_lags+1)*self.model.GRID_STEP, self.model.GRID_STEP)
        self.autocorrelation = smt.acf(Pt, nlags=n_lags, fft=False)
        
        acf_regression = smf.ols(
            formula='np.log(ac) ~ gs',
            data = {'ac':self.autocorrelation, 'gs':self.grid_steps}).fit()
        b, m = acf_regression.params
        
        self.length_scale = -1/m
        self.length_scale_fit = b + m*self.grid_steps
        self.fit_rsquared = acf_regression.rsquared

class PyritePlotter:

    def __init__(self, pickled_model):

        with open(pickled_model.pickled, 'rb') as file:
            self.model = pickle.load(file)

        self.define_colors()
        self.plot_hydrography()
        self.plot_cp_Pt_regression()
        self.plot_zone_length_scales()
        #self.plot_poc_data()

    def define_colors(self):

        self.BLACK = '#000000'
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
        
    #def plot_poc_data():

if __name__ == '__main__':

    start_time = time.time()
    model = PyriteModel()
    plotter = PyritePlotter(model)

    print(f'--- {(time.time() - start_time)/60} minutes ---')

