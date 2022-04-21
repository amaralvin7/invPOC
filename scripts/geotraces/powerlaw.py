#!/usr/bin/env python3
# plot log(POC) profiles (after data have been cleaned)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import statsmodels.formula.api as smf

import src.geotraces.data as data
from src.colors import *

import sys

def power_law(x, a, b):
    return a*np.power(x, b)

def get_b_values(df, tracer):
    
    regression = smf.ols(
        formula=f'np.log10({tracer}) ~ np.log10(depth)',
        data=df).fit()
    ols_a, ols_b = regression.params
    ols_a = 10**ols_a
    
    (cf_a, cf_b), _ = curve_fit(f=power_law, xdata=df['depth'], ydata=df[tracer])
    
    return ols_a, ols_b, cf_a, cf_b

poc_data = data.load_poc_data()
ppz_data = data.load_ppz_data()

for s in poc_data['station'].unique():

    station_poc_full = data.get_station_poc(poc_data, s, 1000)
    ppz = ppz_data[s]
    refdepth = min(station_poc_full['depth'], key=lambda x:abs(x - ppz))  #closest depth to ppz
    station_poc = station_poc_full[station_poc_full['depth'] >= refdepth]
 
    ps_b = get_b_values(station_poc, 'POCS')
    pl_b = get_b_values(station_poc, 'POCL')
    
    fig, [ax1, ax2] = plt.subplots(1, 2, tight_layout=True)
    fig.subplots_adjust(wspace=0.5)

    ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)', fontsize=14)
    ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)', fontsize=14)
    ax1.set_ylabel('Depth (m)', fontsize=14)

    ax1.scatter(station_poc_full['POCS'], station_poc_full['depth'], c=blue)
    ax2.scatter(station_poc_full['POCL'], station_poc_full['depth'], c=blue)
    
    ax1.plot(ps_b[0]*station_poc['depth']**ps_b[1], station_poc['depth'], c=black)
    ax1.plot(ps_b[2]*station_poc['depth']**ps_b[3], station_poc['depth'], c=orange)

    ax2.plot(pl_b[0]*station_poc['depth']**pl_b[1], station_poc['depth'], c=black, label='OLS')
    ax2.plot(pl_b[2]*station_poc['depth']**pl_b[3], station_poc['depth'], c=orange, label='CF')

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_ylim(top=0, bottom=1110)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.axhline(ppz, c=black, ls='--')
    ax2.tick_params(labelleft=False)
    ax2.legend(loc='lower right')
    
    ax1.set_title(f'OLS: {ps_b[1]:0.2f}, CF: {ps_b[3]:0.2f}')
    ax2.set_title(f'OLS: {pl_b[1]:0.2f}, CF: {pl_b[3]:0.2f}')

    plt.savefig(f'../../results/geotraces/powerfit_stn{int(s)}')
    plt.close()
    
