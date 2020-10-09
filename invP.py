#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 09 2020

@author: vamaral

Gamma sensitivity (_gs) studies on invP
"""
import numpy as np
import scipy.linalg as splinalg
import scipy.stats as sstats
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import pandas as pd
import scipy.io as sio
import matplotlib as mpl
import statsmodels.formula.api as smf
import statsmodels.sandbox.stats.runs as smr
import statsmodels.tsa.stattools as smt
import sympy as sym
import os
import sys
import time
import pickle

start_time = time.time()
plt.close('all')

#need for when running on remote server
sys.setrecursionlimit(10000)

#colors
red, green, blue, purple, cyan, orange, teal, navy, olive = '#e6194B', '#3cb44b', '#4363d8', '#911eb4', '#42d4f4', '#f58231', '#469990', '#000075', '#808000'
#colors
black, orange, sky, green, yellow, blue, red, radish = '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'

#read in POC data
cwd = os.getcwd()
df = pd.read_csv(cwd+'/poc_means.csv')

#some parameters
dz = 5 #meters
zmax = 500   #max depth, m
h = 30 #MLD, m
zml = np.arange(h, zmax+dz, dz) #this one DOES include h meters (need for OI)
n = len(zml)

#some conversion factors
mm = 12 #molar mass of C (12 g/mol)
dpy = 365.24 #days per year

#assign df colums to variables
zs = df.Depth
Ps_mean = df.SSF_mean/mm
Ps_sd = df.SSF_sd/mm
Ps_se = df.SSF_se/mm
Pl_mean = df.LSF_mean/mm
Pl_sd = df.LSF_sd/mm
Pl_se = df.LSF_se/mm

gammas = [0.01, 0.05, 0.1, 0.5, 1] #multiplier for weighting model errors

bnd = 112.5 #boundary that separates EZ from UMZ
depthranges = ((h,bnd),(bnd,zmax)) #for integration
dr_str = tuple(map(lambda d: '_'.join((str(d[0]),str(d[1]))),depthranges))

#param info
layers = ['A','B']
params = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l', 'Gh', 'Lp']
params_dv = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l'] #depth-varying params
params_dc = ['Gh', 'Lp'] #depth-constant params
gkeys_pdi = ['xh', 'xhe'] #some keys for building dictionaries later
#make a dictionaries to store param info
pdi = {param:{} for param in params}

#typeset name
p_tset = {'ws':'$w_s$', 'wl':'$w_l$', 'B2p':'$\\beta^,_2$', 'Bm2':'$\\beta_{-2}$', 
                'Bm1s':'$\\beta_{-1,s}$', 'Bm1l':'$\\beta_{-1,l}$', 'Gh':'$\overline{\.P_s}$', 
                'Lp':'$L_{p}$'}

#priors
p_o = {'ws':2, #m/d
        'wl':20, #m/d, reported from Murnane 1990.
        'B2p':0.5*mm/dpy, #m3/(mmol*yr), arbritrary
        'Bm2':400/dpy, #from Murnane 1994, converted to d
        'Bm1s':36/dpy, #from Clegg surface value, converted to d
        'Bm1l':0.15, #based on preliminary RESPIRE data from A Santoro
        'Gh':0.28, #prior set to typical NPP shared data value. mg/m3/d converted to mmol
        'Lp':28} #from NPP data, m
#prior errors
p_oe = {'ws':2, 
        'wl':15, 
        'B2p':0.5*mm/dpy, 
        'Bm2':10000/dpy, 
        'Bm1s':36/dpy, 
        'Bm1l':0.15, 
        'Gh':0.12, 
        'Lp':28*0.5}

#update entries in pdi
for p in pdi.keys():
    pdi[p]['tset'] = p_tset[p]
    pdi[p]['o'] = p_o[p]
    pdi[p]['oe'] = p_oe[p]
    pdi[p]['dv'] = 1 if p in params_dv else 0
    pdi[p]['gammas'] = {g:({k:{l:{} for l in layers} for k in gkeys_pdi} 
                            if pdi[p]['dv'] else {k:{} for k in gkeys_pdi})
                        for g in gammas}

#build tracer dictionary
tracers = ['Ps','Pl']
gkeys_td = ['xh', 'xhe', 'inv', 'ires', 'n', 'nm', 'nma'] #some keys
tkeys_td = ['x', 'xerr', 'y', 'si', 'gammas']
invkeys = ['inv','ires']
td = {t:{k:({g:{gk:({dr:{} for dr in dr_str} if gk in invkeys else {}) 
                for gk in gkeys_td} for g in gammas} if k == 'gammas' 
            else {}) for k in tkeys_td} for t in tracers}
    
#build flux dictionary
flxs = ['ws_Ps','wl_Pl','ws_Psdz','wl_Pldz','Bm1s_Ps','Bm1l_Pl','B2p_Ps2','Bm2_Pl','Psdot']
flxnames = {'ws_Ps':'sinkS', 'wl_Pl':'sinkL', 'ws_Psdz':'sinkS_div', 'wl_Pldz':'sinkL_div',
            'Bm1s_Ps':'SRemin', 'Bm1l_Pl':'LRemin', 'B2p_Ps2':'Agg', 'Bm2_Pl':'Disagg', 'Psdot':'Prod'}
flxpairs = [('ws_Ps','wl_Pl'),('ws_Psdz','wl_Pldz'),('Bm1s_Ps','B2p_Ps2'),('Bm1l_Pl','Bm2_Pl'),('Psdot',)]
#fluxes that we want to integrate
iflxs = ['ws_Psdz','wl_Pldz','Bm1s_Ps','Bm1l_Pl','B2p_Ps2','Bm2_Pl','Psdot']
flxd = {f:{} for f in flxs}
gkeys_flxd = ['xh','xhe','iflx','tau']
intkeys = ['iflx','tau']
for f in flxs:
    flxd[f]['name'] = flxnames[f]
    #assign order
    if f == 'Psdot': flxd[f]['order'] = 0   
    elif f == 'B2p_Ps2': flxd[f]['order'] = 2
    else: flxd[f]['order'] = 1
    if f in iflxs:
        flxd[f]['gammas'] = {g:{k:({dr:{} for dr in dr_str} if k in intkeys 
                                   else {}) for k in gkeys_flxd} for g in gammas}
    else: flxd[f]['gammas'] = {g:{k:{} for k in gkeys_pdi} for g in gammas}

#some parameters for plotting later
ms, lw, elw, cs = 3, 1, 0.5, 2

"""
#SOME FUNCTIONS
"""
#check which columns in a matrix have all zeros
def zcols(A):
    return np.where(~A.any(axis=0))[0]

#add variables to the variable index
#takes as input the idx to be updated and an array of strings of species to be added
def addtovaridx(varidx,species):
    for spec in species:
        tvidx = [''.join([spec,i]) for i in svis] #create a temporary index for that species
        varidx = varidx + tvidx
    return varidx

#find index of required variable and corresponding value in the state vector
#takes as input the species, depth, and variable index. Only returns the index and value in the state vector
def fidx2(s, d, idx, sv):
    idx = [x for x, st in enumerate(idx) if st == f'{s}_{d}']
    val = sv[idx]
    return [idx,val]

#given sampling depths, return indices at which those depths correspond to
#INCLUDES MIXED LAYER, hence zml
def difind(zs):
    sorter = np.argsort(zml)
    zsi = sorter[np.searchsorted(zml, zs, sorter=sorter)]
    return zsi

#some matrices that will be copied
def Rmatx(sampledepths,griddepths,lengthscale):
    m = len(sampledepths)
    n = len(griddepths)
    #print(m,n)
    L = lengthscale
    Rxxmm = np.zeros((m,m))
    for i in np.arange(0,m):
        for j in np.arange(0,m):
            Rxxmm[i,j] = np.exp(-np.abs(sampledepths[i]-sampledepths[j])/L)
    Rxxnn = np.zeros((n,n))
    for i in np.arange(0,n):
        for j in np.arange(0,n):
            Rxxnn[i,j] = np.exp(-np.abs(griddepths[i]-griddepths[j])/L)
    Rxy = np.zeros((n,m))
    for i in np.arange(0,n):
        for j in np.arange(0,m):
            Rxy[i,j] = np.exp(-np.abs(griddepths[i]-sampledepths[j])/L)
    return(Rxxmm,Rxxnn,Rxy,m,n)

#given sampling depths, return layer with which that depth is associated
def lmatch(di):
    d = zml[int(di)]
    if d < bnd: layer = 'A'
    else: layer = 'B'
    return layer

#given statevector, variable idx, param, and layer, find value of a parameter
#inteded just for prior values
def findp_dvp(pvec,vi,param,layer):
    ip = vi.index('_'.join([param,layer]))
    p = pvec[ip]
    return p

#like findp_dvp, but for depth-constant params
def findp_dcp(pvec,vi,param):
    ip = vi.index(param)
    p = pvec[ip]
    return p

#calculate model errors given a state vector
def modresi(sv):
    n_err = np.zeros(P)
    #depth-independent params
    Ghio = findp_dcp(sv,vidxSV,'Gh')
    Lpio = findp_dcp(sv,vidxSV,'Lp')
    for i in np.arange(0,P):
        #what tracer and depth are we on?
        t,d = vidxP[i].split('_')
        d = int(d)
        l = lmatch(d) #what layer does this depth index correspond to?
        iPsi,Psio = fidx2('Ps',d,vidxP,sv)
        iPli,Plio = fidx2('Pl',d,vidxP,sv)
        #depth-dependent parameters
        wsio = findp_dvp(sv,vidxSV,'ws',l)
        wlio = findp_dvp(sv,vidxSV,'wl',l)
        B2pio = findp_dvp(sv,vidxSV,'B2p',l)
        Bm2io = findp_dvp(sv,vidxSV,'Bm2',l)
        Bm1sio = findp_dvp(sv,vidxSV,'Bm1s',l)
        Bm1lio = findp_dvp(sv,vidxSV,'Bm1l',l)
        #POC, SSF
        if t == 'Ps':
            if d == 0: #mixed layer
                n_err[i] = Ghio+Bm2io*Plio-(wsio/h+Bm1sio+B2pio*Psio)*Psio
            elif (d == 1 or d == 2): #first or second grid point
                iPsip1,Psip1o = fidx2('Ps',d+1,vidxP,sv)
                iPsim1,Psim1o = fidx2('Ps',d-1,vidxP,sv)
                n_err[i] = Ghio*np.exp(-(zml[d]-h)/(Lpio))+(Bm2io*Plio)-(Bm1sio+B2pio*Psio)*Psio-wsio/(2*dz)*(Psip1o-Psim1o)
            else: #everywhere else
                iPsim1,Psim1o = fidx2('Ps',d-1,vidxP,sv)
                iPsim2,Psim2o = fidx2('Ps',d-2,vidxP,sv)
                n_err[i] = Ghio*np.exp(-(zml[d]-h)/(Lpio))+(Bm2io*Plio)-(Bm1sio+B2pio*Psio)*Psio-wsio/(2*dz)*(3*Psio-4*Psim1o+Psim2o)
        #POC, LSF
        else:
            if d == 0:
                n_err[i] = (B2pio*(Psio)**2)-(wlio/h+Bm2io+Bm1lio)*Plio                
            elif (d == 1 or d == 2):
                iPlip1,Plip1o = fidx2('Pl',d+1,vidxP,sv)    
                iPlim1,Plim1o = fidx2('Pl',d-1,vidxP,sv)
                n_err[i] = (B2pio*(Psio)**2)-(Bm2io+Bm1lio)*Plio-wlio/(2*dz)*(Plip1o-Plim1o)
            else:
                iPlim1,Plim1o = fidx2('Pl',d-1,vidxP,sv)
                iPlim2,Plim2o = fidx2('Pl',d-2,vidxP,sv)
                n_err[i] = (B2pio*(Psio)**2)-(Bm2io+Bm1lio)*Plio-wlio/(2*dz)*(3*Plio-4*Plim1o+Plim2o)
    #extract errors of individual tracers from diagonal of Cf
    Ps_n, Pl_n = vsli(n_err,td['Ps']['si']), vsli(n_err,td['Pl']['si'])
    Ps_nm, Pl_nm = np.mean(Ps_n), np.mean(Pl_n)
    Ps_nma, Pl_nma = np.mean(np.absolute(Ps_n)), np.mean(np.absolute(Pl_n))
    return (Ps_n, Ps_nm, Ps_nma, Pl_n, Pl_nm, Pl_nma)

def lsqplotsf(model,x,y,z,tracer,logscale): #make a least squares fit and plot the results, using smf offline
    #open fig
    fig, ax =  plt.subplots(1,1)
    fig.subplots_adjust(bottom=0.2,left=0.2)
    #colorbar stuff
    cmap=plt.cm.viridis_r
    normfac = mpl.colors.Normalize(z.min(),z.max())
    axcb = mpl.colorbar.make_axes(ax)[0]
    cbar = mpl.colorbar.ColorbarBase(axcb,norm=normfac,cmap=cmap)
    cbar.set_label('Depth (m)\n',rotation=270, labelpad=20,fontsize=14)
    #plot data
    ax.scatter(x,y,norm=normfac,edgecolors=black,c=z,s=150,marker='o',cmap=cmap)
    ax.set_ylabel(f'{tracer} (mmol/m$^3$)',fontsize=14)
    ax.set_xlabel('$c_{p}$ (m$^{-1}$)',fontsize=14)
    xp = np.arange(0.01,0.14,0.0001)
    c = model.params #extract coefficients, where c[0] is the intercept
    ax.text(0.04,0.2,f'$R^2$ = {model.rsquared:.2f}\nN = {model.nobs:.0f}',fontsize=12)
    if logscale: 
        ax.set_yscale('log'), ax.set_xscale('log') 
        fit = [c[0] + c[1]*np.log(xi) for xi in xp]
    else: fit = [c[0] + c[1]*xi for xi in xp]
    ax.plot(xp, fit, '--', c = 'k', lw=2)
    plt.savefig(f'invP_cpptfit_log{logscale}.pdf')
    plt.close()
    return ax

#autocorrelation function for a specific depth range
def acdrange(drange):
    dmin, dmax = drange #min and max depths within the range
    dslice = zml[difind(dmin):difind(dmax)+1] #slice of zml corresponding to drange
    tslice = cppt['Pt_hat'].values[difind(dmin):difind(dmax)+1] #slice of tracer data array corresponding to drange, in this case Pt
    kn = int(np.ceil(len(dslice)/4)) #number of lags, round up
    kdz = np.arange(0,(kn+1)*dz, dz) #lags in units of m
    ac = smt.acf(tslice,fft=False,nlags=kn) #autocorrelation function
    lmod_dic = {'rk':ac,'kdz':kdz}
    lmod = smf.ols(formula='np.log(rk) ~ kdz', data=lmod_dic).fit()
    l_int, L = lmod.params[0], -(1/lmod.params[1])
    lfit = l_int + -(1/L)*kdz
    l_r2 = lmod.rsquared
    return kdz, ac, l_int, L, lfit, l_r2

#given a vector and starting index, returns a slice of size n
def vsli(vec,sidx):
    sli = vec[sidx:sidx+n]
    return sli

#subsample a larger covariance matrix given list of relavent variabl Strings
def cvmsli(cvm, vlist):
    n = len(vlist)
    cvms = np.zeros((n,n)) #store relevant covarainces
    vids = [] #to store indices of variables corresponding to the state vector
    for v in vlist: vids.append(vidxSV.index(str(v)))
    for i, vi in enumerate(vids):
        for j, vj in enumerate(vids):
            cvms[i,j] = cvm[vi,vj]
    return cvms

#given a depth index and function, evaluate the function and fill in corresponding jacobian values
def Fnf(y,i,di):
    f_x, f_xv, f_xi = Fnf_helper(y,i,di)
    f[i] = sym.lambdify(f_x,y)(*f_xv)
    for j, v in enumerate(f_x):
        dy = y.diff(v)*v #multiplied by v becuase it's atually dy/d(lnv) = v*dy/dv
        F_x, F_xv, F_xi = Fnf_helper(dy,i,di)
        F[i,int(f_xi[j])] = sym.lambdify(F_x,dy)(*F_xv)

def Fnf_helper(y,i,di):
    x = y.free_symbols #gets all (symbolic) variables in function
    nx = len(x) #number of variables
    xv, xi = np.zeros(nx), np.zeros(nx) #numerical values of symbolic variables and their indices in vidxSV
    for j,v in enumerate(x): #for each symbolic variable
        if '_' in v.name: #if it's a tracer
            t, rdi = v.name.split('_') #get tracer and relative (to di) depth index
            adi = str(di+int(rdi)) #the absolute di for this tracer
            iSV = vidxSV.index('_'.join([t,adi])) #index of the state variable
        else: #it it's a parameter
            if pdi[v.name]['dv'] == 1:
                l = lmatch(di)
                iSV = vidxSV.index('_'.join([v.name,l]))
            else: iSV = vidxSV.index(v.name)
        xv[j],xi[j] = np.exp(xk[iSV]), iSV
    return x, xv, xi

#given a mathmatical function for flux at particular depth, return flux and uncertainty
#if err==True, considers uncertainties
#if cov==True, considers covariances
def symfunceval(y,err=True,cov=True):
    #print('stepped in')
    x = y.free_symbols #gets all (symbolic) variables in function
    nx = len(x) #number of variables
    xv = np.zeros(nx) #numerical values of symbolic variables
    #print('obtaining numerical values')
    for i,v in enumerate(x): #for each symbolic variable, get numerical value
        if "_" in str(v): #if the variable varies with depth
            svar, di = str(v).split('_') #what kind of state variable (tracer or param?)
            if svar in td.keys(): xv[i] = td[svar]['gammas'][g]['xh'][int(di)]
            else:
                xv[i] = pdi[svar]['gammas'][g]['xh'][str(di)]
        else: xv[i] = pdi[str(v)]['gammas'][g]['xh'] #if it's a depth-constant variable
    if err == True: #if we are propagating errors
        #print('creating CVM')
        dy = [None]*nx #empty list to store derivatives
        cvms = cvmsli(CVM,x) #covariance matrix for all variables
        #print('cvm created, deriving')
        #calculate derivatives w.r.t each symbolic variable
        for i,v in enumerate(x): dy[i] = y.diff(v) 
        #print('derivations complete, building variance equation')
        u = 0 #initialize value to calculate uncertainties
        #iterate through entries of cvms and calculate relevant term
        for i, r in enumerate(cvms):
            for j, c in enumerate(r):
                if  i>j: continue #only take upper diagonal (including diagonal)
                elif i == j: u += (dy[i]**2)*cvms[i,j]
                else: 
                    if cov == True: u += 2*dy[i]*dy[j]*cvms[i,j]
        flx, unc = sym.lambdify(x,(y,u))(*xv)
        result = (flx, np.sqrt(unc))
    else: result = sym.lambdify(x,y)(*xv) #if just evaluating the function without propagating errors
    #print(y), print(u)
    #print('equations built, evaluating')
    #print('Returned!')
    return result

#given a depth range, calculate integrated fluxes
def iflxcalc(fluxes, deprngs):
    for f in fluxes:
        #print(f'-------{f}-------')
        if '_' in f: #if not Psdot
            p,t = f.split('_')[0], f.split('_')[1][:2]
            ordr = flxd[f]['order'] #get order from the flx dict
        for dr in deprngs:        
            do, dn = dr #unpack start and end depths
            rstr = "_".join([str(do),str(dn)])
            #assign value of dz for first depth
            if do == h:
                dz_b = h+dz/2
            elif do == bnd:
                do += dz/2
                dz_b = dz
            else: dz_b = dz/2
            #assign value of dz for last depth
            if dn == bnd:
                dn -= dz/2
                dz_e = dz
            else: dz_e = dz/2
            doi, dni = zml.tolist().index(do), zml.tolist().index(dn)
            dis = np.arange(doi,dni+1) 
            iF, iI = 0, 0 #initialize variable to collect summation (integrals) of fluxes and inventory
            if 'dz' in f: #if sinking flux divergence term, more complicated
                for i,di in enumerate(dis):
                    if i == 0: dzi = dz_b #first depth in range
                    elif i == len(dis)-1: dzi = dz_e #last depth in range
                    else: dzi = dz #all other depths
                    l = lmatch(di)
                    pwi = "_".join([p,l])
                    twi = "_".join([t,str(di)])
                    w, Pi = sym.symbols(f'{pwi} {twi}')
                    if di == 0: #mixed layer
                        iF += w*Pi/h*dzi
                        iI += Pi*dzi
                    elif (di == 1 or di == 2): #first two points below ML
                        twip1, twim1 = "_".join([t,str(di+1)]), "_".join([t,str(di-1)])
                        Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                        iF += w*(Pip1-Pim1)/(2*dz)*dzi #calculate flux estimate
                        iI += Pi*dzi
                    else: #all other depths
                        twim1, twim2 = "_".join([t,str(di-1)]), "_".join([t,str(di-2)])
                        Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                        iF += w*(3*Pi-4*Pim1+Pim2)/(2*dz)*dzi #calculate flux estimate
                        iI += Pi*dzi
                    #print(i, di, zml[di], dzi)
            elif f == 'Psdot': #if it's the production term
                gh, lp = sym.symbols('Gh Lp')
                for i,di in enumerate(dis):
                    if i == 0: dzi = dz_b #first depth in range
                    elif i == len(dis)-1: dzi = dz_e #last depth in range
                    else: dzi = dz #all other depths
                    Pi = sym.symbols(f'Ps_{di}')
                    iF += gh*sym.exp(-(zml[di]-h)/lp)*dzi
                    iI += Pi*dzi
                    #print(i, di, zml[di], dzi)
            else: #all other terms that are not sinking or production
                for i,di in enumerate(dis):
                    if i == 0: dzi = dz_b #first depth in range
                    elif i == len(dis)-1: dzi = dz_e #last depth in range
                    else: dzi = dz #all other depths
                    l = lmatch(di)
                    pwi = "_".join([p,l])
                    twi = "_".join([t,str(di)])
                    pa, tr = sym.symbols(f'{pwi} {twi}')
                    iF += (pa*tr**ordr)*dzi
                    iI += tr*dzi
                    #print(i, di, zml[di], dzi)
            intflx = symfunceval(iF)
            resT = symfunceval(iI/iF,err=True,cov=True) #error prop takes a long time...
            flxd[f]['gammas'][g]['iflx'][rstr], flxd[f]['gammas'][g]['tau'][rstr] = intflx, resT

#given a depth range, calculate (integrated) inventory and residuals
def inventory(deprngs):
    for t in tracers:
        for dr in deprngs:        
            do, dn = dr #unpack start and end depths
            rstr = "_".join([str(do),str(dn)])
            #assign value of dz for first depth
            if do == h:
                dz_b = h+dz/2
            elif do == bnd:
                do += dz/2
                dz_b = dz
            else: dz_b = dz/2
            #assign value of dz for last depth
            if dn == bnd:
                dn -= dz/2
                dz_e = dz
            else: dz_e = dz/2
            doi, dni = zml.tolist().index(do), zml.tolist().index(dn)
            dis = np.arange(doi,dni+1) 
            I, ir = 0, 0 #initialize variable to collect summation (integrals) of inventory
            for i,di in enumerate(dis):
                if i == 0: dzi = dz_b #first depth in range
                elif i == len(dis)-1: dzi = dz_e #last depth in range
                else: dzi = dz #all other depths
                twi = "_".join([t,str(di)]) #get the tracer at this depth index
                tr = sym.symbols(f'{twi}') #make it a symbolic variable
                I += tr*dzi
                ir += td[t]['gammas'][g]['n'][di]*dzi #integrated residual
            td[t]['gammas'][g]['inv'][rstr] = symfunceval(I)
            td[t]['gammas'][g]['ires'][rstr] = ir
            
####Pt estimates    
#read in cast match data
#cmdp = '/Users/vamaral/GoogleDrive/DOCS/Py/pyEXPORTS/misc/castmatch_v1.csv' #v1 has "exact" matches, but includes ctd cast 39 which has a bad values 
cmdf = pd.read_csv(cwd+'/castmatch_v2.csv') #v2 uses stn 41, at approx. time and place as 39. Also I sub 19 for 22 (has spike at 500m)
pctd = dict(zip(cmdf.pump_cast,cmdf.ctd_cast)) #generate dictionary with pumpcast as key, ctdcast as value

#read in POC data, from EXPORTS_pumps_KB.xlsx, PC&PIC sheet (CONVERTED TO MMOL!!)
pocdf_all = pd.read_excel(cwd+'/poc_all.xlsx') #put into a df

#read in cp data (includes bad casts from Norm)
cp_mdic = sio.loadmat(cwd+'/srcpdata.mat')

#read in updated cp data (20200426) 
cp_bycast = sio.loadmat(cwd+'/cp_bycast.mat')['cp_bycast']

# add a row to the beginning so that row # is depth
cp_bycast = np.vstack((np.ones(cp_bycast.shape[1])*-9999,cp_bycast))

#make a df with POC data that doesn't include corresponding "bad" cp casts
pocdf = pocdf_all.copy()
for pc in pocdf.pump_cast.unique(): #look at each pump cast
    cc = pctd[pc] #find the corresponding ctd cast
    if np.isin(cc,cp_mdic['sr1812bads']):
        print(f'Bad Casts: {cc}')
        pocdf = pocdf[pocdf.pump_cast != pc].copy()

#make a df to store corresponding POC and cp values
combodf = pocdf.copy()
for i, r in combodf.iterrows():
    pc = r.pump_cast
    d = r.depth
    cc = pctd[pc] #find the corresponding ctd cast
    combodf.at[i,'ctd_cast'] = cc
    #combodf.at[i,'cp'] = cp_mdic['sr1812cp'][d][list(cp_mdic['srcast'][0]).index(cc)]
    combodf.at[i,'cp'] = cp_bycast[d][list(cp_mdic['srcast'][0]).index(cc)]
combodf_s = combodf.sort_values(by='cp').copy() #sort by ascending cp values

#linear regression model (and plots)
#converted Pt to mmol here
x = combodf_s.cp.values
y = combodf_s.Pt.values
zdep = combodf_s.depth.values
zcst = combodf_s.ctd_cast.values
ctd_casts = np.unique(zcst)
model = smf.ols(formula='Pt ~ np.log(cp)', data=combodf_s).fit()
model_lin = smf.ols(formula='Pt ~ cp', data=combodf_s).fit()
ax_ls = lsqplotsf(model, x, y, zdep, '$P_T$', logscale=True)
ax_ls_nonlog = lsqplotsf(model_lin, x, y, zdep, '$P_T$', logscale=False)

#make a dataframe of all cp profiles, where index is depth and col name is cast #
allcp = pd.DataFrame(data=cp_bycast, columns=cp_mdic['srcast'][0])
cp_500_cm = allcp.loc[zml,ctd_casts].copy() #cp measurements from casts for which we have poc measurments, only above 500m and at model grid depths
meancp = cp_500_cm.mean(axis=1).values

#obtain Pt estimates at every grid depth
Pt_hat_obj = model.get_prediction(exog=dict(cp=meancp))
Pt_hat = Pt_hat_obj.predicted_mean

#make a df with fitted meancp and Pt_hat values
cppt = pd.DataFrame()
cppt['depth'] = zml
cppt['meancp'] = meancp
cppt['Pt_hat'] = Pt_hat

#runs test
runs_z, runs_p = smr.runstest_1samp(model.resid)
#fig, ax = plt.subplots(1,1)
#ax.scatter(np.arange(0,len(model.resid)),model.resid)
#ax.hlines(np.mean(model.resid),0,65,linestyles='dashed')
#fig.suptitle(f'z = {runs_z:.4f}, p = {runs_p:.4f}')

#make a matrix whose diagonals are MSE of residuals
Cf_addPt = np.diag(np.ones(n)*model.mse_resid)

kdz_A, ac_A, l_int_A, L_A, lfit_A, l_r2_A = acdrange((h,bnd-dz/2))
kdz_B, ac_B, l_int_B, L_B, lfit_B, l_r2_B = acdrange((bnd+dz/2,zmax))
ac_params = (kdz_A, ac_A, l_int_A, L_A, lfit_A, l_r2_A, kdz_B, ac_B, l_int_B, L_B, lfit_B, l_r2_B)
fig, ax = plt.subplots(1,1)
cA, cB = blue, green
ax.scatter(kdz_A,np.log(ac_A),label='A (EZ)',marker='o',color=cA) 
ax.scatter(kdz_B,np.log(ac_B),label='B (MZ)',marker='x',color=cB)
ax.plot(kdz_A,lfit_A,'--',lw=lw,color=cA), ax.plot(kdz_B,lfit_B,'--',lw=lw,color=cB)
ax.set_title(f'int_A = {l_int_A:.2f}, L_A = {L_A:.2f}, R2_A = {l_r2_A:.2f} \n int_B = {l_int_B:.2f}, L_B = {L_B:.2f}, R2_B = {l_r2_B:.2f}')
ax.set_xlabel('lags (m)')
ax.set_ylabel('ln($r_k$)')
ax.legend()
plt.savefig('invP_autocor.png')
plt.close()

"""
#HYDROGRAPHIC FEATURES PLOTS
"""
#load hydrography data
hydro_df = pd.read_excel('T_S_SigT_cast1.xlsx')

#create figure and axes
fig = plt.figure()
host = host_subplot(111, axes_class=AA.Axes, figure=fig)
plt.subplots_adjust(top=0.75)
par1 = host.twiny()
par2 = host.twiny()

#show parasite axes
par1.axis['top'].toggle(all=True)
offset = 40
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis['top'] = new_fixed_axis(loc='top',axes=par2,offset=(0, offset))
par2.axis['top'].toggle(all=True)

host.set_ylim(0, 520)
host.invert_yaxis(), host.grid(axis='y',alpha=0.5)
host.set_xlim(24, 27.4)
par1.set_xlim(3, 14.8)
par2.set_xlim(32, 34.5)

host.set_ylabel('Depth (m)',fontsize=14)
host.set_xlabel('$\sigma_T$ (kg $m^{-3}$)')
par1.set_xlabel('Temperature (°C)')
par2.set_xlabel('Salinity (PSU)')

host.plot(hydro_df['sigT_kgpmc'],hydro_df['depth'],c=orange,marker='o')
par1.plot(hydro_df['t_c'],hydro_df['depth'],c=green,marker='o')
par2.plot(hydro_df['s_psu'],hydro_df['depth'],c=blue,marker='o')
host.axhline(h,c=black,ls=':',zorder=3)
host.axhline(bnd,c=black,ls='--',zorder=3)

host.axis['bottom'].label.set_color(orange)
par1.axis['top'].label.set_color(green)
par2.axis['top'].label.set_color(blue)

host.axis['bottom','left'].label.set_fontsize(14)
par1.axis['top'].label.set_fontsize(14)
par2.axis['top'].label.set_fontsize(14)

host.axis['bottom','left'].major_ticklabels.set_fontsize(12)
par1.axis['top'].major_ticklabels.set_fontsize(12)
par2.axis['top'].major_ticklabels.set_fontsize(12)

host.axis['bottom','left'].major_ticks.set_ticksize(6)
par1.axis['top'].major_ticks.set_ticksize(6)
par2.axis['top'].major_ticks.set_ticksize(6)

plt.savefig('invP_hydrography.pdf')
plt.close()

"""
#INVERSE METHOD (P)
"""
#create variable indexes for three P size fractions, and for the state vector
vidxP = []
vidxPt = []
vidxSV = []

#create depth indices for state variables found at every grid point
gnums = np.arange(0,n)
svis = [''.join(['_',str(i)]) for i in gnums]

#add variables to the index
vidxP = addtovaridx(vidxP,('Ps','Pl'))
vidxPt = addtovaridx(vidxPt,('Pt',)) #just Pt, comma indicates one-element tuple

params_o, params_o_e = np.empty(0), np.empty(0) #group the rate parameters as (p1layA, p1layB, p2layA, p2layB, ...)
p_toidx = [] #and put them in a list to be added to the variable index later
for p in pdi.keys():
    if pdi[p]['dv']: #depth-varying params
        for l in layers:
            params_o = np.append(params_o,pdi[p]['o'])
            params_o_e = np.append(params_o_e,pdi[p]['oe'])
            p_toidx.append('_'.join([p,l]))
    else:
        params_o = np.append(params_o,pdi[p]['o'])
        params_o_e = np.append(params_o_e,pdi[p]['oe'])
        p_toidx.append(f'{p}')
nparams = len(p_toidx) #number of total params (one in each layer for depth-varying)
        
#add params to the vidx's
vidx_allP = vidxP+vidxPt #includes everything in F and f
vidxSV = vidxP+p_toidx #only state variables

#make a dictionary for tracer params for each layer. could be more cleverly coded
oi = {lay:{t:{} for t in tracers} for lay in layers}
oi['A']['Ps']['y'],oi['B']['Ps']['y'] = Ps_mean[0:3].values,Ps_mean[3:].values #mean POC
oi['A']['Pl']['y'],oi['B']['Pl']['y'] = Pl_mean[0:3].values,Pl_mean[3:].values
oi['A']['Pl']['sig_j'],oi['B']['Pl']['sig_j'] = Pl_sd[0:3].values,Pl_sd[3:].values #POC standard deviation
oi['A']['Ps']['sig_j'],oi['B']['Ps']['sig_j'] = Ps_sd[0:3].values,Ps_sd[3:].values
oi['A']['smpd'], oi['A']['grdd'] = zs[0:3].values, zml[difind(h):difind(bnd-dz/2)+1] #sample and grid depths, layer A
oi['B']['smpd'], oi['B']['grdd'] = zs[3:].values, zml[difind(bnd+dz/2):] #sample and grid depths, layer B
oi['A']['L'], oi['B']['L'] = L_A, L_B #interpolation length scales


#calculate OI params
for lay in oi.keys():
    L = oi[lay]['L']
    smpd = oi[lay]['smpd']
    grdd = oi[lay]['grdd']
    for tra in tracers:
        #print(lay,tra)
        Rxxmm,Rxxnn,Rxy,moi,noi = Rmatx(smpd,grdd,L)
        oi[lay][tra]['ym'] = np.mean(oi[lay][tra]['y']) #means
        oi[lay][tra]['ya'] = oi[lay][tra]['y'] - oi[lay][tra]['ym'] #anomalies 
        oi[lay][tra]['var_j'] = oi[lay][tra]['sig_j']**2 #absolute variances
        oi[lay][tra]['var'] = np.var(oi[lay][tra]['y'],ddof=1) + np.sum(oi[lay][tra]['var_j'])/moi #scalar variances
        oi[lay][tra]['Rnn'] = np.zeros((moi,moi)) #diagonal matrix of absolute variances
        np.fill_diagonal(oi[lay][tra]['Rnn'],oi[lay][tra]['var_j'])
        oi[lay][tra]['Rxxmm'] = oi[lay][tra]['var']*Rxxmm
        oi[lay][tra]['Rxxnn'] = oi[lay][tra]['var']*Rxxnn
        oi[lay][tra]['Rxy'] = oi[lay][tra]['var']*Rxy
        oi[lay][tra]['Ryy'] = oi[lay][tra]['Rxxmm'] + oi[lay][tra]['Rnn']
        oi[lay][tra]['Ryyi'] = np.linalg.inv(oi[lay][tra]['Ryy'])
        oi[lay][tra]['xa'] = np.matmul(np.matmul(oi[lay][tra]['Rxy'],oi[lay][tra]['Ryyi']),oi[lay][tra]['ya']) #interpolated anomalies
        oi[lay][tra]['x'] = oi[lay][tra]['xa'] + oi[lay][tra]['ym'] #interpolated estimates
        oi[lay][tra]['P'] = oi[lay][tra]['Rxxnn'] - np.matmul(np.matmul(oi[lay][tra]['Rxy'],oi[lay][tra]['Ryyi']),oi[lay][tra]['Rxy'].T)#covariance matrix
        oi[lay][tra]['xerr'] = np.sqrt(np.diagonal(oi[lay][tra]['P']))

#make an oi matrix with some values concatenated
tpri = np.asarray([]) #initialize an array to collect priors of all tracers AFTER they've been concatenated by layer
for t in tracers:
    for k in ['si','x','xerr']:
        if k == 'si': 
            td[t][k] = vidx_allP.index(f'{t}_0')
            continue
        td[t][k] = np.concatenate((oi['A'][t][k],oi['B'][t][k]))
        if k == 'x': tpri = np.concatenate((tpri,td[t][k]))
                      
#combine xo's to form one xo, take the ln
xo = np.concatenate((tpri,params_o))
xoln = np.log(xo)

#redefine some useful params
N = len(xo)
P = N-len(params_o) #dimension of everything minus params (# of model equations)
M = len(vidx_allP) #dimension that includes all three P size fractions (for F and f)

#construct Co as a (normalized) diagonal matrix, blocks for tracers and diagonals otherwise. Considers the ln()
Co, Coln = np.zeros((N,N)), np.zeros((N,N))
blocks = [oi[l][t]['P'] for t in tracers for l in layers] #tracers is outer loop, layers is inner
Co[:P,:P] = splinalg.block_diag(*blocks)
for i in np.arange(P,N):
    Co[i,i] = (params_o_e[i-P]**2)
Co_cond = np.linalg.cond(Co)
for i, r in enumerate(Coln):
    for j, c in enumerate(r):
        Coln[i,j] = np.log(1+Co[i,j]/(xo[i]*xo[j]))
Co_cond = np.linalg.cond(Co)
Co_neg = (Co<0).any() #checks if any element of Co is negative

#check that the Co inversion is accurate
ColnColninv = np.matmul(Coln,np.linalg.inv(Coln))
Coln_check = np.sum(ColnColninv-np.identity(N))

pdelt = 0.01 #allowable percent change in each state element for convergence
maxiter = 50

for g in gammas:
    Cf_noPt = np.zeros((P,P))
    Cfd_noPt = np.ones(2*n)*pdi['Gh']['o']**2 #Cf from particle production
    Cf_noPt = np.diag(Cfd_noPt)*g
    Cf = splinalg.block_diag(Cf_noPt,Cf_addPt)
    
    #initialize the iterative loop
    F = np.zeros((M, N))
    f = np.zeros(M)
    xk = xoln
    xkp1 = np.ones(N) #initialize this to some dummy value 
    k = 0 #keep a counter for how many steps it takes
    conv_ev = np.empty(0) # keep track of evolution of convergence
    cost_ev = np.empty(0) #keep track of evolution of the cost function, j
    
    
    #define all possible symbolic variables
    svarnames = 'Ps_0 Ps_1 Ps_-1 Ps_-2 \
        Pl_0 Pl_1 Pl_-1 Pl_-2 \
            Bm2 B2p Bm1s Bm1l \
                Gh Lp ws wl'
    Psi, Psip1, Psim1, Psim2, \
        Pli, Plip1, Plim1, Plim2, \
            Bm2i, B2pi, Bm1si, Bm1li, \
                Ghi, Lpi, wsi, wli = sym.symbols(svarnames)
    #ATI
    while True:
        for i in np.arange(0,M):
            #what tracer and depth are we on?
            t,d = vidx_allP[i].split('_')
            d = int(d)
            #POC, SSF
            if t == 'Ps':
                if d == 0: #mixed layer
                    eq = Ghi+Bm2i*Pli-(wsi/h+Bm1si+B2pi*Psi)*Psi
                elif (d == 1 or d == 2): #first or second grid point
                    eq = Ghi*sym.exp(-(zml[d]-h)/(Lpi))+(Bm2i*Pli)-(Bm1si+B2pi*Psi)*Psi-wsi/(2*dz)*(Psip1-Psim1)
                else: #everywhere else
                    eq = Ghi*sym.exp(-(zml[d]-h)/(Lpi))+(Bm2i*Pli)-(Bm1si+B2pi*Psi)*Psi-wsi/(2*dz)*(3*Psi-4*Psim1+Psim2)
            #POC, LSF
            elif t == 'Pl':
                if d == 0:
                    eq = (B2pi*(Psi)**2)-(wli/h+Bm2i+Bm1li)*Pli
                elif (d == 1 or d == 2):
                    eq = B2pi*Psi**2-(Bm2i+Bm1li)*Pli-wli/(2*dz)*(Plip1-Plim1)
                else:
                    eq = B2pi*Psi**2-(Bm2i+Bm1li)*Pli-wli/(2*dz)*(3*Pli-4*Plim1+Plim2)
            #POC, Ps + Pl = Pt
            else: #because this isn't a tracer in the state vector, doesn't play nice with our Fnf function
                Pti = cppt.Pt_hat[vidxPt.index(f'Pt_{d}')] #value of Pt at gridpoint d
                eq = Psi + Pli - Pti
            Fnf(eq,i,d)
        FCoFT = np.matmul(np.matmul(F,Coln),F.T)
        FCoFT_cond = np.linalg.cond(FCoFT)
        FCFCinv = np.matmul(FCoFT,np.linalg.inv(FCoFT))
        FC_check = np.sum(FCFCinv-np.identity(M))
        B = np.matmul(np.matmul(Coln,F.T),np.linalg.inv(FCoFT+Cf))
        xkp1 = xoln + np.matmul(B,np.matmul(F,xk-xoln)-f)
        #convergence criteria based on Murnane 94
        maxchange = np.max(np.abs((np.exp(xkp1)-np.exp(xk))/np.exp(xk)))
        conv_ev = np.append(conv_ev,maxchange)
        #if g = 0, only use equations corresponding to Pt equations for the cost function
        if not g: f_cost, Cf_cost = f[-n:], Cf_addPt
        else: f_cost, Cf_cost = f, Cf
        cost = np.matmul(np.matmul((xk-xoln).T,np.linalg.inv(Coln)),(xk-xoln))+\
            np.matmul(np.matmul(f_cost.T,np.linalg.inv(Cf_cost)),f_cost)
        cost_ev = np.append(cost_ev,cost)
        if maxchange < pdelt or k > maxiter: break
        k += 1
        xk = xkp1
    
    #calculate posterior errors
    I = np.identity(Coln.shape[0])
    CoFT = np.matmul(Coln,F.T)
    FCoFTpCfinv = np.linalg.inv(FCoFT+Cf)
    C = I-np.matmul(np.matmul(CoFT,FCoFTpCfinv),F)
    D = I-np.matmul(np.matmul(np.matmul(F.T,FCoFTpCfinv),F),Coln)
    Ckp1 = np.matmul(np.matmul(C,Coln),D)
    
    #expected value and variance of tracers AND params
    EyP, VyP = xkp1, np.diag(Ckp1)
    
    #recover dimensional values of median, mean, mode, standard deviation
    xhmed = np.exp(EyP)
    xhmod = np.exp(EyP-VyP)
    xhmean = np.exp(EyP+VyP/2)
    xhe = np.sqrt(np.exp(2*EyP+VyP)*(np.exp(VyP)-1))
    
    #calculate covariances of unlogged state variables
    CVM = np.zeros((N,N))
    for i, row in enumerate(CVM):
        for j, unu in enumerate(row):
            mi, mj = EyP[i], EyP[j] #mu's (expected vals)
            vi, vj = VyP[i], VyP[j] #sig2's (variances)
            CVM[i,j] = np.exp(mi+mj)*np.exp((vi+vj)/2)*(np.exp(Ckp1[i,j])-1)
    
    #check that sqrt of diagonals of CVM are equal to xhe
    CVM_xhe_check = np.sqrt(np.diag(CVM)) - xhe
             
    #get errors and estimates for tracers
    for t in tracers:
        td[t]['gammas'][g]['xh'] = vsli(xhmean,td[t]['si'])
        td[t]['gammas'][g]['xhe'] = vsli(xhe,td[t]['si'])
        
    #get model residuals from posterior estimates for tracers
    td['Ps']['gammas'][g]['n'], td['Ps']['gammas'][g]['nm'], \
        td['Ps']['gammas'][g]['nma'], td['Pl']['gammas'][g]['n'], \
            td['Pl']['gammas'][g]['nm'], td['Pl']['gammas'][g]['nma'] \
                = modresi(xhmean)
    
    #get residuals for PT equations
    Pt_resids = Pt_hat - (td['Ps']['gammas'][g]['xh']+td['Pl']['gammas'][g]['xh'])
    
    #propagating errors on Pt
    Pt_xh, Pt_xhe = np.zeros(n), np.zeros(n)
    for i in np.arange(0,n):
        pswi, plwi = "_".join(['Ps',str(i)]), "_".join(['Pl',str(i)])
        ps, pl = sym.symbols(f'{pswi} {plwi}')
        Pt_xh[i], Pt_xhe = symfunceval(ps+pl)
    
    #PDF and CDF calculations
    xg = np.linspace(-2,2,100)
    yg_pdf = sstats.norm.pdf(xg,0,1)
    yg_cdf = sstats.norm.cdf(xg,0,1)
    
    #comparison of estimates to priors (means don't include params, histogram does)
    xdiff = xhmean-xo
    x_osd = np.sqrt(np.diag(Co))
    pdfx = xdiff/x_osd
    # pdfx_Ps_m = np.mean(vsli(pdfx,td['Ps']['si']))
    # pdfx_Pl_m = np.mean(vsli(pdfx,td['Pl']['si']))
    # pdfx_Ps_ma = np.mean(np.absolute(vsli(pdfx,td['Ps']['si'])))
    # pdfx_Pl_ma = np.mean(np.absolute(vsli(pdfx,td['Pl']['si'])))
    
    #comparison of model residuals (posteriors)
    if g:
        nP = np.concatenate((td['Ps']['gammas'][g]['n'],td['Pl']['gammas'][g]['n'],Pt_resids))
        n_sd = np.sqrt(np.diag(Cf))
    else:
        nP = Pt_resids
        n_sd = np.sqrt(np.diag(Cf_addPt)) 
    pdfn = nP/n_sd
    # pdfn_Ps_m = np.mean(vsli(pdfn,td['Ps']['si']))
    # pdfn_Pl_m = np.mean(vsli(pdfn,td['Pl']['si']))
    # pdfn_Ps_ma = np.mean(np.absolute(vsli(pdfn,td['Ps']['si'])))
    # pdfn_Pl_ma = np.mean(np.absolute(vsli(pdfn,td['Pl']['si'])))
    
    #PDFs
    fig, [ax1,ax2] = plt.subplots(1,2,tight_layout=True)
    fig.subplots_adjust(wspace=0.5)
    ax1.set_ylabel('P',size=16)
    ax1.set_xlabel(r'$\frac{\^x-x_{o,i}}{\sigma_{o,i}}$',size=16)
    ax1.hist(pdfx,density=True,bins=20,color=blue)
    ax2.hist(pdfn,density=True,bins=20,color=blue)
    ax2.set_xlabel(r'$\frac{n^{k+1}_{i}}{\sigma_{n^{k+1}_{i}}}$',size=16)
    pdf_params = pdfx, pdfn, xg, yg_pdf
    
    #plot gaussians, show legend
    ax1.plot(xg,yg_pdf,c=red), ax2.plot(xg,yg_pdf,c=red)
    plt.savefig(f'invP_pdfs_gam{str(g).replace(".","")}.png')
    plt.close()
    
    #CDFs
    fig, [ax1,ax2] = plt.subplots(1,2,tight_layout=True)
    fig.subplots_adjust(wspace=0.5)
    ax1.set_ylabel('P',size=16)
    ax1.set_xlabel(r'$\frac{\^x-x_{o,i}}{\sigma_{o,i}}$',size=16), ax2.set_xlabel(r'$\frac{n^{k+1}_{i}}{\sigma_{n^{k+1}_{i}}}$',size=16)
    ax1.plot(xg,yg_cdf,c=red), ax2.plot(xg,yg_cdf,c=red) #plot gaussians
    cdf_dfx, cdf_dfn = pd.DataFrame(), pd.DataFrame()
    cdf_dfx['var_name'] = vidxSV.copy()
    cdf_dfn['var_name'] = vidx_allP.copy() if g else vidxPt.copy()
    cdf_dfx['val'], cdf_dfn['val'] = pdfx.copy(), pdfn.copy() #add values that correspond to those
    cdf_dfx['o_idx'], cdf_dfn['o_idx'] = cdf_dfx.index, cdf_dfn.index #copy original indices
    cdf_dfxs, cdf_dfns = cdf_dfx.sort_values('val').copy(), cdf_dfn.sort_values('val').copy()
    cdf_dfxs.reset_index(inplace=True), cdf_dfns.reset_index(inplace=True) #reset indices
    x1,x2 = cdf_dfxs.val, cdf_dfns.val 
    y1,y2 = np.arange(1,len(x1)+1)/len(x1), np.arange(1,len(x2)+1)/len(x2)
    #Plot params in orange, Ps navy, Pl in green, Pt in teal
    marsize = 16
    mar = 'o'
    fc = 'none'       
    for i, v in enumerate(x1):
        if 'Ps' in cdf_dfxs.var_name[i]: ec = blue
        elif 'Pl' in cdf_dfxs.var_name[i]: ec = green
        else: ec = orange #params
        ax1.scatter(x1[i],y1[i],s=marsize,marker=mar,facecolors=fc,edgecolors=ec)
    #plot posteriors model residuals
    for i, v in enumerate(x2):
        if 'Ps' in cdf_dfns.var_name[i]: ec = blue
        elif 'Pl' in cdf_dfns.var_name[i]: ec = green
        else: ec = teal #Pt
        ax2.scatter(x2[i],y2[i],s=marsize,marker=mar,facecolors=fc,edgecolors=ec)
    plt.savefig(f'invP_cdfs_gam{str(g).replace(".","")}.png')
    plt.close() 
    
    #model residual depth profiles (posteriors)
    fig, [ax1,ax2,ax3] = plt.subplots(1,3)
    fig.subplots_adjust(wspace=0.5)  
    ax1.invert_yaxis(), ax2.invert_yaxis(), ax3.invert_yaxis()
    ax1.set_xlabel('$n^{k+1}_{P_{S}}$ (mmol/m3/d)')
    ax2.set_xlabel('$n^{k+1}_{P_{L}}$ (mmol/m3/d)')
    ax3.set_xlabel('$n^{k+1}_{P_{T}}$ (mmol/m3)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_ylim(top=0,bottom=zmax+dz), ax2.set_ylim(top=0,bottom=zmax+dz), ax3.set_ylim(top=0,bottom=zmax+dz)
    ax1.scatter(td['Ps']['gammas'][g]['n'], zml, marker='o', c=blue, s=ms/2)
    ax2.scatter(td['Pl']['gammas'][g]['n'], zml, marker='o', c=blue, s=ms/2)
    ax3.scatter(Pt_resids, zml, marker='o', c=blue, s=ms/2)
    plt.savefig(f'invP_residprofs_gam{str(g).replace(".","")}.png')
    plt.close()
    
    #plot evolution of convergence
    ms=3
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(0, len(conv_ev)), conv_ev, marker='o',ms=ms)
    ax.set_yscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',size=12)
    plt.savefig(f'invP_conv_gam{str(g).replace(".","")}.png')
    plt.close()
    
    #plot evolution of cost function
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(0, len(cost_ev)),cost_ev,marker='o',ms=ms)
    ax.set_xlabel('k')
    ax.set_ylabel('j')
    ax.set_yscale('log')
    plt.savefig(f'invP_cost_gam{str(g).replace(".","")}.png')
    plt.close()
    
    #comparison plots
    fig, [ax1,ax2,ax3] = plt.subplots(1,3) #P figures
    fig.subplots_adjust(wspace=0.5)  
    ax1.invert_yaxis(), ax2.invert_yaxis(), ax3.invert_yaxis()
    ax1.set_xlabel('$P_{S}$ ($mmol/m^3$)'), ax2.set_xlabel('$P_{L}$ ($mmol/m^3$)'), ax3.set_xlabel('$P_{T}$ ($mmol/m^3$)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_ylim(top=0,bottom=zmax+2*dz), ax2.set_ylim(top=0,bottom=zmax+2*dz), ax3.set_ylim(top=0,bottom=zmax+2*dz)
    ax1.errorbar(td['Ps']['gammas'][g]['xh'], zml, fmt='o', xerr=td['Ps']['gammas'][g]['xhe'], ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='mean', fillstyle='none')
    ax1.errorbar(Ps_mean, zs, fmt='^', xerr=Ps_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
    ax1.errorbar(td['Ps']['x'], zml, fmt='o', xerr=td['Ps']['xerr'], ecolor=blue, elinewidth=elw, c=blue, ms=ms/2, capsize=cs, lw=lw, label='OI')  
    ax1.legend()
    ax2.errorbar(td['Pl']['gammas'][g]['xh'], zml, fmt='o', xerr=td['Pl']['gammas'][g]['xhe'], ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='mean', fillstyle='none')
    ax2.errorbar(Pl_mean, zs, fmt='^', xerr=Pl_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
    ax2.errorbar(td['Pl']['x'], zml, fmt='o', xerr=td['Pl']['xerr'], ecolor=blue, elinewidth=elw, c=blue, ms=ms/2, capsize=cs, lw=lw, label='OI')  
    ax2.legend()
    ax3.errorbar(Pt_xh, zml, fmt='o', xerr=Pt_xhe, ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='Inv', fillstyle='none')
    ax3.errorbar(cppt.Pt_hat, zml+1, fmt='o', xerr=np.ones(n)*np.sqrt(model.mse_resid), ecolor=green, elinewidth=elw, c=green, ms=ms, capsize=cs, lw=lw, label='Data', fillstyle='none')
    ax3.legend()
    ax1.axhline(bnd,c='k',ls='--',lw=lw/2)
    ax2.axhline(bnd,c='k',ls='--',lw=lw/2)
    ax3.axhline(bnd,c='k',ls='--',lw=lw/2)
    plt.savefig(f'invP_Pprofs_gam{str(g).replace(".","")}.png')
    plt.close()
    
    #extract posterior param estimates and errors
    params_ests = xhmean[-nparams:]
    params_errs = xhe[-nparams:]
    for i,stri in enumerate(p_toidx):
        pest, perr = params_ests[i], params_errs[i]
        if '_' in stri: #depth-varying paramters
            p,l = stri.split('_')
            pdi[p]['gammas'][g]['xh'][l], pdi[p]['gammas'][g]['xhe'][l] = pest, perr
        else: pdi[stri]['gammas'][g]['xh'], pdi[stri]['gammas'][g]['xhe'] = pest, perr
    
    #make a plot of parameter priors and posteriors
    elwp, msp, csp, ec = 1, 9, 4, 'k'
    fig, ([ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]) = plt.subplots(2,4)
    fig.subplots_adjust(wspace=0.8, hspace=0.4)
    axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax8]
    for i, p in enumerate(pdi.keys()):
        ax = axs[i]
        ax.set_title(pdi[p]['tset'])
        if pdi[p]['dv']: #if param is depth-varying
            ax.errorbar(1, pdi[p]['o'], yerr=pdi[p]['oe'], fmt='o', ms=msp, c=blue, elinewidth=elwp, ecolor=ec, capsize=csp) #priors with errors
            ax.errorbar(2, pdi[p]['gammas'][g]['xh']['A'], yerr=pdi[p]['gammas'][g]['xhe']['A'], fmt='o', c=teal, ms=msp, elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
            ax.errorbar(3, pdi[p]['gammas'][g]['xh']['B'], yerr=pdi[p]['gammas'][g]['xhe']['B'], fmt='o', c=navy, ms=msp, elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
        else: #if param is depth-constant
            ax.errorbar(1, pdi[p]['o'], yerr=pdi[p]['oe'],fmt='o',ms=msp,c=blue,elinewidth=elwp,ecolor=ec,capsize=csp) #priors with errors
            ax.errorbar(3, pdi[p]['gammas'][g]['xh'], yerr=pdi[p]['gammas'][g]['xhe'],fmt='o',c=cyan,ms=msp,elinewidth=elwp,ecolor=ec,capsize=csp) #posteriors with errors        
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_xticks(np.arange(0,5))
        if p == 'Bm2': ax.set_ylim(-0.5,2.5)
    plt.savefig(f'invP_params_gam{str(g).replace(".","")}.png')
    plt.close()
      
    #calculate fluxes and errors
    for f in flxd.keys():
        #get what parameter, tracer, and order each flux contains
        if '_' in f: #if not Psdot
            p,twordr = f.split('_')
            t = twordr[:2] #requres that tracer be designated by first 2 characters of twordr
            ordr = flxd[f]['order']
        fxh, fxhe = np.zeros(n), np.zeros(n) #calculate estimates and errors 
        if 'dz' in f: #if sinking flux divergence term, more complicated
            for i in np.arange(0,n):
                dzi = dz if i != 0 else h
                l = lmatch(i)
                pwi = "_".join([p,l])
                twi = "_".join([t,str(i)])
                w, Pi = sym.symbols(f'{pwi} {twi}')
                if i == 0: #mixed layer    
                    y = w*Pi/h
                    fxh[i], fxhe[i] = symfunceval(y)
                elif (i == 1 or i == 2): #first two points below ML
                    twip1, twim1 = "_".join([t,str(i+1)]), "_".join([t,str(i-1)])
                    Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                    y = w*(Pip1-Pim1)/(2*dz) #calculate flux estimate
                    fxh[i], fxhe[i]  = symfunceval(y)
                else: #all other depths
                    twim1, twim2 = "_".join([t,str(i-1)]), "_".join([t,str(i-2)])
                    Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                    y = w*(3*Pi-4*Pim1+Pim2)/(2*dz) #calculate flux estimate
                    fxh[i], fxhe[i]  = symfunceval(y)
        else: #all other terms that are not sinking flux divergence
            for i in np.arange(0,n):
                dzi = dz if i != 0 else h
                if f == 'Psdot': #special case
                    gh, lp = sym.symbols('Gh Lp')
                    y = gh*sym.exp(-(zml[i]-h)/lp)
                else:
                    l = lmatch(i)                
                    pwi = "_".join([p,l])
                    twi = "_".join([t,str(i)]) 
                    pa, tr = sym.symbols(f'{pwi} {twi}')
                    y = pa*tr**ordr 
                fxh[i], fxhe[i] = symfunceval(y)
        flxd[f]['gammas'][g]['xh'], flxd[f]['gammas'][g]['xhe'] = fxh, fxhe 
    
    #plot fluxes
    for i, pr in enumerate(flxpairs):
        fig, ax = plt.subplots(1,1) #P figures
        ax.invert_yaxis()
        if ('w' in pr[0]) and ('dz' not in pr[0]):
            ax.set_xlabel('Flux $[mmol/(m^2 \cdot d)]$')
        else: ax.set_xlabel('Vol. Flux $[mmol/(m^3 \cdot d)]$')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(top=0,bottom=zmax+dz)
        c1, c2, c3, c4 = navy, teal, red, purple
        ax.errorbar(flxd[pr[0]]['gammas'][g]['xh'], zml, fmt='o', xerr=flxd[pr[0]]['gammas'][g]['xhe'], ecolor=c1, elinewidth=elw, c=c1, ms=ms, capsize=cs, lw=lw, label=flxnames[pr[0]], fillstyle='none')
        ax.axhline(bnd,c='k',ls='--',lw=lw/2)
        if len(pr) > 1: #if it's actually a pair
            ax.errorbar(flxd[pr[1]]['gammas'][g]['xh'], zml, fmt='o', xerr=flxd[pr[1]]['gammas'][g]['xhe'], ecolor=c2, elinewidth=elw, c=c2, ms=ms, capsize=cs, lw=lw, label=flxnames[pr[1]], fillstyle='none')
        ax.legend()
        plt.savefig(f'invP_flux{i+1}_gam{str(g).replace(".","")}.png')
        plt.close()
    
    iflxcalc(iflxs,depthranges) #calculate integrated fluxes and timescales
    inventory(depthranges) #calculate tracer inventory and integrated residuals

    #Print integrated fluxes and residuals for each depth range
    with open ('invP_out.txt','a') as file:
        print(f'---------- Gamma = {g} ----------', file=file)
        for dr in depthranges:
            rstr = "_".join([str(dr[0]),str(dr[1])])
            print(f'--- Depth range: {dr} ---', file=file)
            for f in iflxs:
                print(f"{f}: {flxd[f]['gammas'][g]['iflx'][rstr][0]:.3f} ± {flxd[f]['gammas'][g]['iflx'][rstr][1]:.3f}", file=file)
            print(f'Ps Residuals: {td["Ps"]["gammas"][g]["ires"][rstr]:.3f} \nPl Residuals: {td["Pl"]["gammas"][g]["ires"][rstr]:.3f}',file=file)

with open ('invP_out.txt','a') as file:
    print(f'--- {time.time() - start_time} seconds ---',file=file)

with open('invP_savedvars.pkl', 'wb') as file:
    pickle.dump((flxd,td,pdi,combodf_s,ac_params,pdf_params),file)

#comparison of integrated fluxes and integrals
bw = 0.15
c1 = [red, green, blue, purple, cyan]
c2 = [red, green, blue, purple, cyan, orange, teal, navy]

barsF = {i:{g:{x:{} for x in ['xh','xhe']} for g in gammas} for i in dr_str}
barsR = {i:{g:{x:{} for x in ['xh','xhe']} for g in gammas} for i in dr_str}
for iv in dr_str:
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig1.suptitle(iv), fig2.suptitle(iv)
    for i, g in enumerate(gammas):
        barsF[iv][g]['xh'] = [flxd[f]['gammas'][g]['iflx'][iv][0] for f in iflxs]
        barsF[iv][g]['xhe'] = [flxd[f]['gammas'][g]['iflx'][iv][1] for f in iflxs]
        barsR[iv][g]['xh'] = [td[t]['gammas'][g]['ires'][iv] for t in tracers]
        if i == 0: 
            r1 = np.arange(len(barsF[iv][g]['xh']))
            r2 = np.arange(len(barsR[iv][g]['xh']))
        ax1.bar(r1, barsF[iv][g]['xh'], width=bw, color=c1[i], edgecolor='k', yerr=barsF[iv][g]['xhe'], capsize=3, label=g)
        ax2.bar(r2, barsR[iv][g]['xh'], width=bw, color=c1[i], edgecolor='k', label=g)
        r1, r2 = [x + bw for x in r1], [x + bw for x in r2]
    ax1.set_xticks(list(map(lambda n: n-bw*len(iflxs)/2,r1)))
    ax1.set_xticklabels([flxd[f]['name'] for f in iflxs])
    ax1.set_ylabel('Integrated Flux (mmol/m2/d)')
    ax1.legend()
    ax2.set_xticks(list(map(lambda n: n-bw*len(iflxs)/2,r2)))
    ax2.set_xticklabels(tracers)
    ax2.set_ylabel('Integrated Residuals (mmol/m2/d)')
    ax2.legend()
    for fig in (fig1,fig2):
        plt.figure(fig.number)
        suf = 'iflxs' if fig == fig1 else 'iresids'
        ivlabel = iv.replace('.','p')
        plt.savefig(f'invP_{suf}_{ivlabel}.png')
        plt.close(fig)

#plots of relative error of rate parameters as a function of gamma
fig, ax = plt.subplots(1,1)
for i,p in enumerate(params):
    if pdi[p]['dv']:
        for l in layers:
            if l == 'A': m, ls = '^', '--'
            else: m, ls = 'o', ':'
            relativeerror = [pdi[p]['gammas'][g]['xhe'][l]/pdi[p]['gammas'][g]['xh'][l] for g in gammas]
            ax.plot(gammas, relativeerror, m, c=c2[i], label=f'{p}_{l}', fillstyle='none', ls=ls)
    else:
        relativeprcsn = [pdi[p]['gammas'][g]['xhe']/pdi[p]['gammas'][g]['xh'] for g in gammas]
        label = p
        ax.plot(gammas, relativeprcsn, 'x', c=c2[i], label=p, ls='-.')
#ax.set_xscale('symlog', linthreshx=0.01)
ax.set_xscale('log')
ax.set_xticks(gammas)
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.legend(loc='lower center', bbox_to_anchor=(0.49, 0.95), ncol=6)
plt.savefig('invP_paramrelerror.png')
plt.close()
