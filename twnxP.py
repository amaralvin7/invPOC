#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 09 2020

@author: vamaral

"""
import numpy as np
import scipy.linalg as splinalg
import scipy.stats as sstats
import matplotlib.pyplot as plt
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

start_time = time.time()
plt.close('all')

#need for when running on remote server
sys.setrecursionlimit(10000)

"""
SETUP FOR GENERATING PSEUDODATA
"""

plt.close('all')

#colors
red, green, blue, purple, cyan, orange, teal, navy, olive = '#e6194B', '#3cb44b', '#4363d8', '#911eb4', '#42d4f4', '#f58231', '#469990', '#000075', '#808000'

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

#first order aggregation estimate from Murnane 1994
B2 = 0.8/dpy

#assign df colums to variables
zs = df.Depth
Ps_sd = df.SSF_sd/mm
Ps_se = df.SSF_se/mm
Pl_sd = df.LSF_sd/mm
Pl_se = df.LSF_se/mm

gam = 0.01 #multiplier for weighting model errors

#param info
layers = ['A','B']
tracers = ['Ps','Pl']
params = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l', 'Gh', 'Lp']
params_dv = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l'] #depth-varying params
params_dc = ['Gh', 'Lp'] #depth-constant params
pdik = ['tset','o','oe','xh','xhe']
#make a dictionaries to store param info
pdi_dv = {param:{lay:{k:{} for k in pdik} for lay in layers} for param in params_dv} 
pdi_dc = {param:{k:{} for k in pdik} for param in params_dc}
pdi = {**pdi_dv,**pdi_dc} #merge the two dictionaries
#add a key for each parameter designating if it is depth-varying or constant
for k in pdi.keys():
    if k in params_dv: pdi[k]['dv'] = 1
    else: pdi[k]['dv'] = 0

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
#target values used to generate pseudo-data
p_tgt = {'ws':{'A':1.8,'B':3.7}, 
        'wl':{'A':12,'B':28}, 
        'B2p':{'A':0.02,'B':0.032}, 
        'Bm2':{'A':0.9,'B':0.03}, 
        'Bm1s':{'A':0.12,'B':0.03}, 
        'Bm1l':{'A':0.02,'B':0.2}, 
        'Gh':0.38, 
        'Lp':16}

#have a single layer seperation
bnd = 112.5

#update entries in pd
for p in pdi.keys():
    if pdi[p]['dv'] == 1: #depth-varying params
        for l in layers:
            pdi[p][l]['tset'] = p_tset[p]
            pdi[p][l]['o'] = p_o[p]
            pdi[p][l]['oe'] = p_oe[p]
            pdi[p][l]['t'] = p_tgt[p][l]
    else:
        pdi[p]['tset'] = p_tset[p]
        pdi[p]['o'] = p_o[p]
        pdi[p]['oe'] = p_oe[p]
        pdi[p]['t'] = p_tgt[p]

#some parameters for plotting later
ms, lw, elw, cs = 3, 1, 0.5, 2

#particle prouction
Ghz = pdi['Gh']['t']*np.exp(-(zml-h)/pdi['Lp']['t'])

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

#LU decomposition. Takes matrix A and b as input, returns x, solution vector
#Also returns bp,  to verify solution 
def LUdecomp(A,b):
    L, U = splinalg.lu(A, permute_l=True)
    y = np.linalg.solve(L,b)
    x = np.linalg.solve(U,y)
    #make sure we can recover b, in this case bp
    bp = np.matmul(A,x)
    return [x,bp]

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

cscheme = plt.cm.jet
def lsqplotsf(model,x,y,z,tracer,logscale): #make a least squares fit and plot the results, using smf offline
    #open fig
    fig, ax =  plt.subplots(1,1)
    #colorbar stuff
    normfac = mpl.colors.Normalize(z.min(),z.max())
    axcb = mpl.colorbar.make_axes(ax)[0]
    cbar = mpl.colorbar.ColorbarBase(axcb, cmap=cscheme, norm=normfac)
    cbar.set_label('Depth (m)\n', rotation=270, labelpad = 14)
    #plot data
    ax.scatter(x, y, norm=normfac, cmap=cscheme, c = z, s=ms)
    ax.set_ylabel(f'{tracer} $(mmol/m^3)$')
    ax.set_xlabel('$c_{p}$ $(m^{-1})$')
    xp = np.arange(0.01,0.14,0.0001)
    c = model.params #extract coefficients, where c[0] is the intercept
    r2 = model.rsquared
    if logscale==True: 
        ax.set_yscale('log'), ax.set_xscale('log') 
        fit = [c[0] + c[1]*np.log(xi) for xi in xp]
    else: fit = [c[0] + c[1]*xi for xi in xp]
    ax.plot(xp, fit, '--', c = 'k', lw=lw)
    #ax.set_title(f' c0 = {c[0]:.2f}, c1 = {c[1]:.2f}, R2 = {r2:.2f}')
    ax.set_title(f'$R^2$ = {r2:.2f}')
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
            if svar in td.keys(): xv[i] = td[svar]['xh'][int(di)]
            else: xv[i] = pdi[svar][str(di)]['xh']
        else: xv[i] = pdi[str(v)]['xh'] #if it's a depth-constant variable
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

#given a depth index and function, evaluate the function and fill in corresponding jacobian values
def Fnf(y,i,di,ln=True):
    f_x, f_xv, f_xi = Fnf_helper(y,i,di,ln)
    f[i] = sym.lambdify(f_x,y)(*f_xv)
    for j, v in enumerate(f_x):
        if ln == True: dy = y.diff(v)*v #multiplied by v becuase it's atually dy/d(lnv) = v*dy/dv
        else: dy = y.diff(v)
        F_x, F_xv, F_xi = Fnf_helper(dy,i,di,ln)
        F[i,int(f_xi[j])] = sym.lambdify(F_x,dy)(*F_xv)

def Fnf_helper(y,i,di,ln):
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
        xv[j] = np.exp(xk[iSV]) if ln == True else xk[iSV]
        xi[j] = iSV
    return x, xv, xi
         
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
ax_ls = lsqplotsf(model, x, y, zdep, '$P_T$', logscale=True)

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

#make a matrix whose diagonals are MSE of residuals
MSE_fit = model.mse_resid
Cf_addPt = np.diag(np.ones(n)*MSE_fit)

kdz_A, ac_A, l_int_A, L_A, lfit_A, l_r2_A = acdrange((h,bnd-dz/2))
kdz_B, ac_B, l_int_B, L_B, lfit_B, l_r2_B = acdrange((bnd+dz/2,zmax))

"""
#LINEAR NUMERICAL SOLUTIONS (need to solve to get priors for nonlinear solutions)
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
    if pdi[p]['dv'] == 1: #depth-varying params
        for l in layers:
            params_o = np.append(params_o,pdi[p][l]['o'])
            params_o_e = np.append(params_o_e,pdi[p][l]['oe'])
            p_toidx.append('_'.join([p,l]))
    else:
        params_o = np.append(params_o,pdi[p]['o'])
        params_o_e = np.append(params_o_e,pdi[p]['oe'])
        p_toidx.append(f'{p}')
nparams = len(p_toidx) #number of total params (one in each layer for depth-varying)
        
#add params to the vidx's
vidx_allP = vidxP+vidxPt #includes everything in F and f
vidxSV = vidxP+p_toidx #only state variables

#some useful matrix dimensions
N = len(vidxSV)
P = N-len(params_o) #dimension of everything minus params (# of model equations)
M = len(vidx_allP) #dimension that includes all three P size fractions (for F and f)

#Construct A matrix and b vector
A, b = np.zeros((P, P)), np.zeros(P)
#loop through and edit rows of AP
for i in np.arange(0,P):
    #what tracer and depth are we on?
    t,d = vidxP[i].split('_')
    d = int(d)
    l = lmatch(d) #what layer does this depth index correspond to?
    iPsi = vidxP.index(f'Ps_{str(d)}')
    iPli = vidxP.index(f'Pl_{str(d)}')
    #pick up all values of rate parameters
    wsi, wli, Bm2i, Bm1si, Bm1li, Ghi = pdi['ws'][l]['t'], pdi['wl'][l]['t'], \
        pdi['Bm2'][l]['t'], pdi['Bm1s'][l]['t'], pdi['Bm1l'][l]['t'], Ghz[int(d)]
    #POC, SSF
    if t == 'Ps':
        A[i,iPli] = -Bm2i
        b[i] = Ghi
        if d == 0: #ML
            A[i,iPsi] = (wsi/h)+Bm1si+B2
        elif (d == 1 or d == 2): #first or second grid point
            iPsip1 = vidxP.index(f'Ps_{str(d+1)}')
            iPsim1 = vidxP.index(f'Ps_{str(d-1)}')
            A[i,iPsip1] = wsi/(2*dz)
            A[i,iPsi] = Bm1si+B2
            A[i,iPsim1] = -wsi/(2*dz)
        else:
            iPsim1 = vidxP.index(f'Ps_{str(d-1)}')
            iPsim2 = vidxP.index(f'Ps_{str(d-2)}')
            A[i,iPsi] = (3*wsi)/(2*dz)+Bm1si+B2
            A[i,iPsim1] = (-2*wsi)/dz
            A[i,iPsim2] =wsi/(2*dz)
    #POC, LSF
    else:       
        A[i,iPsi] = -B2
        if d == 0:
            A[i,iPli] = (wli/h)+Bm1li+Bm2i
        elif (d == 1 or d == 2): #first or second grid point
            iPlip1 = vidxP.index(f'Pl_{str(d+1)}')
            iPlim1 = vidxP.index(f'Pl_{str(d-1)}')
            A[i,iPlip1] = wli/(2*dz)
            A[i,iPli] = Bm2i+Bm1li
            A[i,iPlim1] = -wli/(2*dz)
        else:
            iPlim1 = vidxP.index(f'Pl_{str(d-1)}')
            iPlim2 = vidxP.index(f'Pl_{str(d-2)}')           
            A[i,iPli] = (3*wli)/(2*dz)+Bm2i+Bm1li
            A[i,iPlim1] = (-2*wli)/dz
            A[i,iPlim2] = wli/(2*dz)

#calculate the condition number for A
A_cond = np.linalg.cond(A)
#Find the solution x, and verify the solution looks ok be recovering b
x_numlin, bp_numlin = LUdecomp(A, b)
#assign numerical solutions to variables
Ps_numlin, Pl_numlin = [vsli(x_numlin,vidxSV.index(f'{t}_0')) for t in tracers]

"""
#NONLINEAR NUMERICAL SOLUTIONS (need to solve to get priors for inverse method) 
"""

#initialize matrices and vectors in Fk*xkp1 = Fk*xk-fk+b
F = np.zeros((P, P))
b, f = np.zeros(P), np.zeros(P)
xo = x_numlin
xk = xo
xkp1 = np.ones(P) #initialize this to some dummy value 
k = 0 #keep a counter for how many steps it takes
pdelt = 0.0001 #allowable percent change in each state element for convergence
conv_ev = np.empty(0) # keep track of evolution of convergence

#define all possible symbolic variables
svarnames = 'Ps_0 Ps_1 Ps_-1 Ps_-2 \
    Pl_0 Pl_1 Pl_-1 Pl_-2'
Psi, Psip1, Psim1, Psim2, \
    Pli, Plip1, Plim1, Plim2  = sym.symbols(svarnames)
#iterative solution
while True:
    for i in np.arange(0,P):
        #what tracer and depth are we on?
        t,d = vidxP[i].split('_')
        d = int(d)
        #pick up all values of rate parameters
        wsi, wli, Bm2i, B2pi, Bm1si, Bm1li, Ghi = pdi['ws'][l]['t'], pdi['wl'][l]['t'], \
            pdi['Bm2'][l]['t'], pdi['B2p'][l]['t'], pdi['Bm1s'][l]['t'], pdi['Bm1l'][l]['t'], Ghz[int(d)]
        #POC, SSF
        if t == 'Ps':
            b[i] = -Ghi
            if d == 0: #mixed layer
                eq = Bm2i*Pli-(wsi/h+Bm1si+B2pi*Psi)*Psi
            elif (d == 1 or d == 2): #first or second grid point
                eq = (Bm2i*Pli)-(Bm1si+B2pi*Psi)*Psi-wsi/(2*dz)*(Psip1-Psim1)
            else: #everywhere else
                eq = (Bm2i*Pli)-(Bm1si+B2pi*Psi)*Psi-wsi/(2*dz)*(3*Psi-4*Psim1+Psim2)
        #POC, LSF
        elif t == 'Pl':
            if d == 0:
                eq = (B2pi*(Psi)**2)-(wli/h+Bm2i+Bm1li)*Pli
            elif (d == 1 or d == 2):
                eq = B2pi*Psi**2-(Bm2i+Bm1li)*Pli-wli/(2*dz)*(Plip1-Plim1)
            else:
                eq = B2pi*Psi**2-(Bm2i+Bm1li)*Pli-wli/(2*dz)*(3*Pli-4*Plim1+Plim2)
        Fnf(eq,i,d,ln=False)
    xkp1 = np.linalg.solve(F,(np.matmul(F,xk)-f+b))
    #convergence criteria based on Murnane 94
    maxchange = np.max(np.abs((xkp1-xk)/xk))
    if maxchange < pdelt or k > 2000: break
    conv_ev = np.append(conv_ev,maxchange)    
    k += 1
    xk = xkp1
#assign numerical solutions to variables
Ps_numnl, Pl_numnl = [vsli(xkp1,vidxSV.index(f'{t}_0')) for t in tracers]
#generate Pt pseudodata (with noise)
Pt_numnl = np.random.normal(Ps_numnl+Pl_numnl,np.sqrt(MSE_fit))

"""
#INVERSE METHOD (P)
"""
#define sampling depths, and find indices that they occur at
sorter = np.argsort(zml)
zsi = sorter[np.searchsorted(zml, zs, sorter=sorter)]

#assign observation vectors and data errors
Ps_mean = Ps_numnl[zsi]
Pl_mean = Pl_numnl[zsi]

#make a dictionary for tracer params for each layer. could be more cleverly coded
tracers = ['Ps','Pl']
oi = {lay:{t:{} for t in tracers} for lay in layers}
oi['A']['Ps']['y'],oi['B']['Ps']['y'] = Ps_mean[0:3],Ps_mean[3:] #mean POC
oi['A']['Pl']['y'],oi['B']['Pl']['y'] = Pl_mean[0:3],Pl_mean[3:]
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
td_keys = ['si','x','xerr','y'] 
td = {t:dict.fromkeys(td_keys) for t in tracers}
for tra in tracers:
    for v in td[tra].keys():
        if v == 'si': #what index in vidxSV does each tracer start at?
            td[tra]['si'] = vidxSV.index(tra+'_0') 
            continue
        #all values that need to be concatenated between layers
        td[tra][v] = np.concatenate((oi['A'][tra][v],oi['B'][tra][v]))
        #catch the prior estiamtes and collect them in tpri
        if v == 'x': tpri = np.concatenate((tpri,td[tra][v]))
              
#combine xo's to form one xo, normalize xo and take the ln
xo = np.concatenate((tpri,params_o))
xoln = np.log(xo)

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

#construct Cf
Cf_noPt = np.zeros((P,P))
Cfd_noPt = np.ones(2*n)*pdi['Gh']['o']**2 #Cf from particle production
Cf_noPt = np.diag(Cfd_noPt)*gam
Cf = splinalg.block_diag(Cf_noPt,Cf_addPt)

#initialize the iterative loop
F = np.zeros((M, N))
f = np.zeros(M)
xk = xoln
xkp1 = np.ones(N) #initialize this to some dummy value 
k = 0 #keep a counter for how many steps it takes
conv_ev = np.empty(0) # keep track of evolution of convergence
cost_ev = np.empty(0) #keep track of evolution of the cost function, j
pdelt = 0.0001 #allowable percent change in each state element for convergence

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
        #Total POC
        else:
            Pti = Pt_numnl[vidxPt.index(f'Pt_{d}')] #value of Pt at gridpoint d
            eq = Psi + Pli - Pti
        Fnf(eq,i,d)
    FCoFT = np.matmul(np.matmul(F,Coln),F.T)
    FCoFT_cond = np.linalg.cond(FCoFT)
    FCFCinv = np.matmul(FCoFT,np.linalg.inv(FCoFT))
    FC_check = np.sum(FCFCinv-np.identity(M))
    B = np.matmul(np.matmul(Coln,F.T),np.linalg.inv(FCoFT+Cf))
    xkp1 = xoln + np.matmul(B,np.matmul(F,xk-xoln)-f)
    #convergence criteria based on Murnane 94
    maxchange = np.max(np.abs((xkp1-xk)/xk))
    conv_ev = np.append(conv_ev,maxchange)
    if gam == 0: cost = np.matmul(np.matmul((xk-xoln).T,np.linalg.inv(Coln)),(xk-xoln))
    else: cost = np.matmul(np.matmul((xk-xoln).T,np.linalg.inv(Coln)),(xk-xoln))+np.matmul(np.matmul(f.T,np.linalg.inv(Cf)),f)
    cost_ev = np.append(cost_ev,cost)
    if maxchange < pdelt or k > 2000: break
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
         
#get estimates, errors, and residuals for tracers
for t in td.keys():
    td[t]['xh'] = vsli(xhmean,td[t]['si'])
    td[t]['xhe'] = vsli(xhe,td[t]['si'])
    
#get model residuals from posterior estimates (from means)
td['Ps']['n'], td['Ps']['nm'], td['Ps']['nma'], td['Pl']['n'], td['Pl']['nm'], td['Pl']['nma'] = modresi(xhmean) 

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
pdfx_Ps_m = np.mean(vsli(pdfx,td['Ps']['si']))
pdfx_Pl_m = np.mean(vsli(pdfx,td['Pl']['si']))
pdfx_Ps_ma = np.mean(np.absolute(vsli(pdfx,td['Ps']['si'])))
pdfx_Pl_ma = np.mean(np.absolute(vsli(pdfx,td['Pl']['si'])))

#comparison of model residuals (posteriors)
nP = np.concatenate((td['Ps']['n'],td['Pl']['n']))
n_sd = np.sqrt(np.diag(Cf_noPt))
pdfn = nP/n_sd
pdfn_Ps_m = np.mean(vsli(pdfn,td['Ps']['si']))
pdfn_Pl_m = np.mean(vsli(pdfn,td['Pl']['si']))
pdfn_Ps_ma = np.mean(np.absolute(vsli(pdfn,td['Ps']['si'])))
pdfn_Pl_ma = np.mean(np.absolute(vsli(pdfn,td['Pl']['si'])))

#PDFs
fig, [ax1,ax2] = plt.subplots(1,2,tight_layout=True)
fig.subplots_adjust(wspace=0.5)
ax1.set_ylabel('P',size=16)
ax1.set_xlabel(r'$\frac{\^x-x_{o,i}}{\sigma_{o,i}}$',size=16)
ax1.hist(pdfx,density=True,bins=20,color=blue)
ax2.hist(pdfn,density=True,bins=20,color=blue)
ax2.set_xlabel(r'$\frac{n^{k+1}_{i}}{\sigma_{n^{k+1}_{i}}}$',size=16)

#plot gaussians, show legend
ax1.plot(xg,yg_pdf,c=red), ax2.plot(xg,yg_pdf,c=red)

#CDFs
fig, [ax1,ax2] = plt.subplots(1,2,tight_layout=True)
fig.subplots_adjust(wspace=0.5)
ax1.set_ylabel('P',size=16)
ax1.set_xlabel(r'$\frac{\^x-x_{o,i}}{\sigma_{o,i}}$',size=16), ax2.set_xlabel(r'$\frac{n^{k+1}_{i}}{\sigma_{n^{k+1}_{i}}}$',size=16)
ax1.plot(xg,yg_cdf,c=red), ax2.plot(xg,yg_cdf,c=red) #plot gaussians
cdf_dfx, cdf_dfn = pd.DataFrame(), pd.DataFrame()
cdf_dfx['var_name'], cdf_dfn['var_name'] = vidxSV.copy(), vidxP.copy()
cdf_dfx['val'], cdf_dfn['val'] = pdfx.copy(), pdfn.copy() #add values that correspond to those
cdf_dfx['o_idx'], cdf_dfn['o_idx'] = cdf_dfx.index, cdf_dfn.index #copy original indices
cdf_dfxs, cdf_dfns = cdf_dfx.sort_values('val').copy(), cdf_dfn.sort_values('val').copy()
cdf_dfxs.reset_index(inplace=True), cdf_dfns.reset_index(inplace=True) #reset indices
x1,x2 = cdf_dfxs.val, cdf_dfns.val 
y1,y2 = np.arange(1,len(x1)+1)/len(x1), np.arange(1,len(x2)+1)/len(x2)
#plot estimate residuals, params as orange circles
for i, v in enumerate(x1):
    if 'P' not in cdf_dfxs.var_name[i]:
        marsize = 8
        ec = orange
        mar = 'o'
        fc = 'none'
    else:
        marsize = 4
        ec = blue
        mar = '.'
        fc = ec
    ax1.scatter(x1[i],y1[i],s=marsize,marker=mar,facecolors=fc,edgecolors=ec)
#plot posteriors model residuals
ax2.scatter(x2,y2,s=ms,marker='.',facecolors=blue,edgecolors=blue)
        

#model residual depth profiles (posteriors)
fig1, [axA,axB] = plt.subplots(1,2)
fig1.subplots_adjust(wspace=0.5)  
axA.invert_yaxis(), axB.invert_yaxis()
axA.set_xlabel('$n^{k+1}_{P_{S}}$ (mmol/m3/d)'), axB.set_xlabel('$n^{k+1}_{P_{L}}$ (mmol/m3/d)')
axA.set_ylabel('Depth (m)')
axA.set_ylim(top=0,bottom=zmax+dz), axB.set_ylim(top=0,bottom=zmax+dz)
axA.scatter(td['Ps']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
axB.scatter(td['Pl']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
axA.legend(), axB.legend()

#plot evolution of convergence
ms=3
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(conv_ev)), conv_ev, marker='o',ms=ms)
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',size=12)

#plot evolution of cost function
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(cost_ev)),cost_ev,marker='o',ms=ms)
ax.set_xlabel('k')
ax.set_ylabel('j')
ax.set_yscale('log')

#comparison plots
fig, [ax1,ax2,ax3] = plt.subplots(1,3) #P figures
fig.subplots_adjust(wspace=0.5)  
ax1.invert_yaxis(), ax2.invert_yaxis(), ax3.invert_yaxis()
ax1.set_xlabel('$P_{S}$ ($mmol/m^3$)'), ax2.set_xlabel('$P_{L}$ ($mmol/m^3$)'), ax3.set_xlabel('$P_{T}$ ($mmol/m^3$)')
ax1.set_ylabel('Depth (m)')
ax1.set_ylim(top=0,bottom=zmax+2*dz), ax2.set_ylim(top=0,bottom=zmax+2*dz), ax3.set_ylim(top=0,bottom=zmax+2*dz)
ax1.errorbar(td['Ps']['xh'], zml, fmt='o', xerr=td['Ps']['xhe'], ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='mean', fillstyle='none')
ax1.errorbar(Ps_mean, zs, fmt='^', xerr=Ps_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
ax1.errorbar(td['Ps']['x'], zml, fmt='o', xerr=td['Ps']['xerr'], ecolor=blue, elinewidth=elw, c=blue, ms=ms/2, capsize=cs, lw=lw, label='OI')  
ax1.legend()
ax2.errorbar(td['Pl']['xh'], zml, fmt='o', xerr=td['Pl']['xhe'], ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='mean', fillstyle='none')
ax2.errorbar(Pl_mean, zs, fmt='^', xerr=Pl_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
ax2.errorbar(td['Pl']['x'], zml, fmt='o', xerr=td['Pl']['xerr'], ecolor=blue, elinewidth=elw, c=blue, ms=ms/2, capsize=cs, lw=lw, label='OI')  
ax2.legend()
ax3.errorbar(Pt_xh, zml, fmt='o', xerr=Pt_xhe, ecolor=red, elinewidth=elw, c=red, ms=ms, capsize=cs, lw=lw, label='Inv', fillstyle='none')
ax3.errorbar(Pt_numnl, zml+1, fmt='o', xerr=np.ones(n)*np.sqrt(MSE_fit), ecolor=green, elinewidth=elw, c=green, ms=ms, capsize=cs, lw=lw, label='Data', fillstyle='none')
ax3.legend()
ax1.axhline(bnd,c='k',ls='--',lw=lw/2)
ax2.axhline(bnd,c='k',ls='--',lw=lw/2)
ax3.axhline(bnd,c='k',ls='--',lw=lw/2)

#extract posterior param estimates and errors
params_ests = xhmean[-nparams:]
params_errs = xhe[-nparams:]
for i,stri in enumerate(p_toidx):
    pest, perr = params_ests[i], params_errs[i]
    if '_' in stri: #depth-varying paramters
        p,l = stri.split('_')
        pdi[p][l]['xh'], pdi[p][l]['xhe'] = pest, perr
    else: pdi[stri]['xh'], pdi[stri]['xhe'] = pest, perr

#make a plot of parameter priors and posteriors
elwp, msp, csp, ec = 1, 9, 4, 'k'
fig, ([ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]) = plt.subplots(2,4)
fig.subplots_adjust(wspace=0.8, hspace=0.4)
axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax8]
for i, p in enumerate(pdi.keys()):
    ax = axs[i]
    if pdi[p]['dv'] == 1: #if param is depth-varying
        ax.set_title(pdi[p]['A']['tset'])
        ax.errorbar(1, pdi[p]['A']['o'], yerr=pdi[p]['A']['oe'],fmt='o',ms=msp,c=blue,label='$x_{o}$',elinewidth=elwp,ecolor=ec,capsize=csp) #priors with errors
        ax.scatter(2, pdi[p]['A']['t'],marker='+',s=msp*10,c=teal,label='$x_{T}$') #target value
        ax.errorbar(3, pdi[p]['A']['xh'], yerr=pdi[p]['A']['xhe'], fmt='o', c=teal, ms=msp, label='$x^{A}_{k+1}$', elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
        ax.scatter(4, pdi[p]['B']['t'],marker='+',s=msp*10,c=navy,label='$x_{T}$') #target value
        ax.errorbar(5, pdi[p]['B']['xh'], yerr=pdi[p]['B']['xhe'], fmt='o', c=navy, ms=msp, label='$x^{B}_{k+1}$', elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
    else: #if param is depth-constant
        ax.set_title(pdi[p]['tset'])
        ax.errorbar(2, pdi[p]['o'], yerr=pdi[p]['oe'],fmt='o',ms=msp,c=blue,label='$x_{o}$',elinewidth=elwp,ecolor=ec,capsize=csp) #priors with errors
        ax.scatter(3, pdi[p]['t'],marker='+',s=msp*10,c=cyan,label='$x_{T}$') #target value
        ax.errorbar(4, pdi[p]['xh'], yerr=pdi[p]['xhe'],fmt='o',c=cyan,ms=msp,label='$x_{k+1}$',elinewidth=elwp,ecolor=ec,capsize=csp) #posteriors with errors        
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_xticks(np.arange(0,7))
    
print(f'--- {time.time() - start_time} seconds ---')

plt.show()