#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 09 2020

@author: vamaral

-having flux at every depth and integrated fluxes with residence times in each 
layer calculated in the same loop takes too long if we also want to integrate
depth horizons that are not contained within each layer, so I removed
flux integral and residence time calculations from the flux calculation loop 

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
import time

start_time = time.time()

plt.close('all')

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
phi = 10**-13 #convergence criteria

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

gam = 0.01 #multiplier for weighting model errors

#param info
layers = ['A','B']
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
       'Gh':180/h/mm, #prior set to typical NPP shared data value (divided by h), error is 25% of that. mg/m2/d converted to mmol
       'Lp':28} #from NPP data, m
#prior errors
p_oe = {'ws':2, 
       'wl':15, 
       'B2p':0.5*mm/dpy, 
       'Bm2':1000/dpy, 
       'Bm1s':36/dpy, 
       'Bm1l':0.15, 
       'Gh':180/h/mm*0.25, 
       'Lp':28*0.5}
#depth layer boundaries
dA, dB = 115, zmax

#update entries in pd
for p in pdi.keys():
    if pdi[p]['dv'] == 1: #depth-varying params
        for l in layers:
            pdi[p][l]['tset'] = p_tset[p]
            pdi[p][l]['o'] = p_o[p]
            pdi[p][l]['oe'] = p_oe[p]
    else:
        pdi[p]['tset'] = p_tset[p]
        pdi[p]['o'] = p_o[p]
        pdi[p]['oe'] = p_oe[p]

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

#find index of required variable
#takes as input the species, depth, and variable index. Only returns a single index
def fidx1(s, d, idx):
    idxs = [x for x, st in enumerate(idx) if st == f'{s}_{d}']
    return idxs

#find index of required variable and corresponding value in the state vector
#takes as input the species, depth, and variable index. Only returns the index and value in the state vector
def fidx2(s, d, idx, sv):
    idx = [x for x, st in enumerate(idx) if st == f'{s}_{d}']
    val = sv[idx]
    return [idx,val]

#find index of required variable and corresponding value in the state vector. Also the corresponding prior value
#takes as input the species, depth, variable index, state vector, and prior vector. Returns idx #, and corresponding values in state and prior vectors
def fidx3(s, d, idx, sv, pv):
    idx = [x for x, st in enumerate(idx) if st == f'{s}_{d}']
    val = sv[idx]
    pri = pv[idx]
    return [idx,val,pri]

#like fidx3, but returns the exp of the state variable instead.
def fidx3e(s, d, idx, sv, pv):
    idx = [x for x, st in enumerate(idx) if st == f'{s}_{d}']
    val = np.exp(sv[idx])
    pri = pv[idx]
    return [idx,val,pri]

#LU decomposition. Takes matrix A and b as input, returns x, solution vector
#Also returns bp,  to verify solution 
def LUdecomp(A,b):
    L, U = splinalg.lu(A, permute_l=True)
    y = np.linalg.solve(L,b)
    x = np.linalg.solve(U,y)
    #make sure we can recover b, in this case bp
    bp = np.matmul(A,x)
    return [x,bp]

#some parameters for plotting later
ms, lw = 4, 1
elw = 0.5
cs = 2

#define a function to check convergence
def convergecheck(e, diff):
    epsilon = e
    abs_diff = diff
    diff_vector = e-diff
    diff_mag = np.linalg.norm(diff_vector)
    for i, val in enumerate(epsilon):
        #if any absolute difference is too large, method has not converged
        if (abs_diff[i] >= epsilon[i]) : return([False, diff_mag])
    return([True, diff_mag])

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
#shallower than 115m is 'A', 115 and deeper is 'B'
    d = zml[int(di)]
    if d < dA: layer = 'A'
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

#like findp_dvp, but also returns exponential of state variable
def findp_dv(pvec,svec,vi,param,layer):
    ip = vi.index('_'.join([param,layer]))
    pp = pvec[ip]
    ps = np.exp(svec[ip])
    return ip,pp,ps

#like findp_dvp, but also returns exponential of state variable
def findp_dc(pvec,svec,vi,param):
    ip = vi.index(param)
    pp = pvec[ip]
    ps = np.exp(svec[ip])
    return ip,pp,ps

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
                n_err[i] = (wsio/h+Bm1sio+B2pio*Psio)*Psio-Bm2io*Plio-Ghio
            elif (d == 1 or d == 2): #first or second grid point
                iPsip1,Psip1o = fidx2('Ps',d+1,vidxP,sv)
                iPsim1,Psim1o = fidx2('Ps',d-1,vidxP,sv)
                n_err[i] = (wsio/(2*dz))*Psip1o+(Bm1sio+B2pio*Psio)*Psio-(wsio/(2*dz))*Psim1o-(Bm2io*Plio)-Ghio*np.exp(-(zml[d]-h)/(Lpio))
            else: #everywhere else
                iPsim1,Psim1o = fidx2('Ps',d-1,vidxP,sv)
                iPsim2,Psim2o = fidx2('Ps',d-2,vidxP,sv)
                n_err[i] = ((3*wsio)/(2*dz)+Bm1sio+B2pio*Psio)*Psio-(2*wsio/dz)*Psim1o+(wsio/(2*dz))*Psim2o-(Bm2io*Plio)-Ghio*np.exp(-(zml[d]-h)/(Lpio))
        #POC, LSF
        else:
            if d == 0:
                n_err[i] = (wlio/h+Bm2io+Bm1lio)*Plio-(B2pio*(Psio)**2)                
            elif (d == 1 or d == 2):
                iPlip1,Plip1o = fidx2('Pl',d+1,vidxP,sv)    
                iPlim1,Plim1o = fidx2('Pl',d-1,vidxP,sv)
                n_err[i] = wlio/(2*dz)*Plip1o+(Bm2io+Bm1lio)*Plio-wlio/(2*dz)*Plim1o-(B2pio*(Psio)**2)
            else:
                iPlim1,Plim1o = fidx2('Pl',d-1,vidxP,sv)
                iPlim2,Plim2o = fidx2('Pl',d-2,vidxP,sv)
                n_err[i] = ((3*wlio)/(2*dz)+Bm2io+Bm1lio)*Plio-(2*wlio/dz*Plim1o)+(wlio/(2*dz))*Plim2o-(B2pio*(Psio)**2)
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

#calculate prediction intervals for a given (stats)model
def p_int(model, alpha, xstar, x):
    a = alpha #significance level
    m = model
    df = m.df_resid
    t = sstats.t.ppf(1-a/2,df) #critical t-value
    #print(t)
    sdr = np.std(m.resid,ddof=2) #stdev of residuals, must be equal to k+1 where k is # of predictors
    #sdr = np.sqrt(m.mse_resid) #equivalent to the statement above
    n = m.nobs #number of observations
    xbar = np.mean(x)
    pint = t*sdr*np.sqrt(1+1/n+(xstar-xbar)**2/np.sum((x-xbar)**2))
    return pint

#autocorrelation function for a specific depth range
def acdrange(drange):
    dmin, dmax = drange #min and max depths within the range
    dslice = zml[difind(dmin):difind(dmax)+1] #slice of zml corresponding to drange
    tslice = cppt['Pt_hat'].values[difind(dmin):difind(dmax)+1]#slice of tracer data array corresponding to drange, in this case Pt
    kn = np.ceil(len(dslice)/4) #number of lags, round up
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
#if cov==True, considers covariances
def flxep(y, cov):
    #print('stepped in, creating CVM')
    x = y.free_symbols #gets all (symbolic) variables in function
    nx = len(x) #number of variables
    xv = np.zeros(nx) #numerical values of symbolic variables
    dy = [None]*nx #empty list to store derivatives
    cvms = cvmsli(CVM,x) #covariance matrix for all variables
    #print('cvm created, deriving')
    for i,v in enumerate(x): #for each symbolic variable
        dy[i] = y.diff(v) #calculate derivatives w.r.t x
        if "_" in str(v): #if the variable varies with depth
            svar, di = str(v).split('_') #what kind of state variable (tracer or param?)
            if svar in td.keys(): xv[i] = td[svar]['xh'][int(di)]
            else: xv[i] = pdi[svar][str(di)]['xh']
        else: xv[i] = pdi[str(v)]['xh']
    #print('derivations complete, building variance equation')
    u = 0 #initialize value to calculate uncertainties
    #iterate through entries of cvms and calculate relevant term
    for i, r in enumerate(cvms):
        for j, c in enumerate(r):
            if  i>j: continue #only take upper diagonal (including diagonal)
            elif i == j: u += (dy[i]**2)*cvms[i,j]
            else: 
                if cov == True: u += 2*dy[i]*dy[j]*cvms[i,j]
    #print(y), print(u)
    #print('equation built, evaluating')
    flx, unc = sym.lambdify(x,(y,u))(*xv) #asterisks unpacks array values into function arguments
    #print('Returned!')
    return (flx, np.sqrt(unc))

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
#print(model.summary())
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

#get prediction intervals on Pt ests
alph = 1-0.68
cppt['Pt_hat_pint'] = p_int(model,alph,meancp,x)

#runs test
runs_z, runs_p = smr.runstest_1samp(model.resid)
#fig, ax = plt.subplots(1,1)
#ax.scatter(np.arange(0,len(model.resid)),model.resid)
#ax.hlines(np.mean(model.resid),0,65,linestyles='dashed')
#fig.suptitle(f'z = {runs_z:.4f}, p = {runs_p:.4f}')

#make a matrix whose diagonals are square of prediction intervals
#Cf_addPt = np.diag(cppt['Pt_hat_pint']**2)
Cf_addPt = np.diag(np.ones(n)*model.mse_resid)

kdz_A, ac_A, l_int_A, L_A, lfit_A, l_r2_A = acdrange((h,dA))
kdz_B, ac_B, l_int_B, L_B, lfit_B, l_r2_B = acdrange((dA,zmax))
fig, ax = plt.subplots(1,1)
cA, cB = blue, green
ax.scatter(kdz_A,np.log(ac_A),label='A (EZ)',marker='o',color=cA) 
ax.scatter(kdz_B,np.log(ac_B),label='B (MZ)',marker='x',color=cB)
ax.plot(kdz_A,lfit_A,'--',lw=lw,color=cA), ax.plot(kdz_B,lfit_B,'--',lw=lw,color=cB)
ax.set_title(f'int_A = {l_int_A:.2f}, L_A = {L_A:.2f}, R2_A = {l_r2_A:.2f} \n int_B = {l_int_B:.2f}, L_B = {L_B:.2f}, R2_B = {l_r2_B:.2f}')
ax.set_xlabel('lags (m)')
ax.set_ylabel('ln($r_k$)')
ax.legend()


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

#make a dictionary for tracer params for each layer. could be more cleverly coded
tracers = ['Ps','Pl']
oi = {lay:{t:{} for t in tracers} for lay in layers}
oi['A']['Ps']['y'],oi['B']['Ps']['y'] = Ps_mean[0:3].values,Ps_mean[3:].values #mean POC
oi['A']['Pl']['y'],oi['B']['Pl']['y'] = Pl_mean[0:3].values,Pl_mean[3:].values
oi['A']['Pl']['sig_j'],oi['B']['Pl']['sig_j'] = Pl_sd[0:3].values,Pl_sd[3:].values #POC standard deviation
oi['A']['Ps']['sig_j'],oi['B']['Ps']['sig_j'] = Ps_sd[0:3].values,Ps_sd[3:].values
oi['A']['smpd'], oi['A']['grdd'] = zs[0:3].values, zml[difind(h):difind(dA)] #sample and grid depths, layer A
oi['B']['smpd'], oi['B']['grdd'] = zs[3:].values, zml[difind(dA):] #sample and grid depths, layer B
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
        oi[lay][tra]['Pn'] = np.zeros((noi,noi)) #normalized covariance matrix
        for i in np.arange(0,noi):
            for j in np.arange(0,noi):
                oi[lay][tra]['Pn'][i,j] = oi[lay][tra]['P'][i,j]/(oi[lay][tra]['x'][i]*oi[lay][tra]['x'][j])
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
xon = np.log(xo/xo)

#redefine some useful params
N = len(xo)
P = N-len(params_o) #dimension of everything minus params (# of model equations)
M = len(vidx_allP) #dimension that includes all three P size fractions (for F and f)

#construct Co as a (normalized) diagonal matrix, blocks for tracers and diagonals otherwise. Considers the ln()
Co_noln = np.zeros((N,N))
Co_noln[:P,:P] = splinalg.block_diag(oi['A']['Ps']['Pn'],oi['B']['Ps']['Pn'],oi['A']['Ps']['Pn'],oi['B']['Pl']['Pn'])
for i in np.arange(P,N):
    Co_noln[i,i] = (params_o_e[i-P]**2)/xo[i]**2
Co_noln_cond = np.linalg.cond(Co_noln)
Co = np.log(1+Co_noln)
Co_cond = np.linalg.cond(Co)
Co_neg = (Co<0).any() #checks if any element of Co is negative

#check that the Co inversion is accurate
CoCoinv = np.matmul(Co,np.linalg.inv(Co))
Co_check = np.sum(CoCoinv-np.identity(N))

#construct Cf
#initialize model equation errors and matrix
#for k in oi.keys(): oi[k]['n'] = np.zeros(n)
Cf_noPt = np.zeros((P,P))
#Ps_nvar_o, Pl_nvar_o = np.mean(oi['Ps']['n_o']**2)-(oi['Ps']['nm_o']**2), np.mean(oi['Pl']['n_o']**2)-(oi['Pl']['nm_o']**2)
#Cfd = np.concatenate((np.ones(n)*Ps_nvar_o,np.ones(n)*Pl_nvar_o)) #Cf by variance of residuals
Cfd_noPt = np.ones(2*n)*pdi['Gh']['o']**2 #Cf from particle production
Cf_noPt = np.diag(Cfd_noPt)*gam
Cf = splinalg.block_diag(Cf_noPt,Cf_addPt)

#initialize the iterative loop
F = np.zeros((M, N))
f = np.zeros(M)
xk = xon
eps = phi*(xo/xo)
xkp1 = xon + 1 #initialize this to some dummy value 
x_diff = np.abs(xkp1-xk) #initialize this to some dummy value
k = 0 #keep a counter for how many steps it takes
conv_ev = np.empty(0) # keep track of evolution of convergence
cost_ev = np.empty(0) #keep track of evolution of the cost function, j


#while k < 100:
while convergecheck(eps, x_diff)[0] == False:
    #depth-independent params
    iGhi,Ghio,Ghi = findp_dc(xo,xk,vidxSV,'Gh')
    iLpi,Lpio,Lpi = findp_dc(xo,xk,vidxSV,'Lp')
    for i, r in enumerate(F):
        #what tracer and depth are we on?
        t,d = vidx_allP[i].split('_')
        d = int(d)
        l = lmatch(d) #what layer does this depth index correspond to?
        iPsi,Psi,Psio = fidx3e('Ps',d,vidx_allP,xk,xo)
        iPli,Pli,Plio = fidx3e('Pl',d,vidx_allP,xk,xo)
        #depth-dependent parameters
        iwsi,wsio,wsi = findp_dv(xo,xk,vidxSV,'ws',l)
        iwli,wlio,wli = findp_dv(xo,xk,vidxSV,'wl',l)
        iB2pi,B2pio,B2pi = findp_dv(xo,xk,vidxSV,'B2p',l)
        iBm2i,Bm2io,Bm2i = findp_dv(xo,xk,vidxSV,'Bm2',l)
        iBm1si,Bm1sio,Bm1si = findp_dv(xo,xk,vidxSV,'Bm1s',l)
        iBm1li,Bm1lio,Bm1li = findp_dv(xo,xk,vidxSV,'Bm1l',l)
        #POC, SSF
        if t == 'Ps':
            F[i,iBm2i] = -Pli*Plio*Bm2io*Bm2i
            F[i,iBm1si] = Psi*Psio*Bm1sio*Bm1si
            F[i,iB2pi]  = B2pio*(Psi*Psio)**2*B2pi
            F[i,iPli] = -Bm2i*Bm2io*Plio*Pli
            if d == 0: #mixed layer
                F[i,iPsi] = (wsi*wsio/h+Bm1si*Bm1sio+2*B2pi*B2pio*Psi*Psio)*Psi*Psio
                F[i,iwsi] = Psi*Psio*wsio/h*wsi
                F[i,iGhi] = -Ghio*Ghi
                f[i] = (wsi*wsio/h+Bm1si*Bm1sio+B2pi*B2pio*Psi*Psio)*Psi*Psio-Bm2i*Bm2io*Pli*Plio-Ghi*Ghio
            elif (d == 1 or d == 2): #first or second grid point
                iPsip1,Psip1,Psip1o = fidx3e('Ps',d+1,vidx_allP,xk,xo)
                iPsim1,Psim1,Psim1o = fidx3e('Ps',d-1,vidx_allP,xk,xo)
                F[i,iPsip1] = wsi*wsio*Psip1o/(2*dz)*Psip1
                F[i,iPsi] = (Bm1si*Bm1sio+2*B2pi*B2pio*Psi*Psio)*Psi*Psio
                F[i,iPsim1] = -wsi*wsio*Psim1o/(2*dz)*Psim1
                F[i,iwsi] = wsio*(Psip1*Psip1o-Psim1*Psim1o)/(2*dz)*wsi
                F[i,iGhi] = -Ghio*np.exp(-(zml[d]-h)/(Lpi*Lpio))*Ghi
                F[i,iLpi] = -(Ghi*Ghio*(zml[d]-h))/(Lpio*Lpi)*np.exp(-(zml[d]-h)/(Lpio*Lpi))
                f[i] = (wsi*wsio/(2*dz))*Psip1*Psip1o+(Bm1si*Bm1sio+B2pi*B2pio*Psi*Psio)*Psi*Psio-(wsi*wsio/(2*dz))*Psim1*Psim1o-(Bm2i*Bm2io*Pli*Plio)-Ghi*Ghio*np.exp(-(zml[d]-h)/(Lpi*Lpio))
            else: #everywhere else
                iPsim1,Psim1,Psim1o = fidx3e('Ps',d-1,vidx_allP,xk,xo)
                iPsim2,Psim2,Psim2o = fidx3e('Ps',d-2,vidx_allP,xk,xo)
                F[i,iPsi] = ((3*wsi*wsio)/(2*dz)+Bm1si*Bm1sio+2*B2pi*B2pio*Psi*Psio)*Psi*Psio
                F[i,iPsim1] = -2*wsi*wsio*Psim1o/dz*Psim1
                F[i,iPsim2] = wsi*wsio*Psim2o/(2*dz)*Psim2
                F[i,iwsi] = wsio*(Psim2*Psim2o-4*Psim1*Psim1o+3*Psi*Psio)/(2*dz)*wsi
                F[i,iGhi] = -Ghio*np.exp(-(zml[d]-h)/(Lpi*Lpio))*Ghi
                F[i,iLpi] = -(Ghi*Ghio*(zml[d]-h))/(Lpio*Lpi)*np.exp(-(zml[d]-h)/(Lpio*Lpi))
                f[i] = ((3*wsi*wsio)/(2*dz)+Bm1si*Bm1sio+B2pi*B2pio*Psi*Psio)*Psi*Psio-(2*wsi*wsio/dz)*Psim1*Psim1o+(wsi*wsio/(2*dz))*Psim2*Psim2o-(Bm2i*Bm2io*Pli*Plio)-Ghi*Ghio*np.exp(-(zml[d]-h)/(Lpi*Lpio))
        #POC, LSF
        elif t == 'Pl':
            F[i,iPsi] = -2*B2pi*B2pio*(Psi*Psio)**2
            F[i,iBm2i], F[i,iBm1li] = Pli*Plio*Bm2io*Bm2i, Pli*Plio*Bm1lio*Bm1li
            F[i,iB2pi] = -B2pio*(Psi*Psio)**2*B2pi
            if d == 0:
                F[i,iPli] = (wli*wlio/h+Bm2i*Bm2io+Bm1li*Bm1lio)*Plio*Pli
                F[i,iwli] = Pli*Plio*wlio/h*wli
                f[i] = (wli*wlio/h+Bm2i*Bm2io+Bm1li*Bm1lio)*Pli*Plio-(B2pi*B2pio*(Psi*Psio)**2)                
            elif (d == 1 or d == 2):
                iPlip1,Plip1,Plip1o = fidx3e('Pl',d+1,vidx_allP,xk,xo)    
                iPlim1,Plim1,Plim1o = fidx3e('Pl',d-1,vidx_allP,xk,xo)
                F[i,iPlip1] = wli*wlio*Plip1o/(2*dz)*Plip1
                F[i,iPli] = Plio*(Bm2i*Bm2io+Bm1li*Bm1lio)*Pli
                F[i,iPlim1] = -wli*wlio*Plim1o/(2*dz)*Plim1
                F[i,iwli] = wlio*(Plip1*Plip1o-Plim1*Plim1o)/(2*dz)*wli
                f[i] = wli*wlio/(2*dz)*Plip1*Plip1o+(Bm2i*Bm2io+Bm1li*Bm1lio)*Pli*Plio-wli*wlio/(2*dz)*Plim1*Plim1o-(B2pi*B2pio*(Psi*Psio)**2)
            else:
                iPlim1,Plim1,Plim1o = fidx3e('Pl',d-1,vidx_allP,xk,xo)
                iPlim2,Plim2,Plim2o = fidx3e('Pl',d-2,vidx_allP,xk,xo)
                F[i,iPli] = ((3*wli*wlio)/(2*dz)+Bm2i*Bm2io+Bm1li*Bm1lio)*Plio*Pli
                F[i,iPlim1] = -2*wli*wlio*Plim1o/dz*Plim1
                F[i,iPlim2] = wli*wlio*Plim2o/(2*dz)*Plim2
                F[i,iwli] = wlio*(Plim2*Plim2o-4*Plim1*Plim1o+3*Pli*Plio)/(2*dz)*wli
                f[i] = ((3*wli*wlio)/(2*dz)+Bm2i*Bm2io+Bm1li*Bm1lio)*Pli*Plio-(2*wli*wlio/dz*Plim1*Plim1o)+(wli*wlio/(2*dz))*Plim2*Plim2o-(B2pi*B2pio*(Psi*Psio)**2)
        #POC, Ps + Pl = Pt
        else:
            iPti,Pti = fidx2('Pt',d,vidxPt,cppt.Pt_hat)
            F[i,iPsi] = Psi*Psio
            F[i,iPli] = Pli*Plio
            f[i] = Psi*Psio+Pli*Plio-Pti
    FCoFT = np.matmul(np.matmul(F,Co),F.T)
    FCoFT_cond = np.linalg.cond(FCoFT)
    FCFCinv = np.matmul(FCoFT,np.linalg.inv(FCoFT))
    FC_check = np.sum(FCFCinv-np.identity(M))
    #print(FC_check)
    B = np.matmul(np.matmul(Co,F.T),np.linalg.inv(FCoFT+Cf))
    xkp1 = xon + np.matmul(B,np.matmul(F,xk-xon)-f)
    conv_ev = np.append(conv_ev,convergecheck(eps, x_diff)[1])
    if gam == 0: cost = np.matmul(np.matmul((xk-xon).T,np.linalg.inv(Co)),(xk-xon))
    else: cost = np.matmul(np.matmul((xk-xon).T,np.linalg.inv(Co)),(xk-xon))+np.matmul(np.matmul(f.T,np.linalg.inv(Cf)),f)
    cost_ev = np.append(cost_ev,cost)
    x_diff = np.abs((np.exp(xkp1)-np.exp(xk)))
    #print(np.linalg.norm(x_diff))
    k += 1
    xk = xkp1
#print(k)
#calculate posterior errors
I = np.identity(Co.shape[0])
CoFT = np.matmul(Co,F.T)
FCoFTpCfinv = np.linalg.inv(FCoFT+Cf)
C = I-np.matmul(np.matmul(CoFT,FCoFTpCfinv),F)
D = I-np.matmul(np.matmul(np.matmul(F.T,FCoFTpCfinv),F),Co)
Ckp1 = np.matmul(np.matmul(C,Co),D)

#expected value and variance of tracers AND params
EyP, VyP = xkp1, np.diag(Ckp1)
#recover dimensional values of median, mean, mode, standard deviation
xhmed = np.exp(EyP)*xo
xhmod = np.exp(EyP-VyP)*xo
xhmean = np.exp(EyP+VyP/2)*xo
xhe = np.sqrt(np.exp(2*EyP+VyP)*(np.exp(VyP)-1))*xo

#calculate covariances of unlogged state variables 
CVM = np.zeros((N,N))
for i, row in enumerate(CVM):
    for j, unu in enumerate(row):
        mi, mj = EyP[i], EyP[j] #mu's (expected vals)
        vi, vj = VyP[i], VyP[j] #sig2's (variances)
        CVM[i,j] = np.exp(mi+mj)*np.exp((vi+vj)/2)*(np.exp(Ckp1[i,j])-1)*xo[i]*xo[j]

#check that sqrt of diagonals of CVM are equal to xhe
CVM_xhe_check = np.sqrt(np.diag(CVM)) - xhe
         
#get estimates, errors, and residuals for tracers
for t in td.keys():
    td[t]['xh'] = vsli(xhmean,td[t]['si'])
    td[t]['xhe'] = vsli(xhe,td[t]['si'])
    
#get model residuals from posterior estimates (from means)
td['Ps']['n'], td['Ps']['nm'], td['Ps']['nma'], td['Pl']['n'], td['Pl']['nm'], td['Pl']['nma'] = modresi(xhmean) 

#propagating errors on Pt
Pt_xh = td['Ps']['xh'] + td['Pl']['xh'] 
Pt_xhe = np.sqrt((td['Ps']['xhe']**2)+(td['Pl']['xhe']**2))

#PDF and CDF calculations
xg = np.linspace(-2,2,100)
yg_pdf = sstats.norm.pdf(xg,0,1)
yg_cdf = sstats.norm.cdf(xg,0,1)

#comparison of estimates to priors (means don't include params, histogram does)
xdiff = xhmean-xo
x_osd = np.sqrt(np.diag(Co_noln))*xo
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
ax1.axvline(pdfx_Ps_m,c=purple,ls=':',label='Ps_m',lw=lw*2), ax1.axvline(pdfx_Ps_ma,c=green,ls=':',label='Ps_ma',lw=lw*2)
ax1.axvline(pdfx_Pl_m,c=purple,ls='--',label='Pl_m',lw=lw*2), ax1.axvline(pdfx_Pl_ma,c=green,ls='--',label='Pl_ma',lw=lw*2)
ax2.hist(pdfn,density=True,bins=20,color=blue)
ax2.axvline(pdfn_Ps_m,c=purple,ls=':',label='Ps_m',lw=lw*2), ax2.axvline(pdfn_Ps_ma,c=green,ls=':',label='Ps_ma',lw=lw*2)
ax2.axvline(pdfn_Pl_m,c=purple,ls='--',label='Pl_m',lw=lw*2), ax2.axvline(pdfn_Pl_ma,c=green,ls='--',label='Pl_ma',lw=lw*2)
ax2.set_xlabel(r'$\frac{n^{k+1}_{i}}{\sigma_{n^{k+1}_{i}}}$',size=16)

#plot gaussians, show legend
ax1.plot(xg,yg_pdf,c=red), ax2.plot(xg,yg_pdf,c=red)
ax1.legend(), ax2.legend()

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
        marsize = ms*2
        ec = orange
        mar = 'o'
        fc = 'none'
    else:
        marsize = ms
        ec = blue
        mar = '.'
        fc = ec
    ax1.scatter(x1[i],y1[i],s=marsize,marker=mar,facecolors=fc,edgecolors=ec)
#plot posteriors model residuals
ax2.scatter(x2,y2,s=ms,marker='.',facecolors=blue,edgecolors=blue)
        

#model residual depth profiles (ppsteriors)
fig1, [axA,axB] = plt.subplots(1,2)
fig1.subplots_adjust(wspace=0.5)  
axA.invert_yaxis(), axB.invert_yaxis()
axA.set_xlabel('$n^{k+1}_{P_{S}}$ (mmol/m3/d)'), axB.set_xlabel('$n^{k+1}_{P_{L}}$ (mmol/m3/d)')
axA.set_ylabel('Depth (m)')
axA.set_ylim(top=0,bottom=zmax+dz), axB.set_ylim(top=0,bottom=zmax+dz)
axA.scatter(td['Ps']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
axA.axvline(td['Ps']['nm'],c=purple,ls='--',label='mean'), axA.axvline(td['Ps']['nma'],c=green,ls=':',label='mean_a')
axB.scatter(td['Pl']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
axB.axvline(td['Pl']['nm'],c=purple,ls='--',label='mean'), axB.axvline(td['Pl']['nma'],c=green,ls=':',label='mean_a')
axA.legend(), axB.legend()

#plot evolution of convergence
ms=3
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(conv_ev)), np.log10(conv_ev),marker='o',ms=ms)
ax.set_xlabel('k')
ax.set_ylabel('$log(||\epsilon - |x_{i,k+1}-x_{i,k}| ||)$')

#plot evolution of cost function
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(cost_ev)),cost_ev,marker='o',ms=ms)
ax.set_xlabel('k')
ax.set_ylabel('j')

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
ax3.errorbar(cppt.Pt_hat, zml+1, fmt='o', xerr=np.ones(n)*np.sqrt(model.mse_resid), ecolor=green, elinewidth=elw, c=green, ms=ms, capsize=cs, lw=lw, label='Data', fillstyle='none')
ax3.legend()
ax1.axhline(dA-2.5,c='k',ls='--',lw=lw/2)
ax2.axhline(dA-2.5,c='k',ls='--',lw=lw/2)
ax3.axhline(dA-2.5,c='k',ls='--',lw=lw/2)

#just the pump data (for presentation)
fig, [ax1,ax2,ax3] = plt.subplots(1,3) #P figures
fig.subplots_adjust(wspace=0.3) 
ax1.invert_yaxis(), ax2.invert_yaxis(), ax3.invert_yaxis()
ax1.set_xlabel('$P_{S}$ ($mmol/m^3$)'), ax2.set_xlabel('$P_{L}$ ($mmol/m^3$)'), ax3.set_xlabel('$P_{T}$ ($mmol/m^3$)')
ax1.set_ylabel('Depth (m)')
ax1.set_ylim(top=0,bottom=zmax+2*dz), ax2.set_ylim(top=0,bottom=zmax+2*dz), ax3.set_ylim(top=0,bottom=zmax+2*dz)
ax1.errorbar(Ps_mean, zs, fmt='^', xerr=Ps_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
ax2.errorbar(Pl_mean, zs, fmt='^', xerr=Pl_se, ecolor=green, elinewidth=elw, c=green, ms=ms*2, capsize=cs, lw=lw, label='Data', fillstyle='full')
ax3.errorbar(cppt.Pt_hat, zml, fmt='o', xerr=np.ones(n)*np.sqrt(model.mse_resid), ecolor=blue, elinewidth=elw, c=blue, ms=ms, capsize=cs, lw=lw, label='from $c_P$', fillstyle='none', zorder=1)
ax3.scatter(Ps_mean+Pl_mean, zs, marker='^', c=green, s=ms*10, label='pump', zorder=2)
ax3.legend()


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
elwp=1
ec='k'
msp = 9
csp = 4
fig, ([ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]) = plt.subplots(2,4)
fig.subplots_adjust(wspace=0.8, hspace=0.4)
axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax8]
for i, p in enumerate(pdi.keys()):
    ax = axs[i]
    if pdi[p]['dv'] == 1: #if param is depth-varying
        ax.set_title(pdi[p]['A']['tset'])
        ax.errorbar(1, pdi[p]['A']['o'], yerr=pdi[p]['A']['oe'],fmt='o',ms=msp,c=blue,label='$x_{o}$',elinewidth=elwp,ecolor=ec,capsize=csp) #priors with errors
        ax.errorbar(2, pdi[p]['A']['xh'], yerr=pdi[p]['A']['xhe'], fmt='o', c=teal, ms=msp, label='$x^{A}_{k+1}$', elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
        ax.errorbar(3, pdi[p]['B']['xh'], yerr=pdi[p]['B']['xhe'], fmt='o', c=navy, ms=msp, label='$x^{B}_{k+1}$', elinewidth=elwp, ecolor=ec,capsize=csp) #posteriors with errors
    else: #if param is depth-constant
        ax.set_title(pdi[p]['tset'])
        ax.errorbar(1, pdi[p]['o'], yerr=pdi[p]['oe'],fmt='o',ms=msp,c=blue,label='$x_{o}$',elinewidth=elwp,ecolor=ec,capsize=csp) #priors with errors
        ax.errorbar(3, pdi[p]['xh'], yerr=pdi[p]['xhe'],fmt='o',c=cyan,ms=msp,label='$x_{k+1}$',elinewidth=elwp,ecolor=ec,capsize=csp) #posteriors with errors        
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_xticks(np.arange(0,5))
  
#calculate fluxes and errors
flxs = ['ws_Ps','wl_Pl','ws_Psdz','wl_Pldz','Bm1s_Ps','Bm1l_Pl','B2p_Ps2','Bm2_Pl','Psdot']
flxnames = {'ws_Ps':'sinkS', 'wl_Pl':'sinkL', 'ws_Psdz':'sinkS_div', 'wl_Pldz':'sinkL_div',
            'Bm1s_Ps':'SRemin', 'Bm1l_Pl':'LRemin', 'B2p_Ps2':'Agg', 'Bm2_Pl':'Disagg', 'Psdot':'Prod'}
flxd = {f:{} for f in flxs}
for f in flxd.keys():
    flxd[f]['name'] = flxnames[f]
    #get what parameter, tracer, and order each flux contains
    if '_' in f: #if not Psdot
        p,twordr = f.split('_')
        if any(map(str.isdigit, twordr)): flxd[f]['o'] = int(twordr[-1]) #if tracer is not first order      
        else: flxd[f]['o'] = 1
        t = twordr[:2] #requres that tracer be designated by first 2 characters of twordr
        ordr = flxd[f]['o']
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
                fxh[i], fxhe[i] = flxep(y,cov=True) 
            elif (i == 1 or i == 2): #first two points below ML
                twip1, twim1 = "_".join([t,str(i+1)]), "_".join([t,str(i-1)])
                Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                y = w*(Pip1-Pim1)/(2*dz) #calculate flux estimate
                fxh[i], fxhe[i]  = flxep(y,cov=True) 
            else: #all other depths
                twim1, twim2 = "_".join([t,str(i-1)]), "_".join([t,str(i-2)])
                Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                y = w*(3*Pi-4*Pim1+Pim2)/(2*dz) #calculate flux estimate
                fxh[i], fxhe[i]  = flxep(y,cov=True)
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
            fxh[i], fxhe[i] = flxep(y,cov=True)
    flxd[f]['xh'], flxd[f]['xhe'] = fxh, fxhe 


#plot fluxes
flxpairs = [('ws_Ps','wl_Pl'),('ws_Psdz','wl_Pldz'),('Bm1s_Ps','B2p_Ps2'),('Bm1l_Pl','Bm2_Pl'),('Psdot',)]
for pr in flxpairs:
    fig, ax = plt.subplots(1,1) #P figures
    ax.invert_yaxis()
    if ('w' in pr[0]) and ('dz' not in pr[0]):
        ax.set_xlabel('Flux $[mmol/(m^2 \cdot d)]$')
    else: ax.set_xlabel('Vol. Flux $[mmol/(m^3 \cdot d)]$')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(top=0,bottom=zmax+dz)
    c1, c2, c3, c4 = navy, teal, red, purple
    ax.errorbar(flxd[pr[0]]['xh'], zml, fmt='o', xerr=flxd[pr[0]]['xhe'], ecolor=c1, elinewidth=elw, c=c1, ms=ms, capsize=cs, lw=lw, label=flxnames[pr[0]], fillstyle='none')
    ax.axhline(dA-2.5,c='k',ls='--',lw=lw/2)
    if len(pr) > 1: #if it's actually a pair
        ax.errorbar(flxd[pr[1]]['xh'], zml, fmt='o', xerr=flxd[pr[1]]['xhe'], ecolor=c2, elinewidth=elw, c=c2, ms=ms, capsize=cs, lw=lw, label=flxnames[pr[1]], fillstyle='none')
    ax.legend()
        
#integrated fluxes (stored in a separate dict)
#iflxs = ['ws_Psdz','wl_Pldz','Bm1s_Ps','Bm1l_Pl','B2p_Ps2','Bm2_Pl','Psdot']
iflxs = ['Bm1s_Ps','Bm1l_Pl','ws_Psdz','wl_Pldz']
def iflxcalc(fluxes, deprngs):
    for f in fluxes:
        if '_' in f: #if not Psdot
            p,t = f.split('_')[0], f.split('_')[1][:2]
            ordr = flxd[f]['o'] #get order from the flx dict
        for dr in deprngs:        
            do, dn = dr #unpack start and end depths
            doi, dni = zml.tolist().index(do), zml.tolist().index(dn)
            rstr = "_".join([str(do),str(dn)])
            flxd[f][rstr] = {} #create dict to store integrated flux and res time for this depthrange
            dis = np.arange(doi,dni+1) 
            iF, iI = 0, 0 #initialize variable to collect summation (integrals) of fluxes and inventory
            if 'dz' in f: #if sinking flux divergence term, more complicated
                for i in dis:
                    dzi = dz if i != 0 else h #if it's the ML, multiply by h instead of dz
                    l = lmatch(i)
                    pwi = "_".join([p,l])
                    twi = "_".join([t,str(i)])
                    w, Pi = sym.symbols(f'{pwi} {twi}')
                    if i == 0: #mixed layer
                        iF += w*Pi/h*dzi
                        iI += Pi*dzi
                    elif (i == 1 or i == 2): #first two points below ML
                        twip1, twim1 = "_".join([t,str(i+1)]), "_".join([t,str(i-1)])
                        Pip1, Pim1 = sym.symbols(f'{twip1} {twim1}')
                        iF += w*(Pip1-Pim1)/(2*dz)*dzi #calculate flux estimate
                        iI += Pi*dzi
                    else: #all other depths
                        twim1, twim2 = "_".join([t,str(i-1)]), "_".join([t,str(i-2)])
                        Pim1, Pim2 = sym.symbols(f'{twim1} {twim2}')
                        iF += w*(3*Pi-4*Pim1+Pim2)/(2*dz)*dzi #calculate flux estimate
                        iI += Pi*dzi
            elif f == 'Psdot': #if it's the production term
                gh, lp = sym.symbols('Gh Lp')
                for i in dis:
                    dzi = dz if i != 0 else h
                    Pi = sym.symbols(f'Ps_{i}')
                    iF += gh*sym.exp(-(zml[i]-h)/lp)*dzi
                    iI += Pi*dzi
            else: #all other terms that are not sinking or production
                for i in dis:
                    dzi = dz if i != 0 else h
                    l = lmatch(i)
                    pwi = "_".join([p,l])
                    twi = "_".join([t,str(i)])
                    pa, tr = sym.symbols(f'{pwi} {twi}')
                    iF += (pa*tr**ordr)*dzi
                    iI += tr*dzi
            intflx = flxep(iF,cov=True)
            resT = flxep(iI/iF,cov=True)
            flxd[f][rstr]['iflx'], flxd[f][rstr]['tau'] = intflx, resT
            #return(intflx)

#should be equal to the fluxes integrated in A and B (WORKS)
iflxcalc(iflxs,((95,zmax),))

# #one test for iflxcalc with two layers (WORKS!)
# tid, tid1, pid, pid1 = vidxSV.index('Pl_16'), vidxSV.index('Pl_17'), vidxSV.index('Bm2_A'), vidxSV.index('Bm2_B')
# tv, te = td['Pl']['xh'][16], td['Pl']['xhe'][16]
# tv1, te1 = td['Pl']['xh'][17], td['Pl']['xhe'][17]
# pv, pe = pdi['Bm2']['A']['xh'], pdi['Bm2']['A']['xhe']
# pv1, pe1 = pdi['Bm2']['B']['xh'], pdi['Bm2']['B']['xhe']
# print((pv*tv+pv1*tv1)*dz, np.sqrt((pv*dz*te)**2+(pv1*dz*te1)**2+(tv*dz*pe)**2+(tv1*dz*pe1)**2+
#                             2*dz**2*(tv*tv1*CVM[pid,pid1]+pv*pv1*CVM[tid,tid1]+tv*pv*CVM[pid,tid]+
#                             tv1*pv*CVM[pid1,tid]+tv*pv1*CVM[pid,tid1]+tv1*pv1*CVM[pid1,tid1])))
# v1_st, v2_st, v3_st, v4_st = '_'.join(['Bm2','A']), '_'.join(['Bm2','B']), '_'.join(['Pl','16']), '_'.join(['Pl','17'])
# v1, v2, v3, v4 = sym.symbols(f'{v1_st} {v2_st} {v3_st} {v4_st}')
# y = (v1*v3+v2*v4)*dz
# print(flxep(y,cov=True))
# print(y)
# print(iflxcalc(['Bm2_Pl'],((zml[16],zml[17]),)))

# #checking calculation of timescales (LOOKS GOOD)       
# invPs_A = np.sum(td['Ps']['xh'][1:17])*dz+td['Ps']['xh'][0]*h
# invPl_A = np.sum(td['Pl']['xh'][1:17])*dz+td['Pl']['xh'][0]*h
# invPs_B = np.sum(td['Ps']['xh'][17:])*dz
# invPl_B = np.sum(td['Pl']['xh'][17:])*dz
# for f in iflxs:
#     if 'Ps' in f:
#         print(flxd[f]['30_110']['iflx'][0]-invPs_A/flxd[f]['30_110']['tau'][0])
#         print(flxd[f]['115_500']['iflx'][0]-invPs_B/flxd[f]['115_500']['tau'][0])
#     else:
#         print(flxd[f]['30_110']['iflx'][0]-invPl_A/flxd[f]['30_110']['tau'][0])
#         print(flxd[f]['115_500']['iflx'][0]-invPl_B/flxd[f]['115_500']['tau'][0])  

        
print(f'--- {time.time() - start_time} seconds ---')