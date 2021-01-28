#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 09 2020

@author: vamaral


Twin experiments. Noiseless total POC pseudodata. Uses posterior estimates
from invP.py as target values
"""
import numpy as np
import scipy.linalg as splinalg
import scipy.stats as sstats
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import sympy as sym
import os
import sys
import time
import pickle

start_time = time.time()
plt.close('all')

#need for when running on remote server
sys.setrecursionlimit(10000)

"""
SETUP FOR GENERATING PSEUDODATA
"""

plt.close('all')

#unpickle vars
with open('invP_savedvars.pkl','rb') as file:
    flxd,td,pdi,combodf_s,ac_params,pdf_params = pickle.load(file)
model = smf.ols(formula='Pt ~ np.log(cp)',data=combodf_s).fit()
model = smf.ols(formula='Pt ~ np.log(cp)',data=combodf_s).fit()
pdfx,pdfn,xg,yg_pdf = pdf_params
kdz_A,ac_A,l_int_A,L_A,lfit_A,l_r2_A,kdz_B,ac_B,l_int_B,L_B,lfit_B,l_r2_B = ac_params

#colors
red, green, blue, purple, cyan, orange, teal, navy, olive = '#e6194B', '#3cb44b', '#4363d8', '#911eb4', '#42d4f4', '#f58231', '#469990', '#000075', '#808000'
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

#first order aggregation estimate from Murnane 1994
B2 = 0.8/dpy

#assign df columns to variables
zs = df.Depth
Ps_mean_real = df.SSF_mean/mm
#Ps_sd_real = df.SSF_sd/mm
Ps_se_real = df.SSF_se/mm
Ps_re = Ps_se_real/Ps_mean_real
Pl_mean_real = df.LSF_mean/mm
#Pl_sd_real = df.LSF_sd/mm
Pl_se_real = df.LSF_se/mm
Pl_re = Pl_se_real/Pl_mean_real

gam = 0.02 #multiplier for weighting model errors

#param info
layers = ['A','B']
tracers = ['Ps','Pl']
params = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l', 'Gh', 'Lp']
params_dv = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l'] #depth-varying params
params_dc = ['Gh', 'Lp'] #depth-constant params

#typeset name
p_tset = {'ws':'$w_S$', 'wl':'$w_L$', 'B2p':'$\\beta^,_2$', 'Bm2':'$\\beta_{-2}$', 
                'Bm1s':'$\\beta_{-1,S}$', 'Bm1l':'$\\beta_{-1,L}$', 'Gh':'$\.P_{S,30}$',
                'Lp':'$L_{P}$'}

#extract information for priors from invP.py results
p_o = {'ws':pdi['ws']['o'], #m/d
        'wl':pdi['wl']['o'], #m/d, reported from Murnane 1990.
        'B2p':pdi['B2p']['o'], #m3/(mg*yr) converted to m3/(mmol*d). Divided B2 from Murnane 94 (0.8 1/y) by average Ps from Bishop 99 (1.6 mmol/m3)
        'Bm2':pdi['Bm2']['o'], #from Murnane 1994, converted to d
        'Bm1s':pdi['Bm1s']['o'], #from Clegg 91 (Fig. 6) surface average
        'Bm1l':pdi['Bm1l']['o'], #based on preliminary RESPIRE data from A. Santoro
        'Gh':pdi['Gh']['o'], #prior set to typical NPP shared data value
        'Lp':pdi['Lp']['o']} #from NPP data, m
#prior errors
p_oe = {'ws':pdi['ws']['oe'], #m/d
        'wl':pdi['wl']['oe'], #m/d, reported from Murnane 1990.
        'B2p':pdi['B2p']['oe'], #m3/(mg*yr) converted to m3/(mmol*d). Divided B2 from Murnane 94 (0.8 1/y) by average Ps from Bishop 99 (1.6 mmol/m3)
        'Bm2':pdi['Bm2']['oe'], #from Murnane 1994, converted to d
        'Bm1s':pdi['Bm1s']['oe'], #from Clegg 91 (Fig. 6) surface average
        'Bm1l':pdi['Bm1l']['oe'], #based on preliminary RESPIRE data from A. Santoro
        'Gh':pdi['Gh']['oe'], #prior set to typical NPP shared data value
        'Lp':pdi['Lp']['oe']} #from NPP data, m
#target values used to generate pseudo-data
p_tgt = {'ws':{'A':pdi['ws']['gammas'][gam]['xh']['A'],'B':pdi['ws']['gammas'][gam]['xh']['B']},
        'wl':{'A':pdi['wl']['gammas'][gam]['xh']['A'],'B':pdi['wl']['gammas'][gam]['xh']['B']},
        'B2p':{'A':pdi['B2p']['gammas'][gam]['xh']['A'],'B':pdi['B2p']['gammas'][gam]['xh']['B']},
        'Bm2':{'A':pdi['Bm2']['gammas'][gam]['xh']['A'],'B':pdi['Bm2']['gammas'][gam]['xh']['B']},
        'Bm1s':{'A':pdi['Bm1s']['gammas'][gam]['xh']['A'],'B':pdi['Bm1s']['gammas'][gam]['xh']['B']},
        'Bm1l':{'A':pdi['Bm1l']['gammas'][gam]['xh']['A'],'B':pdi['Bm1l']['gammas'][gam]['xh']['B']},
        'Gh':pdi['Gh']['gammas'][gam]['xh'],
        'Lp':pdi['Lp']['gammas'][gam]['xh']}

#have a single layer seperation
bnd = 112.5

#some parameters for plotting later
ms, lw, elw, cs = 3, 1, 0.5, 2

#delete entries in pdi to save data from twnx
pdi = {param:{} for param in params}
#add a key for each parameter designating if it is depth-varying or constant
for k in pdi.keys():
    if k in params_dv: pdi[k]['dv'] = 1
    else: pdi[k]['dv'] = 0
#update entries in pdi
for p in pdi.keys():
    dv = pdi[p]['dv']
    pdi[p]['tset'] = p_tset[p]
    pdi[p]['o'] = p_o[p]
    pdi[p]['oe'] = p_oe[p]
    pdi[p]['t'] = p_tgt[p]
    for k in ['xh','xhe']:
        pdi[p][k] = {} if not dv else {l:{} for l in layers}

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
            else: xv[i] = pdi[svar]['xh'][str(di)]
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
            if pdi[v.name]['dv']:
                l = lmatch(di)
                iSV = vidxSV.index('_'.join([v.name,l]))
            else: iSV = vidxSV.index(v.name)
        xv[j] = np.exp(xk[iSV]) if ln == True else xk[iSV]
        xi[j] = iSV
    return x, xv, xi

#make a matrix whose diagonals are MSE of residuals
MSE_fit = model.mse_resid
Cf_addPt = np.diag(np.ones(n)*MSE_fit)

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
    wsi, wli, Bm2i, Bm1si, Bm1li, Ghi = pdi['ws']['t'][l], pdi['wl']['t'][l], \
        pdi['Bm2']['t'][l], pdi['Bm1s']['t'][l], pdi['Bm1l']['t'][l], Ghz[int(d)]
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
pdelt = 0.01 #allowable percent change in each state element for convergence
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
        l = lmatch(d)
        #pick up all values of rate parameters
        wsi, wli, Bm2i, B2pi, Bm1si, Bm1li, Ghi = pdi['ws']['t'][l], pdi['wl']['t'][l], \
            pdi['Bm2']['t'][l], pdi['B2p']['t'][l], pdi['Bm1s']['t'][l], pdi['Bm1l']['t'][l], Ghz[int(d)]
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
Pt_numnl = Ps_numnl + Pl_numnl

"""
#INVERSE METHOD (P)
"""
#define sampling depths, and find indices that they occur at
sorter = np.argsort(zml)
zsi = sorter[np.searchsorted(zml, zs, sorter=sorter)]

#assign observation vectors and data errors
Ps_mean = Ps_numnl[zsi]
Pl_mean = Pl_numnl[zsi]
Ps_se = Ps_re*Ps_mean
Pl_se = Pl_re*Pl_mean

#make a dictionary for tracer params for each layer. could be more cleverly coded
tracers = ['Ps','Pl']
oi = {lay:{t:{} for t in tracers} for lay in layers}
oi['A']['Ps']['y'],oi['B']['Ps']['y'] = Ps_mean[0:3],Ps_mean[3:] #mean POC
oi['A']['Pl']['y'],oi['B']['Pl']['y'] = Pl_mean[0:3],Pl_mean[3:]
oi['A']['Pl']['sig_j'],oi['B']['Pl']['sig_j'] = Pl_se[0:3].values,Pl_se[3:].values #POC standard deviation
oi['A']['Ps']['sig_j'],oi['B']['Ps']['sig_j'] = Ps_se[0:3].values,Ps_se[3:].values
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
    maxchange = np.max(np.abs((np.exp(xkp1)-np.exp(xk))/np.exp(xk)))
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
ax1.set_xlabel(r'$\frac{\^x_{i}-x_{o,i}}{\sigma_{o,i}}$',size=24)
ax1.hist(pdfx,density=True,bins=20,color=blue)
ax2.hist(pdfn,density=True,bins=20,color=blue)
ax2.set_xlabel(r'$\frac{f(\^x)_{i}}{\sigma_{f(\^x)_{i}}}$',size=24)
[ax.set_xlim([-1,1]) for ax in (ax1,ax2)]
plt.savefig('twnxP_pdfs.png')
plt.close()

#CDFs
fig, [ax1,ax2] = plt.subplots(1,2,tight_layout=True)
fig.subplots_adjust(wspace=0.5)
ax1.set_ylabel('P',size=16)
ax1.set_xlabel(r'$\frac{\^x_{i}-x_{o,i}}{\sigma_{o,i}}$',size=24), ax2.set_xlabel(r'$\frac{f(\^x)_{i}}{\sigma_{f(\^x)_{i}}}$',size=24)
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
plt.savefig('twnxP_cdfs.png')
plt.close()     

#model residual depth profiles (posteriors)
fig, [ax1,ax2] = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)  
ax1.invert_yaxis(), ax2.invert_yaxis()
ax1.set_xlabel('$n^{k+1}_{P_{S}}$ (mmol m$^{-3}$ d$^{-1}$)'), ax2.set_xlabel('$n^{k+1}_{P_{L}}$ (mmol m$^{-3}$ d$^{-1}$)')
ax1.set_ylabel('Depth (m)')
ax1.set_ylim(top=0,bottom=zmax+dz), ax2.set_ylim(top=0,bottom=zmax+dz)
ax1.scatter(td['Ps']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
ax2.scatter(td['Pl']['n'], zml, marker='o', c=blue, s=ms/2, label='MRes')
ax1.legend(), ax2.legend()
plt.savefig('twnxP_residprofs.png')
plt.close()

#plot evolution of convergence
ms=3
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(conv_ev)), conv_ev, marker='o',ms=ms)
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',size=12)
plt.savefig('twnxP_conv.png')
plt.close()

#plot evolution of cost function
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(cost_ev)),cost_ev,marker='o',ms=ms)
ax.set_xlabel('k')
ax.set_ylabel('j')
ax.set_yscale('log')
plt.savefig('twnxP_cost.png')
plt.close()

#comparison plots
fig,[ax1,ax2,ax3] = plt.subplots(1,3,tight_layout=True) #P figures
fig.subplots_adjust(wspace=0.5)  
ax1.invert_yaxis(),ax2.invert_yaxis(),ax3.invert_yaxis()
ax1.set_xlabel('$P_{S}$ (mmol m$^{-3}$)',fontsize=14),ax2.set_xlabel('$P_{L}$ (mmol m$^{-3}$)',fontsize=14),ax3.set_xlabel('$P_{T}$ (mmol m$^{-3}$)',fontsize=14)
ax1.set_ylabel('Depth (m)',fontsize=14)
ax1.set_ylim(top=0,bottom=zmax+30),ax2.set_ylim(top=0,bottom=zmax+30),ax3.set_ylim(top=0,bottom=zmax+30)
ax1.errorbar(Ps_mean,zs,fmt='^',xerr=Ps_se,ecolor=blue,elinewidth=1,c=blue,ms=10,capsize=5,label='Data',fillstyle='full')
ax1.errorbar(td['Ps']['xh'],zml,fmt='o',xerr=td['Ps']['xhe'],ecolor=orange,elinewidth=0.5,c=orange,ms=3,capsize=2,label='TE',fillstyle='none',zorder=3,markeredgewidth=0.5)
ax1.errorbar(td['Ps']['x'],zml,fmt='o',xerr=td['Ps']['xerr'],ecolor=sky,elinewidth=0.5,c=sky,ms=2,capsize=2,label='OI',markeredgewidth=0.5)
ax1.set_xticks([0,1,2])
ax2.errorbar(Pl_mean,zs,fmt='^',xerr=Pl_se,ecolor=blue,elinewidth=1,c=blue,ms=10,capsize=5,label='Data',fillstyle='full')
ax2.errorbar(td['Pl']['xh'],zml,fmt='o',xerr=td['Pl']['xhe'],ecolor=orange,elinewidth=0.5,c=orange,ms=3,capsize=2,label='TE',fillstyle='none',zorder=3,markeredgewidth=0.5)
ax2.errorbar(td['Pl']['x'],zml,fmt='o',xerr=td['Pl']['xerr'],ecolor=sky,elinewidth=0.5,c=sky,ms=2,capsize=2,label='OI',markeredgewidth=0.5)
ax2.set_xticks([0,0.05,0.1])
ax2.set_xticklabels(['0','0.05','0.1'])
ax3.errorbar(Pt_numnl,zml+1,fmt='o',xerr=np.ones(n)*np.sqrt(MSE_fit),ecolor=blue,elinewidth=0.5,c=blue,ms=2,capsize=2,label='Data',markeredgewidth=0.5)
ax3.errorbar(Pt_xh,zml,fmt='o',xerr=Pt_xhe,ecolor=orange,elinewidth=0.5,c=orange,ms=3,capsize=2,label='TE',fillstyle='none',zorder=3,markeredgewidth=0.5)
ax3.set_xticks([0,1,2])
[ax.legend(fontsize=12,borderpad=0.2) for ax in (ax1,ax2,ax3)]
[ax.tick_params(labelleft=False) for ax in (ax2,ax3)]
[ax.tick_params(axis='both',which='major',labelsize=12) for ax in (ax1,ax2,ax3)]
[ax.axhline(bnd,c=black,ls='--',lw=1) for ax in (ax1,ax2,ax3)]
plt.savefig('twnxP_Pprofs.pdf')
plt.close()

#extract posterior param estimates and errors
params_ests = xhmean[-nparams:]
params_errs = xhe[-nparams:]
for i,stri in enumerate(p_toidx):
    pest, perr = params_ests[i], params_errs[i]
    if '_' in stri: #depth-varying paramters
        p,l = stri.split('_')
        pdi[p]['xh'][l], pdi[p]['xhe'][l] = pest, perr
    else: pdi[stri]['xh'], pdi[stri]['xhe'] = pest, perr

#make a plot of parameter priors and posteriors
elwp, msp, csp, ec = 1, 9, 4, 'k'
fig, ([ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]) = plt.subplots(2,4)
fig.subplots_adjust(wspace=0.8, hspace=0.4)
axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
for i,p in enumerate(pdi.keys()):
    ax = axs[i]
    ax.set_title(pdi[p]['tset'])
    if pdi[p]['dv']: #if param is depth-varying
        ax.errorbar(1,pdi[p]['o'],yerr=pdi[p]['oe'],fmt='o',ms=9,c=blue,elinewidth=1.5,ecolor=blue,capsize=6,markeredgewidth=1.5,label='Prior') #priors with errors
        ax.scatter(2,pdi[p]['t']['A'],marker='+',s=90,c=green) #target value
        ax.errorbar(3,pdi[p]['xh']['A'],yerr=pdi[p]['xhe']['A'],fmt='o',c=green,ms=9,elinewidth=1.5,ecolor=green,capsize=6,markeredgewidth=1.5,label='EZ') #posteriors with errors
        ax.scatter(4,pdi[p]['t']['B'],marker='+',s=90,c=orange) #target value
        ax.errorbar(5,pdi[p]['xh']['B'],yerr=pdi[p]['xhe']['B'],fmt='o',c=orange,ms=9,elinewidth=1.5,ecolor=orange,capsize=6,markeredgewidth=1.5,label='UMZ') #posteriors with errors
        if i == 5: ax.legend(loc='upper center',bbox_to_anchor=(1.38,-0.07),ncol=3,fontsize=12,frameon=False)
    else: #if param is depth-constant
        ax.errorbar(2,pdi[p]['o'],yerr=pdi[p]['oe'],fmt='o',ms=9,c=blue,elinewidth=1.5,ecolor=blue,capsize=6,markeredgewidth=1.5,label='Prior') #priors with errors
        ax.scatter(3,pdi[p]['t'],marker='+',s=90,c=radish) #target value
        ax.errorbar(4,pdi[p]['xh'],yerr=pdi[p]['xhe'],fmt='o',c=radish,ms=9,elinewidth=1.5,ecolor=radish,capsize=6,markeredgewidth=1.5) #posteriors with errors        
    ax.tick_params(bottom=False,labelbottom=False)
    ax.set_xticks(np.arange(0,7))
    if p == 'Bm2': ax.set_ylim(-0.5,2)
plt.savefig('twnxP_params.pdf')
plt.close()
    
print(f'--- {time.time() - start_time} seconds ---')