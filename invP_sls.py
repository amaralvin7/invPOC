#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 09 2020

@author: vamaral

A simple least-squares inversion of POC data. Assumes no errors, and that
data are perfectly known.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as splinalg
import scipy.optimize as spopt
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

#param info
layers = ['A','B']
tracers = ['Ps','Pl']
params = ['ws', 'wl', 'B2p', 'Bm2', 'Bm1s', 'Bm1l']
#make a dictionaries to store param estimates
pdi = {param:{lay:0 for lay in layers} for param in params} 
#pdi = {param:0 for param in params} 

#have a single layer seperation
bnd = 112.5

#some parameters for plotting later
ms, lw, elw, cs = 3, 1, 0.5, 2

#particle prouction
Ghz = 0.28*np.exp(-(zml-h)/28) #numerical values from prior values used for Gh, and Lp respectively in ATi inversion

#first order aggregation estimate from Murnane 1994
B2 = 0.8/dpy

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

#given a vector and starting index, returns a slice of size n
def vsli(vec,sidx):
    sli = vec[sidx:sidx+n]
    return sli

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
            iSV = vidxP.index('_'.join([t,adi])) #index of the state variable
        xv[j] = np.exp(xk[iSV]) if ln == True else xk[iSV]
        xi[j] = iSV
    return x, xv, xi
         
"""
#OBTAIN ESTIMATES FOR PARAMS USING A SIMPLE LS SOLUTION
"""

#create depth indices for state variables found at every grid point
svis = [''.join(['_',str(i)]) for i in np.arange(0,n)]

#create variable indexes for P size fractions and params
vidxP = []
vidxP = addtovaridx(vidxP,('Ps','Pl'))
pidx = ['_'.join([p,l]) for p in params for l in layers]
#pidx = [p for p in params]

#some useful matrix dimensions
N = len(vidxP)
P = len(pidx)

#do OI to get "data" for Ps and Pl
oi = {lay:{t:{} for t in tracers} for lay in layers}
oi['A']['Ps']['y'],oi['B']['Ps']['y'] = Ps_mean[0:3],Ps_mean[3:] #mean POC
oi['A']['Pl']['y'],oi['B']['Pl']['y'] = Pl_mean[0:3],Pl_mean[3:]
oi['A']['Pl']['sig_j'],oi['B']['Pl']['sig_j'] = Pl_sd[0:3].values,Pl_sd[3:].values #POC standard deviation
oi['A']['Ps']['sig_j'],oi['B']['Ps']['sig_j'] = Ps_sd[0:3].values,Ps_sd[3:].values
oi['A']['smpd'], oi['A']['grdd'] = zs[0:3].values, zml[difind(h):difind(bnd-dz/2)+1] #sample and grid depths, layer A
oi['B']['smpd'], oi['B']['grdd'] = zs[3:].values, zml[difind(bnd+dz/2):] #sample and grid depths, layer B
oi['A']['L'], oi['B']['L'] = 13.39, 67.9 #interpolation length scales

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
pocdata = np.asarray([]) #initialize an array to collect priors of all tracers AFTER they've been concatenated by layer
td_keys = ['si','x','y'] 
td = {t:dict.fromkeys(td_keys) for t in tracers}
for tra in tracers:
    for k in td[tra].keys():
        if k == 'si': #what index in vidxSV does each tracer start at?
            td[tra]['si'] = vidxP.index(tra+'_0') 
            continue
        #all values that need to be concatenated between layers
        td[tra][k] = np.concatenate((oi['A'][tra][k],oi['B'][tra][k]))
        #catch the prior estiamtes and collect them in tpri
        if k == 'x': pocdata = np.concatenate((pocdata,td[tra][k]))
Ps_oi, Pl_oi = [vsli(pocdata,vidxP.index(f'{t}_0')) for t in tracers]

#solve Ex = y problem
E, x, y = np.zeros((N, P)), np.zeros(P), np.zeros(N)
y = y.reshape(-1, 1)
#loop through and edit rows of AP
for i in np.arange(0,N):
    #what tracer and depth are we on?
    t,d = vidxP[i].split('_')
    d = int(d)
    l = lmatch(d) #what layer does this depth index correspond to?
    Psi = Ps_oi[d]
    Pli = Pl_oi[d]
    #depth-dependent parameters
    iwsi, iwli, iB2pi, iBm2i, iBm1si, iBm1li = [pidx.index('_'.join([p,l])) for p in params]
    #iwsi, iwli, iB2pi, iBm2i, iBm1si, iBm1li = [pidx.index(p) for p in params]
    #POC, SSF
    if t == 'Ps':
        E[i,iBm2i] = -Pli
        E[i,iBm1si] = Psi
        E[i,iB2pi]  = Psi**2
        y[i] = Ghz[d]
        if d == 0: #mixed layer
            E[i,iwsi] = Psi/h
        elif (d == 1 or d == 2): #first or second grid point
            Psip1 = Ps_oi[d+1]
            Psim1 = Ps_oi[d-1]
            E[i,iwsi] = (Psip1-Psim1)/(2*dz)
        else: #everywhere else
            Psim1 = Ps_oi[d-1]
            Psim2 = Ps_oi[d-2]
            E[i,iwsi] = (Psim2-4*Psim1+3*Psi)/(2*dz)
    #POC, LSF
    else:
        E[i,iBm2i], E[i,iBm1li] = Pli, Pli
        E[i,iB2pi] = -Psi**2
        if d == 0:
            E[i,iwli] = Pli/h              
        elif (d == 1 or d == 2):
            Plip1 = Pl_oi[d+1]
            Plim1 = Pl_oi[d-1]
            E[i,iwli] = (Plip1-Plim1)/(2*dz)
        else:
            Plim1 = Pl_oi[d-1]
            Plim2 = Pl_oi[d-2]
            E[i,iwli] = (Plim2-4*Plim1+3*Pli)/(2*dz)
E_cond = np.linalg.cond(E)
ETEinv = np.linalg.inv(np.matmul(E.T,E))
ETEinv_cond = np.linalg.cond(ETEinv)
ETy = np.matmul(E.T,y)
ETycond = np.linalg.cond(ETy)
x, sum_of_squares = spopt.nnls(E,y.flatten()) #non-negative least squares

#store estimates for each layer in the dictionary
for p in params:
    for l in layers:
        pdi[p][l] = x[pidx.index('_'.join([p,l]))]
    #pdi[p] = x[pidx.index(p)]
    
"""
#GENERATE PSEUDODATA WITH LS ESTIMATES
#LINEAR NUMERICAL SOLUTIONS (need to solve to get priors for nonlinear solutions)
"""

#Construct A matrix and b vector
A, b = np.zeros((N, N)), np.zeros(N)
#loop through and edit rows of AP
for i in np.arange(0,N):
    #what tracer and depth are we on?
    t,d = vidxP[i].split('_')
    d = int(d)
    l = lmatch(d) #what layer does this depth index correspond to?
    iPsi = vidxP.index(f'Ps_{str(d)}')
    iPli = vidxP.index(f'Pl_{str(d)}')
    #pick up all values of rate parameters
    wsi, wli, Bm2i, Bm1si, Bm1li, B2i = pdi['ws'][l], pdi['wl'][l], \
        pdi['Bm2'][l], pdi['Bm1s'][l], pdi['Bm1l'][l], B2
    #wsi, wli, Bm2i, Bm1si, Bm1li, B2i = pdi['ws'], pdi['wl'], \
    #    pdi['Bm2'], pdi['Bm1s'], pdi['Bm1l'], B2    
    #POC, SSF
    if t == 'Ps':
        A[i,iPli] = -Bm2i
        b[i] = Ghz[d]
        if d == 0: #ML
            A[i,iPsi] = (wsi/h)+Bm1si+B2i
        elif (d == 1 or d == 2): #first or second grid point
            iPsip1 = vidxP.index(f'Ps_{str(d+1)}')
            iPsim1 = vidxP.index(f'Ps_{str(d-1)}')
            A[i,iPsip1] = wsi/(2*dz)
            A[i,iPsi] = Bm1si+B2i
            A[i,iPsim1] = -wsi/(2*dz)
        else:
            iPsim1 = vidxP.index(f'Ps_{str(d-1)}')
            iPsim2 = vidxP.index(f'Ps_{str(d-2)}')
            A[i,iPsi] = (3*wsi)/(2*dz)+Bm1si+B2i
            A[i,iPsim1] = (-2*wsi)/dz
            A[i,iPsim2] =wsi/(2*dz)
    #POC, LSF
    else:       
        A[i,iPsi] = -B2i
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
Ps_numlin, Pl_numlin = [vsli(x_numlin,vidxP.index(f'{t}_0')) for t in tracers]

# """
# #NONLINEAR NUMERICAL SOLUTIONS (need to solve to get priors for inverse method) 
# """

#initialize matrices and vectors in Fk*xkp1 = Fk*xk-fk+b
F = np.zeros((N, N))
b, f = np.zeros(N), np.zeros(N)
xo = x_numlin
xk = xo
k = 0 #keep a counter for how many steps it takes
pdelt = 0.0001 #allowable percent change in each state element for convergence
conv_ev = np.empty(0) # keep track of evolution of convergence

#define all possible symbolic variables
svarnames = 'Ps_0 Ps_1 Ps_-1 Ps_-2 \
    Pl_0 Pl_1 Pl_-1 Pl_-2'
Psi, Psip1, Psim1, Psim2, \
    Pli, Plip1, Plim1, Plim2  = sym.symbols(svarnames)
#iterative solution
#while True:
while k < 5:
    for i in np.arange(0,N):
        #what tracer and depth are we on?
        t,d = vidxP[i].split('_')
        d = int(d)
        #pick up all values of rate parameters
        wsi, wli, Bm2i, B2pi, Bm1si, Bm1li = pdi['ws'][l], pdi['wl'][l], \
            pdi['Bm2'][l], pdi['B2p'][l], pdi['Bm1s'][l], pdi['Bm1l'][l]
        # wsi, wli, Bm2i, B2pi, Bm1si, Bm1li = pdi['ws'], pdi['wl'], \
        #     pdi['Bm2'], pdi['B2p'], pdi['Bm1s'], pdi['Bm1l']       
        #POC, SSF
        if t == 'Ps':
            b[i] = -Ghz[d]
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
    #if maxchange < pdelt or k > 2000: break
    conv_ev = np.append(conv_ev,maxchange)    
    k += 1
    xk = xkp1
#assign numerical solutions to variables
Ps_numnl, Pl_numnl = [vsli(xkp1,vidxP.index(f'{t}_0')) for t in tracers]

#plot evolution of convergence
ms=3
fig, ax = plt.subplots(1)
ax.plot(np.arange(0, len(conv_ev)), conv_ev, marker='o',ms=ms)
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('max'+r'$(\frac{|x_{i,k+1}-x_{i,k}|}{x_{i,k}})$',size=12)

#comparison plots
fig, [ax1,ax2] = plt.subplots(1,2) #P figures
ax1.axhline(bnd, 0, 1, c='k', ls='--')
ax2.axhline(bnd, 0, 1, c='k', ls='--')
fig.subplots_adjust(wspace=0.5)  
ax1.invert_yaxis(), ax2.invert_yaxis()
ax1.set_xlabel('$P_{s}$ [mg/m3]'), ax2.set_xlabel('$P_{l}$ [mg/m3]')
ax1.set_ylabel('Depth [m]')
ax1.set_ylim(top=0,bottom=zmax+dz), ax2.set_ylim(top=0,bottom=zmax+dz)
ax1.plot(Ps_numnl, zml, 'o', c=red, ms=ms, lw=lw, label = 'Model', fillstyle='none')
ax1.plot(Ps_oi, zml, 'o', c=blue, ms=ms/2, lw=lw, label = 'OI', fillstyle='none')
ax2.plot(Pl_numnl, zml, 'o', c=red, ms=ms, lw=lw, label = 'Model', fillstyle='none')
ax2.plot(Pl_oi, zml, 'o', c=blue, ms=ms/2, lw=lw, label = 'OI', fillstyle='none')
ax1.plot(Ps_mean, zs, '^', c=green, ms=ms*2, lw=lw, label = 'Data', fillstyle='full')
ax2.plot(Pl_mean, zs, '^', c=green, ms=ms*2, lw=lw, label = 'Data', fillstyle='full')
ax1.legend(loc='lower right')
    
print(f'--- {time.time() - start_time} seconds ---')

plt.show()