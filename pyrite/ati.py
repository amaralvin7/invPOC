#!/usr/bin/env python3
"""
To do:
- convergence evolution and cost evolution
"""
from constants import LAYERS
from modelequations import evaluate_model_equations
import numpy as np

def calculate_xkp1(Co, xo, xk, f, F):

    CoFT = Co @ F.T
    FCoFT = F @ CoFT
    FCoFTi = np.linalg.inv(FCoFT)
    xkp1 = xo + CoFT @ FCoFTi @ (F @ (xk - xo) - f)

    return xkp1, CoFT, FCoFTi

def check_convergence(xk, xkp1):

    converged = False
    max_change_limit = 10**-6
    change = np.abs((xkp1 - xk)/xk)
    # Convergence_evolution.append(np.max(change))

    if np.max(change) < max_change_limit:
        converged = True

    return converged

def calculate_cost(Co, xo, x):

    cost = (x - xo).T @ np.linalg.inv(Co) @ (x - xo)

    # Cost_evolution.append(cost)

def find_solution(tracers, state_elements, equation_elements, xo, Co):

    max_iterations = 100

    xk = xo
    xkp1 = np.ones(len(xk))  # at iteration k+1
    for count in range(max_iterations):
        f, F = evaluate_model_equations(tracers, state_elements,
                                        equation_elements, xk)
        xkp1, CoFT, FCoFTi = calculate_xkp1(Co, xo, xk, f, F)
        calculate_cost(Co, xo, xkp1)
        if count > 0:
            converged = check_convergence(xk, xkp1)
            if converged:
                break
        xk = xkp1

    Ckp1 = Co - CoFT @ FCoFTi @ F @ Co
    xhat = xkp1
    
    return xhat, Ckp1
