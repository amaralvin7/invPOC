import numpy as np

from src.modelequations import evaluate_model_equations

def calculate_xkp1(Co, xo, xk, f, F):

    CoFT = Co @ F.T
    FCoFT = F @ CoFT
    FCoFTi = np.linalg.inv(FCoFT)
    xkp1 = xo + CoFT @ FCoFTi @ (F @ (xk - xo) - f)

    return xkp1, CoFT, FCoFTi

def check_convergence(xk, xkp1):

    converged = False
    convergence = []
    max_change_limit = 10**-6
    change = np.abs((xkp1 - xk)/xk)
    convergence.append(np.max(change))

    if np.max(change) < max_change_limit:
        converged = True

    return converged, convergence

def calculate_cost(Co, xo, x):

    cost = (x - xo).T @ np.linalg.inv(Co) @ (x - xo)

    return cost

def find_solution(
    tracers, state_elements, equation_elements, xo, Co, grid, zg, mld,
    productionbool, umz_start, priors_from, station):

    max_iterations = 200
    convergence_evolution = []
    cost_evolution = []

    xk = xo
    xkp1 = np.ones(len(xk))  # at iteration k+1
    for count in range(max_iterations):
        f, F = evaluate_model_equations(
            tracers, state_elements, equation_elements, xk, grid, zg, mld,
            productionbool, umz_start)

        xkp1, CoFT, FCoFTi = calculate_xkp1(Co, xo, xk, f, F)
        cost = calculate_cost(Co, xo, xkp1)

        cost_evolution.append(cost)
        if count > 0:  # xk contains 0's for residuals when k=0
            converged, convergence = check_convergence(xk, xkp1)
            convergence_evolution.append(convergence)
            if converged:
                break
        xk = xkp1

    Ckp1 = Co - CoFT @ FCoFTi @ F @ Co
    xhat = xkp1

    if not converged:
        print(priors_from, station)

    return xhat, Ckp1, convergence_evolution, cost_evolution
