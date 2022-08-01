"""Algorithm of total inversion (Tarantola and Valette, 1982)."""
import numpy as np

from src.modelequations import evaluate_model_equations


def calculate_xkp1(Co, xo, xk, f, F):
    """Calculate an estimate of the state vector.""

    Args:
        Co (np.ndarray): Error covariance matrix of prior estimates.
        xo (np.ndarray): State vector of prior estimates.
        xk (np.ndarray): State vector of estimates at the beginning of an
        iteration, k.
        f (np.ndarray): Vector of functions containing the model equations.
        F (np.ndarray): The Jacobian matrix.

    Returns:
        xkp1 (np.ndarray): State vector of new estimates produced after
        iteration k.
        CoFT, FCoFTi (np.ndarray): Matrix products used in subsequent steps.
    """
    CoFT = Co @ F.T
    FCoFT = F @ CoFT
    FCoFTi = np.linalg.inv(FCoFT)
    xkp1 = xo + CoFT @ FCoFTi @ (F @ (xk - xo) - f)

    return xkp1, CoFT, FCoFTi


def check_convergence(xk, xkp1):
    """Check if the model has converged.

    Args:
        xk (np.ndarray): State vector of estimates at the beginning of an
        iteration, k.
        xkp1 (np.ndarray): State vector of new estimates produced after
        iteration k.

    Returns:
        converged (bool): True if model converged.
        max_change (float): The magnitude of the largest change in a single
        state element before and after iteration k.
    """
    converged = False
    max_change_limit = 10**-6
    change = np.abs((xkp1 - xk) / xk)
    max_change = np.max(change)

    if max_change < max_change_limit:
        converged = True

    return converged, max_change


def calculate_cost(Co, xo, x):
    """Evaluate the cost function given an estimate of the state vector, x"""
    cost = (x - xo).T @ np.linalg.inv(Co) @ (x - xo)

    return cost


def find_solution(equation_elements, xo, Co, grid, zg, umz_start, mld=None,
                  state_elements=None, soft_constraint=False):
    """An iterative approach for finding a solution to a nonlinear system.

    Args:
        state_elements (list[str]): Names of state elements.
        equation_elements (list[str]): Names of state elements that have
        associated equations (i.e., the tracers).
        xo (np.ndarray): State vector of prior estimates.
        Co (np.ndarray): Error covariance matrix of prior estimates.
        grid (list[float]): The model grid.
        zg (float): The maximum grazing depth, also the base of the euphotic
        zone.
        umz_start (int): Index of grid which corresponds to the depth of the
        base of the first layer in the upper mesopelagic zone.
        mld (float): Mixed layer depth.

    Returns:
        xhat (np.ndarray): Estimate of the state vector produced once the model
        has converged (i.e., the solution).
        Ckp1 (np.ndarray): Error covariance matrix of posterior estimates.
        convergence_evolution (list): A history of the magnitude of the
        maximum elementwise change in the state vector before and after an
        iteration.
        cost_evolution (list): A history of the cost.
        converged (bool): True if model converged.
    """
    max_iterations = 50
    convergence_evolution = []
    cost_evolution = []
    xhat = np.full(xo.shape, -9999)
    Ckp1 = np.full(Co.shape, -9999)

    xk = xo
    xkp1 = np.ones(len(xk))  # at iteration k+1
    for count in range(max_iterations):
        f, F = evaluate_model_equations(
            equation_elements, xk, grid, zg, umz_start, mld,
            state_elements=state_elements, soft_constraint=soft_constraint)

        xkp1, CoFT, FCoFTi = calculate_xkp1(Co, xo, xk, f, F)
        cost = calculate_cost(Co, xo, xkp1)

        cost_evolution.append(cost)
        if count > 0:  # xk contains 0's for residuals when k=0
            converged, max_change = check_convergence(xk, xkp1)
            convergence_evolution.append(max_change)
            if converged:
                Ckp1 = Co - CoFT @ FCoFTi @ F @ Co
                xhat = xkp1
                break
        xk = xkp1

    return xhat, Ckp1, convergence_evolution, cost_evolution, converged


def normalized_state_residuals(xhat, xo, Co):
    """Calculate residuals of state estimates relative to prior estimates."""
    x_resids = list((xhat - xo) / np.sqrt(np.diag(Co)))

    return x_resids


def success_check(converged, state_elements, xhat, Ckp1, zg):
    """Check for negative concentrations and model parameters in a solution."""
    if not converged:
        return False
    
    indexes = [i for i, s in enumerate(state_elements) if 'R' not in s]
    nonresidual_estimates = [xhat[i] for i in indexes]
    negative_estimates = any(i < 0 for i in nonresidual_estimates)
    if negative_estimates:
        return False
    
    variances = np.diag(Ckp1)
    nonresidual_variances = [variances[i] for i in indexes]
    negative_variances = any(i < 0 for i in nonresidual_variances)
    if negative_variances:
        return False
    
    zm = xhat[state_elements.index('zm')]
    if zm < zg:
        return False

    return True
