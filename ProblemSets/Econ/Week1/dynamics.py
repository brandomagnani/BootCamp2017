# ILDEBRANDO MAGNANI, joint with Francesco Furno


# Problem 5.3:


import numpy as np
import scipy.optimize as opt
from ss import equilibriumss, Wss, Rss, U_prime, c1, c2, c3
from numpy.linalg import norm




# Specify Parameters

period = 20
annual_beta1 = 0.96
beta1 = annual_beta1**period
annual_beta2 = 0.98
beta2 = annual_beta2**period
annual_delta = 0.05
delta = 1 - (1 - annual_delta)**period
sigma = 3
A = 1
alpha = 0.35
L = 2.2




# Get SS values

b_ss_guess = np.array([0.1, 0.1])
b_ss1  = opt.fsolve(equilibriumss, b_ss_guess, args=(beta1, sigma, alpha, delta))




#Specify TPI Parameters

b_initial = np.array([0.8*b_ss1[0], 1.1*b_ss1[1]])
K_0 = np.sum(b_initial)
K_ss = np.sum(b_ss1)

eps = 10**(-9)
xi = 0.2
T = 55
params = (beta1, A, delta, sigma, alpha)
K_guess = np.linspace(K_0, K_ss, T)




# Define functions for Marginal Utility, Wage, Interest Rate, and MU of consumption for all cohorts

def MU(c, params):
    return c**(-sigma)


def get_W(K, L, params):
    return (1-alpha) * A * (K/L)**alpha


def get_R(K, L, params):
    return alpha * A * (L/K)**(1-alpha) - delta


def get_WR(K, L, params):
    w = get_W(K, L, params)
    r = get_R(K, L, params)
    return w, r


def MU_C1(w_minus, b2):
    return MU(w_minus - b2, params)


def MU_C2(w, r, b2, b3_plus):
    return MU(w + (1+r)*b2 - b3_plus, params)


def MU_C3(w_plus, r_plus, b3_plus):
    return MU((1+r_plus)*b3_plus + 0.2*w_plus, params)




# Specify functions that return errors for equilibrium conditions

def b32_error(b32, args):
    params, K, L, b21 = args
    w, r = get_WR(K, L, params)
    w_1  = w[0]
    w_2  = w[1]
    r_1  = r[0]
    r_2  = r[1]
    error = MU_C2(w_1, r_1, b21, b32) - beta1 * (1+r_2) * MU_C3(w_2, r_2, b32)
    return error


def b_error(b, args):
    params, K, L, t = args
    error = np.empty((2))
    w, r = get_WR(K, L, params)
    w_minus = w[t-1]
    w_t     = w[t]
    w_plus  = w[t+1]
    r_t     = r[t]
    r_plus  = r[t+1]
    b2, b3_plus = b
    error_0 = MU_C1(w_minus, b2) - beta1 * (1+r_t) * MU_C2(w_t, r_t, b2, b3_plus)
    error_1 = MU_C2(w_t, r_t, b2, b3_plus) - beta1 * (1+r_plus) * MU_C3(w_plus, r_plus, b3_plus)
    error[0] = error_0
    error[1] = error_1
    return error




# Define functions that carries out TPI Iteration until convergence. It returns the path for K

def Tpi(b_start, args):
    params, K, L, eps, xi = args
    b21 = b_start[0]
    b32_guess = 0.1
    b32_error_params = [params, K, L, b21]
    b32 = opt.fsolve(b32_error, b32_guess, args=b32_error_params)[0]
    b2_list = [b_start[0]]
    b3_list = [b_start[1], b32]
    valid = False
    iterations = 1

    while valid == False:
        
        for i in range(T-1):

            b_error_params = [params, K, L, i]
            b_error_root_guess = (0.1, 0.1)
            b2t, b3t_plus  = opt.fsolve(b_error, b_error_root_guess, b_error_params)
            b2_list.append(b2t)
            b3_list.append(b3t_plus)     
        
        K_prime_list = [x + y for x, y in zip(b2_list, b3_list)]
        K_prime_array = np.asarray(K_prime_list)
        K_difference = K - K_prime_array
        K_distance = norm(K_difference)
        
        b2_list = [b_start[0]]
        b3_list = [b_start[1], b32]
        K_prime_list = []
        
        
        if K_distance < eps:
            print("Number of Iterations: ", iterations+1)
            valid = True
            
        
        else:
            K_guess1 = xi * K_prime_array + (1 - xi) * K
            K  = K_guess1
            iterations += 1

    return K_prime_array


