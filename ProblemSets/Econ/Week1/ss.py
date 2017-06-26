# ILDEBRANDO MAGNANI, joint with Francesco Furno


# Exercises 5.1, 5.2 : 


import numpy as np
import scipy.optimize as opt



# Parameters

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




# Marginal Utility, SS

def U_prime(c, sigma):
 
    return c**(-sigma)




# Wage, SS

def Wss(b, alpha):
    b2 = b[0]
    b3 = b[1]
    
    return (1-alpha) * A * ((b2 + b3)/2.2)**alpha




# Interest Rate, SS

def Rss(b, alpha, delta):
    b2 = b[0]
    b3 = b[1]
    
    return alpha * A * (2.2/(b2 + b3))**(1-alpha) - delta




# Equilibrium conditions, returns Errors

def equilibriumss(b, beta, sigma, alpha, delta):
    b2 = b[0]
    b3 = b[1]
    
    error = np.empty((2))
    
    error[0] = U_prime(Wss(b, alpha) - b2, sigma) - beta * (1+Rss(b, alpha, delta)) * \
               U_prime(Wss(b, alpha) + (1+Rss(b, alpha, delta))*b2 - b3, sigma)
    
    error[1] = U_prime(Wss(b, alpha) + (1+Rss(b, alpha, delta))*b2 - b3, sigma) - beta * (1+Rss(b, alpha, delta)) * \
               U_prime((1 + Rss(b, alpha, delta))*b3 + 0.2*Wss(b, alpha), sigma)
    
    return error




# Consumption 1

def c1(vec):
    b2 = vec[0]
    w = vec[1]
    return w - b2




# Consumption 2

def c2(vec):
    b2 = vec[0]
    b3 = vec[1]
    r  = vec[2]
    w  = vec[3]
    return w + (1 + r)*b2 - b3




# Consumption 3

def c3(vec):
    b3 = vec[0]
    r  = vec[1]
    w  = vec[2]
    return 0.2*w + (1 + r)*b3

