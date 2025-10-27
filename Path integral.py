"""
Path integral.py

This script numerically evaluates the Euclidean path integral for a one-dimensional harmonic oscillator 
using a lattice discretization approach combined with multidimensional numerical integration. 

Specifically, it computes the quantum propagator K(x, x; T) for closed paths with fixed endpoints x, 
where T corresponds to the total Euclidean time (related to lattice spacing and number of sites). 
The propagator is evaluated by integrating over all intermediate positions on the lattice, 
yielding the transition amplitude for a particle starting and ending at position x. 

This implementation serves as a brute-force integration demonstration (not a Monte Carlo simulation), 
illustrating the direct numerical computation of the path integral on a discretized lattice.

"""

import math
import matplotlib.pyplot as plt
from scipy.integrate import nquad

# Number of lattice sites (including fixed endpoints) discretizing the Euclidean time interval
N = 4

# Lattice spacing 'a' corresponds to the discretized Euclidean time step size
lattice_spacing_a = 1/2  

# Mass of the quantum particle (in natural units where ħ=1)
particle_mass = 1.0  

# Integration bounds limit for each intermediate lattice coordinate
bound_limit = 5  

# Integration bounds for the N-1 intermediate coordinates (excluding fixed endpoints)
# Fixed endpoints are held constant; integration is performed over intermediate positions only
bounds = [(-bound_limit, bound_limit)] * (N - 1)  

# List to store the computed propagator values for different fixed endpoint positions
propagator = []  

# Normalization factor for the path integral measure on the lattice
normalization_A = (particle_mass / (2 * math.pi * lattice_spacing_a)) ** (N / 2)  


def potential_V(x):
    """
    Harmonic oscillator potential function V(x) = (1/2) * x^2.
    
    This function defines the potential energy at position x. It can be modified to represent 
    different potentials for other quantum systems.
    """
    return 0.5 * x**2


def S_lat(x_list, x_fixed, *args):
    """
    Compute the Euclidean lattice action S for a discretized path with fixed endpoints.
    
    The action is the sum over lattice sites of kinetic and potential energy contributions:
        S = Σ_j [ (m / 2a) * (x_{j+1} - x_j)^2 + a * V(x_j) ]
    
    Here, 'a' is the lattice spacing, 'm' is the particle mass, and V(x) is the potential.
    
    The path is specified by intermediate positions x_list, with endpoints fixed at x_fixed.
    This corresponds to evaluating the propagator ⟨x_fixed| e^{-H T} |x_fixed⟩, where T = N * a.
    
    Parameters:
        x_list : list of float
            Intermediate lattice positions (excluding endpoints).
        x_fixed : float
            Fixed endpoint position (x_0 = x_N = x_fixed).
        *args : additional arguments (not used).
    
    Returns:
        float: The Euclidean lattice action S for the path.
    """
    x = [x_fixed] + list(x_list) + [x_fixed]  # Complete path including fixed endpoints
    Action_S = 0
    for j in range(0, N - 1):
        # Discrete derivative approximating the Euclidean time derivative (x_{j+1} - x_j)
        x_derivative = x[j + 1] - x[j]
        # Sum kinetic and potential contributions at each lattice site
        Action_S += (particle_mass / (2 * lattice_spacing_a)) * x_derivative**2 + lattice_spacing_a * potential_V(x[j])
    return Action_S


# Compute the propagator for multiple fixed endpoint positions x_fixed
# For each fixed endpoint, integrate over all intermediate coordinates to evaluate the path integral
x_values = [i * 0.25 for i in range(-10, 11)]
for x_fixed in x_values:
    # Define the integrand representing the Euclidean path integral weight exp(-S_E[x])
    def integrand(*x_list):
        return math.exp(-S_lat(x_list, x_fixed))
    
    # Perform multidimensional integration over all intermediate coordinates x_1,...,x_{N-1}
    result, error = nquad(integrand, bounds)
    
    # Store the normalized propagator value for the given fixed endpoint
    propagator.append(normalization_A * result)  


# Plot the propagator as a function of the fixed endpoint position x_fixed
# X-axis: fixed endpoints; Y-axis: corresponding normalized propagator values
plt.plot(x_values, propagator)
plt.xlabel("Fixed endpoint position x")
plt.ylabel("Normalized propagator K(x, x; T)")
plt.title(f"Numerical Path Integral (N={N})")
plt.grid(True)
plt.show()