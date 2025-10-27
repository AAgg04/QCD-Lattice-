"""
@mainpage Numerical Euclidean Path Integral for the Harmonic Oscillator
@file path_integral.py
@brief Brute-force multidimensional Euclidean path integral evaluation.

@details
This demonstration numerically evaluates the diagonal propagator
K(x, x; T) = ⟨x| e^(−H T) |x⟩
for a 1D harmonic oscillator using a discretized Euclidean action:

    S_E = Σ_j [ m/(2a) (x_{j+1} − x_j)² + a V(x_j) ],

with fixed endpoints x₀ = x_N = x_fixed and (N−1) internal lattice points
integrated over a finite domain. The integral is computed using
SciPy’s multi-dimensional quadrature (`nquad`), which scales
exponentially with N and serves only as a pedagogical reference
(not an efficient Monte Carlo method).
"""

import math
import matplotlib.pyplot as plt
from scipy.integrate import nquad

# ===============================================================
#              Physical and Discretization Parameters
# ===============================================================

N = 4
#: int: number of lattice sites including fixed endpoints (integration dimension = N−1).

lattice_spacing_a = 1 / 2
#: float: Euclidean time spacing, total extent T = N · a.

particle_mass = 1.0
#: float: mass of particle (natural units ℏ = 1).

bound_limit = 5
#: float: |x| integration domain bound for intermediate positions.

bounds = [(-bound_limit, bound_limit)] * (N - 1)
#: list(tuple): integration bounds for each of the (N−1) intermediate coordinates.

propagator = []
#: list(float): evaluated propagator K(x, x; T) at each fixed endpoint x.

normalization_A = (particle_mass / (2 * math.pi * lattice_spacing_a)) ** (N / 2)
#: float: Gaussian normalization prefactor from discretized measure.


# ===============================================================
#                      Potential Energy
# ===============================================================

def potential_V(x):
    """
    @brief Harmonic oscillator potential.
    @param x float: position value.
    @return float: potential V(x) = 1/2 x².
    """
    return 0.5 * x**2


# ===============================================================
#                        Euclidean Action
# ===============================================================

def S_lat(x_list, x_fixed, *args):
    """
    @brief Compute discretized Euclidean path action.
    @param x_list list(float): internal coordinates, length (N−1).
    @param x_fixed float: fixed boundary value x₀ = x_N.
    @return float: Euclidean action S_E for given path.
    @details
    Constructs full path:
        x = [x_fixed, x₁, x₂, ..., x_{N−1}, x_fixed]
    and applies:
        S_E = Σ_j m/(2a)(x_{j+1} − x_j)² + a V(x_j)
    without periodic BCs since endpoints are fixed.
    """
    x = [x_fixed] + list(x_list) + [x_fixed]
    Action_S = 0
    for j in range(0, N - 1):
        x_derivative = x[j + 1] - x[j]
        Action_S += (particle_mass / (2 * lattice_spacing_a)) * x_derivative**2 \
                    + lattice_spacing_a * potential_V(x[j])
    return Action_S


# ===============================================================
#               Propagator Evaluation Loop
# ===============================================================

x_values = [i * 0.25 for i in range(-10, 11)]
#: list(float): fixed endpoint values used to evaluate K(x, x; T).

for x_fixed in x_values:

    def integrand(*x_list):
        """
        @brief Integrand exp(−S_E[x]) for numerical quadrature.
        @param x_list variadic float: internal lattice points.
        @return float: value of exp(−S_E).
        @details
        This closure captures `x_fixed` from the loop scope.
        """
        return math.exp(-S_lat(x_list, x_fixed))

    result, error = nquad(integrand, bounds)
    propagator.append(normalization_A * result)


# ===============================================================
#                      Visualization
# ===============================================================

plt.plot(x_values, propagator)
plt.xlabel("Fixed endpoint position x")
plt.ylabel("Normalized propagator K(x, x; T)")
plt.title(f"Numerical Path Integral (N={N})")
plt.grid(True)
plt.show()
