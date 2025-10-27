"""
@mainpage Monte Carlo Euclidean Path Integral for the 1D Anharmonic Oscillator
@file Monte_carlo_path_integral.py
@brief Metropolis Monte Carlo sampling of discretized Euclidean paths to estimate the ground-state energy.
@details
This script samples discretized Euclidean-time paths x(τ) for a 1D quantum anharmonic oscillator
with potential V(x) = 1/2 x² + λ x⁴ using the Metropolis algorithm. The two-point correlator

    C(τ) = ⟨ x(0) x(τ) ⟩

is accumulated after thermalization and its large-τ exponential decay, C(τ) ∼ exp(−E₀ τ), is used
to estimate the ground-state energy E₀ via a linear fit to log C(τ). Comments and docstrings
are written in Doxygen-friendly style.
"""

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
#                   Physical and Simulation Parameters
# ===============================================================

particle_mass = 1.0
#: float: Mass of the quantum particle (natural units ℏ = 1).

lattice_spacing_a = 0.1
#: float: Discrete Euclidean time step Δτ; smaller values approach the continuum limit.

number_of_sites = 100
#: int: Number of lattice points along Euclidean time; total Euclidean extent T = N * a.

total_monte_carlo_steps = 30000
#: int: Total number of Monte Carlo sweeps (includes thermalization and sampling).

thermalization_steps = 5000
#: int: Number of initial sweeps discarded to allow the Markov chain to equilibrate (burn-in).

proposal_step_size = 0.5
#: float: Maximum magnitude of uniform proposal displacement for local Metropolis updates.

# List of anharmonicity parameters λ to sweep over
anharmonicity_values = [0.0, 0.1, 0.3, 0.5, 1.0]
#: list(float): Anharmonicity strengths to simulate (λ).

# Fix random seed for reproducibility of Monte Carlo results
np.random.seed(42)


# ===============================================================
#                   Potential Energy Function
# ===============================================================

def potential_energy_V(position, lambda_parameter):
    """
    @brief Anharmonic potential V(x) = 1/2 x² + λ x⁴.
    @param position float: particle position x.
    @param lambda_parameter float: anharmonicity λ (λ = 0 → harmonic).
    @return float: potential energy V(x).
    @details
    This function is vectorizable (accepts numpy arrays) and used in the local action.
    """
    return 0.5 * position**2 + lambda_parameter * position**4


# ===============================================================
#                Local Euclidean Action Contributions
# ===============================================================

def local_action(x_prev, x_current, x_next, lambda_parameter):
    """
    @brief Local contribution to the discretized Euclidean action at a lattice site.
    @param x_prev float: x at site j−1.
    @param x_current float: x at site j.
    @param x_next float: x at site j+1.
    @param lambda_parameter float: anharmonicity λ.
    @return float: local action contribution S_E(j).
    @details
    The discretized action uses a symmetric kinetic term averaged from forward and backward
    finite differences:
        S_kin = (m / (4a)) [ (x_{j+1}-x_j)^2 + (x_j-x_{j-1})^2 ]
    plus the local potential term S_pot = a V(x_j).
    """
    # Symmetric discretization of kinetic term (averaging forward/backward differences)
    S_kinetic_local = 0.5 * particle_mass / lattice_spacing_a * ((x_next - x_current)**2 + (x_current - x_prev)**2) / 2
    S_potential_local = lattice_spacing_a * potential_energy_V(x_current, lambda_parameter)
    return S_kinetic_local + S_potential_local


def delta_action_change(x_path, j, x_new, lambda_parameter):
    """
    @brief Compute the local change ΔS_E for a proposed update at site j.
    @param x_path ndarray: current path array of length `number_of_sites`.
    @param j int: index of the lattice site being updated.
    @param x_new float: proposed new value for x[j].
    @param lambda_parameter float: anharmonicity λ.
    @return float: ΔS = S_new − S_old for the local neighborhood {j-1, j, j+1}.
    @details
    Only terms involving x_j appear in ΔS, so we evaluate the local action before and after
    the proposal using periodic boundary conditions for neighbors.
    """
    j_minus = (j - 1) % number_of_sites
    j_plus = (j + 1) % number_of_sites

    S_old = local_action(x_path[j_minus], x_path[j], x_path[j_plus], lambda_parameter)
    S_new = local_action(x_path[j_minus], x_new, x_path[j_plus], lambda_parameter)

    return S_new - S_old


# ===============================================================
#                Monte Carlo Simulation for One λ
# ===============================================================

def run_monte_carlo(lambda_parameter):
    """
    @brief Run Metropolis Monte Carlo sampling of Euclidean paths for a fixed λ.
    @param lambda_parameter float: anharmonicity strength λ.
    @return tuple: (G_correlator, acceptance_fraction)
        - G_correlator (ndarray): averaged two-point correlator C(τ) for τ = 0..N/2-1.
        - acceptance_fraction (float): overall acceptance ratio of proposed updates.
    @details
    The algorithm:
      1. Initialize x_path (zeros).
      2. For each Monte Carlo sweep, propose local updates at every site.
      3. Accept/reject via Metropolis: accept if ΔS < 0 or with probability exp(−ΔS).
      4. After thermalization, accumulate the correlator every 10 sweeps by averaging
         x_j x_{j+τ} over time origins using np.roll (translational averaging).
    """
    # Initialize path to zeros (arbitrary initial condition)
    x_path = np.zeros(number_of_sites)

    # Correlator accumulator for τ up to half the lattice (exploit reflection/periodicity)
    G_correlator = np.zeros(number_of_sites // 2)
    N_measure = 0
    accepted_updates = 0

    for monte_carlo_step in range(total_monte_carlo_steps):

        # Full lattice sweep proposing local updates site-by-site
        for j in range(number_of_sites):
            # Propose local displacement uniformly in [-proposal_step_size, proposal_step_size]
            x_new = x_path[j] + np.random.uniform(-proposal_step_size, proposal_step_size)

            # Compute local ΔS for proposed update
            delta_S_local = delta_action_change(x_path, j, x_new, lambda_parameter)

            # Metropolis acceptance: accept if ΔS < 0 or with probability exp(-ΔS)
            if delta_S_local < 0 or np.exp(-delta_S_local) > np.random.rand():
                x_path[j] = x_new
                accepted_updates += 1

        # Measurements: start after thermalization and sample every 10 sweeps
        if monte_carlo_step >= thermalization_steps and monte_carlo_step % 10 == 0:
            # Accumulate translationally averaged two-point correlator for τ up to N/2
            for t_index in range(number_of_sites // 2):
                G_correlator[t_index] += np.mean(x_path * np.roll(x_path, -t_index))
            N_measure += 1

    # Normalize correlator by number of measurements and compute acceptance fraction
    G_correlator /= N_measure
    acceptance_fraction = accepted_updates / (total_monte_carlo_steps * number_of_sites)

    return G_correlator, acceptance_fraction


# ===============================================================
#                   Main Loop over λ Values
# ===============================================================

estimated_ground_energies = []
#: list(float): extracted ground-state energies E₀ for each λ via log-linear fit to correlator.

for lambda_parameter in anharmonicity_values:
    correlation_function, acceptance_rate = run_monte_carlo(lambda_parameter)

    # Normalize correlator by its τ=0 value to form C(τ)/C(0)
    correlation_function /= correlation_function[0]
    euclidean_times = np.arange(len(correlation_function)) * lattice_spacing_a

    # Fit log C(τ) vs τ in a short τ window to estimate the slope ≈ −E₀
    fit_slice = slice(1, 6)  # chosen small-τ window for approximate single-exponential behavior
    slope, intercept = np.polyfit(euclidean_times[fit_slice], np.log(correlation_function[fit_slice]), 1)
    estimated_E0 = -slope
    estimated_ground_energies.append(estimated_E0)

    print(f"λ = {lambda_parameter:.2f} | Estimated E0 = {estimated_E0:.4f} | Acceptance = {acceptance_rate:.3f}")


# ===============================================================
#                   Plot Results
# ===============================================================

plt.plot(anharmonicity_values, estimated_ground_energies, 'o-', lw=2)
plt.xlabel("Anharmonicity Parameter λ")
plt.ylabel("Estimated Ground-State Energy $E_0$")
plt.title("Quantum Anharmonic Oscillator: Ground-State Energy vs λ")
plt.grid(True)
plt.show()
