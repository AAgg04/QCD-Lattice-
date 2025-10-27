"""
@mainpage Monte Carlo Simulation for the 1D Quantum Anharmonic Oscillator
@file monte_carlo_anharmonic.py
@brief Monte Carlo Euclidean path-integral estimator for the ground-state energy E₀.

@details
This script estimates the ground-state energy of a 1D quantum anharmonic oscillator
with Euclidean action

    S_E = Σ_j [ m/(2a) (x_{j+1}-x_j)² + a V(x_j) ],

where
    V(x) = 1/2 x² + λ x⁴.

Configurations of x(τ) are sampled using local Metropolis updates with periodic
boundary conditions. The two-point correlator

    C(τ) = ⟨x(0)x(τ)⟩ ∼ e^(−E₀ τ),

is accumulated and fitted to a single exponential in the plateau region to extract E₀.
"""

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
#                 Physical and Simulation Parameters
# ===============================================================

particle_mass = 1.0
#: float: mass of particle (in natural units ℏ = 1).

lattice_spacing_a = 0.1
#: float: Euclidean lattice spacing a = Δτ.

number_of_sites = 100
#: int: number of lattice sites (total imaginary time extent T = N·a).

total_monte_carlo_steps = 30000
#: int: total Metropolis sweeps (equilibration + measurement).

thermalization_steps = 5000
#: int: number of initial sweeps discarded before measurements.

proposal_step_size = 0.5
#: float: amplitude of uniform proposal displacement for local updates.

anharmonicity_values = [0.0, 0.1, 0.3, 0.5, 1.0]
#: list(float): list of anharmonicity parameters λ to simulate.

np.random.seed(42)  # reproducibility


# ===============================================================
#                Potential Energy Function
# ===============================================================

def potential_energy_V(position, lambda_parameter):
    """
    @brief Anharmonic potential energy V(x).
    @param position float or ndarray: spatial coordinate(s) x.
    @param lambda_parameter float: anharmonicity λ (λ → 0 gives harmonic limit).
    @return float or ndarray: potential energy V(x).
    @details
    Implements:
        V(x) = 1/2 x² + λ x⁴.
    """
    return 0.5 * position**2 + lambda_parameter * position**4


# ===============================================================
#         Local Euclidean Action Contributions
# ===============================================================

def local_action(x_prev, x_current, x_next, lambda_parameter):
    """
    @brief Local contribution to discretized Euclidean action around site j.
    @param x_prev float: x at site j−1.
    @param x_current float: x at site j.
    @param x_next float: x at site j+1.
    @param lambda_parameter float: anharmonicity λ.
    @return float: local action S_E(j).
    @details
    Uses symmetric discretized kinetic term:
        S_kin(j) = m/(4a)[(x_{j+1}-x_j)² + (x_j - x_{j-1})²]
    plus potential:
        S_pot(j) = a V(x_j).
    Periodic boundary conditions handled externally.
    """
    S_kinetic_local = 0.5 * particle_mass / lattice_spacing_a * \
                      ((x_next - x_current)**2 + (x_current - x_prev)**2) / 2
    S_potential_local = lattice_spacing_a * potential_energy_V(x_current, lambda_parameter)
    return S_kinetic_local + S_potential_local


def delta_action_change(x_path, j, x_new, lambda_parameter):
    """
    @brief Compute local change ΔS_E for a proposed update at site j.
    @param x_path ndarray(float): full current path configuration.
    @param j int: lattice site index for update.
    @param x_new float: proposed new value for x[j].
    @param lambda_parameter float: anharmonicity λ.
    @return float: ΔS = S_new − S_old.
    @details
    Only the action terms involving sites {j−1, j, j+1} contribute to ΔS.
    Indices wrap via periodic boundary conditions.
    """
    j_minus = (j - 1) % number_of_sites
    j_plus = (j + 1) % number_of_sites

    S_old = local_action(x_path[j_minus], x_path[j], x_path[j_plus], lambda_parameter)
    S_new = local_action(x_path[j_minus], x_new, x_path[j_plus], lambda_parameter)

    return S_new - S_old


# ===============================================================
#         Monte Carlo Simulation for Given λ
# ===============================================================

def run_monte_carlo(lambda_parameter):
    """
    @brief Perform local Metropolis updates to sample Euclidean paths for fixed λ.
    @param lambda_parameter float: anharmonicity value for this simulation.
    @return tuple: (C_tau, acceptance_fraction)
        - C_tau: ndarray(float) correlator C(τ) for τ up to T/2
        - acceptance_fraction: overall acceptance rate of updates
    @details
    • Initializes x(τ)=0 path
    • Local updates at every site each sweep
    • Observables recorded every 10 steps post-thermalization
    • Correlator estimator:
            C(τ) = ⟨ x_j x_{j+τ} ⟩ averaged over j and Monte Carlo samples
    """
    x_path = np.zeros(number_of_sites)
    G_correlator = np.zeros(number_of_sites // 2)
    N_measure = 0
    accepted_updates = 0

    for monte_carlo_step in range(total_monte_carlo_steps):
        for j in range(number_of_sites):
            x_new = x_path[j] + np.random.uniform(-proposal_step_size, proposal_step_size)
            delta_S_local = delta_action_change(x_path, j, x_new, lambda_parameter)

            if delta_S_local < 0 or np.exp(-delta_S_local) > np.random.rand():
                x_path[j] = x_new
                accepted_updates += 1

        if monte_carlo_step >= thermalization_steps and monte_carlo_step % 10 == 0:
            for t_index in range(number_of_sites // 2):
                G_correlator[t_index] += np.mean(x_path * np.roll(x_path, -t_index))
            N_measure += 1

    G_correlator /= N_measure
    acceptance_fraction = accepted_updates / (total_monte_carlo_steps * number_of_sites)
    return G_correlator, acceptance_fraction


# ===============================================================
#          Main Loop — Extract Ground-State Energy
# ===============================================================

estimated_ground_energies = []
#: list(float): extracted ground-state energies E₀(λ) from exponential fits.

for lambda_parameter in anharmonicity_values:
    correlation_function, acceptance_rate = run_monte_carlo(lambda_parameter)

    correlation_function /= correlation_function[0]
    euclidean_times = np.arange(len(correlation_function)) * lattice_spacing_a

    fit_slice = slice(1, 6)  # small τ region where log(C) approx linear
    slope, intercept = np.polyfit(euclidean_times[fit_slice],
                                  np.log(correlation_function[fit_slice]), 1)
    estimated_E0 = -slope
    estimated_ground_energies.append(estimated_E0)

    print(f"λ = {lambda_parameter:.2f} | Estimated E0 = {estimated_E0:.4f} "
          f"| Acceptance = {acceptance_rate:.3f}")


# ===============================================================
#                   Visualization
# ===============================================================

plt.plot(anharmonicity_values, estimated_ground_energies, 'o-', lw=2)
plt.xlabel("Anharmonicity Parameter λ")
plt.ylabel("Estimated Ground-State Energy $E_0$")
plt.title("Quantum Anharmonic Oscillator: Ground-State Energy vs λ")
plt.grid(True)
plt.show()
