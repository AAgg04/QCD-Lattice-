"""
Lattice Field Theory.py

This script implements a lattice scalar ϕ⁴ field theory simulation in (d+1)-dimensional Euclidean spacetime. 
Using the Metropolis Monte Carlo algorithm, it generates field configurations with periodic boundary conditions 
and computes physical observables such as the two-point correlation function. The primary goal is to extract 
the effective mass of the scalar particle by analyzing the exponential decay of the correlator in Euclidean time.

The code performs the following steps:
- Initializes the lattice field and parameters (mass, coupling, lattice spacing).
- Updates field configurations through local Metropolis sweeps respecting detailed balance.
- Measures the two-point correlation function averaged over all spatial and temporal origins to improve statistics.
- Extracts the effective mass from the correlator decay using a cosh fit that accounts for periodic boundary conditions.

Methods:
- Discretized Euclidean scalar field theory with ϕ⁴ self-interaction.
- Periodic boundary conditions in all lattice directions.
- Statistical analysis using bootstrap and error propagation techniques.
- Optional multi-run averaging to enhance statistical reliability.

Output:
- Log-scaled plot of the two-point correlation function G(t) illustrating exponential decay.
- Plot of the effective mass m_eff(t) versus Euclidean time showing plateau behavior.
- Printed fitted mass extracted from the cosh fit to the correlator.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
spatial_dims = 1           # Number of spatial dimensions
lattice_spacing_a = 1.0    # Lattice spacing (physical distance between adjacent lattice points)
lattice_size_L = 8         # Number of lattice points per dimension (spatial and temporal)
particle_mass = 1.0        # Bare mass parameter of the scalar particle (in natural units)
lambda_coupling = 0.1      # Quartic self-coupling constant (ϕ^4 interaction strength)

# Derived quantities
D = spatial_dims + 1                       # Total spacetime dimensions (spatial + Euclidean time)
phi = np.zeros((lattice_size_L,)*D)       # Scalar field array defined on the lattice; last axis corresponds to Euclidean time by convention
eps_tuned = 0.7                            # Proposal step size for Metropolis updates

# ---------------- Local physics functions ----------------

def potential_energy_V(field_phi_value):
    """Compute the local potential energy density at a single lattice site:
    V(ϕ) = 1/2 m^2 ϕ^2 + (λ/4) ϕ^4
    """
    return 0.5 * particle_mass**2 * field_phi_value**2 + (lambda_coupling / 4.0) * field_phi_value**4

def x_neighbor(x, mu, shift=1):
    """Return the lattice coordinate tuple with the mu-th component shifted by +shift,
    enforcing periodic boundary conditions in all directions.
    """
    x_new = list(x)
    x_new[mu] = (x_new[mu] + shift) % lattice_size_L
    return tuple(x_new)

def local_action(x, field_phi):
    """
    Compute the local action contribution at lattice site x, including the potential energy 
    and symmetric kinetic terms from both forward and backward nearest neighbors. 
    This symmetric coupling ensures proper discretization of the Laplacian operator.
    """
    phi_site = field_phi[x]
    action_S = potential_energy_V(phi_site)
    for mu in range(field_phi.ndim):
        # Include contributions from both forward and backward nearest neighbors to maintain symmetry
        forward_neighbor = x_neighbor(x, mu, shift=1)
        backward_neighbor = x_neighbor(x, mu, shift=-1)
        action_S += 0.5 * (field_phi[forward_neighbor] - phi_site)**2 / (lattice_spacing_a**2)
        action_S += 0.5 * (field_phi[backward_neighbor] - phi_site)**2 / (lattice_spacing_a**2)
    return action_S

# ---------------- Monte Carlo / Metropolis ----------------

def metropolis_update_field(field_phi, proposal_delta=0.5):
    """
    Perform one Metropolis sweep over the entire lattice.
    For each lattice site, propose a new field value by adding a random perturbation.
    Accept or reject the proposal according to the Boltzmann factor exp(-ΔS), where ΔS is the change in local action.
    The input array 'field_phi' is modified in-place.
    Returns the number of accepted moves and total proposals.
    """
    accepted_moves = 0
    proposals_count = 0
    for x in np.ndindex(*field_phi.shape):
        old_value = field_phi[x]
        old_local_action = local_action(x, field_phi)

        # Propose a new field value by a uniform random shift within [-proposal_delta, proposal_delta]
        new_value = old_value + np.random.uniform(-proposal_delta, proposal_delta)
        field_phi[x] = new_value
        new_local_action = local_action(x, field_phi)

        delta_S = new_local_action - old_local_action
        proposals_count += 1
        # Accept the move with probability min(1, exp(-ΔS))
        if delta_S < 0 or np.exp(-delta_S) > np.random.rand():
            accepted_moves += 1
        else:
            # Reject the move and revert to old value
            field_phi[x] = old_value
    return accepted_moves, proposals_count

# ---------------- Measurements & Statistics ----------------

def measure_field_correlation_all_origins(field_phi):
    """
    Compute the two-point correlation function G(Δt) defined as the average over all spatial and temporal origins:
    G(Δt) = (1 / (V * Lt)) * sum_{t0, x} < ϕ(x, t0) ϕ(x, t0 + Δt) >
    where V = L^d is the number of spatial lattice points and Lt = L is the temporal extent.
    Averaging over all origins reduces statistical noise and exploits translational invariance.
    Assumes the last axis of 'field_phi' corresponds to Euclidean time.
    """
    temporal = field_phi.shape[-1]
    spatial = field_phi.shape[:-1]
    G_correlator = np.zeros(temporal)
    for delta_t in range(temporal):
        correlation_sum = 0.0
        count = 0
        for t_0 in range(temporal):
            t_new = (t_0 + delta_t) % temporal
            for X in np.ndindex(*spatial):
                idx0 = X + (t_0,)
                idx1 = X + (t_new,)
                correlation_sum += field_phi[idx0] * field_phi[idx1]
                count += 1
        G_correlator[delta_t] = correlation_sum / count
    return G_correlator

def compute_effective_mass(G_correlator, G_correlator_err=None):
    """
    Compute the effective mass m_eff(t) from the correlator via:
    m_eff(t) ≈ ln(G(t)/G(t+1))
    This quantity approximates the energy gap between the ground state and first excited state on the lattice.
    If error estimates are provided, exclude points where signal-to-noise is poor.
    """
    m_eff = np.full(len(G_correlator) - 1, np.nan)
    for t in range(len(G_correlator) - 1):
        if G_correlator[t] > 0 and G_correlator[t + 1] > 0:
            if G_correlator_err is not None and (G_correlator[t] < 2*G_correlator_err[t] or G_correlator[t + 1] < 2*G_correlator_err[t + 1]):
                continue
            m_eff[t] = np.log(G_correlator[t] / G_correlator[t + 1])
    return m_eff

# ---------------- Simulation runner ----------------

def run_field_simulation(field_phi, MC_sweeps=2000, correlation_length=10, eps=0.5, burn_in_sweeps=500):
    """
    Run the Monte Carlo simulation:
    - Thermalization phase: perform burn-in sweeps to reach equilibrium.
    - Measurement phase: generate field configurations and measure correlators at intervals defined by correlation_length.
    The input field is mutated in-place.
    Returns a tuple containing the array of measurements, mean correlator, correlator error, and effective mass.
    """
    # Thermalize the system to reach equilibrium distribution
    accepted_moves = proposals_count = 0
    for i in range(burn_in_sweeps):
        accept, propose = metropolis_update_field(field_phi, eps)
        accepted_moves += accept; proposals_count += propose
    print(f"Post-thermalization acceptance fraction: {accepted_moves/proposals_count:.3f}")

    # Collect measurements after thermalization
    measurements = []
    accepted_moves = proposals_count = 0
    for sweep in range(MC_sweeps):
        accept, propose = metropolis_update_field(field_phi, eps)
        accepted_moves += accept; proposals_count += propose
        # Measure correlator every 'correlation_length' sweeps to reduce autocorrelation
        if sweep % correlation_length == 0:
            G_correlator = measure_field_correlation_all_origins(field_phi)
            measurements.append(G_correlator)
    print(f"Measurement acceptance fraction: {accepted_moves/proposals_count:.3f}")

    measurement_array = np.array(measurements)
    G_mean = np.mean(measurement_array, axis=0)
    G_err = np.std(measurement_array, axis=0, ddof=1)
    m_eff = compute_effective_mass(G_mean, G_err)
    return measurement_array, G_mean, G_err, m_eff

def run_multiple_simulations(number_of_runs, MC_sweeps, correlation_length, eps, burn_in_sweeps):
    """
    Perform multiple independent simulation runs to improve statistical accuracy.
    Each run starts from a different random initial field configuration.
    Results are averaged over runs, and errors are reduced by sqrt(number_of_runs).
    """
    all_measurement = []
    for i in range(number_of_runs):
        # Initialize field with small Gaussian noise to break symmetry
        field = 0.01 * np.random.randn(*(lattice_size_L,)*D)
        measured_correlation, i, i, i = run_field_simulation(field, MC_sweeps, correlation_length, eps, burn_in_sweeps)
        all_measurement.append(measured_correlation)

    all_measurement = np.vstack(all_measurement)
    G_mean = np.mean(all_measurement, axis=0)
    G_err = np.std(all_measurement, axis=0, ddof=1)/np.sqrt(all_measurement.shape[0])
    m_eff = compute_effective_mass(G_mean, G_err)

    return G_mean, G_err, m_eff

# ---------------- Main execution ----------------

G_mean, G_err, m_eff = run_multiple_simulations(number_of_runs=5, MC_sweeps=4000, correlation_length=20, eps=eps_tuned, burn_in_sweeps=1000)

print("G_mean:", G_mean)
print("G_err:", G_err)
print("m_eff:", m_eff)



# ----------- Cosh fit section -----------
def cosh_model(t, A, m):
    # Cosh model accounts for periodic boundary conditions in Euclidean time,
    # modeling the correlator as G(t) = A * (exp(-m t) + exp(-m (L - t)))
    return A * (np.exp(-m*t) + np.exp(-m*(lattice_size_L - t)))

# Symmetrize G_mean to enforce time-reversal symmetry in Euclidean time correlators
G_mean = 0.5 * (G_mean + G_mean[::-1])

# Fit the correlator to the cosh model in the region where signal dominates noise (G_mean > 5 * G_err)
time_data = np.arange(lattice_size_L)
mask = G_mean > 5 * G_err
optimized_parameter, covariance_parameter = curve_fit(cosh_model, time_data[mask], G_mean[mask], p0=[0.2, 1.0])
A_fit, m_fit = optimized_parameter
print(f"Fitted mass m = {m_fit:.3f}")

# Plot: Two-point correlation function G(t)
# The plot shows exponential decay modulated by periodic boundary conditions.
G_plot = np.abs(G_mean)
plt.errorbar(time_data, G_plot, yerr=G_err, fmt='o', capsize=3, label='|G(t)|')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel('Euclidean time t')
plt.ylabel('G(t)')
plt.title('Two-point correlation function')
plt.grid(True)
plt.legend()
plt.show()

# Plot: Effective mass m_eff(t)
# The effective mass plot shows the energy gap plateau extracted from correlator ratios.
plt.plot(np.arange(len(m_eff)), m_eff, 'o-')
plt.xlabel('t')
plt.ylabel('m_eff(t)')
plt.title('Effective mass vs time')
plt.grid(True)
plt.show()

# Plot: Cosh fit over the correlator data
G_fit = cosh_model(time_data, A_fit, m_fit)

plt.errorbar(time_data, G_plot, yerr=G_err, fmt='o', capsize=3, label='|G(t)| data')
plt.plot(time_data, G_fit, '-', label=f'Cosh fit (m = {m_fit:.3f})')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel('Euclidean time t')
plt.ylabel('G(t)')
plt.title('Two-point correlation function with cosh fit')
plt.grid(True)
plt.legend()
plt.show()