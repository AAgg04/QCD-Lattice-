"""
Lattice field theory simulation for a scalar φ⁴ field in (d+1) dimensions.

Implements Metropolis Monte-Carlo updates for a real scalar field, computes
correlation functions, extracts effective mass via logarithmic and cosh fits.
Includes thermalization, autocorrelation spacing, and multi-run averaging.
"""

import numpy as np
from scipy.optimize import curve_fit
import itertools
import matplotlib.pyplot as plt


# --- Parameters ---
d = 1
"""Spatial dimensions."""

a = 1.0
"""Lattice spacing."""

L = 32
"""Sites per spatial and temporal direction."""

m = 1.0
"""Bare mass parameter."""

lambda_ = 0.1
"""φ⁴ interaction coupling. Set 0 for free-field tests."""

D = d + 1
"""Spacetime dimensions."""

phi = np.zeros((L,) * D)
"""Scalar field array; last axis = Euclidean time."""


# --- Local physics functions ---
def potential_at_site(phi_val):
    """
    Compute potential energy: V(φ) = ½ m²φ² + λ/4 φ⁴.
    """
    return 0.5 * m**2 * phi_val**2 + (lambda_/4.0) * phi_val**4


def neighbor_index(idx, mu, shift=1):
    """
    Periodic-BC neighbor index shift by ±1 in direction μ.
    """
    new = list(idx)
    new[mu] = (new[mu] + shift) % L
    return tuple(new)


def action_total(field):
    """
    Full lattice action including kinetic nearest-neighbor term.
    """
    S = 0.0
    for idx in np.ndindex(*field.shape):
        phi_site = field[idx]
        S += potential_at_site(phi_site)
        for mu in range(field.ndim):
            neigh = neighbor_index(idx, mu, shift=1)
            diff = field[neigh] - phi_site
            S += 0.5 * (diff**2) / (a**2)
    return S


def local_action_contribution(idx, field):
    """
    Local action contribution at a given lattice site incl. neighbors.
    Used for fast ΔS in Metropolis updates.
    """
    phi_site = field[idx]
    S = potential_at_site(phi_site)
    for mu in range(field.ndim):
        neigh_f = neighbor_index(idx, mu, shift=1)
        neigh_b = neighbor_index(idx, mu, shift=-1)
        S += 0.5 * (field[neigh_f] - phi_site)**2 / (a**2)
        S += 0.5 * (field[neigh_b] - phi_site)**2 / (a**2)
    return S


# --- Metropolis updates ---
def metropolis_update_field(field, eps=0.5):
    """
    Perform one Metropolis sweep; return accepted/proposed counts.
    """
    accepted = 0
    proposals = 0
    for idx in np.ndindex(*field.shape):
        old_val = field[idx]
        old_loc = local_action_contribution(idx, field)

        new_val = old_val + np.random.uniform(-eps, eps)
        field[idx] = new_val
        new_loc = local_action_contribution(idx, field)

        dS = new_loc - old_loc
        proposals += 1
        if dS > 0 and np.exp(-dS) < np.random.rand():
            field[idx] = old_val
        else:
            accepted += 1
    return accepted, proposals


# --- Measurements ---
def measure_field_correlation_all_origins(field):
    """
    Compute averaged two-point function G(Δt) over all origins.
    """
    Lt = field.shape[-1]
    spatial = field.shape[:-1]
    G = np.zeros(Lt)
    for dt in range(Lt):
        corr = 0.0
        count = 0
        for t0 in range(Lt):
            t1 = (t0 + dt) % Lt
            for idx in np.ndindex(*spatial):
                corr += field[idx + (t0,)] * field[idx + (t1,)]
                count += 1
        G[dt] = corr / count
    return G


def compute_effective_mass(G, G_err=None):
    """
    Effective mass via m_eff(t) = log(G(t)/G(t+1)).
    Masked if too noisy.
    """
    m_eff = np.full(len(G) - 1, np.nan)
    for t in range(len(G) - 1):
        if G[t] > 0 and G[t+1] > 0:
            if G_err is not None and (G[t] < 2*G_err[t] or G[t+1] < 2*G_err[t+1]):
                continue
            m_eff[t] = np.log(G[t] / G[t+1])
    return m_eff


# --- Simulation driver ---
def run_field_simulation(field, N_sweeps=2000, N_cor=10, eps=0.5, thermal_sweeps=500):
    """
    Run simulation: thermalize, measure correlations, return G and m_eff.
    """
    accepted = proposed = 0
    for _ in range(thermal_sweeps):
        a, p = metropolis_update_field(field, eps)
        accepted += a; proposed += p
    print(f"Post-thermalization acceptance: {accepted/proposed:.3f}")

    measurements = []
    accepted = proposed = 0
    for sweep in range(N_sweeps):
        a, p = metropolis_update_field(field, eps)
        accepted += a; proposed += p
        if sweep % N_cor == 0:
            measurements.append(measure_field_correlation_all_origins(field))
    print(f"Measurement acceptance: {accepted/proposed:.3f}")

    meas_arr = np.array(measurements)
    G_mean = np.mean(meas_arr, axis=0)
    G_err = np.std(meas_arr, axis=0, ddof=1)
    m_eff = compute_effective_mass(G_mean, G_err)
    return meas_arr, G_mean, G_err, m_eff


def run_multiple_simulations(num_runs, N_sweeps, N_cor, eps, thermal_sweeps):
    """
    Run multiple independent simulations and average observed correlators.
    """
    all_meas = []
    for _ in range(num_runs):
        field = 0.01 * np.random.randn(*(L,) * D)
        meas_arr, *_ = run_field_simulation(field, N_sweeps, N_cor, eps, thermal_sweeps)
        all_meas.append(meas_arr)

    all_meas = np.vstack(all_meas)
    G_mean = np.mean(all_meas, axis=0)
    G_err = np.std(all_meas, axis=0, ddof=1) / np.sqrt(all_meas.shape[0])
    m_eff = compute_effective_mass(G_mean, G_err)
    return G_mean, G_err, m_eff


# --- Analysis & visualization ---
eps_tuned = 0.5
G_mean, G_err, m_eff = run_multiple_simulations(
    num_runs=5, N_sweeps=4000, N_cor=20, eps=eps_tuned, thermal_sweeps=1000
)

print("G_mean:", G_mean)

def cosh_model(t, A, m):
    return A * (np.exp(-m*t) + np.exp(-m*(L - t)))

G_mean = 0.5 * (G_mean + G_mean[::-1])  # symmetrize

tdata = np.arange(L)
mask = G_mean > 5 * G_err
A_fit, m_fit = curve_fit(cosh_model, tdata[mask], G_mean[mask], p0=[0.2, 1.0])[0]
print(f"Fitted mass m = {m_fit:.3f}")

G_plot = np.abs(G_mean)
plt.errorbar(tdata, G_plot, yerr=G_err, fmt='o', capsize=3)
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel('Euclidean time t')
plt.ylabel('G(t)')
plt.title('Two-point correlation function')
plt.grid(True)
plt.show()

plt.plot(np.arange(len(m_eff)), m_eff, 'o-')
plt.xlabel('t')
plt.ylabel('m_eff(t)')
plt.title('Effective mass vs time')
plt.grid(True)
plt.show()
