"""
@mainpage SU(3) Lattice Gauge Theory (Wilson action) — Cabibbo–Marinari Implementation

@file QCD_Lattice_SU3.py
@brief SU(3) lattice gauge theory simulation using Cabibbo–Marinari SU(2) subgroup updates.
@details
This module implements a pragmatic SU(3) lattice gauge theory code based on the Wilson action,
using Cabibbo–Marinari updates (embedded SU(2) rotations) together with local ΔS computations
(only plaquettes touching a link are recomputed) and reprojection to SU(3) via SVD to maintain
unitarity and $\det U = 1$.

It contains:
 - Local Metropolis updates applying a sequence of small SU(2) rotations embedded into SU(3).
 - Efficient local plaquette recomputation for ΔS evaluations.
 - Utilities for plaquette measurement, bootstrap error estimation, Wilson loops and static potential.
 - Helpers to extract approximate gauge fields (A^a) and field-strength components F^{\mu\nu}_a
   from link matrices for diagnostic/classical analysis.

@section references Key references
 - K. G. Wilson, "Confinement of quarks," Phys. Rev. D 10, 2445 (1974).
 - N. Cabibbo and E. Marinari, "A new method for updating SU(N) matrices," Phys. Lett. B119 (1982).
 - G. P. Lepage lecture notes for pragmatic algorithmic choices.
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


# ---------------- Parameters (global simulation controls) ----------------
spatial_dims = 1           #: int: number of spatial dimensions (d)
lattice_size_L = 8         #: int: lattice extent per dimension (L)
beta = 6.0                 #: float: gauge coupling parameter (Wilson action: β = 6/g²)
eps_initial = 0.06         #: float: initial SU(2) rotation amplitude for subgroup updates
burn_in_sweeps = 500       #: int: number of thermalization sweeps
MC_sweeps = 2000           #: int: number of Monte Carlo sweeps for measurements
MC_measure_interval = 5    #: int: sweeps between stored measurements (decorrelation interval)
n_boot = 300               #: int: bootstrap samples for error estimation

# Derived geometry / storage
D = spatial_dims + 1       #: int: total spacetime dimensions (d + 1)
x_shape = (lattice_size_L,) * D
# link_matrix: shape (D, L, L, ..., 3, 3) storing SU(3) link matrices for each direction mu and site x
link_matrix = np.zeros((D,) + x_shape + (3, 3), dtype=np.complex128)


# ---------------- Utilities ----------------
def x_neighbor(x, mu, shift=1):
    """
    @brief Periodic lattice neighbor coordinate.
    @param x tuple: Lattice coordinate (length D).
    @param mu int: Direction index (0..D-1).
    @param shift int: Integer shift (positive forward, negative backward).
    @return tuple: New lattice coordinate (with periodic wrap).
    @details Implements periodic boundary conditions: (x_mu + shift) mod L.
    """
    x_new = list(x)
    x_new[mu] = (x_new[mu] + shift) % lattice_size_L
    return tuple(x_new)


def su3_matrices(M):
    """
    @brief Project a general complex 3x3 matrix to SU(3) via unitary polar/SVD projection.
    @param M (ndarray): 3x3 complex matrix (candidate link).
    @return ndarray: Unitary 3x3 matrix with det = 1 (projection of M into SU(3)).
    @details
    We perform an SVD: M = U S V^H and set U_proj = U V^H (closest unitary in Frobenius norm).
    A global phase is then removed to enforce det(U_proj) = 1. If the projection yields
    a near-singular matrix we add a tiny perturbation as fallback.
    """
    U, s, Vh = LA.svd(M)
    U_projection = U @ Vh
    determinant = LA.det(U_projection)
    if determinant == 0 or np.isnan(determinant):
        # Numerical fallback: small perturbation then reproject
        U_projection = U_projection + 1e-12 * np.eye(3, dtype=complex)
        determinant = LA.det(U_projection)
    # Remove global phase to ensure unit determinant
    phase = determinant ** (1.0 / 3.0)
    U_projection /= phase
    return U_projection


# ---------------- SU(2) small updater (embedded in SU(3)) ----------------
def su2_random_unitary(eps):
    """
    @brief Generate a small random SU(2) rotation matrix using Gaussian parameters.
    @param eps float: amplitude controlling rotation angle scale (a = eps * |r|).
    @return ndarray: 2x2 complex SU(2) matrix.
    @details
    The parametrization uses R = cos(a) I + i sin(a) n·σ where n is a unit 3-vector
    and σ_i are the Pauli matrices. We project via SVD to correct numerical drift and
    ensure exact unitarity, then enforce det=1.
    """
    r = np.random.normal(size=3)
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.eye(2, dtype=complex)
    a = eps * r_norm
    n = r / r_norm
    # Pauli matrices
    sigma1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma2 = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma3 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    ndotsigma = n[0] * sigma1 + n[1] * sigma2 + n[2] * sigma3
    R = np.cos(a) * np.eye(2, dtype=complex) + 1j * np.sin(a) * ndotsigma
    # Project R to exact SU(2) via SVD/polar projection and fix determinant
    U, s, Vh = LA.svd(R)
    R_projection = U @ Vh
    det = LA.det(R_projection)
    R_projection /= (det ** 0.5)
    return R_projection


def embed_su2_into_su3(R2, i, j):
    """
    @brief Embed a 2x2 SU(2) matrix into SU(3) acting on indices (i, j).
    @param R2 ndarray: 2x2 SU(2) matrix.
    @param i int: first SU(2) index (0..2).
    @param j int: second SU(2) index (0..2), must satisfy i < j.
    @return ndarray: 3x3 matrix equal to identity except the 2x2 block at (i,j) replaced by R2.
    @details This is the standard Cabibbo–Marinari embedding that extends SU(2) subgroup rotations
             to SU(3) by acting non-trivially on a chosen 2D subspace.
    """
    R = np.eye(3, dtype=complex)
    R[i, i] = R2[0, 0]
    R[i, j] = R2[0, 1]
    R[j, i] = R2[1, 0]
    R[j, j] = R2[1, 1]
    return R


# ---------------- Plaquette helpers & local ΔS computation ----------------
def plaquette_matrix(x, mu, nu, link_sites):
    """
    @brief Construct the plaquette matrix U_mu(x) U_nu(x+mu) U_mu^\dagger(x+nu) U_nu^\dagger(x).
    @param x tuple: lattice coordinate.
    @param mu int: direction index mu.
    @param nu int: direction index nu.
    @param link_sites ndarray: link variable array.
    @return ndarray: 3x3 plaquette matrix P_{mu,nu}(x).
    """
    x_plus_mu = x_neighbor(x, mu, 1)
    x_plus_nu = x_neighbor(x, nu, 1)
    U_mu = link_sites[(mu,) + x]
    U_nu_xmu = link_sites[(nu,) + x_plus_mu]
    U_mu_xnu = link_sites[(mu,) + x_plus_nu]
    U_nu = link_sites[(nu,) + x]
    P = U_mu @ U_nu_xmu @ U_mu_xnu.conj().T @ U_nu.conj().T
    return P


def real_trace_plaquette(x, mu, nu, link_sites):
    """
    @brief Compute the real part of the trace of the plaquette matrix.
    @return float: Re Tr[P_{mu,nu}(x)].
    """
    P = plaquette_matrix(x, mu, nu, link_sites)
    trace = np.trace(P)
    return float(np.real(trace))


def plaquettes_touching_link(x, mu, link_sites):
    """
    @brief List plaquettes that include the link at (mu, x).
    @param x tuple: lattice coordinate of the starting site of the link.
    @param mu int: link direction.
    @param link_sites ndarray: array of link matrices.
    @return list: entries [((x_plaq, mu, nu), real_trace), ...] for all plaquettes touching the link.
    @details
    For each nu != mu, the link (mu,x) sits in two elementary plaquettes:
      - the plaquette at x in the (mu,nu) plane,
      - the plaquette at x - e_nu in the (mu,nu) plane.
    Only these plaquettes are required to compute the local change in action when U_mu(x) is updated.
    """
    p_list = []
    for nu in range(D):
        if nu == mu:
            continue
        trace_1 = real_trace_plaquette(x, mu, nu, link_sites)
        p_list.append(((x, mu, nu), trace_1))
        x_minus_nu = x_neighbor(x, nu, -1)
        trace_2 = real_trace_plaquette(x_minus_nu, mu, nu, link_sites)
        p_list.append(((x_minus_nu, mu, nu), trace_2))
    return p_list


# ---------------- Local update: Metropolis with embedded SU(2) updates ----------------
def metropolis_update(link_sites, eps_sub=0.06):
    """
    @brief Perform a single Metropolis sweep over all links applying embedded SU(2) updates.
    @param link_sites ndarray: link variable array (modified in-place).
    @param eps_sub float: SU(2) proposal amplitude for each embedded sub-update.
    @return tuple: (accepted int, proposals int)
    @details
    For each link U_mu(x) we cycle through the three SU(2) subgroups (0,1), (0,2), (1,2).
    For each subgroup:
      1. compute sum_old = Σ Re Tr(P) over plaquettes touching the link,
      2. propose an SU(2) rotation R2, embed into SU(3) → R3,
      3. set U_candidate = R3 @ U_old and reproject to SU(3),
      4. compute sum_new and ΔS = - (β/3) (sum_new - sum_old),
      5. accept/reject with Metropolis probability.
    Using only touching plaquettes makes ΔS computation local and efficient.
    """
    accepted = 0
    proposals = 0
    su2_pairs = [(0, 1), (0, 2), (1, 2)]
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            U_old = link_sites[(mu,) + x].copy()
            for (i, j) in su2_pairs:
                plist = plaquettes_touching_link(x, mu, link_sites)
                sum_old = sum(trace for (_meta, trace) in plist)

                R2 = su2_random_unitary(eps_sub)
                R3 = embed_su2_into_su3(R2, i, j)
                link_sites[(mu,) + x] = R3 @ U_old
                # Reproject to SU(3) to correct numerical drift
                link_sites[(mu,) + x] = su3_matrices(link_sites[(mu,) + x])

                new_p_list = plaquettes_touching_link(x, mu, link_sites)
                sum_new = sum(trace for (_meta, trace) in new_p_list)

                dS = - (beta / 3.0) * (sum_new - sum_old)
                proposals += 1
                # Metropolis acceptance: accept if dS <= 0 or with probability exp(-dS)
                if dS > 0 and np.exp(-dS) < np.random.rand():
                    # reject: revert this subgroup update (resume next subgroup from U_old)
                    link_sites[(mu,) + x] = U_old.copy()
                else:
                    # accept: update U_old so subsequent subgroup multiplications act on accepted matrix
                    U_old = link_sites[(mu,) + x].copy()
                    accepted += 1
    return accepted, proposals


# ---------------- Observables ----------------
def average_plaquette_su3(link_sites):
    """
    @brief Compute the normalized average plaquette ⟨Re Tr P⟩/3 over the lattice.
    @param link_sites ndarray: link array.
    @return float: average plaquette normalized by color factor (3).
    @details The Wilson action density per plaquette is proportional to (1 - Re Tr P / 3).
    """
    total = 0.0
    count = 0
    for x in np.ndindex(*x_shape):
        for mu in range(D):
            for nu in range(mu + 1, D):
                trace = real_trace_plaquette(x, mu, nu, link_sites)
                total += trace
                count += 1
    # Normalize by color dimension (Tr 1 = 3)
    return (total / count) / 3.0


def bootstrap_mean_std(values, nboot=300):
    """
    @brief Estimate mean and bootstrap standard error for a 1D array of samples.
    @param values array-like: measurement samples.
    @param nboot int: number of bootstrap resamples.
    @return tuple: (boot_mean, boot_std)
    @details We resample with replacement and compute sample means for each bootstrap
             realization; the returned std is the bootstrap estimate of the error.
    """
    vals = np.asarray(values)
    N = len(vals)
    boots = np.zeros(nboot)
    for i in range(nboot):
        inds = np.random.randint(0, N, size=N)
        boots[i] = np.mean(vals[inds])
    return boots.mean(), boots.std(ddof=1)


# ---------------- Tuner & Runner ----------------
def tune_eps_su3(matrix0, target=0.5, initial_eps=0.06, tries=10, test_sweeps=150):
    """
    @brief Tune the SU(2) proposal amplitude eps so that acceptance fraction ~ target.
    @param matrix0 ndarray: initial link matrix copy for tuning (will be copied internally).
    @param target float: desired acceptance fraction (e.g., 0.5).
    @param initial_eps float: starting amplitude.
    @param tries int: maximum adjustment attempts.
    @param test_sweeps int: sweeps per tuning test.
    @return float: tuned eps value.
    @details We perform a small number of sweeps and adjust eps multiplicatively to move acceptance
             fraction towards target. This is a heuristic tuner used before a production run.
    """
    eps = initial_eps
    for attempt in range(tries):
        matrix_copy = matrix0.copy()
        # quick thermalize copy
        for i in range(50):
            metropolis_update(matrix_copy, eps_sub=eps)
        accepted = proposed = 0
        for i in range(test_sweeps):
            a, p = metropolis_update(matrix_copy, eps_sub=eps)
            accepted += a; proposed += p
        fraction = accepted / proposed if proposed > 0 else 0.0
        if abs(fraction - target) < 0.05:
            break
        eps *= 1.2 if fraction > target else 0.8
    return eps


def run_su3_simulation(link_sites, eps_sub=0.06, burn_in_sweeps=500, MC_sweeps=2000, N_correlator=5):
    """
    @brief Run SU(3) Metropolis simulation collecting plaquette samples.
    @param link_sites ndarray: initial link configuration (modified in-place).
    @param eps_sub float: SU(2) subgroup proposal amplitude.
    @param burn_in_sweeps int: thermalization sweeps.
    @param MC_sweeps int: measurement sweeps.
    @param N_correlator int: interval between stored measurements.
    @return tuple: (plaquette_samples ndarray, mean_plaquette float, error_plaquette float)
    @details After burn-in we perform MC_sweeps sweeps and measure the average plaquette every
             N_correlator sweeps. Bootstrap error estimation is applied to the set of plaquette samples.
    """
    accepted = proposed = 0
    for i in range(burn_in_sweeps):
        a, p = metropolis_update(link_sites, eps_sub=eps_sub)
        accepted += a; proposed += p
    plaquette_samples = []
    accepted = proposed = 0
    for sweep in range(MC_sweeps):
        a, p = metropolis_update(link_sites, eps_sub=eps_sub)
        accepted += a; proposed += p
        if sweep % N_correlator == 0:
            plaquette = average_plaquette_su3(link_sites)
            plaquette_samples.append(plaquette)
    mean_plaquette, error_plaquette = bootstrap_mean_std(plaquette_samples, nboot=n_boot)
    return np.array(plaquette_samples), mean_plaquette, error_plaquette


# ---------------- Initialization helpers ----------------
def init_links_identity(link_sites):
    """
    @brief Initialize all links to the identity matrix.
    @param link_sites ndarray: link array to initialize (modified in-place).
    """
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            link_sites[(mu,) + x] = np.eye(3, dtype=complex)


def randomize_links_small(link_sites, amplitude=0.02):
    """
    @brief Apply small random SU(3) rotations (via embedded SU(2)) to each link for breaking symmetry.
    @param link_sites ndarray: link array (modified in-place).
    @param amplitude float: small rotation amplitude used for initial randomization.
    @details Useful to seed the Markov chain with a slightly randomized starting configuration.
    """
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            for (i, j) in [(0, 1), (0, 2), (1, 2)]:
                R2 = su2_random_unitary(amplitude)
                R3 = embed_su2_into_su3(R2, i, j)
                link_sites[(mu,) + x] = su3_matrices(R3 @ link_sites[(mu,) + x])


# ---------------- Wilson loop helper ----------------
def measure_wilson_loop_RT(link_sites, R, T, spatial_direction=0, time_direction=None):
    """
    @brief Measure the average Wilson loop W(R,T) for rectangular loops of spatial size R and temporal extent T.
    @param link_sites ndarray: link configuration.
    @param R int: spatial extent (number of spatial steps).
    @param T int: temporal extent (number of temporal steps).
    @param spatial_direction int: spatial direction index used for the R side.
    @param time_direction int or None: time direction index; defaults to D-1 (last axis).
    @return float: average Re Tr[W(R,T)] / 3 over all possible loop origins.
    @details
    The loop path starts at each lattice site x and multiplies the link matrices along the rectangular contour.
    Backward traversals multiply by Hermitian conjugate of the traversed link.
    """
    if time_direction is None:
        time_direction = D - 1
    total = 0.0
    count = 0
    for x in np.ndindex(*x_shape):
        current_x = x
        W = np.eye(3, dtype=complex)
        # R steps + spatial_direction
        for i in range(R):
            U = link_sites[(spatial_direction,) + current_x]
            W = W @ U
            current_x = x_neighbor(current_x, spatial_direction, 1)
        # T steps + time_direction
        for i in range(T):
            U = link_sites[(time_direction,) + current_x]
            W = W @ U
            current_x = x_neighbor(current_x, time_direction, 1)
        # R steps - spatial_direction (backwards)
        for i in range(R):
            current_x = x_neighbor(current_x, spatial_direction, -1)
            U = link_sites[(spatial_direction,) + current_x]
            W = W @ U.conj().T
        # T steps - time_direction (backwards)
        for i in range(T):
            current_x = x_neighbor(current_x, time_direction, -1)
            U = link_sites[(time_direction,) + current_x]
            W = W @ U.conj().T
        total += np.real(np.trace(W)) / 3.0
        count += 1
    return total / count


def su3_simulation_with_wilson_loops(link_sites, eps_sub=0.06, burn_in_sweeps=500, MC_sweeps=2000, N_correlator=5, max_R=None, max_T=None):
    """
    @brief Run full SU(3) simulation storing Wilson loop matrices for each measurement.
    @param link_sites ndarray: initial link configuration (modified in-place).
    @param eps_sub float: SU(2) subgroup proposal amplitude.
    @param burn_in_sweeps int: thermalization sweeps.
    @param MC_sweeps int: measurement sweeps.
    @param N_correlator int: interval between stored measurements.
    @param max_R int or None: maximum spatial size to measure (defaults to L/2).
    @param max_T int or None: maximum temporal size to measure (defaults to L/2).
    @return tuple: (wilson_loops_samples ndarray [n_meas, n_R, n_T], R_values ndarray, T_values ndarray)
    @details
    Measures a grid of Wilson loops W(R,T) for R in [1..max_R], T in [1..max_T] at each stored configuration.
    """
    if max_R is None:
        max_R = lattice_size_L // 2
    if max_T is None:
        max_T = lattice_size_L // 2
    R_values = np.arange(1, max_R + 1)
    T_values = np.arange(1, max_T + 1)
    n_R = len(R_values)
    n_T = len(T_values)
    # Thermalize
    for i in range(burn_in_sweeps):
        metropolis_update(link_sites, eps_sub=eps_sub)
    wilson_loops_samples = []
    for sweep in range(MC_sweeps):
        metropolis_update(link_sites, eps_sub=eps_sub)
        if sweep % N_correlator == 0:
            W_sample = np.zeros((n_R, n_T))
            for i, R in enumerate(R_values):
                for j, T in enumerate(T_values):
                    W_sample[i, j] = measure_wilson_loop_RT(link_sites, R, T)
            wilson_loops_samples.append(W_sample)
    wilson_loops_samples = np.array(wilson_loops_samples)
    return wilson_loops_samples, R_values, T_values


# ============================================================
# ===============  PLAQUETTE CALCULATION SECTION  ==============
# ============================================================
init_links_identity(link_matrix)
randomize_links_small(link_matrix, amplitude=0.02)

eps_tuned = tune_eps_su3(link_matrix, initial_eps=eps_initial)
samples, plaq_mean, plaq_err = run_su3_simulation(
    link_matrix, eps_sub=eps_tuned, burn_in_sweeps=burn_in_sweeps,
    MC_sweeps=MC_sweeps, N_correlator=MC_measure_interval)

print(f"Average plaquette = {plaq_mean:.6f} ± {plaq_err:.6f}")

# Plot plaquette history
plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(samples)), samples, marker='o', linestyle='-')
plt.xlabel('Measurement Index')
plt.ylabel('Plaquette Value (Re Tr P / 3)')
plt.title('Plaquette History')
plt.tight_layout()
plt.show()

# Plot histogram of plaquette samples
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=30, alpha=0.75)
plt.xlabel('Plaquette Value')
plt.ylabel('Frequency')
plt.title('Histogram of Plaquette Samples')
plt.tight_layout()
plt.show()


# ============================================================
# ===============  WILSON LOOP CALCULATION SECTION  ============
# ============================================================
wilson_loops_samples, R_values, T_values = su3_simulation_with_wilson_loops(
    link_matrix, eps_sub=eps_tuned, burn_in_sweeps=burn_in_sweeps,
    MC_sweeps=MC_sweeps, N_correlator=MC_measure_interval,
    max_R=lattice_size_L // 2, max_T=lattice_size_L // 2)

avg_wilson_loops = np.mean(wilson_loops_samples, axis=0)  # shape (n_R, n_T)

# Extract static potential V(R) via effective mass from Wilson loops
V_R = np.zeros(len(R_values))
for i in range(len(R_values)):
    potentials = []
    for j in range(len(T_values) - 1):
        W_T = avg_wilson_loops[i, j]
        W_Tp1 = avg_wilson_loops[i, j + 1]
        if W_T > 0 and W_Tp1 > 0:
            potentials.append(-np.log(W_Tp1 / W_T))
    V_R[i] = np.mean(potentials) if potentials else np.nan

print("Wilson loop and static potential calculation complete.")
print("R values:", R_values)
print("V(R):", V_R)

plt.figure(figsize=(8, 5))
plt.plot(R_values, V_R, marker='o', linestyle='-')
plt.xlabel('Spatial Separation R')
plt.ylabel('Static Quark-Antiquark Potential V(R)')
plt.title('Static Potential from Averaged Wilson Loops')
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------- Classical gluon-field diagnostics ----------------
def gell_mann_matrices():
    """
    @brief Return the eight Gell-Mann matrices λ^a (3x3).
    @return list: eight 3x3 numpy arrays forming a basis for su(3).
    @details These are used to project Lie-algebra components from SU(3) link matrices.
    """
    lambda_ = []
    lambda_.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    lambda_.append(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex))
    lambda_.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
    lambda_.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    lambda_.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))
    lambda_.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    lambda_.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))
    lambda_.append((1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex))
    return lambda_


def extract_gluon_field(U, g=1.0, a=1.0):
    """
    @brief Extract approximate local gauge field components A^a from a single SU(3) link.
    @param U ndarray: SU(3) link matrix.
    @param g float: gauge coupling (default 1.0).
    @param a float: lattice spacing (default 1.0).
    @return ndarray: array shape (8,) containing A^a components (real).
    @details For small lattice spacing we approximate U ≈ exp(i g a A) => A ≈ (U - U†)/(2 i g a).
             We then project the traceless anti-Hermitian part onto the Gell-Mann basis.
    """
    difference = (U - U.conj().T) / (2j * g * a)
    difference -= np.trace(difference).real / 3.0 * np.eye(3)
    lambda_ = gell_mann_matrices()
    A_components = np.array([np.real(np.trace(difference @ lambda_a)) / 2.0 for lambda_a in lambda_])
    return A_components


def field_strength_tensor(link_sites, x, mu, nu, g=1.0, a=1.0):
    """
    @brief Compute the lattice field-strength components F_{mu,nu}^a at site x from the plaquette.
    @param link_sites ndarray: link variables.
    @param x tuple: lattice coordinate.
    @param mu int: direction mu.
    @param nu int: direction nu.
    @param g float: gauge coupling.
    @param a float: lattice spacing.
    @return ndarray: shape (8,) F^a components (real).
    @details Uses the anti-Hermitian traceless projection of the plaquette:
             F ~ (P - P†)/(2 i g a^2) projected on Gell-Mann matrices.
    """
    P = plaquette_matrix(x, mu, nu, link_sites)
    difference = (P - P.conj().T) / (2j * g * a ** 2)
    difference -= np.trace(difference).real / 3.0 * np.eye(3)
    lambda_ = gell_mann_matrices()
    F_components = np.array([np.real(np.trace(difference @ lambda_a)) / 2.0 for lambda_a in lambda_])
    return F_components


def measure_avg_A2_and_F2(link_sites, g=1.0, a=1.0):
    """
    @brief Compute averages ⟨A^2⟩ and ⟨F^2⟩ over the entire lattice as diagnostics.
    @param link_sites ndarray: link configuration.
    @param g float: gauge coupling.
    @param a float: lattice spacing.
    @return tuple: (A2_avg float, F2_avg float)
    @details A^2 and F^2 are computed by summing squares of components and normalizing by counts.
    """
    A2_sum = 0.0
    F2_sum = 0.0
    nA = 0
    nF = 0
    for x in np.ndindex(*x_shape):
        for mu in range(D):
            U = link_sites[(mu,) + x]
            A = extract_gluon_field(U, g=g, a=a)
            A2_sum += np.dot(A, A)
            nA += 1
        for mu in range(D):
            for nu in range(mu + 1, D):
                F = field_strength_tensor(link_sites, x, mu, nu, g=g, a=a)
                F2_sum += np.dot(F, F)
                nF += 1
    return A2_sum / nA, F2_sum / nF


A2_avg, F2_avg = measure_avg_A2_and_F2(link_matrix)
print(f"⟨A²⟩ = {A2_avg:.6e}, ⟨F²⟩ = {F2_avg:.6e}")
