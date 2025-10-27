"""
QCD_Lattice_SU3.py

SU(3) lattice gauge theory (Wilson action) using Cabibbo-Marinari updates (embedded SU(2) subgroups).
- Local ΔS (only recompute plaquettes touching a link) for speed.
- Metropolis proposals implemented by applying several small SU(2) rotations embedded inside SU(3).
- Reprojection to SU(3) via SVD to maintain unitarity and det=1.
- Measurements: average plaquette (normalized), bootstrap error, simple Wilson-loop helper.

This follows the pragmatic approach described in Lepage's notes: extend SU(2) methods to SU(3)
via SU(2) subgroups (Cabibbo-Marinari).
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
spatial_dims = 1           # spatial dimensions
lattice_size_L = 8         # sites per direction (spatial and time extent are both L)
beta = 6.0                 # typical SU(3) beta ~ 5.5-6.5 for coarse lattices; tune as desired
eps_initial = 0.06         # SU(2) rotation amplitude for each sub-update (small)
burn_in_sweeps = 500       # Number of initial thermalization sweeps before measurements
MC_sweeps = 2000           # Total number of Monte Carlo sweeps for measurement
MC_measure_interval = 5    # Number of sweeps between stored measurements (decorrelation interval)
n_boot = 300               # Number of bootstrap samples for error estimation

# Derived
D = spatial_dims + 1       # spacetime dimensions
x_shape = (lattice_size_L,) * D
link_matrix = np.zeros((D,) + x_shape + (3, 3), dtype=np.complex128)

# ---------------- Utilities ----------------
def x_neighbor(x, mu, shift=1):
    x_new = list(x)
    x_new[mu] = (x_new[mu] + shift) % lattice_size_L
    return tuple(x_new)

def su3_matrices(M):
    """
    Project a (3x3) complex link_sites M to SU(3) via SVD polar-type projection:
      U, s, Vh = svd(M); Uproj = U @ Vh; enforce det = 1 via global phase.
    """
    U, diagonal, V_hermitian = LA.svd(M)   # Singular Value Decomposition of M
    U_projection = U @ V_hermitian
    determinant = LA.det(U_projection)
    if determinant == 0:
        # fallback: small perturbation then project
        U_projection = U_projection + 1e-12 * np.eye(3)
        determinant = LA.det(U_projection)
    phase = determinant ** (1/3)
    U_projection /= phase
    return U_projection

# ---------------- SU(2) small updater (used inside SU(3) embeddings) ----------------
def su2_random_unitary(eps):
    """
    Return 2x2 SU(2) link_sites R = cos(a) I + i sin(a) n·sigma,
    where a = eps * |r| and n = r/|r| for gaussian r.
    We'll return as 2x2 numpy complex array.
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
    # project to SU(2) via SVD
    U, diagonal, V_hermitian = LA.svd(R)   # Singular Value Decomposition of R
    R_projection = U @ V_hermitian
    # ensure det=1
    determinant = LA.det(R_projection)
    R_projection /= (determinant ** 0.5)
    return R_projection

def embed_su2_into_su3(R2, i, j):
    """
    Embed a 2x2 SU(2) link_sites R2 into 3x3 SU(3) acting on indices (i,j), with other row/col untouched.
    i,j in {0,1,2}, i<j.
    """
    R = np.eye(3, dtype=complex)
    R[i, i] = R2[0, 0]
    R[i, j] = R2[0, 1]
    R[j, i] = R2[1, 0]
    R[j, j] = R2[1, 1]
    return R

# ---------------- Plaquette helpers ----------------
def plaquette_matrix(x, mu, nu, link_sites):
    x_plus_mu = x_neighbor(x, mu, 1)
    x_plus_nu = x_neighbor(x, nu, 1)
    U_mu = link_sites[(mu,) + x]
    U_nu_xmu = link_sites[(nu,) + x_plus_mu]
    U_mu_xnu = link_sites[(mu,) + x_plus_nu]
    U_nu = link_sites[(nu,) + x]
    P = U_mu @ U_nu_xmu @ U_mu_xnu.conj().T @ U_nu.conj().T
    return P

def real_trace_plaquette(x, mu, nu, link_sites):
    P = plaquette_matrix(x, mu, nu, link_sites)
    trace = np.trace(P)
    return float(np.real(trace))

def plaquettes_touching_link(x, mu, link_sites):
    """
    Return list of plaquettes (x, mu, nu, real trace P) that include the link (mu, x).
    For each nu != mu: plaquette at x and at x - e_nu.
    """
    p_list = []
    for nu in range(D):
        if nu == mu: continue
        trace_1 = real_trace_plaquette(x, mu, nu, link_sites)
        p_list.append(((x, mu, nu), trace_1))
        x_minus_nu = x_neighbor(x, nu, -1)
        trace_2 = real_trace_plaquette(x_minus_nu, mu, nu, link_sites)
        p_list.append(((x_minus_nu, mu, nu), trace_2))
    return p_list

# ---------------- Local update: Metropolis ----------------
def metropolis_update(link_sites, eps_sub=0.06):
    """
    One Metropolis sweep: for each link (mu, x) we apply sequential SU(2) embedded updates
    acting on subgroups (0,1), (0,2), (1,2). For each embedded rotation:
      - compute sum_old = sum ReTr(P) for affected plaquettes,
      - propose R = embed_su2_into_su3(R2),
      - set U' = R @ U, reproject, compute sum_new, accept/reject based on ΔS = -beta/3*(new-old).
    Returns (accepted, proposals).
    """
    accepted = 0
    proposals = 0
    su2_pairs = [(0,1), (0,2), (1,2)]
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            # base link
            U_old = link_sites[(mu,) + x].copy()
            # For each SU(2) subgroup do proposal/accept
            for (i, j) in su2_pairs:
                # compute old sum of ReTr(P) for plaquettes touching this link
                plist = plaquettes_touching_link(x, mu, link_sites)
                sum_old = sum(trace for (_meta, trace) in plist)

                R2 = su2_random_unitary(eps_sub)
                R3 = embed_su2_into_su3(R2, i, j)
                link_sites[(mu,) + x] = R3 @ U_old
                # reproject to SU(3)
                link_sites[(mu,) + x] = su3_matrices(link_sites[(mu,) + x])

                new_p_list = plaquettes_touching_link(x, mu, link_sites)
                sum_new = sum(trace for (_meta, trace) in new_p_list)

                dS = - (beta / 3.0) * (sum_new - sum_old)
                proposals += 1
                if dS > 0 and np.exp(-dS) < np.random.rand():
                    # reject: restore U_old (note: this discards the current subgroup update, but we still continue next subgroup starting from U_old)
                    link_sites[(mu,) + x] = U_old.copy()
                else:
                    # accept: update U_old to new (so next subgroup multiplies on it)
                    U_old = link_sites[(mu,) + x].copy()
                    accepted += 1
    return accepted, proposals

# ---------------- Observables ----------------
def average_plaquette_su3(link_sites):
    total = 0.0
    count = 0
    for x in np.ndindex(*x_shape):
        for mu in range(D):
            for nu in range(mu+1, D):
                trace = real_trace_plaquette(x, mu, nu, link_sites)
                total += trace
                count += 1
    # Normalize by 3: trace(1_3)=3
    return (total / count) / 3.0

def bootstrap_mean_std(values, nboot=300):
    vals = np.asarray(values)
    N = len(vals)
    boots = np.zeros(nboot)
    for i in range(nboot):
        inds = np.random.randint(0, N, size=N)
        boots[i] = np.mean(vals[inds])
    return boots.mean(), boots.std(ddof=1)

# ---------------- Tuner & Runner ----------------
def tune_eps_su3(matrix0, target=0.5, initial_eps=0.06, tries=10, test_sweeps=150):
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
        fraction = accepted / proposed if proposed>0 else 0.0
        if abs(fraction - target) < 0.05:
            break
        eps *= 1.2 if fraction > target else 0.8
    return eps

def run_su3_simulation(link_sites, eps_sub=0.06, burn_in_sweeps=500, MC_sweeps=2000, N_correlator=5):
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

# ---------------- Initialization ----------------
def init_links_identity(link_sites):
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            link_sites[(mu,) + x] = np.eye(3, dtype=complex)

def randomize_links_small(link_sites, amplitude=0.02):
    # apply a few small SU(3) multiplications (via embedded SU(2))
    for mu in range(D):
        for x in np.ndindex(*x_shape):
            # make a small combined SU(3) rotation
            for (i, j) in [(0,1), (0,2), (1,2)]:
                R2 = su2_random_unitary(amplitude)
                R3 = embed_su2_into_su3(R2, i, j)
                link_sites[(mu,) + x] = su3_matrices(R3 @ link_sites[(mu,) + x])

# ---------------- Simple Wilson loop helper for rectangles R x T (spatial x temporal) ----------------
def measure_wilson_loop_RT(link_sites, R, T, spatial_direction=0, time_direction=None):
    """
    Average Wilson loop for rectangle (R x T). Assumes last axis is time if time_direction is None.
    Returns average Re trace(W)/3.
    """
    if time_direction is None:
        time_direction = D - 1
    total = 0.0
    count = 0
    for x in np.ndindex(*x_shape):
        # build loop path (lower-left corner at x)
        # path: R steps +spatial_direction, T steps +time_direction, R steps -spatial_direction, T steps -time_direction
        # sum of link matrices along loop -> multiply them in order and take trace
        current_x = x
        W = np.eye(3, dtype=complex)
        # R steps + spatial_direction
        for i in range(R):
            U = link_sites[(spatial_direction,) + current_x]
            W = W @ U
            current_x = x_neighbor(current_x, spatial_direction, 1)
        # T steps +time_direction
        for i in range(T):
            U = link_sites[(time_direction,) + current_x]
            W = W @ U
            current_x = x_neighbor(current_x, time_direction, 1)
        # R steps -spatial_direction: multiply by U^\dagger of links we traverse backward
        for i in range(R):
            current_x = x_neighbor(current_x, spatial_direction, -1)
            U = link_sites[(spatial_direction,) + current_x]
            W = W @ U.conj().T
        # T steps -time_direction
        for i in range(T):
            current_x = x_neighbor(current_x, time_direction, -1)
            U = link_sites[(time_direction,) + current_x]
            W = W @ U.conj().T
        total += np.real(np.trace(W)) / 3.0
        count += 1
    return total / count

# ---- Wilson Loop Averaging and Static Potential Extraction ----
def su3_simulation_with_wilson_loops(link_sites, eps_sub=0.06, burn_in_sweeps=500, MC_sweeps=2000, N_correlator=5, max_R=None, max_T=None):
    """
    Run SU(3) simulation, storing Wilson loops at each MC measurement.
    Returns:
      - wilson_loops_samples: shape (n_measurements, n_R, n_T)
      - R_values, T_values
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
            # Measure all Wilson loops for this configuration
            W_sample = np.zeros((n_R, n_T))
            for i, R in enumerate(R_values):
                for j, T in enumerate(T_values):
                    W_sample[i, j] = measure_wilson_loop_RT(link_sites, R, T)
            wilson_loops_samples.append(W_sample)
    wilson_loops_samples = np.array(wilson_loops_samples)  # shape (n_meas, n_R, n_T)
    return wilson_loops_samples, R_values, T_values

# ============================================================
# ===============  PLAQUETTE CALCULATION SECTION  ==============
# ============================================================

init_links_identity(link_matrix)
randomize_links_small(link_matrix, amplitude=0.02)

eps_tuned = tune_eps_su3(link_matrix, initial_eps=eps_initial)
samples, plaq_mean, plaq_err = run_su3_simulation(link_matrix, eps_sub=eps_tuned, burn_in_sweeps=burn_in_sweeps, MC_sweeps=MC_sweeps, N_correlator=MC_measure_interval)

# Print the average plaquette and its error
print(f"Average plaquette = {plaq_mean:.6f} ± {plaq_err:.6f}")

# Plot the plaquette values as a function of measurement index
plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(samples)), samples, marker='o', linestyle='-', color='b')
plt.xlabel('Measurement Index')
plt.ylabel('Plaquette Value')
plt.title('Plaquette History')
plt.tight_layout()
plt.show()

# Plot the histogram of plaquette measurements to show their statistical distribution
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=30, color='g', alpha=0.75)
plt.xlabel('Plaquette Value')
plt.ylabel('Frequency')
plt.title('Histogram of Plaquette Samples')
plt.tight_layout()
plt.show()


#
# ============================================================
# ===============  WILSON LOOP CALCULATION SECTION  ============
# ============================================================

# Run the Monte Carlo simulation to collect Wilson loop measurements
wilson_loops_samples, R_values, T_values = su3_simulation_with_wilson_loops( link_matrix, eps_sub=eps_tuned, 
                                                                            burn_in_sweeps=burn_in_sweeps, MC_sweeps=MC_sweeps, 
                                                                            N_correlator=MC_measure_interval, max_R=lattice_size_L // 2, 
                                                                            max_T=lattice_size_L // 2)

# Compute the average Wilson loop ⟨W(R,T)⟩ over Monte Carlo samples
avg_wilson_loops = np.mean(wilson_loops_samples, axis=0)  # shape (n_R, n_T)

# Extract the static quark-antiquark potential V(R) using the effective mass method from Wilson loops
V_R = np.zeros(len(R_values))
for i in range(len(R_values)):
    potentials = []
    for j in range(len(T_values) - 1):
        W_T = avg_wilson_loops[i, j]
        W_Tp1 = avg_wilson_loops[i, j + 1]
        if W_T > 0 and W_Tp1 > 0:
            potentials.append(-np.log(W_Tp1 / W_T))
    if potentials:
        V_R[i] = np.mean(potentials)
    else:
        V_R[i] = np.nan

# Print summary of Wilson loop averages and static potential extraction
print("Wilson loop and static potential calculation complete.")
print("R values:", R_values)
print("V(R):", V_R)

# Plot the extracted static potential V(R) as a function of the spatial separation R
plt.figure(figsize=(8, 5))
plt.plot(R_values, V_R, marker='o', linestyle='-', color='r')
plt.xlabel('Spatial Separation R')
plt.ylabel('Static Quark-Antiquark Potential V(R)')
plt.title('Static Potential from Averaged Wilson Loops')
plt.grid(True)
plt.tight_layout()
plt.show()




#     # ---------------- Classical Gluon Field Extraction ----------------
def gell_mann_matrices():
    """Return list of 8 Gell-Mann matrices λ^a."""
    lambda_ = []
    lambda_.append(np.array([[0,1,0],[1,0,0],[0,0,0]],dtype=complex))
    lambda_.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]],dtype=complex))
    lambda_.append(np.array([[1,0,0],[0,-1,0],[0,0,0]],dtype=complex))
    lambda_.append(np.array([[0,0,1],[0,0,0],[1,0,0]],dtype=complex))
    lambda_.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]],dtype=complex))
    lambda_.append(np.array([[0,0,0],[0,0,1],[0,1,0]],dtype=complex))
    lambda_.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]],dtype=complex))
    lambda_.append((1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],dtype=complex))
    return lambda_

def extract_gluon_field(U, g=1.0, a=1.0):
    """
    Given one SU(3) link link_sites U, return its gauge field components A^a (a=1..8).
    U ≈ exp(i g a A), so for small a: A ≈ (U - U†)/(2 i g a), projected onto su(3).
    """
    difference = (U - U.conj().T) / (2j * g * a)
    difference -= np.trace(difference).real/3.0 * np.eye(3)  # traceless part
    lambda_ = gell_mann_matrices()
    A_components = np.array([np.real(np.trace(difference @ lambda_a))/2.0 for lambda_a in lambda_])
    return A_components  # shape (8,)

def field_strength_tensor(link_sites, x, mu, nu, g=1.0, a=1.0):
    """
    Compute F_{mu,nu}^a(x) = (1/(i g a^2)) (U_mu_nu - U_mu_nu†)_traceless projected on λ^a.
    """
    P = plaquette_matrix(x, mu, nu, link_sites)
    difference = (P - P.conj().T) / (2j * g * a**2)
    difference -= np.trace(difference).real/3.0 * np.eye(3)
    lambda_ = gell_mann_matrices()
    F_components = np.array([np.real(np.trace(difference @ lambda_a))/2.0 for lambda_a in lambda_])
    return F_components  # shape (8,)

def measure_avg_A2_and_F2(link_sites, g=1.0, a=1.0):
    """Average ⟨A^2⟩ and ⟨F^2⟩ over all sites and directions."""
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
            for nu in range(mu+1, D):
                F = field_strength_tensor(link_sites, x, mu, nu, g=g, a=a)
                F2_sum += np.dot(F, F)
                nF += 1
    return A2_sum/nA, F2_sum/nF

A2_avg, F2_avg = measure_avg_A2_and_F2(link_matrix)
print(f"⟨A²⟩ = {A2_avg:.6e}, ⟨F²⟩ = {F2_avg:.6e}")