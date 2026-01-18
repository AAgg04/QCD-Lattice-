# Lattice Path Integrals and Monte Carlo Simulations  
**From Quantum Mechanics to SU(3) Lattice QCD**

## Overview

This project is a unified computational study of **Euclidean path integrals and lattice field theory**, progressing from elementary quantum-mechanical systems to non-Abelian gauge theory. It demonstrates how quantum observables emerge from discretized Euclidean spacetime using both **direct numerical integration** and **Monte Carlo sampling**.

The code is intentionally explicit and pedagogical: all algorithms are implemented from first principles, mirroring the structure used in professional lattice field theory calculations while remaining readable and self-contained.

---

## Scope and Progression

The project advances conceptually in four stages, each building on the previous one:

1. **Quantum Mechanics (Harmonic Oscillator)**  
2. **Interacting Quantum Systems (Anharmonic Oscillator)**  
3. **Relativistic Scalar Field Theory (ϕ⁴ Theory)**  
4. **Non-Abelian Gauge Theory (SU(3) Lattice QCD)**  

All simulations are performed in **Euclidean spacetime**, where quantum dynamics map naturally onto statistical mechanics.

---

## File Contents

### 1. Euclidean Path Integral — Harmonic Oscillator  
**Section: `Path_integral.py`**

- Direct numerical evaluation of the Euclidean path integral
- Lattice discretization of time
- Computation of the propagator \( K(x,x;T) \) via multidimensional integration
- No Monte Carlo sampling (brute-force integration)

**Purpose:**  
To explicitly demonstrate how the path integral formulation reduces to a weighted sum over discretized paths and how quantum propagation arises from the Euclidean action.

---

### 2. Monte Carlo Path Integral — Anharmonic Oscillator  
**Section: `Monte_carlo_path_integral.py`**

- Metropolis Monte Carlo sampling of Euclidean paths
- Anharmonic potential  
<img width="302" height="82" alt="Screenshot 2026-01-18 at 8 24 59 PM" src="https://github.com/user-attachments/assets/0f88c9a7-5617-4d85-a2a5-85126f1f03d5" />

- Computation of the two-point correlator  
<img width="308" height="71" alt="Screenshot 2026-01-18 at 8 25 36 PM" src="https://github.com/user-attachments/assets/966263df-5aeb-43d3-955e-d6f40c3290b0" />

- Extraction of the ground-state energy from exponential decay

**Purpose:**  
To show how Monte Carlo methods replace infeasible integrals with statistically controlled sampling, enabling scalable quantum calculations.

---

### 3. Scalar ϕ⁴ Lattice Field Theory  
**Section: `Lattice_Field_Theory.py`**

- Relativistic scalar field on a Euclidean spacetime lattice
- Periodic boundary conditions
- Local Metropolis updates
- Measurement of two-point correlation functions
- Effective mass extraction and cosh fitting

**Purpose:**  
To transition from quantum mechanics to quantum field theory, illustrating how particles emerge as poles in correlation functions rather than as fundamental inputs.

---

### 4. SU(3) Lattice Gauge Theory (QCD Prototype)  
**Section: `QCD_Lattice_SU3.py`**

- Wilson gauge action for SU(3)
- Cabibbo–Marinari updates using embedded SU(2) subgroups
- Local Metropolis acceptance using plaquette contributions
- Measurement of:
  - Average plaquette
  - Wilson loops
  - Static quark–antiquark potential
  - Gluon field observables

**Purpose:**  
To implement the essential numerical machinery of lattice QCD and demonstrate confinement-related observables in a simplified setting.

---

## Scientific Principles

- Euclidean path integral formulation
- Lattice regularization of spacetime
- Markov Chain Monte Carlo sampling
- Gauge invariance and local updates
- Extraction of physical observables from correlation functions

---

## Dependencies

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib  

All simulations run on a standard CPU without external lattice libraries.

---
