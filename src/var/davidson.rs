//! # Davidson Diagonalization (`var::davidson`)
//!
//! This module orchestrates the Davidson diagonalization process for the sparse
//! Hamiltonian matrix constructed within the variational space during HCI iterations.
//! It uses the `Davidson` implementation from the `eigenvalues` submodule.

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::utils::read_input::Global;
use crate::var::eigenvalues::algorithms::davidson::{Davidson, DavidsonError};
use crate::var::eigenvalues::algorithms::{DavidsonCorrection, SpectrumTarget};
use crate::var::ham_gen::gen_sparse_ham_fast;
use crate::var::fast_ham::gen_sparse_ham_fast_lookup;
use crate::wf::VarWf;
use nalgebra::DMatrix;
use std::time::Instant;

/// Constructs the sparse Hamiltonian and solves the eigenvalue problem using Davidson.
///
/// This function is called within each iteration of the main `variational` loop.
/// It performs two main steps:
/// 1. **Build Sparse Hamiltonian:** Calls either `gen_sparse_ham_fast` (original) or
///    `gen_sparse_ham_fast_lookup` (fast algorithm) to compute and store the necessary 
///    off-diagonal elements of the Hamiltonian matrix within the current variational 
///    space (`wf.wf`) into `wf.sparse_ham`.
/// 2. **Davidson Diagonalization:** Initializes and runs the `Davidson` algorithm
///    using the constructed `wf.sparse_ham`. It targets the lowest eigenvalue
///    (ground state) and uses diagonal preconditioning (DPR). An initial guess
///    vector based on the previous iteration's coefficients can be provided via `init_last_iter`.
///
/// Upon successful convergence of the Davidson algorithm, it updates the variational
/// energy (`wf.wf.energy`) and coefficients (`wf.wf.dets[*].coeff`) in the `VarWf` struct.
///
/// # Arguments
/// * `global`: Global calculation parameters (unused here, but potentially needed by `gen_sparse_ham_fast`).
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator.
/// * `wf`: Mutable reference to the variational wavefunction structure. The sparse Hamiltonian
///   is built into `wf.sparse_ham`, and the resulting energy/coefficients are stored in `wf.wf`.
/// * `coeff_eps`: Convergence threshold for the eigenvector residuals in Davidson.
/// * `energy_eps`: Convergence threshold for the eigenvalue change in Davidson.
/// * `init_last_iter`: If `true`, use the coefficients from the previous iteration as the initial guess for Davidson.
pub fn sparse_optimize(
    global: &Global, // Potentially unused, passed to gen_sparse_ham_fast
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    wf: &mut VarWf,
    coeff_eps: f64,
    energy_eps: f64,
    init_last_iter: bool,
) {
    let start_gen_sparse_ham: Instant = Instant::now();
    
    // Use fast algorithm for larger systems, original for smaller ones
    if wf.wf.n > 100 {
        println!("Using fast Hamiltonian construction (n={})...", wf.wf.n);
        gen_sparse_ham_fast_lookup(global, wf, ham);
    } else {
        println!("Using original Hamiltonian construction for small system (n={})...", wf.wf.n);
        gen_sparse_ham_fast(global, wf, ham, excite_gen);
    }
    
    println!(
        "Time to generate sparse H: {:?}",
        start_gen_sparse_ham.elapsed()
    );

    // Davidson
    let start_dav: Instant = Instant::now();
    let dav: Result<Davidson, DavidsonError>;
    let init: Option<DMatrix<f64>> = {
        if init_last_iter {
            // Use inital guess
            let mut init = DMatrix::from_vec(wf.wf.n, 1, vec![0.0; wf.wf.n]);
            for (i, det) in wf.wf.dets.iter().enumerate() {
                init[(i, 0)] = det.coeff;
            }
            Some(init)
        } else {
            // No initial guess
            None
        }
    };
    // let eps_hpr = wf.eps;
    dav = Davidson::new(
        &wf.sparse_ham,
        1,
        init,
        DavidsonCorrection::DPR, //HPR(eps_hpr),
        SpectrumTarget::Lowest,
        coeff_eps,
        energy_eps,
        ham,
        excite_gen,
        &wf.wf,
    );
    println!(
        "Time to perform Davidson diagonalization: {:?}",
        start_dav.elapsed()
    );

    match dav {
        Ok(eig) => {
            wf.wf.energy = eig.eigenvalues[0];
            for i in 0..wf.wf.n {
                wf.wf.dets[i].coeff = eig.eigenvectors[(i, 0)];
            }
        }
        Err(err) => {
            println!("Error! {}", err);
        }
    }
}