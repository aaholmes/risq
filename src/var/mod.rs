//! # Variational HCI Module (`var`)
//!
//! This module implements the variational stage of the Heat-bath Configuration Interaction (HCI) algorithm.
//! HCI is a selected CI method that iteratively builds a compact and important subset of the
//! full CI space.
//!
//! ## Algorithm Overview:
//! 1. **Initialization:** Start with a reference wavefunction (e.g., Hartree-Fock).
//! 2. **Iteration:** Repeat until convergence:
//!    a. **Search/Add:** Identify important determinants connected to the current variational
//!       space but not yet included, based on a screening threshold (`eps_var`). Add these
//!       new determinants to the variational space (`find_new_dets`).
//!    b. **Diagonalize:** Construct the Hamiltonian matrix within the current (expanded)
//!       variational space and solve the eigenvalue problem (using the Davidson algorithm
//!       implemented in `davidson::sparse_optimize`) to obtain the updated variational
//!       energy and wavefunction coefficients.
//!    c. **Update Epsilon:** Decrease the screening threshold `eps_var` according to a schedule
//!       (managed by `VarWf::eps_iter`).
//!    d. **Check Convergence:** Stop if no new determinants are added at the target `eps_var`,
//!       or if the energy converges, or if a maximum iteration count is reached.
//!
//! ## Submodules:
//! *   `davidson`: Implementation of the Davidson diagonalization algorithm for sparse matrices.
//! *   `eigenvalues`: (Potentially external or older) eigenvalue solver components.
//! *   `ham_gen`: Functions for generating Hamiltonian matrix elements for the variational space.
//! *   `off_diag`: Utilities for handling off-diagonal elements in the sparse matrix.
//! *   `sparse`: Custom sparse matrix representations.
//! *   `utils`: Helper functions specific to the variational stage.

mod davidson;
pub mod davidson_unified;
pub mod eigenvalues;
pub mod fast_ham;
mod fast_ham_tests;
pub mod debug_fast_ham;
mod ham_gen;
pub mod off_diag;
pub(crate) mod sparse;
mod utils;

use super::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::utils::read_input::Global;
use crate::var::davidson::sparse_optimize;
use crate::wf::VarWf;
use std::time::Instant;
use crate::excite::iterator::dets_excites_and_excited_dets;

/// Performs the iterative variational Heat-bath Configuration Interaction (HCI) calculation.
///
/// Implements the main HCI loop: iteratively finding important connected determinants,
/// adding them to the variational space (`var_wf`), and diagonalizing the Hamiltonian
/// in the current space using the Davidson algorithm (`sparse_optimize`). The screening
/// threshold `eps_var` is controlled by `var_wf.eps_iter`.
///
/// # Arguments
/// * `global`: Global calculation parameters (e.g., `eps_var` target).
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator.
/// * `var_wf`: Mutable reference to the variational wavefunction structure, which is
///   updated in place throughout the iterations.
pub fn variational(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, var_wf: &mut VarWf) {
    let mut iter: i32 = 0;

    println!(
        "Start of variational stage: Wavefunction has {} det with energy {:.14}",
        var_wf.wf.n, var_wf.wf.energy
    );

    // let eps_energy_converged: f64 = 2.5e-4;
    let mut last_energy: Option<f64>;

    while !var_wf.converged {
        iter += 1;

        if iter > 20 {
            println!("Too many iterations! Stopping");
            break;
        }

        let start_find_new_dets: Instant = Instant::now();
        // --- Search/Add Step ---
        // Find new determinants connected to the current space that are important
        // according to the current screening threshold `var_wf.eps`.
        let no_new_dets_added = var_wf.find_new_dets(global, ham, excite_gen); // Modifies var_wf.wf
        println!("Time to find new dets: {:?}", start_find_new_dets.elapsed());

        // Check for convergence: If we are at the target epsilon and no new determinants were added.
        if var_wf.eps == global.eps_var && no_new_dets_added {
            println!("No new dets added at target eps_var; variational stage converged.");
            var_wf.converged = true;
            break; // Exit the loop
        }

        last_energy = Some(var_wf.wf.energy);

        let coeff_eps: f64 = 1e-4; // Davidson convergence epsilon for coefficients
        let energy_eps: f64 = 1e-8; // Davidson convergence epsilon for energy

        // --- Diagonalization Step ---
        // Solve the eigenvalue problem Hc = Ec within the current variational space
        // using the Davidson algorithm. Updates `var_wf.wf.energy` and `var_wf.wf.dets[*].coeff`.
        println!("\nOptimizing coefficients of wf with {} dets", var_wf.wf.n);
        let start_optimize_coeffs: Instant = Instant::now();
        sparse_optimize( // Calls the Davidson solver
            global,     // Pass global parameters (might be needed by ham_gen)
            ham,
            excite_gen,
            var_wf,     // Passed mutably to update energy/coeffs and build sparse H
            coeff_eps,  // Davidson convergence threshold for eigenvector change
            energy_eps, // Davidson convergence threshold for eigenvalue change
            iter > 1,   // Flag indicating whether to reuse previous guess vectors
        );
        println!(
            "Time to optimize wf coefficients: {:?}",
            start_optimize_coeffs.elapsed()
        );
        // Removed commented-out call to dense_optimize

        println!("End of iteration {} (eps = {:.1e}): Wavefunction has {} determinants with energy {:.14}", iter, var_wf.eps, var_wf.wf.n, var_wf.wf.energy);
        if var_wf.wf.n <= 10 {
            var_wf.print();
        } else {
            var_wf.print_largest(10);
        }

        // if iter == 2 { panic!("Debug!") }

        // --- Update Epsilon & Check Other Convergence ---
        // Get the epsilon for the *next* iteration from the Eps iterator.
        var_wf.eps = var_wf.eps_iter.next().unwrap_or(global.eps_var);

        // Optional: Add energy convergence check here if desired
        // if let Some(last_e) = last_energy {
        //     if (var_wf.wf.energy - last_e).abs() < some_energy_threshold {
        //         println!("Variational energy converged.");
        //         var_wf.converged = true;
        //         break;
        //     }
        // }
    }
}
