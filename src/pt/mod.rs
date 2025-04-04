//! # Epstein-Nesbet Perturbation Theory Module (`pt`)
//!
//! This module implements the second-order Epstein-Nesbet perturbation theory (ENPT2)
//! correction, often used in conjunction with Heat-bath Configuration Interaction (HCI)
//! to form the Semistochastic HCI (SHCI) method.
//!
//! It includes functions for both purely deterministic PT (using a screening threshold `eps`)
//! and semistochastic approaches that combine deterministic treatment of important
//! contributions with stochastic sampling of the remaining perturbative space.

use crate::excite::init::ExciteGenerator;
use crate::excite::iterator::dets_excites_and_excited_dets;
use crate::excite::{Excite, Orbs};
use crate::ham::Ham;
use crate::rng::{init_rand, Rand};
use crate::semistoch::{new_semistoch_enpt2_dtm_diag_singles, old_semistoch_enpt2};
use crate::utils::read_input::Global;
use crate::wf::det::{Config, Det};
use crate::wf::Wf;
use itertools::enumerate;
use std::collections::HashMap;
use crate::utils::bits::valence_elecs_and_epairs;

/// Orchestrates the calculation of the second-order Epstein-Nesbet perturbation theory correction (PT2).
///
/// Based on the `global.use_new_semistoch` flag, it dispatches to either the older
/// semistochastic method (`old_semistoch_enpt2`) or a newer importance-sampled variant
/// (`new_semistoch_enpt2_dtm_diag_singles`). It then prints the final variational energy,
/// PT2 correction (with standard deviation if sampled), and total energy.
///
/// # Arguments
/// * `global`: Global calculation parameters.
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator data.
/// * `wf`: The converged variational wavefunction (from HCI).
pub fn perturbative(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, wf: &Wf) {
    // Initialize random number genrator
    let mut rand: Rand = init_rand();

    let e_pt2: f64;
    let std_dev: f64;
    // e_pt2 = dtm_pt(wf, excite_gen, ham, 1e-6);
    // println!("Variational energy: {}, Deterministic PT: {}, Total energy: {}", wf.energy, e_pt2, wf.energy + e_pt2);
    if !global.use_new_semistoch {
        // Old SHCI (2017 paper)
        println!("\nCalling semistoch ENPT2 the old way with p ~ |c|");
        let out = old_semistoch_enpt2(wf, global, ham, excite_gen, false, &mut rand);
        e_pt2 = out.0;
        std_dev = out.1;
    } else {
        println!("\nCalling semistoch ENPT2 the new way with importance sampling");
        let out = new_semistoch_enpt2_dtm_diag_singles(wf, global, ham, excite_gen, &mut rand);
        e_pt2 = out.0;
        std_dev = out.1;
    }
    println!("Variational energy: {:.6}", wf.energy);
    println!("PT energy: {:.6} +- {:.6}", e_pt2, std_dev);
    println!("Total energy: {:.6} +- {:.6}", wf.energy + e_pt2, std_dev);
}

/// Calculates the deterministic second-order Epstein-Nesbet PT correction.
///
/// 1. Generates the first-order interacting wavefunction `|psi_1> = sum_a |a><a|H|psi_0> / (E_0 - E_a)`,
///    where `|psi_0>` is the input variational wavefunction `wf`, and `|a>` are external determinants.
///    It uses `dets_excites_and_excited_dets` to find external determinants `|a>` connected
///    to `|psi_0>` with `|H_ai * c_i| >= eps`.
/// 2. Efficiently computes the diagonal elements `E_a = <a|H|a>` for these external determinants.
/// 3. Calculates the PT2 energy using the formula: `E_PT2 = sum_a |<a|H|psi_0>|^2 / (E_0 - E_a)`.
///
/// # Arguments
/// * `wf`: The converged variational wavefunction (`|psi_0>`).
/// * `excite_gen`: Pre-computed excitation generator.
/// * `ham`: The Hamiltonian operator.
/// * `eps`: The screening threshold for including perturbative contributions.
///
/// # Returns
/// The deterministic PT2 energy correction.
pub fn dtm_pt(wf: &Wf, excite_gen: &ExciteGenerator, ham: &Ham, eps: f64) -> f64 {
    println!("Start of deterministic PT");
    let mut h_psi: Wf = Wf::default();
    let mut n: usize = 0;
    let mut old_var_det: Config = wf.dets[0].config;
    for (var_det, excite, pt_config) in dets_excites_and_excited_dets(wf, excite_gen, eps) {
        if var_det.config != old_var_det {
            n += 1;
            old_var_det = var_det.config;
            if n % 1000 == 0 {
                println!("Var det {} of {}", n, wf.n);
            }
        }
        // println!("Var det: {}, PT det: {}", var_det, pt_config);

        // Compute off-diagonal element times var_det.coeff
        let h_ai_c_i: f64 = ham.ham_off_diag(&var_det.config, &pt_config, &excite) * var_det.coeff;

        // For single excitaitons: Check whether this excite actually exceeds eps
        if let Orbs::Single(_) = excite.init {
            if h_ai_c_i.abs() < eps {
                continue;
            }
        }

        // Compute diagonal element in O(N) time only if necessary
        if let Some(ind) = h_psi.inds.get_mut(&pt_config) {
            h_psi.dets[*ind].coeff += h_ai_c_i;
        } else {
            h_psi.dets.push(Det {
                config: pt_config,
                coeff: h_ai_c_i,
                diag: Some(var_det.new_diag(ham, &excite)),
            });
        }
    }
    println!("Preparing to calculate PT energy from generated PT dets");
    pt(&h_psi, wf.energy)
}

// Removed commented-out functions `dtm_pt_basic` and `dtm_pt_batches`

/// Evaluate the PT expression given H * Psi and E0
/// Calculates the PT2 energy sum given the first-order interacting wavefunction `h_psi`.
///
/// Computes `sum_a |(H*psi)_a|^2 / (E0 - E_a)`, where `a` iterates over the determinants
/// in `h_psi`, `(H*psi)_a` is `det.coeff`, `E_a` is `det.diag`, and `E0` is the reference energy.
/// Assumes `det.diag` is `Some` for all determinants in `h_psi`.
pub fn pt(h_psi: &Wf, e0: f64) -> f64 {
    h_psi.dets.iter().fold(0.0f64, |e_pt, det| {
        e_pt + det.coeff * det.coeff / (e0 - det.diag.unwrap())
    })
}

/// Evaluate off-diagonal component of the PT expression given H * Psi, sum_sq, and E0
/// Calculates the off-diagonal component of a PT2 estimator (used in stochastic methods).
///
/// Computes `sum_a [ ((H*psi)_a)^2 - (sum_sq)_a ] / (E0 - E_a)`.
/// This form often appears in variance calculations or specific stochastic PT estimators
/// where `sum_sq` might represent the sum of squared individual contributions `(H_ai * c_i)^2`.
/// Assumes `h_psi` and `sum_sq` have corresponding determinants with valid diagonal energies.
pub fn pt_off_diag(h_psi: &Wf, sum_sq: &Wf, e0: f64) -> f64 {
    // h_psi.dets.iter().zip(sum_sq.dets.iter()).fold(0.0f64, |e_pt, (sum_hc, sum_hc_sq)| {
    //     e_pt + (sum_hc.coeff * sum_hc.coeff - sum_hc_sq.coeff) / (e0 - sum_hc.diag.unwrap())
    // })
    let mut out: f64 = 0.0;
    for (sum_hc, sum_hc_sq) in h_psi.dets.iter().zip(sum_sq.dets.iter()) {
        out += (sum_hc.coeff * sum_hc.coeff - sum_hc_sq.coeff) / (e0 - sum_hc.diag.unwrap());
    }
    out
}

/// Stores samples collected during stochastic or semistochastic PT calculations.
///
/// Used to accumulate information needed to compute the unbiased PT2 energy estimator
/// as described in the SHCI papers.
///
/// The main `samples` HashMap maps a perturbative determinant configuration (`Config`)
/// to a tuple containing:
/// 1.  `f64`: The diagonal energy (E_a) of the perturbative determinant `a`.
/// 2.  `HashMap<Config, (f64, f64, i32)>`: A nested map where keys are the configurations
///     of *variational* determinants (`i`) that generated this perturbative determinant `a`.
///     The values are tuples `(H_ai * c_i, p_ai, w_ai)`:
///     *   `H_ai * c_i`: The matrix element connecting `a` and `i` times the coefficient of `i`.
///     *   `p_ai`: The probability with which the excitation `i -> a` was sampled.
///     *   `w_ai`: The number of times this specific excitation `i -> a` was sampled (weight).
#[derive(Default)]
pub struct PtSamples {
    /// Total number of samples collected.
    pub n: i32,
    /// The main storage for collected samples.
    pub samples: HashMap<Config, (f64, HashMap<Config, (f64, f64, i32)>)>,
}

impl PtSamples {
    // pub fn print(&self) {
    //     for (pt_key, pt_val) in self.samples.iter() {
    //         println!("Perturbative det: {} with diag elem: {}", pt_key, pt_val.0);
    //         for (var_key, var_val) in pt_val.1.iter() {
    //             println!(
    //                 "   Variational det: {}, H_ai c_i: {}, p_ai: {}, w_ai: {}",
    //                 var_key, var_val.0, var_val.1, var_val.2
    //             )
    //         }
    //     }
    // }

    /// Clears the collected samples to start a new batch or calculation.
    pub fn clear(&mut self) {
        self.n = 0;
        self.samples.clear(); // Use clear() instead of Default::default() to avoid reallocation
    }

    /// Adds a new sample to the collection, computing the PT determinant's diagonal if needed.
    ///
    /// Takes a sampled excitation from `var_det` to `pt_det` (via `excite`) with probability
    /// `sampled_prob`. It updates the nested HashMaps in `self.samples`. If `pt_det.config`
    /// is encountered for the first time, its diagonal energy `E_a` is computed efficiently
    /// using `var_det.new_diag` and stored.
    pub fn add_sample_compute_diag(
        &mut self,
        var_det: Det,        // The variational determinant (i) that was sampled from
        excite: &Excite,     // The excitation (i -> a) that was sampled
        pt_det: Det,         // The resulting perturbative determinant (a), contains H_ai * c_i in coeff field
        sampled_prob: f64,   // The probability p_ai with which this excitation was sampled
        ham: &Ham,           // Hamiltonian needed for new_diag
    ) {
        // Add a new sample to PtSamples
        // Compute diagonal element of perturbative determinant if it hasn't already been computed
        self.n += 1;
        // match excite.init {
        //     Orbs::Double(_) => {
        //         println!("Doubles in add_sample_compute_diag: (Hc)^2 / p = {}", pt_det.coeff * pt_det.coeff / sampled_prob);
        //     },
        //     Orbs::Single(_) => {
        //         println!("Singles in add_sample_compute_diag: (Hc)^2 / p = {}", pt_det.coeff * pt_det.coeff / sampled_prob);
        //     }
        // }
        match self.samples.get_mut(&pt_det.config) {
            None => {
                // New PT det was sampled: compute diagonal element and create new variational det map
                let pt_diag_elem: f64 = var_det.new_diag(ham, excite);
                let mut var_det_map: HashMap<Config, (f64, f64, i32)> = Default::default();
                var_det_map.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                self.samples
                    .insert(pt_det.config, (pt_diag_elem, var_det_map));
            }
            Some(pt_det_info) => {
                // No need to recompute the diagonal element
                Self::process_resampled_pt_det(&var_det, pt_det, sampled_prob, pt_det_info)
            }
        }
    }

    /// Helper function to update the sample information when a perturbative determinant `pt_det`
    /// has already been encountered (i.e., its diagonal energy is known).
    ///
    /// Updates the inner HashMap associated with `pt_det.config`. If the specific variational
    /// determinant `var_det` is new for this `pt_det`, it's added. If `var_det` has contributed
    /// to this `pt_det` before, only its weight (`w_ai`) is incremented.
    fn process_resampled_pt_det(
        var_det: &Det,          // The variational determinant (i)
        pt_det: Det,            // The perturbative determinant (a)
        sampled_prob: f64,      // The sampling probability p_ai
        pt_det_info: &mut (f64, HashMap<Config, (f64, f64, i32)>), // Existing info for pt_det
    ) {
        match pt_det_info.1.get_mut(&var_det.config) {
            None => {
                // New var det to reach this PT det; add to variational det map
                pt_det_info
                    .1
                    .insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
            }
            Some(var_det_info) => {
                // Already have this var det; just increment number of times it has been sampled
                var_det_info.2 += 1;
            }
        }
    }

    /// Adds a new sample, assuming the PT determinant's diagonal energy is already stored.
    ///
    /// Similar to `add_sample_compute_diag`, but assumes `pt_det.diag` contains the
    /// correct diagonal energy `E_a`, avoiding the call to `new_diag`.
    pub fn add_sample_diag_already_stored(&mut self, var_det: Det, pt_det: Det, sampled_prob: f64) {
        // Add a new sample to PtSamples
        // Assumes that pt_det's diagonal element already stored
        self.n += 1;
        match self.samples.get_mut(&pt_det.config) {
            None => {
                // New PT det was sampled
                let mut var_det_map: HashMap<Config, (f64, f64, i32)> = Default::default();
                var_det_map.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                self.samples
                    .insert(pt_det.config, (pt_det.diag.unwrap(), var_det_map));
            }
            Some(pt_det_info) => {
                // No need to recompute the diagonal element
                Self::process_resampled_pt_det(&var_det, pt_det, sampled_prob, pt_det_info)
            }
        }
    }

    /// Computes the unbiased semistochastic PT2 energy estimator from the collected samples.
    ///
    /// Implements the estimator formula from the SHCI papers (e.g., Eq. 10 in JCP 149, 214105 (2018)),
    /// which corrects for sampling bias.
    /// E_PT2 = (1 / (N_v * (N_v - 1))) * sum_a [ sum_i (w_ai / p_ai * (H_ai*c_i)) ]^2 / (E0 - E_a)
    ///         + (1 / (N_v * (N_v - 1))) * sum_a [ sum_i ( (N_v-1) - w_ai/p_ai ) * (w_ai/p_ai) * (H_ai*c_i)^2 ] / (E0 - E_a)
    /// where N_v (`n_det`) is the number of variational determinants sampled from in total.
    ///
    /// # Arguments
    /// * `e0`: The variational reference energy.
    /// * `n_det`: The total number of samples drawn from the variational space (N_v).
    ///
    /// # Returns
    /// The estimated PT2 energy correction.
    pub fn pt_estimator(&self, e0: f64, n_det: i32) -> f64 {
        // TODO: Figure out why PT energy is wrong and why contributions vary so much

        let mut out: f64 = 0.0;
        let mut diag_term: f64;
        let mut to_square: f64;
        let mut w_over_p: f64;

        // TODO: Exclude perturbers that only have large contributions

        for (_pt_det, (pt_det_diag, var_det_map)) in &self.samples {
            // println!("\nPT det {}: {}\n", ind, pt_det);
            diag_term = 0.0;
            to_square = 0.0;
            for (hai_ci, p, w) in var_det_map.values() {
                // println!("New energy sample! H_ai c_i = {}, p = {}, (H_ai c_i)^2 / p = {}, w = {}, E0 = {}, E_a = {}", hai_ci, p, hai_ci * hai_ci / p, w, e0, pt_det_diag);
                if *p < 1e-9 {
                    // println!("Warning! Sample probability very small! p = {}", p);
                } else {
                    w_over_p = (*w as f64) / p;
                    // println!("p = {:.2e}", p);
                    // println!("(H_ai c_i)^2 / p_i = {:.3}", hai_ci * hai_ci / p);
                    diag_term += ((n_det - 1) as f64 - w_over_p) * w_over_p * hai_ci * hai_ci;
                    to_square += hai_ci * w_over_p;
                }
            }
            // println!(
            //     "Diag term = {:.3}, off-diag term = {:.3}, diag + off-diag^2 = {:.3}",
            //     diag_term,
            //     to_square,
            //     diag_term + to_square * to_square
            // );
            // println!(
            //     "Energy estimator: {}",
            //     (diag_term + to_square * to_square) / (e0 - pt_det_diag)
            // );
            out += (diag_term + to_square * to_square) / (e0 - pt_det_diag);
            // tmp += diag_term + to_square * to_square;
        }

        // println!(
        //     "Unbiased estimator ({} sampled var dets, {} sampled PT dets) = {}",
        //     n_det,
        //     self.n,
        //     out / (n_det as f64 * (n_det - 1) as f64)
        // );
        // println!(
        //     "Component of unbiased estimator that should be constant = {:.3}",
        //     tmp / (n_det as f64 * (n_det - 1) as f64)
        // );
        out / (n_det as f64 * (n_det - 1) as f64)
    }
}
