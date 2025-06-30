//! # Matrix-Vector Operations
//!
//! This module provides a unified interface for the various approx_matmul_* functions
//! using an enum-based approach instead of traits to avoid lifetime complexity.

use super::{Wf, det::Config};
use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs};
use crate::ham::Ham;
use crate::stoch::ScreenedSampler;
use crate::utils::bits::{bit_pairs, bits};
use itertools::enumerate;

/// Strategy for handling single excitations in matrix-vector products
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SinglesStrategy {
    /// Include all singles deterministically
    Include,
    /// Skip singles entirely
    Skip,
    /// Handle singles stochastically (semi-stochastic approach)
    Semistochastic,
    /// No singles processing
    None,
}

/// Strategy for handling double excitations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoublesStrategy {
    /// Separate processing for doubles and singles
    Separate,
    /// Standard double excitation processing
    Standard,
    /// Deterministic-only processing
    DeterministicOnly,
}

/// Configuration for matrix-vector product operations
#[derive(Debug, Clone)]
pub struct MatVecConfig {
    /// Screening threshold (epsilon)
    pub eps: f64,
    /// Strategy for handling single excitations
    pub singles_strategy: SinglesStrategy,
    /// Strategy for handling double excitations
    pub doubles_strategy: DoublesStrategy,
    /// Whether to compute diagonal elements
    pub compute_diagonals: bool,
    /// Input coefficients for variational operations
    pub input_coeffs: Option<Vec<f64>>,
}

/// Result of matrix-vector multiplication operations
pub enum MatVecResult<'a> {
    /// External space result: new wavefunction + matrix elements vector
    ExternalWithElements(Wf, Vec<f64>),
    /// External space result: new wavefunction + screened sampler
    ExternalWithSampler(Wf, ScreenedSampler<'a>),
    /// Variational space result: coefficient vector
    Variational(Vec<f64>),
    /// Diagonal computation result: wavefunction with computed diagonals
    WithDiagonals(Wf),
}

/// Matrix-vector product operations for wavefunctions
pub struct MatVecOperations;

impl MatVecOperations {
    /// Apply the Hamiltonian to the wavefunction using specified configuration
    pub fn apply_hamiltonian<'a>(
        wf: &'a Wf,
        ham: &Ham,
        excite_gen: &'a ExciteGenerator,
        config: &MatVecConfig,
    ) -> MatVecResult<'a> {
        match (&config.input_coeffs, config.singles_strategy, config.doubles_strategy) {
            // Variational space operations (use input coefficients)
            (Some(coeffs), _, _) => {
                let result = Self::apply_variational_internal(wf, coeffs, ham, excite_gen, config);
                MatVecResult::Variational(result)
            }
            
            // External space operations (perturbative)
            (None, SinglesStrategy::Include, DoublesStrategy::Separate) => {
                let (new_wf, sampler) = wf.approx_matmul_external_separate_doubles_and_singles(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithSampler(new_wf, sampler)
            }
            
            (None, SinglesStrategy::Skip, DoublesStrategy::Standard) => {
                let (new_wf, sampler) = wf.approx_matmul_external_skip_singles(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithSampler(new_wf, sampler)
            }
            
            (None, SinglesStrategy::Semistochastic, DoublesStrategy::Standard) => {
                let (new_wf, sampler) = wf.approx_matmul_external_semistoch_singles(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithSampler(new_wf, sampler)
            }
            
            (None, SinglesStrategy::None, DoublesStrategy::Standard) => {
                let (new_wf, sampler) = wf.approx_matmul_external_no_singles(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithSampler(new_wf, sampler)
            }
            
            (None, _, DoublesStrategy::DeterministicOnly) => {
                let (new_wf, elements) = wf.approx_matmul_external_dtm_only(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithElements(new_wf, elements)
            }
            
            // Default case - use standard external approach
            (None, _, _) => {
                let (new_wf, sampler) = wf.approx_matmul_external_separate_doubles_and_singles(ham, excite_gen, config.eps);
                MatVecResult::ExternalWithSampler(new_wf, sampler)
            }
        }
    }

    /// Convenience method for external perturbative space operations
    pub fn apply_external<'a>(
        wf: &'a Wf,
        ham: &Ham,
        excite_gen: &'a ExciteGenerator,
        eps: f64,
        singles_strategy: SinglesStrategy,
    ) -> MatVecResult<'a> {
        let config = MatVecConfig {
            eps,
            singles_strategy,
            doubles_strategy: DoublesStrategy::Standard,
            compute_diagonals: false,
            input_coeffs: None,
        };
        Self::apply_hamiltonian(wf, ham, excite_gen, &config)
    }

    /// Convenience method for variational space operations
    pub fn apply_variational(
        wf: &Wf,
        input_coeffs: &[f64],
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> Vec<f64> {
        let config = MatVecConfig {
            eps,
            singles_strategy: SinglesStrategy::Include,
            doubles_strategy: DoublesStrategy::Standard,
            compute_diagonals: false,
            input_coeffs: Some(input_coeffs.to_vec()),
        };
        
        match Self::apply_hamiltonian(wf, ham, excite_gen, &config) {
            MatVecResult::Variational(coeffs) => coeffs,
            _ => panic!("Expected variational result from apply_variational"),
        }
    }

    /// Internal implementation for variational space matrix-vector product
    fn apply_variational_internal(
        wf: &Wf,
        input_coeffs: &[f64],
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        config: &MatVecConfig,
    ) -> Vec<f64> {
        // Unified implementation that consolidates approx_matmul_variational and 
        // approx_matmul_off_diag_variational_no_singles logic
        
        let mut out: Vec<f64> = vec![0.0; wf.n];
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // Diagonal component (skip if config specifies off-diagonal only)
        if config.singles_strategy != SinglesStrategy::None {
            for (i_det, det) in enumerate(wf.dets.iter()) {
                if let Some(diag) = det.diag {
                    out[i_det] = diag * input_coeffs[i_det];
                }
            }
        }

        // Off-diagonal component
        for (i_det, det) in enumerate(wf.dets.iter()) {
            if input_coeffs[i_det].abs() < 1e-12 {
                continue; // Skip negligible coefficients
            }
            
            local_eps = config.eps / input_coeffs[i_det].abs();
            
            // Double excitations - opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if let Some(&j_det) = wf.inds.get(&d) {
                                    // Valid excitation within variational space
                                    out[j_det] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det];
                                }
                            }
                        }
                    }
                }
            }

            // Double excitations - same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config_bits, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config_bits) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if let Some(&j_det) = wf.inds.get(&d) {
                                    // Valid excitation within variational space
                                    out[j_det] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det];
                                }
                            }
                        }
                    }
                }
            }

            // Single excitations (if strategy includes them)
            if config.singles_strategy == SinglesStrategy::Include && excite_gen.max_sing >= local_eps {
                for (config_bits, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config_bits) {
                        for stored_excite in excite_gen
                            .sing_sorted_list
                            .get(&Orbs::Single(i))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if let Some(&j_det) = wf.inds.get(&d) {
                                    // Valid excitation within variational space
                                    out[j_det] += ham.ham_sing(&det.config, &d) * input_coeffs[i_det];
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }
}