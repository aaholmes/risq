//! # Stochastic Sampling Utilities (`stoch`)
//!
//! This module provides general utilities and data structures for performing
//! stochastic sampling, particularly importance sampling, as required by
//! semistochastic methods.
//!
//! ## Key Components:
//! *   `alias`: Submodule implementing Alias sampling for efficient O(1) sampling from
//!     discrete distributions.
//! *   `utils`: Submodule containing other sampling helpers like CDF sampling.
//! *   `ImpSampleDist`: Enum to specify the desired importance sampling distribution
//!     (e.g., proportional to |Hc| or (Hc)^2).
//! *   `DetOrbSample`: Represents a potential starting point for sampling an excitation
//!     (a specific determinant and initial orbital(s)).
//! *   `ScreenedSampler`: A structure holding pre-computed Alias tables for efficiently
//!     sampling `DetOrbSample` instances according to different distributions.
//! *   `matmul_sample_remaining`: Function to perform a full importance sampling step,
//!     first sampling a `DetOrbSample` and then sampling a target excitation.

use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::ham::Ham;
use crate::stoch::utils::sample_cdf;
use crate::wf::det::Det;
use std::hash::{Hash, Hasher};

pub(crate) mod alias;
pub(crate) mod utils;
use crate::rng::Rand;
use alias::Alias;

/// Specifies the desired probability distribution for importance sampling.
#[derive(Clone, Copy, Debug)] // Added Debug
pub enum ImpSampleDist {
    /// Sample with probability proportional to `|H_ai * c_i|`.
    AbsHc,
    /// Sample with probability proportional to `(H_ai * c_i)^2`.
    HcSquared,
}

/// For importance sampling the component of the matmul that is screened out by the eps threshold

/// Matmul_sample_remaining performs the whole excitation sampling (exciting pair and target), but
/// this contains only data structures sampling the exciting det and its exciting electron pair;
/// sampling the target electron pair uses CDF searching which only requires ExciteGenerator

/// Lifetime 'a must last as long as the vector being semistochastically multiplied, since this
/// struct has pointers to its components

/// For doubles:
/// Contains two samplers, for sampling with p ~ |Hc| and with p ~ (Hc)^2

/// For singles:
/// If uniform_singles is true, enables uniform sampling of the singles from small-magnitude
/// determinants that were not already treated deterministically
/// Else, singles are sampled alongside doubles using max |H| instead of |H| as their
/// relative prob

/// Also, contains elements, the list of elements being sampled (e.g. det/orb pairs)
/// Facilitates efficient importance sampling of determinant/initial-orbital pairs.
///
/// This structure is generated from a list of `DetOrbSample` instances, which represent
/// all possible starting points (determinant `i` and initial orbitals `init`) for excitations
/// that fall below the deterministic screening threshold (`eps_pt_dtm`).
///
/// It pre-computes Alias tables (`det_orb_sampler_*`) allowing for O(1) sampling of a
/// `DetOrbSample` according to either the `|H_ai * c_i|` or `(H_ai * c_i)^2` distribution
/// (summed over potential targets `a`).
pub struct ScreenedSampler<'a> {
    /// Vector storing the actual `DetOrbSample` elements that can be sampled.
    /// The index corresponds to the sample returned by the Alias samplers.
    pub elements: Vec<DetOrbSample<'a>>,
    /// Alias sampler for drawing samples with probability proportional to `sum_a |H_ai * c_i|`.
    pub det_orb_sampler_abs_hc: Alias,
    /// Alias sampler for drawing samples with probability proportional to `sum_a (H_ai * c_i)^2`.
    pub det_orb_sampler_hc_squared: Alias,
    /// The global sum of `sum_abs_hc` over all elements, used for normalization or total estimates.
    pub sum_abs_hc_all_dets_orbs: f64,
    /// The global sum of `sum_hc_squared` over all elements, used for normalization or total estimates.
    pub sum_hc_squared_all_dets_orbs: f64,
}

/// Represents a potential starting point for sampling an excitation.
///
/// Contains a reference to a source determinant (`det`) and the specific initial
/// orbital(s) (`init`) and spin (`is_alpha`) from which an excitation originates.
/// It also stores pre-calculated sums of Hamiltonian estimates (`sum_*_h`) and
/// wavefunction-weighted estimates (`sum_*_hc`) over all possible *target* excitations
/// originating from this specific `det`/`init`/`is_alpha` combination that fall *below*
/// the deterministic threshold (`eps_pt_dtm`). These sums are used to build the
/// `ScreenedSampler`'s Alias tables.
#[derive(Clone, Copy, Debug)]
pub struct DetOrbSample<'a> {
    /// Reference to the source determinant (`i`). Lifetime `'a` ensures it lives as long as the source `Wf`.
    pub det: &'a Det,
    /// The initial orbital(s) the excitation originates from.
    pub init: Orbs,
    /// The spin channel of the excitation (None for opposite-spin doubles).
    pub is_alpha: Option<bool>,
    /// Sum of `|H_ai|` over all target excitations `a` originating from this `det`/`init`/`is_alpha`
    /// that were screened out (i.e., below `eps_pt_dtm`).
    pub sum_abs_h: f64,
    /// Sum of `H_ai^2` over all target excitations `a` originating from this `det`/`init`/`is_alpha`
    /// that were screened out.
    pub sum_h_squared: f64,
    /// Sum of `|H_ai * c_i|` over all target excitations `a` originating from this `det`/`init`/`is_alpha`
    /// that were screened out. Used for `AbsHc` importance sampling distribution.
    pub sum_abs_hc: f64,
    /// Sum of `(H_ai * c_i)^2` over all target excitations `a` originating from this `det`/`init`/`is_alpha`
    /// that were screened out. Used for `HcSquared` importance sampling distribution.
    pub sum_hc_squared: f64,
}

impl PartialEq for DetOrbSample<'_> {
    /// Equality based on determinant configuration, initial orbitals, and spin channel.
    fn eq(&self, other: &Self) -> bool {
        self.det.config == other.det.config // Use Config's PartialEq
            && self.init == other.init
            && self.is_alpha == other.is_alpha
    }
}
impl Eq for DetOrbSample<'_> {}

impl Hash for DetOrbSample<'_> {
    /// Hashes based on determinant configuration, initial orbitals, and spin channel. Consistent with `PartialEq`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.det.config.hash(state);
        self.init.hash(state);
        self.is_alpha.hash(state);
    }
}

/// Creates a `ScreenedSampler` from a vector of `DetOrbSample` instances.
///
/// Calculates the total sums (`sum_*_all_dets_orbs`) and constructs the Alias tables
/// (`det_orb_sampler_*`) based on the normalized `sum_abs_hc` and `sum_hc_squared` values
/// from the input `det_orbs`. Filters out any `DetOrbSample` with zero probability (`sum_abs_h == 0.0`).
///
/// # Arguments
/// * `det_orbs`: A vector containing all `DetOrbSample` instances representing the space to be sampled.
///
/// # Returns
/// An initialized `ScreenedSampler` ready for O(1) sampling.
pub fn generate_screened_sampler(det_orbs: Vec<DetOrbSample>) -> ScreenedSampler {
    // pub fn generate_screened_sampler<'a>(eps: f64, det_orbs: &'a Vec<DetOrbSample>) -> ScreenedSampler<'a> {

    println!("Generating screened sampler of size {}", det_orbs.len());

    // Normalize probs (both |Hc| and (Hc)^2)
    let mut det_orbs_nonzero: Vec<DetOrbSample> = Vec::with_capacity(det_orbs.len());
    let mut probs_abs_hc: Vec<f64> = Vec::with_capacity(det_orbs.len());
    let mut probs_hc_squared: Vec<f64> = Vec::with_capacity(det_orbs.len());

    let sum_abs_hc_all_dets_orbs: f64 = det_orbs
        .iter()
        .fold(0f64, |sum, &val| sum + val.sum_abs_hc as f64);
    let sum_hc_squared_all_dets_orbs: f64 = det_orbs
        .iter()
        .fold(0f64, |sum, &val| sum + val.sum_hc_squared as f64);

    for val in det_orbs.iter() {
        if val.sum_abs_h != 0.0 {
            det_orbs_nonzero.push(*val);
            probs_abs_hc.push(val.sum_abs_hc / sum_abs_hc_all_dets_orbs);
            probs_hc_squared.push(val.sum_hc_squared / sum_hc_squared_all_dets_orbs);
        }
    }

    println!("Done generating screened sampler of size {}", det_orbs.len());

    ScreenedSampler {
        elements: det_orbs_nonzero,
        det_orb_sampler_abs_hc: Alias::new(probs_abs_hc),
        det_orb_sampler_hc_squared: Alias::new(probs_hc_squared),
        sum_abs_hc_all_dets_orbs,
        sum_hc_squared_all_dets_orbs,
    }
}

/// Performs one full importance sampling step for the screened (stochastic) part of H*psi.
///
/// This function combines two sampling steps:
/// 1. **Sample Source:** Samples a `DetOrbSample` (representing source determinant `i` and
///    initial orbitals `init`) from the `screened_sampler` using Alias sampling according
///    to the specified `imp_sample_dist` (`|Hc|` or `(Hc)^2`). This gives `det_orb_sample`
///    and its sampling probability `det_orb_prob`.
/// 2. **Sample Target:** Samples a specific target excitation (`StoredExcite`) from the list
///    associated with the chosen `det_orb_sample` (using `excite_gen`). This step uses
///    CDF sampling (`sample_cdf`) weighted by the same `imp_sample_dist`. This gives
///    `sampled_excite` and its conditional probability `sampled_excite_prob`.
///
/// It then reconstructs the full `Excite` object, applies it to the source determinant's
/// configuration, calculates the resulting determinant's coefficient (`H_ai * c_i`), and
/// returns the source `Det`, the `Excite`, the resulting `Det` (with coefficient but `diag=None`),
/// and the total probability (`det_orb_prob * sampled_excite_prob`).
/// Returns `None` for the `Det` tuple if the sampled excitation was invalid (e.g., target occupied).
///
/// # Arguments
/// * `screened_sampler`: The pre-computed sampler for source det/orb pairs.
/// * `imp_sample_dist`: The importance sampling distribution to use for both steps.
/// * `excite_gen`: Contains the lists of target excitations.
/// * `ham`: The Hamiltonian operator (needed to compute the final coefficient).
/// * `rand`: Mutable random number generator state.
///
/// # Returns
/// A tuple `(Option<(source_det, excitation, target_det)>, total_probability)`.
pub fn matmul_sample_remaining(
    screened_sampler: &ScreenedSampler,
    imp_sample_dist: ImpSampleDist,
    excite_gen: &ExciteGenerator,
    ham: &Ham,
    rand: &mut Rand,
) -> (Option<(Det, Excite, Det)>, f64) {
    let det_orb_sample: DetOrbSample;
    let det_orb_prob: f64;
    let sampled_excite: &StoredExcite;
    let sampled_excite_prob: f64;

    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            // First, sample a (determinant, orbs) pair using Alias sampling with prob |Hc|
            let sample = screened_sampler
                .det_orb_sampler_abs_hc
                .sample_with_prob(rand);
            det_orb_sample = screened_sampler.elements[sample.0];
            det_orb_prob = sample.1;
            // println!("Sampled det {}, prob {}", det_orb_sample, det_orb_prob);

            // Sample excitation from this det/orb pair by binary search the stored cdf with prob |H|
            match det_orb_sample.is_alpha {
                None => {
                    // Opposite spin double
                    let sample2 = sample_cdf(
                        excite_gen
                            .opp_doub_sorted_list
                            .get(&det_orb_sample.init)
                            .unwrap(),
                        &ImpSampleDist::AbsHc,
                        Some(det_orb_sample.sum_abs_h),
                        rand,
                    )
                    .unwrap();
                    sampled_excite = sample2.0;
                    sampled_excite_prob = sample2.1;
                }
                Some(_) => {
                    match det_orb_sample.init {
                        Orbs::Double(_) => {
                            // Same spin double
                            let sample2 = sample_cdf(
                                excite_gen
                                    .same_doub_sorted_list
                                    .get(&det_orb_sample.init)
                                    .unwrap(),
                                &ImpSampleDist::AbsHc,
                                Some(det_orb_sample.sum_abs_h),
                                rand,
                            )
                            .unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                        Orbs::Single(_) => {
                            // Single
                            let sample2 = sample_cdf(
                                excite_gen
                                    .sing_sorted_list
                                    .get(&det_orb_sample.init)
                                    .unwrap(),
                                &ImpSampleDist::AbsHc,
                                Some(det_orb_sample.sum_abs_h),
                                rand,
                            )
                            .unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                    }
                }
            }
        }
        ImpSampleDist::HcSquared => {
            // First, sample a (determinant, orbs) pair using Alias sampling with prob (Hc)^2
            let sample = screened_sampler
                .det_orb_sampler_hc_squared
                .sample_with_prob(rand);
            det_orb_sample = screened_sampler.elements[sample.0];
            det_orb_prob = sample.1;

            // Sample excitation from this det/orb pair by binary search the stored cdf with prob H^2
            match det_orb_sample.is_alpha {
                None => {
                    // Opposite spin double
                    let sample2 = sample_cdf(
                        excite_gen
                            .opp_doub_sorted_list
                            .get(&det_orb_sample.init)
                            .unwrap(),
                        &ImpSampleDist::HcSquared,
                        Some(det_orb_sample.sum_h_squared),
                        rand,
                    )
                    .unwrap();
                    sampled_excite = sample2.0;
                    sampled_excite_prob = sample2.1;
                }
                Some(_) => {
                    match det_orb_sample.init {
                        Orbs::Double(_) => {
                            // Same spin double
                            let sample2 = sample_cdf(
                                excite_gen
                                    .same_doub_sorted_list
                                    .get(&det_orb_sample.init)
                                    .unwrap(),
                                &ImpSampleDist::HcSquared,
                                Some(det_orb_sample.sum_h_squared),
                                rand,
                            )
                            .unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                        Orbs::Single(_) => {
                            // Single
                            let sample2 = sample_cdf(
                                excite_gen
                                    .sing_sorted_list
                                    .get(&det_orb_sample.init)
                                    .unwrap(),
                                &ImpSampleDist::HcSquared,
                                Some(det_orb_sample.sum_h_squared),
                                rand,
                            )
                            .unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                    }
                }
            }
        }
    }

    // Construct the excitation (for output)
    let excite = Excite {
        init: det_orb_sample.init,
        target: sampled_excite.target,
        abs_h: sampled_excite.abs_h,
        is_alpha: det_orb_sample.is_alpha,
    };

    // Apply the excitation to the sampled det
    let sampled_det = det_orb_sample.det.config.safe_excite_det(&excite);

    // Compute total probability
    let prob_sampled_det: f64 = det_orb_prob * sampled_excite_prob;

    match sampled_det {
        None => (None, prob_sampled_det), // Proposed excitation would excite to already-occupied orbs
        Some(d) => {
            // Compute new det coefficient
            let mut new_det_coeff = det_orb_sample.det.coeff;
            match det_orb_sample.init {
                Orbs::Double(_) => new_det_coeff *= ham.ham_doub(&det_orb_sample.det.config, &d),
                Orbs::Single(_) => new_det_coeff *= ham.ham_sing(&det_orb_sample.det.config, &d),
            }
            (
                Some((
                    *det_orb_sample.det,
                    excite,
                    Det {
                        config: d,
                        coeff: new_det_coeff,
                        diag: None, // Compute diagonal element later, only if needed (since it would be the most expensive step)
                    },
                )),
                prob_sampled_det,
            )
        }
    }
}
