//! General stochastic functions

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

/// Importance sampling distribution: proportional to either |Hc| or (Hc)^2
#[derive(Clone, Copy)]
pub enum ImpSampleDist {
    AbsHc,
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
pub struct ScreenedSampler<'a> {
    pub eps: f64,
    // pub uniform_singles: bool,
    pub elements: Vec<DetOrbSample<'a>>,
    pub det_orb_sampler_abs_hc: Alias,
    pub det_orb_sampler_hc_squared: Alias,
    pub sum_abs_hc_all_dets_orbs: f64,
    pub sum_hc_squared_all_dets_orbs: f64,
    // pub single_excitable_dets: Vec<&'a Det>, // Dets that can have single excites sampled
    // pub single_excitable_det_sampler: Alias, // Alias sampler for selecting dets to perform single excites on
}

/// Individual sample of a det and an electron or electron pair to excite from
#[derive(Clone, Copy, Debug)]
pub struct DetOrbSample<'a> {
    pub det: &'a Det, // just the pointers to the wf's dets because we don't want to copy them; hence the lifetime parameter
    pub init: Orbs,
    pub is_alpha: Option<bool>,
    pub sum_abs_h: f64,
    pub sum_h_squared: f64,
    pub sum_abs_hc: f64,
    pub sum_hc_squared: f64,
}

impl PartialEq for DetOrbSample<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.det.config.up == other.det.config.up
            && self.det.config.dn == other.det.config.dn
            && self.init == other.init
            && self.is_alpha == other.is_alpha
    }
}
impl Eq for DetOrbSample<'_> {}

impl Hash for DetOrbSample<'_> {
    // Hash using only the config, orbs, and is_alpha
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.det.config.hash(state);
        self.init.hash(state);
        self.is_alpha.hash(state);
    }
}

/// Generate a screened sampler object for sampling (det, orbs) pairs using Alias sampling
/// The input det_orbs will contain the information needed for CDF-searching as a separate step
/// (sum_abs_hc, sum_hc_squared are the sums of remaining terms to be sampled)
pub fn generate_screened_sampler(eps: f64, det_orbs: Vec<DetOrbSample>) -> ScreenedSampler {
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

    ScreenedSampler {
        eps,
        elements: det_orbs_nonzero,
        det_orb_sampler_abs_hc: Alias::new(probs_abs_hc),
        det_orb_sampler_hc_squared: Alias::new(probs_hc_squared),
        sum_abs_hc_all_dets_orbs,
        sum_hc_squared_all_dets_orbs,
    }
}

/// Importance-sample the remaining component of a screened matmul using the given epsilon
/// Returns tuple containing (option(exciting det, excitation, and sampled determinant (with coeff attached)), and probability of that sample
/// O(log M) time
pub fn matmul_sample_remaining(
    screened_sampler: &mut ScreenedSampler,
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
            println!("Sampled det/orb {}, prob {}", det_orb_sample, det_orb_prob);

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
                        diag: 0.0, // Compute diagonal element later, only if needed (since it would be the most expensive step)
                    },
                )),
                prob_sampled_det,
            )
        }
    }
}
