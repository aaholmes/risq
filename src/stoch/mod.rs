// Module containing stochastic functions

use vose_alias::VoseAlias;
use crate::wf::det::Det;
use crate::excite::{Orbs, Excite, StoredExcite};
use crate::excite::init::ExciteGenerator;
use crate::stoch::utils::sample_cdf;
use crate::ham::Ham;
use std::hash::{Hash, Hasher};

mod utils;

pub struct ScreenedSampler<'a> {
    // For importance sampling the component of the matmul that is screened out by the eps threshold
    // Lifetime 'a must last as long as the vector being semistochastically multiplied, since this
    // struct has pointers to its components
    pub eps: f64,
    pub det_orb_sampler: VoseAlias<DetOrbSample<'a>>,
    pub sum_abs_hc_all_dets_orbs: f64
}

#[derive(Clone, Copy, Debug)]
pub struct DetOrbSample<'a> {
    // Individual sample of a det and an electron or electron pair to excite from
    pub det: &'a Det,
    pub init: Orbs,
    pub is_alpha: Option<bool>,
    pub sum_abs_hc: f64
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

pub fn generate_screened_sampler(eps: f64, det_orbs: Vec<DetOrbSample>) -> ScreenedSampler {
    // Generate a screened sampler object (using Alias)

    // Normalize probs
    let sum_hc_all_dets_orbs: f64 = det_orbs.iter().fold(0f64, |sum, &val| sum + val.sum_abs_hc);
    let mut probs: Vec<f32> = vec![];
    for prob in det_orbs.iter() {
        probs.push((prob.sum_abs_hc / sum_hc_all_dets_orbs) as f32);
    }

    ScreenedSampler{
        eps: eps,
        det_orb_sampler: VoseAlias::new(det_orbs, probs),
        sum_abs_hc_all_dets_orbs: sum_hc_all_dets_orbs
    }
}

pub fn matmul_sample_remaining(screened_sampler: &ScreenedSampler, excite_gen: &ExciteGenerator, ham: &Ham) -> (Option<Det>, f64) {
    // Importance-sample the remaining component of a screened matmul using the given epsilon
    // Returns sampled determinant (with coeff attached), and probability of that sample
    // O(log M) time

    // First, sample a (determinant, orbs) pair using Alias sampling
    let det_orb_sample = screened_sampler.det_orb_sampler.sample();

    // Sample excitation from this det/orb pair by binary search the stored cdf
    let sampled_excite: &StoredExcite;
    match det_orb_sample.is_alpha {
        None => {
            // Opposite spin double
            sampled_excite = sample_cdf(excite_gen.opp_doub_generator.get(&det_orb_sample.init).unwrap(), det_orb_sample.sum_abs_hc);
        },
        Some(_) => {
            match det_orb_sample.init {
                Orbs::Double(_) => {
                    // Same spin double
                    sampled_excite = sample_cdf(excite_gen.same_doub_generator.get(&det_orb_sample.init).unwrap(), det_orb_sample.sum_abs_hc);
                },
                Orbs::Single(_) => {
                    // Single
                    sampled_excite = sample_cdf(excite_gen.sing_generator.get(&det_orb_sample.init).unwrap(), det_orb_sample.sum_abs_hc);
                }
            }
        }
    }

    // Apply the excitation to the sampled det
    let sampled_det = det_orb_sample.det.config.safe_excite_det(
        &Excite {
            init: det_orb_sample.init,
            target: sampled_excite.target,
            abs_h: sampled_excite.abs_h,
            is_alpha: det_orb_sample.is_alpha
        }
    );

    // Compute total probability
    let prob_sampled_det = sampled_excite.abs_h * det_orb_sample.det.coeff.abs() / screened_sampler.sum_abs_hc_all_dets_orbs;

    match sampled_det {
        None => (None, prob_sampled_det), // Proposed excitation would excite to already-occupied orbs
        Some(d) => {
            // Compute new det coefficient
            let mut new_det_coeff = det_orb_sample.det.coeff;
            match det_orb_sample.init {
                Orbs::Double(_) => new_det_coeff *= ham.ham_doub(&det_orb_sample.det.config, &d),
                Orbs::Single(_) => new_det_coeff *= ham.ham_sing(&det_orb_sample.det.config, &d),
            }
            (Some(
                Det {
                    config: d,
                    coeff: new_det_coeff,
                    diag: 0.0 // Compute diagonal element later, only if needed (since it would be the most expensive step)
                }
            ), prob_sampled_det)
        }
    }
}
