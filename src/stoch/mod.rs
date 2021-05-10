// Module containing stochastic functions

use vose_alias::VoseAlias;
use crate::wf::det::{Config, Det};
use crate::excite::{Orbs, Excite};
use crate::excite::init::ExciteGenerator;
use crate::stoch::utils::sample_cdf;
use crate::ham::Ham;

mod utils;

struct ScreenedSampler {
    // For importance sampling the component of the matmul that is screened out by the eps threshold
    eps: f64,
    det_orb_sampler: VoseAlias<DetOrbSample>,
    sum_abs_hc_all_dets_orbs: f64
}

struct DetOrbSample {
    // Individual sample
    det: Det,
    init: Orbs,
    is_alpha: Option<bool>,
    sum_abs_hc: f64
}


pub fn matmul_sample_remaining(screened_sampler: &ScreenedSampler, excite_gen: &ExciteGenerator, ham: &Ham) -> (Option<Det>, f64) {
    // Importance-sample the remaining component of a screened matmul using the given epsilon
    // Returns sampled determinant (with coeff attached), and probability of that sample
    // O(log M) time

    // First, sample a (determinant, orbs) pair using Alias sampling
    let det_orb_sample = screened_sampler.det_orb_sampler.sample();

    // Sample excitation from this det/orb pair by binary search the stored cdf
    match is_alpha {
        None => {
            // Opposite spin double
            let sampled_excite = sample_cdf(excite_gen.opp_doub_generator.get(&sampled_orbs).unwrap(), det_orb_sample.sum_abs_hc);
        },
        Some(alpha) => {
            match sampled_orbs {
                Orbs::Double(pq) => {
                    // Same spin double
                    let sampled_excite = sample_cdf(excite_gen.same_doub_generator.get(&sampled_orbs).unwrap(), det_orb_sample.sum_abs_hc);
                },
                Orbs::Single(p) => {
                    // Single
                    let sampled_excite = sample_cdf(excite_gen.sing_generator.get(&sampled_orbs).unwrap(), det_orb_sample.sum_abs_hc);
                }
            }
        }
    }

    // Apply the excitation to the sampled det
    let sampled_det = sampled_exciting_det.safe_excite_det(
        Excite{
            init:det_orb_sample.init,
            target: sampled_excite.target,
            abs_h: sampled_excite.abs_h,
            is_alpha: det_orb_sample.is_alpha
        }
    );

    // Compute total probability
    let prob_sampled_det = sampled_excite.abs_h * det_orb_sample.det.coeff / screened_sampler.sum_abs_hc_all_dets_orbs;

    match sampled_det {
        None => (None, prob_sampled_det), // Proposed excitation would excite to already-occupied orbs
        Some(d) => {
            // Compute new det coefficient
            let mut new_det_coeff = det_orb_sample.det.coeff;
            match det_orb_sample.init {
                Orbs::Double(_) => new_det_coeff *= ham.ham_doub(&det_orb_sample.det.config, d),
                Orbs::Single(_) => new_det_coeff *= ham.ham_sing(&det_orb_sample.det.config, d),
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
