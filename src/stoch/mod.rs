// Module containing stochastic functions

use vose_alias::VoseAlias;
use crate::wf::det::{Config, Det};
use crate::excite::Orbs;
use crate::excite::init::ExciteGenerator;
use crate::stoch::utils::sample_cdf;

mod utils;


pub fn matmul_sample_remaining(alias: VoseAlias<(Config, Orbs, Option<bool>, f64)>, excite_gen: &ExciteGenerator) -> Option<(Det, f64)> {
    // Importance-sample the remaining component of a screened matmul using the given epsilon
    // Returns sampled determinant (with coeff attached), and probability of that sample
    // O(log M) time

    // First, sample a (determinant, epair) pair using Alias sampling
    let (sampled_exciting_det, sampled_orbs, is_alpha, prob_det_orbs) = alias.sample();

    // Then, sample excitations for this epair by binary search the stored cdf
    match is_alpha {
        None => {
            // Opposite spin double
            let sampled_excite = sample_cdf(excite_gen.opp_doub_generator.get(&sampled_orbs).unwrap(), r);
            let prob_excite = sampled_excite.abs_h / r;
            let sampled_det = sampled_exciting_det.safe_excite_det(sampled_excite);
            match sampled_det {
                None => None,
                Some(d) => Some((Det{config: d, coeff: ??, diag: 0.0}, prob_excite))
            }
        },
        Some(alpha) => {
            match sampled_orbs {
                Orbs::Double(pq) => {
                    // Same spin double
                },
                Orbs::Single(p) => {
                    // Single
                }
            }
        }
    }

    todo!()
}