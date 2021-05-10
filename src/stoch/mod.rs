// Module containing stochastic functions

use vose_alias::VoseAlias;
use crate::wf::det::{Config, Det};
use crate::excite::Orbs;
use crate::excite::init::ExciteGenerator;

mod utils;


pub fn matmul_sample_remaining(alias: VoseAlias<(Config, Orbs, Option<bool>, f64)>, excite_gen: &ExciteGenerator) -> (Det, f64) {
    // Importance-sample the remaining component of a screened matmul using the given epsilon
    // Returns sampled determinant (with coeff attached), and probability of that sample
    // O(log M) time

    // First, sample a (determinant, epair) pair using Alias sampling
    (sampled_exciting_det, sampled_orbs, is_alpha, prob_det_orbs) = alias.sample();

    // Then, sample excitations for this epair by binary search the stored cdf
    match is_alpha {
        None => {
            // Opposite spin double
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