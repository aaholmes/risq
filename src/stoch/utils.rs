//! # Miscellaneous Stochastic Utilities (`stoch::utils`)
//!
//! This module contains helper functions for stochastic sampling procedures.

extern crate rand;

use rand::prelude::*;

use crate::excite::StoredExcite;
use crate::rng::Rand;
use crate::stoch::ImpSampleDist;
// use std::intrinsics::offset;

/// Samples an element from a discrete distribution represented by a pre-sorted list
/// using the inverse transform sampling method via binary search on the cumulative sums.
///
/// This function assumes `cdf` is a slice of `StoredExcite` sorted in descending order
/// by `abs_h`. It uses the pre-computed `sum_remaining_*` fields, which act as a
/// cumulative distribution function (CDF) but in reverse order (sum of probabilities
/// *after* the current element).
///
/// # Arguments
/// * `cdf`: A slice of `StoredExcite`, sorted descending by `abs_h`. Contains the
///   cumulative sums needed for sampling.
/// * `imp_sample_dist`: Specifies whether to use `sum_remaining_abs_h` (for sampling ~|H|)
///   or `sum_remaining_h_squared` (for sampling ~H^2).
/// * `max_cdf`: An optional upper bound for the sampling range. If `Some(m)`, samples
///   uniformly from `[0, m)`. If `None`, samples uniformly from `[0, total_sum)`, where
///   `total_sum` is the sum corresponding to the first element (representing the sum over
///   the entire distribution).
/// * `rand`: Mutable random number generator state.
///
/// # Returns
/// `Some((sampled_excite, sample_probability))` if successful, where `sampled_excite`
/// is a reference to the chosen `StoredExcite` and `sample_probability` is the probability
/// mass associated with that specific element according to the chosen distribution and range.
/// Returns `None` if the total sum of the distribution is zero.
///
/// # Complexity
/// O(log M), where M is the length of the `cdf` slice, due to the binary search (`partition_point`).
///
/// # Panics
/// Panics if `cdf` is empty or if `max_cdf` is `Some(0.0)`.
pub fn sample_cdf<'a>(
    cdf: &'a [StoredExcite],            // Pre-sorted slice with cumulative sums
    imp_sample_dist: &ImpSampleDist,    // Distribution type (|H| or H^2)
    max_cdf: Option<f64>,               // Optional upper bound for sampling range
    rand: &mut Rand,                    // RNG state
) -> Option<(&'a StoredExcite, f64)> { // Returns (Sampled element, Its probability) or None
    let n = cdf.len();
    // println!("CDF has size: {}", n);

    if n == 0 {
        panic!("Attempted to sample CDF with zero elements!");
    } else if n == 1 {
        return Some((&cdf[0], 1.0));
    }

    let max: f64;
    match max_cdf {
        None => match imp_sample_dist {
            ImpSampleDist::AbsHc => {
                max = cdf[0].sum_remaining_abs_h;
            }
            ImpSampleDist::HcSquared => {
                max = cdf[0].sum_remaining_h_squared;
            }
        },
        Some(m) => {
            if m == 0.0 {
                panic!("Attempted to search for zero in a CDF!");
            }
            max = m;
        }
    }

    if max == 0.0 {
        // CDF is all zeros (e.g., no single excitations from this orb)
        return None;
    }

    let mut target: f64 = rand.rng.gen();
    // println!("rng, max: {}, {}", target, max);
    target *= max;

    // println!("Sampling excitation with max_cdf {} from the CDF: ", max_cdf);
    // for c in cdf.iter() {
    //     match imp_sample_dist {
    //         ImpSampleDist::AbsHc => {
    //             if c.sum_remaining_abs_h == 0.0 {break;}
    //             println! ("{}", c.sum_remaining_abs_h);
    //         },
    //         ImpSampleDist::HcSquared => {
    //             if c.sum_remaining_h_squared == 0.0 {break;}
    //             println! ("{}", c.sum_remaining_h_squared);
    //         },
    //     }
    // }
    // println!("Target (sampled value) = {}", target);

    // Binary-search for target
    let ind: usize;
    let sample_prob: f64;
    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            // println!("target: {}", target);
            ind = cdf.partition_point(|x| x.sum_remaining_abs_h > target) - 1;
            // println!("Computing abs_h sample_prob, {}, {}", cdf[ind].sum_remaining_abs_h, cdf[ind + 1].sum_remaining_abs_h);
            sample_prob = (cdf[ind].sum_remaining_abs_h - cdf[ind + 1].sum_remaining_abs_h) / max;
            // println!("Selected element {} with probability {}", cdf[ind].sum_remaining_abs_h, sample_prob);
        }
        ImpSampleDist::HcSquared => {
            ind = cdf.partition_point(|x| x.sum_remaining_h_squared > target) - 1;
            // println!("Computing h_sq sample_prob, {}, {}", cdf[ind].sum_remaining_h_squared, cdf[ind + 1].sum_remaining_h_squared);
            sample_prob =
                (cdf[ind].sum_remaining_h_squared - cdf[ind + 1].sum_remaining_h_squared) / max;
            // println!("Selected element {} with probability {}", cdf[ind].sum_remaining_h_squared, sample_prob);
        }
    }

    // Return sampled excite
    Some((&cdf[ind], sample_prob))
}

// Removed commented-out test function `test_cdf`
