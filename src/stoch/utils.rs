use rand::prelude::*;

use crate::excite::StoredExcite;
use std::intrinsics::offset;

pub fn sample_cdf_decreasing_order(cdf: Vec<StoredExcite>, r: f64) -> usize {
    // Sample a CDF (in decreasing order) by sampling a uniform random number up to r
    // and binary searching the CDF
    // Returns the sampled index

    // TODO: Move this rng out of this fn
    let mut rng = rand::thread_rng();
    let target: f64 = r * rng.gen();

    // Binary-search for target
    cdf.partition_point(|&x| x.sum_remaining_abs_h > target)
}

// struct Alias {
//     // Contains Alias sampling arrays
// }
//
// pub fn setup_alias(pdf: Vec<f64>) -> Alias {
//     // Set up the Alias method for a given pdf in O(N) time
//
//     todo!()
// }
//
// impl Alias {
//     pub fn sample_alias(&self) -> usize {
//         // Sample using the Alias sample arrays in O(1) time
//         todo!()
//     }
// }