extern crate rand;
use rand::prelude::*;

use crate::excite::StoredExcite;
// use std::intrinsics::offset;

pub fn sample_cdf(cdf: &Vec<StoredExcite>, max_cdf: f64) -> &StoredExcite {
    // Sample a CDF (in decreasing order) by sampling a uniform random number up to max_cdf
    // and binary searching the CDF
    // max_cdf is chosen such that CDF(elem) = max_cdf for the first elem that is a valid sample
    // Returns the sampled excite and the probability of the sample
    // O(log M)

    // TODO: Move this rng def out of this fn
    let mut rng = rand::thread_rng();
    let mut target: f64 = rng.gen();
    target *= max_cdf;

    // Binary-search for target
    let ind = cdf.partition_point(|x| x.sum_remaining_abs_h > target);

    // Return sampled excite
    &cdf[ind]
}
