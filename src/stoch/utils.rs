extern crate rand;
use rand::prelude::*;

use crate::excite::StoredExcite;
use crate::stoch::ImpSampleDist;
// use std::intrinsics::offset;

pub fn sample_cdf(cdf: &Vec<StoredExcite>, imp_sample_dist: ImpSampleDist, max_cdf: f64) -> (&StoredExcite, f64) {
    // Sample a CDF (in decreasing order) by sampling a uniform random number up to max_cdf
    // and binary searching the CDF
    // max_cdf is chosen such that CDF(elem) = max_cdf for the first elem that is a valid sample
    // Returns the sampled excite and the probability of the sample
    // O(log M)

    // TODO: Move this rng def out of this fn
    let mut rng = rand::thread_rng();
    let mut target: f64 = rng.gen();
    target *= max_cdf;
    println!("Sampling excitation with max_cdf {} from the CDF: ", max_cdf);
    for c in cdf.iter() {
        match imp_sample_dist {
            ImpSampleDist::AbsHc => {
                if c.sum_remaining_abs_h == 0.0 {break;}
                println! ("{}", c.sum_remaining_abs_h);
            },
            ImpSampleDist::HcSquared => {
                if c.sum_remaining_h_squared == 0.0 {break;}
                println! ("{}", c.sum_remaining_h_squared);
            },
        }
    }
    println!("Target (sampled value) = {}", target);

    // Binary-search for target
    let mut ind: usize = 0;
    let mut sample_prob: f64 = 0.0;
    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            ind = cdf.partition_point(|x| x.sum_remaining_abs_h > target) - 1;
            sample_prob = (cdf[ind].sum_remaining_abs_h - cdf[ind + 1].sum_remaining_abs_h) / max_cdf;
            println!("Selected element {} with probability {}", cdf[ind].sum_remaining_abs_h, sample_prob);
        },
        ImpSampleDist::HcSquared => {
            ind = cdf.partition_point(|x| x.sum_remaining_h_squared > target) - 1;
            sample_prob = (cdf[ind].sum_remaining_h_squared - cdf[ind + 1].sum_remaining_h_squared) / max_cdf;
            println!("Selected element {} with probability {}", cdf[ind].sum_remaining_h_squared, sample_prob);
        }
    }

    // Return sampled excite
    (&cdf[ind], sample_prob)
}
