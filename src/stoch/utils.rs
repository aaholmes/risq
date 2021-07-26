extern crate rand;
use rand::prelude::*;

use crate::excite::StoredExcite;
use crate::stoch::ImpSampleDist;
use std::collections::HashMap;
// use std::intrinsics::offset;

pub fn sample_cdf<'a>(cdf: &'a Vec<StoredExcite>, imp_sample_dist: &ImpSampleDist, max_cdf: Option<f64>) -> Option<(&'a StoredExcite, f64)> {
    // Sample a CDF (in decreasing order) by sampling a uniform random number up to max_cdf
    // and binary searching the CDF
    // cdf is a vector of StoredExcites, and we access their stored cumulative sums depending on imp_sample_dist
    // max_cdf is chosen such that CDF(elem) = max_cdf.unwrap() for the first elem that is a valid sample
    // (max_cdf = None samples the whole distribution)
    // Returns the sampled excite and the probability of the sample
    // O(log M)

    let n = cdf.len();
    // println!("CDF has size: {}", n);

    if n == 0 {
        panic!("Attempted to sample CDF with zero elements!");
    } else if n == 1 {
        return Some((&cdf[0], 1.0));
    }

    let mut max: f64 = 0.0;
    match max_cdf {
        None => {
            match imp_sample_dist {
                ImpSampleDist::AbsHc => { max = cdf[0].sum_remaining_abs_h; }
                ImpSampleDist::HcSquared => { max = cdf[0].sum_remaining_h_squared; }
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

    // TODO: Move this rng def out of this fn
    let mut rng = rand::thread_rng();
    let mut target: f64 = rng.gen();
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
    let mut ind: usize = 0;
    let mut sample_prob: f64 = 0.0;
    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            // println!("target: {}", target);
            ind = cdf.partition_point(|x| x.sum_remaining_abs_h > target) - 1;
            // println!("Computing abs_h sample_prob, {}, {}", cdf[ind].sum_remaining_abs_h, cdf[ind + 1].sum_remaining_abs_h);
            sample_prob = (cdf[ind].sum_remaining_abs_h - cdf[ind + 1].sum_remaining_abs_h) / max;
            // println!("Selected element {} with probability {}", cdf[ind].sum_remaining_abs_h, sample_prob);
        },
        ImpSampleDist::HcSquared => {
            ind = cdf.partition_point(|x| x.sum_remaining_h_squared > target) - 1;
            // println!("Computing h_sq sample_prob, {}, {}", cdf[ind].sum_remaining_h_squared, cdf[ind + 1].sum_remaining_h_squared);
            sample_prob = (cdf[ind].sum_remaining_h_squared - cdf[ind + 1].sum_remaining_h_squared) / max;
            // println!("Selected element {} with probability {}", cdf[ind].sum_remaining_h_squared, sample_prob);
        }
    }

    // Return sampled excite
    Some((&cdf[ind], sample_prob))
}


pub fn test_cdf(cdf: &Vec<StoredExcite>, imp_sample_dist: &ImpSampleDist, max_cdf: Option<f64>, n_samples: i32) {
    // Test the sample_cdf routine by taking n_samples samples and showing frequency of samples vs probability for full distribution
    println!("Calling test_cdf");
    let n: usize = cdf.len();
    let mut expected: Vec<f64> = vec![];
    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            for i in 1..n {
                expected.push(cdf[i - 1].sum_remaining_abs_h - cdf[i].sum_remaining_abs_h);
            }
            expected.push(cdf[n - 1].sum_remaining_abs_h);
        },
        ImpSampleDist::HcSquared => {
            for i in 1..n {
                expected.push(cdf[i - 1].sum_remaining_h_squared - cdf[i].sum_remaining_h_squared);
            }
            expected.push(cdf[n - 1].sum_remaining_h_squared);
        }
    }
    let mut freq: HashMap<StoredExcite, i32> = HashMap::default();
    for i in cdf {
        freq.insert(*i, 0);
    }
    for _i_sample in 0..n_samples {
        let sample = sample_cdf(cdf, imp_sample_dist, max_cdf).unwrap();
        *freq.get_mut(&sample.0).unwrap() += 1;
    }
    println!("Target prob, sampled prob");
    for (i, exc) in cdf.iter().enumerate() {
        println!("{:.6},   {:.6},   {:.6}", expected[i], (freq[exc] as f64) / (n_samples as f64), (freq[exc] as f64) / (n_samples as f64) / expected[i]);
    }
}
