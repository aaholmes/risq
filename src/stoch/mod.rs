// Module containing stochastic functions

use vose_alias::VoseAlias;
use crate::wf::det::{Det, Config};
use crate::excite::{Orbs, Excite, StoredExcite};
use crate::excite::init::ExciteGenerator;
use crate::stoch::utils::{sample_cdf, test_cdf};
use crate::ham::Ham;
use std::hash::{Hash, Hasher};

pub(crate) mod utils;

#[derive(Clone, Copy)]
pub enum ImpSampleDist {
    // Either importance sample proportional to |Hc| or to (Hc)^2
    AbsHc,
    HcSquared,
}

pub struct ScreenedSampler<'a> {
    // For importance sampling the component of the matmul that is screened out by the eps threshold
    // Matmul_sample_remaining performs the whole excitation sampling (exciting pair and target), but
    // this contains only data structures sampling the exciting det and its exciting electron pair;
    // sampling the target electron pair uses CDF searching which only requires ExciteGenerator
    // Lifetime 'a must last as long as the vector being semistochastically multiplied, since this
    // struct has pointers to its components
    // Contains two samplers, for sampling with p ~ |Hc| and with p ~ (Hc)^2
    pub eps: f64,
    pub det_orb_sampler_abs_hc: VoseAlias<DetOrbSample<'a>>,
    pub det_orb_sampler_hc_squared: VoseAlias<DetOrbSample<'a>>,
    pub sum_abs_hc_all_dets_orbs: f64,
    pub sum_hc_squared_all_dets_orbs: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DetOrbSample<'a> {
    // Individual sample of a det and an electron or electron pair to excite from
    pub det: &'a Det,
    pub init: Orbs,
    pub is_alpha: Option<bool>,
    pub sum_abs_h: f64,
    pub sum_h_squared: f64,
    pub sum_abs_hc: f64,
    pub sum_hc_squared: f64
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

    // Normalize probs (both |Hc| and (Hc)^2)
    let mut det_orbs_nonzero: Vec<DetOrbSample> = vec![];
    let mut probs_abs_hc: Vec<f64> = vec![];
    let mut probs_hc_squared: Vec<f64> = vec![];

    let sum_hc_all_dets_orbs: f64 = det_orbs.iter().fold(0f64, |sum, &val| sum + val.sum_abs_hc as f64);
    let sum_hc_squared_all_dets_orbs: f64 = det_orbs.iter().fold(0f64, |sum, &val| sum + val.sum_hc_squared as f64);

    for val in det_orbs.iter() {
        // println!("Det_orb: {}, prob: {}", val, val.sum_abs_hc / sum_hc_all_dets_orbs);
        if val.sum_abs_h != 0.0 {
            det_orbs_nonzero.push(*val);
            probs_abs_hc.push(val.sum_abs_hc / sum_hc_all_dets_orbs);
            probs_hc_squared.push(val.sum_hc_squared / sum_hc_squared_all_dets_orbs);
        }
    }

    // println!("Total number of stored det orbs: {}", det_orbs_nonzero.len());
    // println!("Normalization abs: {}", probs_abs_hc.iter().sum::<f64>());
    // println!("Normalization sq: {}", probs_hc_squared.iter().sum::<f64>());

    ScreenedSampler{
        eps: eps,
        det_orb_sampler_abs_hc: VoseAlias::new(det_orbs_nonzero.clone(), probs_abs_hc), // TODO: Remove clone here?
        det_orb_sampler_hc_squared: VoseAlias::new(det_orbs_nonzero, probs_hc_squared),
        sum_abs_hc_all_dets_orbs: sum_hc_all_dets_orbs,
        sum_hc_squared_all_dets_orbs: sum_hc_squared_all_dets_orbs,
    }
}


pub fn matmul_sample_remaining(screened_sampler: &ScreenedSampler, imp_sample_dist: ImpSampleDist, excite_gen: &ExciteGenerator, ham: &Ham) -> (Option<(Det, Excite, Det)>, f64) {
    // Importance-sample the remaining component of a screened matmul using the given epsilon
    // Returns tuple containing (option(exciting det, excitation, and sampled determinant (with coeff attached)), and probability of that sample
    // O(log M) time

    let mut det_orb_sample: DetOrbSample = DetOrbSample {
        det: &Det {
            config: Config { up: 0, dn: 0 },
            coeff: 0.0,
            diag: 0.0
        },
        init: Orbs::Single(0),
        is_alpha: None,
        sum_abs_h: 0.0,
        sum_h_squared: 0.0,
        sum_abs_hc: 0.0,
        sum_hc_squared: 0.0
    };
    let mut det_orb_prob: f64 = 0.0;
    let mut sampled_excite: &StoredExcite = &StoredExcite {
        target: Orbs::Single(0),
        abs_h: 0.0,
        sum_remaining_abs_h: 0.0,
        sum_remaining_h_squared: 0.0
    };
    let mut sampled_excite_prob: f64 = 0.0;

    match imp_sample_dist {
        ImpSampleDist::AbsHc => {
            // First, sample a (determinant, orbs) pair using Alias sampling with prob |Hc|
            // (det_orb_sample, det_orb_prob) = screened_sampler.det_orb_sampler_abs_hc.sample_with_prob();
            let sample = screened_sampler.det_orb_sampler_abs_hc.sample_with_prob();
            det_orb_sample = sample.0;
            det_orb_prob = sample.1;
            // println!("Det: {}", det_orb_sample.det);
            // match det_orb_sample.init {
            //     Orbs::Double((p, q)) => println!("Orbs: {}, {}", p, q),
            //     Orbs::Single(p) => println!("Orb: {}", p)
            // }
            // println!("Det orb prob: {}, CDF target: {}", det_orb_prob, det_orb_sample.sum_abs_h);

            // Sample excitation from this det/orb pair by binary search the stored cdf with prob |H|
            match det_orb_sample.is_alpha {
                None => {
                    // Opposite spin double
                    let sample2 = sample_cdf(excite_gen.opp_doub_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::AbsHc, Some(det_orb_sample.sum_abs_h)).unwrap();
                    sampled_excite = sample2.0;
                    sampled_excite_prob = sample2.1;
                },
                Some(_) => {
                    match det_orb_sample.init {
                        Orbs::Double(_) => {
                            // Same spin double
                            let sample2 = sample_cdf(excite_gen.same_doub_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::AbsHc, Some(det_orb_sample.sum_abs_h)).unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        },
                        Orbs::Single(_) => {
                            // Single
                            let sample2 = sample_cdf(excite_gen.sing_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::AbsHc, Some(det_orb_sample.sum_abs_h)).unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                    }
                }
            }
        }
        ImpSampleDist::HcSquared => {
            // First, sample a (determinant, orbs) pair using Alias sampling with prob (Hc)^2
            // (det_orb_sample, det_orb_prob) = screened_sampler.det_orb_sampler_hc_squared.sample_with_prob();
            let sample = screened_sampler.det_orb_sampler_hc_squared.sample_with_prob();
            det_orb_sample = sample.0;
            det_orb_prob = sample.1;
            // println!("Det: {}", det_orb_sample.det);
            // match det_orb_sample.init {
            //     Orbs::Double((p, q)) => println!("Orbs: {}, {}", p, q),
            //     Orbs::Single(p) => println!("Orb: {}", p)
            // }
            // println!("Det orb prob: {}, CDF target: {}", det_orb_prob, det_orb_sample.sum_h_squared);

            // Sample excitation from this det/orb pair by binary search the stored cdf with prob H^2
            match det_orb_sample.is_alpha {
                None => {
                    // Opposite spin double
                    // test_cdf(&excite_gen.opp_doub_generator.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::HcSquared, 1.0, 1000000);
                    // test_cdf(&excite_gen.opp_doub_generator.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::HcSquared, det_orb_sample.sum_h_squared, 10000);
                    // panic!("Done testing CDF");
                    let sample2 = sample_cdf(excite_gen.opp_doub_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::HcSquared, Some(det_orb_sample.sum_h_squared)).unwrap();
                    sampled_excite = sample2.0;
                    sampled_excite_prob = sample2.1;
                },
                Some(_) => {
                    match det_orb_sample.init {
                        Orbs::Double(_) => {
                            // Same spin double
                            let sample2 = sample_cdf(excite_gen.same_doub_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::HcSquared, Some(det_orb_sample.sum_h_squared)).unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        },
                        Orbs::Single(_) => {
                            // Single
                            let sample2 = sample_cdf(excite_gen.sing_sorted_list.get(&det_orb_sample.init).unwrap(), &ImpSampleDist::HcSquared, Some(det_orb_sample.sum_h_squared)).unwrap();
                            sampled_excite = sample2.0;
                            sampled_excite_prob = sample2.1;
                        }
                    }
                }
            }
        }
    }

    // println!("Sampled excite prob: {}", sampled_excite_prob);

    // Construct the excitation (for output)
    let excite = Excite {
        init: det_orb_sample.init,
        target: sampled_excite.target,
        abs_h: sampled_excite.abs_h,
        is_alpha: det_orb_sample.is_alpha
    };

    // Apply the excitation to the sampled det
    let sampled_det = det_orb_sample.det.config.safe_excite_det(&excite);

    // Compute total probability
    let prob_sampled_det: f64 = det_orb_prob * sampled_excite_prob;
    // match imp_sample_dist {
    //     ImpSampleDist::AbsHc => {
    //         println!("Prob det_orbs: {}, prob excite: {}, product: {}, |H|: {}, |c|: {}, sum|Hc|: {}, overall prob: {}",
    //         det_orb_prob, sampled_excite_prob, det_orb_prob * sampled_excite_prob, sampled_excite.abs_h, det_orb_sample.det.coeff.abs(),
    //         screened_sampler.sum_abs_hc_all_dets_orbs, sampled_excite.abs_h * det_orb_sample.det.coeff.abs() / screened_sampler.sum_abs_hc_all_dets_orbs);
    //         prob_sampled_det = det_orb_prob * sampled_excite_prob;
    //         // prob_sampled_det = sampled_excite.abs_h * det_orb_sample.det.coeff.abs() / screened_sampler.sum_abs_hc_all_dets_orbs;
    //     }
    //     ImpSampleDist::HcSquared => {
    //         prob_sampled_det = det_orb_prob * sampled_excite_prob;
    //         // prob_sampled_det = sampled_excite.abs_h * sampled_excite.abs_h * det_orb_sample.det.coeff * det_orb_sample.det.coeff / screened_sampler.sum_hc_squared_all_dets_orbs;
    //         println!("Sampled abs_h = {}, sampled coeff = {}", sampled_excite.abs_h, det_orb_sample.det.coeff);
    //         // println!("Probability of this det_orb pair: {}, Probability of this target pair: {}, total probability of this excitation: {}", );
    //     }
    // }

    match sampled_det {
        None => (None, prob_sampled_det), // Proposed excitation would excite to already-occupied orbs
        Some(d) => {
            // Compute new det coefficient
            let mut new_det_coeff = det_orb_sample.det.coeff;
            match det_orb_sample.init {
                Orbs::Double(_) => new_det_coeff *= ham.ham_doub(&det_orb_sample.det.config, &d),
                Orbs::Single(_) => new_det_coeff *= ham.ham_sing(&det_orb_sample.det.config, &d),
            }
            // println!("Sampled excite with prob = {}", prob_sampled_det);
            (
                Some(
                    (
                        *det_orb_sample.det,
                        excite,
                        Det {
                            config: d,
                            coeff: new_det_coeff,
                            diag: 0.0 // Compute diagonal element later, only if needed (since it would be the most expensive step)
                        }
                    )
                ), prob_sampled_det
            )
        }
    }
}
