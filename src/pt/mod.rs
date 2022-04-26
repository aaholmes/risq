//! Epstein-Nesbet perturbation theory

use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs};
use crate::ham::Ham;
use crate::rng::{init_rand, Rand};
use crate::semistoch::{new_semistoch_enpt2, new_semistoch_enpt2_no_diag_singles, old_semistoch_enpt2};
use crate::utils::read_input::Global;
use crate::wf::det::{Config, Det};
use crate::wf::Wf;
use itertools::enumerate;
use std::collections::HashMap;
use crate::excite::iterator::dets_excites_and_excited_dets;

/// Perform the perturbative stage (Epstein-Nesbet perturbation theory, that is)
pub fn perturbative(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, wf: &Wf) {
    // Initialize random number genrator
    let mut rand: Rand = init_rand();

    let mut e_pt2: f64;
    let std_dev: f64;
    e_pt2 = dtm_pt(wf, excite_gen, ham, global.eps_pt_dtm);
    println!("Variational energy: {}, Deterministic PT: {}, Total energy: {}", wf.energy, e_pt2, wf.energy + e_pt2);
    panic!("DEBUG");
    if global.n_cross_term_samples == 0 {
        // Old SHCI (2017 paper)
        println!("\nCalling semistoch ENPT2 the old way with p ~ |c|");
        let out = old_semistoch_enpt2(wf, global, ham, excite_gen, false, &mut rand);
        e_pt2 = out.0;
        std_dev = out.1;
    } else {
        println!("\nCalling semistoch ENPT2 the new way with importance sampling");
        let out = new_semistoch_enpt2_no_diag_singles(wf, global, ham, excite_gen, &mut rand);
        // let out = new_semistoch_enpt2(wf, global, ham, excite_gen, &mut rand);
        // let out = importance_sampled_semistoch_enpt2(wf, global, ham, excite_gen, &mut rand);
        // let out = fast_stoch_enpt2(wf, global, ham, excite_gen, &mut rand);
        // let out = faster_semistoch_enpt2(wf, global, ham, excite_gen);
        e_pt2 = out.0;
        std_dev = out.1;
    }
    println!("Variational energy: {:.6}", wf.energy);
    println!("PT energy: {:.6} +- {:.6}", e_pt2, std_dev);
    println!("Total energy: {:.6} +- {:.6}", wf.energy + e_pt2, std_dev);
}

/// Deterministic PT
pub fn dtm_pt(wf: &Wf, excite_gen: &ExciteGenerator, ham: &Ham, eps: f64) -> f64 {
    println!("Start of deterministic PT");
    let mut h_psi: Wf = Wf::default();
    let mut n: i16 = 0;
    let mut old_var_det: Config = wf.dets[0].config;
    for (var_det, excite, pt_config) in dets_excites_and_excited_dets(wf, excite_gen, eps) {
        if var_det.config != old_var_det {
            n += 1;
            old_var_det = var_det.config;
            if n % 1000 == 0 {
                println!("Var det {} of {}", n, wf.n);
            }
        }
        // println!("Var det: {}, PT det: {}", var_det, pt_config);

        // Compute off-diagonal element times var_det.coeff
        let h_ai_c_i: f64 = ham.ham_off_diag(&var_det.config, &pt_config, &excite) * var_det.coeff;

        // For single excitaitons: Check whether this excite actually exceeds eps
        if let Orbs::Single(_) = excite.init {
            if h_ai_c_i.abs() < eps {
                continue;
            }
        }

        // Compute diagonal element in O(N) time only if necessary
        if let Some(ind) = h_psi.inds.get_mut(&pt_config) {
            h_psi.dets[*ind].coeff += h_ai_c_i;
        } else {
            h_psi.dets.push(Det{
                config: pt_config,
                coeff: h_ai_c_i,
                diag: Some(var_det.new_diag(ham, &excite))
            });
        }
    }
    println!("Preparing to calculate PT energy from generated PT dets");
    pt(&h_psi, wf.energy)
}

pub fn dtm_pt_basic(wf: &Wf, ham: &Ham, eps: f64) -> f64
{
    let mut h_psi: Wf = Wf::default();
    for det in &wf.dets {

    }
    todo!()
}

/// Deterministic PT in batches (on average one excite per variational det per batch)
// pub fn dtm_pt_batches(wf: &Wf, excite_gen: &ExciteGenerator, ham: &Ham, e0: f64, eps: f64) -> f64 {
//     let mut e_pt: f64 = 0.0f64;
//     let mut h_psi: Wf;
//     for batch in 0..n_batches {
//         for (var_det, excite, pt_config) in dets_excites_and_excited_dets_batched(wf, excite_gen, eps, batch) {
//
//             // Compute off-diagonal element times var_det.coeff
//             let h_ai_c_i: f64 = ham.ham_off_diag(&var_det.config, &pt_config, &excite) * var_det.coeff;
//
//             // For single excitaitons: Check whether this excite actually exceeds eps
//             if let Orbs::Single(_) = excite.init {
//                 if h_ai_c_i.abs() < eps {
//                     continue;
//                 }
//             }
//
//             // Compute diagonal element in O(N) time only if necessary
//             if let Some(ind) = h_psi.inds.get_mut(&pt_config) {
//                 h_psi.dets[&ind].coeff += h_ai_c_i;
//             } else {
//                 h_psi.dets.push(Det {
//                     config: pt_config,
//                     coeff: h_ai_c_i,
//                     diag: Some(var_det.new_diag(ham, &excite))
//                 });
//             }
//         }
//         e_pt += pt(&h_psi, e0);
//     }
//     e_pt
// }


/// Evaluate the PT expression given H * Psi and E0
pub fn pt(h_psi: &Wf, e0: f64) -> f64 {
    h_psi.dets.iter().fold(0.0f64, |e_pt, det| e_pt + det.coeff * det.coeff / (e0 - det.diag.unwrap()))
}

/// Sampled contributions to the ENPT2 correction
/// Samples stored in a hashmap:
/// Key: perturbative det (Config)
/// Value: (perturbative det's diag, HashMap (key: variational det's config, value: (H_ai c_i, p_ai, w_ai)))
#[derive(Default)]
pub struct PtSamples {
    pub n: i32,
    pub samples: HashMap<Config, (f64, HashMap<Config, (f64, f64, i32)>)>,
}

impl PtSamples {
    // pub fn print(&self) {
    //     for (pt_key, pt_val) in self.samples.iter() {
    //         println!("Perturbative det: {} with diag elem: {}", pt_key, pt_val.0);
    //         for (var_key, var_val) in pt_val.1.iter() {
    //             println!(
    //                 "   Variational det: {}, H_ai c_i: {}, p_ai: {}, w_ai: {}",
    //                 var_key, var_val.0, var_val.1, var_val.2
    //             )
    //         }
    //     }
    // }

    pub fn clear(&mut self) {
        // Clear data structure to start collecting a new batch of samples
        self.n = 0;
        self.samples = Default::default();
    }

    pub fn add_sample_compute_diag(
        &mut self,
        var_det: Det,
        excite: &Excite,
        pt_det: Det,
        sampled_prob: f64,
        ham: &Ham,
    ) {
        // Add a new sample to PtSamples
        // Compute diagonal element of perturbative determinant if it hasn't already been computed
        self.n += 1;
        // match excite.init {
        //     Orbs::Double(_) => {
        //         println!("Doubles in add_sample_compute_diag: (Hc)^2 / p = {}", pt_det.coeff * pt_det.coeff / sampled_prob);
        //     },
        //     Orbs::Single(_) => {
        //         println!("Singles in add_sample_compute_diag: (Hc)^2 / p = {}", pt_det.coeff * pt_det.coeff / sampled_prob);
        //     }
        // }
        match self.samples.get_mut(&pt_det.config) {
            None => {
                // New PT det was sampled: compute diagonal element and create new variational det map
                let pt_diag_elem: f64 = var_det.new_diag(ham, excite);
                let mut var_det_map: HashMap<Config, (f64, f64, i32)> = Default::default();
                var_det_map.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                self.samples
                    .insert(pt_det.config, (pt_diag_elem, var_det_map));
            }
            Some(pt_det_info) => {
                // No need to recompute the diagonal element
                Self::process_resampled_pt_det(&var_det, pt_det, sampled_prob, pt_det_info)
            }
        }
    }

    fn process_resampled_pt_det(
        var_det: &Det,
        pt_det: Det,
        sampled_prob: f64,
        pt_det_info: &mut (f64, HashMap<Config, (f64, f64, i32)>),
    ) {
        match pt_det_info.1.get_mut(&var_det.config) {
            None => {
                // New var det to reach this PT det; add to variational det map
                pt_det_info
                    .1
                    .insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
            }
            Some(var_det_info) => {
                // Already have this var det; just increment number of times it has been sampled
                var_det_info.2 += 1;
            }
        }
    }

    pub fn add_sample_diag_already_stored(&mut self, var_det: Det, pt_det: Det, sampled_prob: f64) {
        // Add a new sample to PtSamples
        // Assumes that pt_det's diagonal element already stored
        self.n += 1;
        match self.samples.get_mut(&pt_det.config) {
            None => {
                // New PT det was sampled
                let mut var_det_map: HashMap<Config, (f64, f64, i32)> = Default::default();
                var_det_map.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                self.samples
                    .insert(pt_det.config, (pt_det.diag.unwrap(), var_det_map));
            }
            Some(pt_det_info) => {
                // No need to recompute the diagonal element
                Self::process_resampled_pt_det(&var_det, pt_det, sampled_prob, pt_det_info)
            }
        }
    }

    pub fn pt_estimator(&self, e0: f64, n_det: i32) -> f64 {
        // Computes the unbiased PT estimator as in the original SHCI paper
        // Needs to input the variational energy e0
        // n_det is either the number of sampled variational dets (old method),
        // or the number of sampled perturbative dets (new method)
        // TODO: Figure out why PT energy is wrong and why contributions vary so much

        let mut out: f64 = 0.0;
        let mut tmp: f64 = 0.0;
        let mut diag_term: f64;
        let mut to_square: f64;
        let mut w_over_p: f64;

        // TODO: Exclude perturbers that only have large contributions

        for (ind, (pt_det, (pt_det_diag, var_det_map))) in enumerate(&self.samples) {
            // println!("\nPT det {}: {}\n", ind, pt_det);
            diag_term = 0.0;
            to_square = 0.0;
            for (hai_ci, p, w) in var_det_map.values() {
                // println!("New energy sample! H_ai c_i = {}, p = {}, (H_ai c_i)^2 / p = {}, w = {}, E0 = {}, E_a = {}", hai_ci, p, hai_ci * hai_ci / p, w, e0, pt_det_diag);
                if *p < 1e-9 {
                    // println!("Warning! Sample probability very small! p = {}", p);
                } else {
                    w_over_p = (*w as f64) / p;
                    // println!("p = {:.2e}", p);
                    // println!("(H_ai c_i)^2 / p_i = {:.3}", hai_ci * hai_ci / p);
                    diag_term += ((n_det - 1) as f64 - w_over_p) * w_over_p * hai_ci * hai_ci;
                    to_square += hai_ci * w_over_p;
                }
            }
            // println!(
            //     "Diag term = {:.3}, off-diag term = {:.3}, diag + off-diag^2 = {:.3}",
            //     diag_term,
            //     to_square,
            //     diag_term + to_square * to_square
            // );
            // println!(
            //     "Energy estimator: {}",
            //     (diag_term + to_square * to_square) / (e0 - pt_det_diag)
            // );
            out += (diag_term + to_square * to_square) / (e0 - pt_det_diag);
            tmp += diag_term + to_square * to_square;
        }

        // println!(
        //     "Unbiased estimator ({} sampled var dets, {} sampled PT dets) = {}",
        //     n_det,
        //     self.n,
        //     out / (n_det as f64 * (n_det - 1) as f64)
        // );
        // println!(
        //     "Component of unbiased estimator that should be constant = {:.3}",
        //     tmp / (n_det as f64 * (n_det - 1) as f64)
        // );
        out / (n_det as f64 * (n_det - 1) as f64)
    }
}
