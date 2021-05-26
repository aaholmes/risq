// Module for functions specific to Epstein-Nesbet perturbation theory

use std::collections::HashMap;
use crate::wf::det::{Config, Det};
use crate::excite::Excite;
use crate::ham::Ham;

#[derive(Default)]
pub struct PtSamples {
    // Sampled contributions to the ENPT2 correction
    // Samples stored in a hashmap:
    // Key: perturbative det (Config)
    // Value: (perturbative det's diag, HashMap (key: variational det's config, value: (H_ai c_i, p_ai, w_ai)))
    pub n: i32,
    pub samples: HashMap<
        Config, (
            f64, HashMap<
                Config, (
                    f64, f64, i32
                )
            >
        )
    >
}

impl PtSamples {
    pub fn add_sample(&mut self, var_det: Det, excite: &Excite, pt_det: Det, sampled_prob: f64, ham: &Ham) {
        // Add a new sample to PtSamples
        // Compute diagonal element of perturbative determinant if it hasn't already been computed
        self.n += 1;
        match self.samples.get_mut(&pt_det.config) {
            None => {
                // New PT det was sampled: compute diagonal element and create new variational det map
                let pt_diag_elem: f64 = var_det.new_diag(ham, excite);
                let mut var_det_map: HashMap<Config, (f64, f64, i32)> = Default::default();
                var_det_map.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                self.samples.insert(pt_det.config, (pt_diag_elem, var_det_map));
            }
            Some(pt_det_info) => {
                // No need to recompute the diagonal element
                match pt_det_info.1.get_mut(&var_det.config) {
                    None => {
                        // New var det to reach this PT det; add to variational det map
                        pt_det_info.1.insert(var_det.config, (pt_det.coeff, sampled_prob, 1));
                    }
                    Some(var_det_info) => {
                        // Already have this var det; just increment number of times it has been sampled
                        var_det_info.2 += 1;
                    }
                }
            }
        }
    }

    pub fn unbiased_pt_estimator(&self, e0: f64) -> f64 {
        // Computes the unbiased PT estimator as in the original SHCI paper
        // Needs to input the variational energy e0
        let mut out: f64 = 0.0;
        let mut diag_term: f64;
        let mut to_square: f64;
        let mut w_over_p: f64;

        for (pt_det_diag, var_det_map) in self.samples.values() {
            diag_term = 0.0;
            to_square = 0.0;
            for (hai_ci, p, w) in var_det_map.values() {
                //println!("New energy sample! H_ai c_i = {}, p = {}, w = {}, E0 = {}, E_a = {}", hai_ci, p, w, e0, pt_det_diag);
                w_over_p = (*w as f64) / p;
                diag_term += ((self.n - 1) as f64 - w_over_p) * w_over_p * hai_ci * hai_ci;
                to_square += hai_ci * w_over_p;
                //println!("Diag term = {}, off-diag term = {}", diag_term, to_square);
            }
            //println!("Incrementing output by... {}", (diag_term + to_square * to_square) / (e0 - pt_det_diag));
            out += (diag_term + to_square * to_square) / (e0 - pt_det_diag);
        }

        //println!("Unbiased estimator = {}", out / (self.n as f64 * (self.n - 1) as f64));
        out / (self.n as f64 * (self.n - 1) as f64)
    }
}