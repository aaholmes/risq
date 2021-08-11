// Wavefunction data structure:
// Includes both the variational wf and the variational H
// Includes functions for initializing, printing, and adding new determinants

pub mod det;
mod eps;

use std::collections::HashMap;

use super::utils::read_input::Global;
use super::ham::Ham;
use crate::excite::{Excite, Orbs};
use crate::excite::init::ExciteGenerator;
use det::{Det, Config};
use eps::{Eps, init_eps};
use crate::utils::bits::{bit_pairs, bits};
use crate::stoch::{DetOrbSample, ScreenedSampler, generate_screened_sampler};
use itertools::enumerate;
use crate::var::sparse::{SparseMat, SparseMatUpperTri};
use crate::var::off_diag::OffDiagElemsNoHash;

#[derive(Default)]
pub struct Wf {
    pub n_states: i32, // number of states - same as in input file, but want it attached to the wf
    pub converged: bool, // whether variational wf is converged. Update at end of each HCI iteration
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    pub eps: f64, // current eps value
    pub n: usize,                         // number of dets
    pub energy: f64,                    // variational energy
    pub inds: HashMap<Config, usize>,            // hashtable : det -> usize for looking up index by det
    pub dets: Vec<Det>,                     // for looking up det by index
    n_stored_h: usize, // n for which the variational H has already been stored
    pub sparse_ham: SparseMatUpperTri, // store the sparse variational H off-diagonal elements here
}

impl Wf {

    pub fn push(&mut self, d: Det) {
        // Just adds a new det to the wf
        if !self.inds.contains_key(&d.config) {
            self.inds.insert(d.config, self.n);
            self.n += 1;
            self.dets.push(d);
        }
    }

    pub fn add_det_with_coeff(&mut self, exciting_det: &Det, ham: &Ham, excite: &Excite, new_det: Config, coeff: f64) {
        // Add det with its coefficient
        // If det already exists in wf, add its coefficient to that det
        // Also, computes diagonal element if necessary (that's what exciting_det, ham and excite are needed for)
        let ind = self.inds.get(&new_det);
        match ind {
            Some(k) => self.dets[*k].coeff += coeff,
            None => {
                self.inds.insert(new_det, self.n);
                self.n += 1;
                self.dets.push(
                    Det {
                        config: new_det,
                        coeff: coeff,
                        diag: exciting_det.new_diag(ham, excite)
                    }
                );
            }
        }
    }

    pub fn approx_matmul_external_dtm_only(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> (Wf, Vec<f64>) {
        // Approximate matrix-vector multiplication
        // Uses eps as a cutoff for both singles and doubles, as in SHCI (but faster of course)
        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
        // Also returns a vector of sum of remaining (Hc)^2 for each input det, since this is the
        // optimal probability for sampling variational dets in the old SHCI algorithm (rather than |c|)

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // For making screened sampler
        let mut out_sum_remaining: Vec<f64> = vec![];

        // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
        let mut out_wf: Wf = Wf::default();

        // Off-diagonal component
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            let mut sum_remaining_this_det: f64 = 0.0;
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        let sing: f64 = ham.ham_sing(&det.config, &d);
                                        if sing.abs() >= local_eps {
                                            // TODO: Do this in a cache efficient way
                                            out_wf.add_det_with_coeff(det, ham, &excite, d, sing * det.coeff);
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
            out_sum_remaining.push(det.coeff * det.coeff * sum_remaining_this_det);
        } // for det in self.dets

        (out_wf, out_sum_remaining)
    }

    pub fn approx_matmul_external(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> (Wf, ScreenedSampler) {
        // Approximate matrix-vector multiplication
        // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
        // they meet the cutoff is as expensive as actually calculating the matrix element)
        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // For making screened sampler
        let mut det_orbs: Vec<DetOrbSample> = vec![];

        // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
        let mut out_wf: Wf = Wf::default();
        // for det in &self.dets {
        //     out_wf.push(Det{config: det.config, coeff: det.diag * det.coeff, diag: det.diag});
        // }

        // Off-diagonal component
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample{
                                    det: det,
                                    init: Orbs::Double((i, j)),
                                    is_alpha: None,
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample{
                                    det: det,
                                    init: Orbs::Double((i, j)),
                                    is_alpha: Some(*is_alpha),
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample{
                                    det: det,
                                    init: Orbs::Single(i),
                                    is_alpha: Some(*is_alpha),
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_sing(&det.config, &d) * det.coeff);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        // Now, convert det_orbs to a screened_sampler
        (out_wf, generate_screened_sampler(eps, det_orbs))
    }

    pub fn approx_matmul_external_semistoch_singles(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> (Wf, ScreenedSampler) {
        // Approximate matrix-vector multiplication
        // Deterministic step uses eps as a cutoff for both doubles and singles
        // Prepares screened sampler that is intended to sample singles and doubles separately

        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
        // ScreenedSampler object is able to sample according to multiple probabilities
        // Note: we can't use the max_sing and max_doub values here, because we have to create the sampler,
        // which requires iterating over exciting electrons even if they will all be screened out deterministically

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // For making screened sampler
        let mut det_orbs: Vec<DetOrbSample> = vec![];

        // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
        let mut out_wf: Wf = Wf::default();

        // Off-diagonal component
        for det in &self.dets {

            local_eps = eps / det.coeff.abs();

            // Double excitations

            // Opposite spin
            for i in bits(excite_gen.valence & det.config.up) {
                for j in bits(excite_gen.valence & det.config.dn) {
                    for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {

                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {

                            None => {}, // If not a valid excitation, do nothing

                            Some(d) => {

                                if stored_excite.abs_h >= local_eps {

                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }

                                } else {

                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det: det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: None,
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                for (i, j) in bit_pairs(excite_gen.valence & *config) {
                    for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {

                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha)
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {

                            None => {}, // If not a valid excitation, do nothing

                            Some(d) => {

                                if stored_excite.abs_h >= local_eps {

                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                    }

                                } else {

                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det: det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: Some(*is_alpha),
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }


            // Single excitations

            // Set up sampling *all* remaining single excitations starting from the first one whose value (not max value!)
            // is smaller than eps
            // For the sampler, also keep going (after max |H| < eps) to skip over the invalid excitations here so they won't be sampled later!
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        let mut stored_sampler: bool = false;
                        for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h >= local_eps {
                                // *max* |H| >= eps: deterministic component
                                excite = Excite {
                                    init: Orbs::Single(i),
                                    target: stored_excite.target,
                                    abs_h: stored_excite.abs_h,
                                    is_alpha: Some(*is_alpha)
                                };
                                // First check whether this is a valid excite
                                new_det = det.config.safe_excite_det(&excite);
                                match new_det {
                                    Some(d) => {
                                        // Make sure excite is to a perturbative det (not a variational one)
                                        if !self.inds.contains_key(&d) {
                                            // Valid excite: add to H*psi
                                            // Compute matrix element and compare its magnitude to eps
                                            let sing: f64 = ham.ham_sing(&det.config, &d);
                                            if sing.abs() >= local_eps {
                                                // *exact* |H| >= eps; add to output wf
                                                // TODO: Do this in a cache efficient way
                                                out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_sing(&det.config, &d) * det.coeff);
                                            } else {
                                                // *exact* |H| < eps; store remaining excites in sampler so this valid excite and subsequent ones can be sampled
                                                if !stored_sampler {
                                                    // println!("Setup single det_orb sampler when exact |H| < eps, orb = {}", i);
                                                    det_orbs.push(DetOrbSample {
                                                        det: det,
                                                        init: Orbs::Single(i),
                                                        is_alpha: Some(*is_alpha),
                                                        sum_abs_h: stored_excite.sum_remaining_abs_h,
                                                        sum_h_squared: stored_excite.sum_remaining_h_squared,
                                                        sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                                        sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                                    });
                                                    stored_sampler = true;
                                                }
                                            }
                                        }
                                    }
                                    None => {}
                                }
                            } else if stored_sampler {

                                // max |H| < eps: no more deterministic pieces allowed, and stochastic sampler
                                // already stored, so done with this orb
                                break;

                            } else {

                                // max |H| < eps: no more deterministic pieces allowed

                                // Update the screened sampler if this excite is valid and points to
                                // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                excite = Excite {
                                    init: Orbs::Single(i),
                                    target: stored_excite.target,
                                    abs_h: stored_excite.abs_h,
                                    is_alpha: Some(*is_alpha)
                                };

                                // First check whether this is a valid excite
                                new_det = det.config.safe_excite_det(&excite);
                                match new_det {
                                    None => {},
                                    Some(d) => {
                                        // Make sure excite is to a perturbative det (not a variational one)
                                        if !self.inds.contains_key(&d) {
                                            // println!("Setup single det_orb sampler when max |H| < eps, orb = {}", i);
                                            det_orbs.push(DetOrbSample {
                                                det: det,
                                                init: Orbs::Single(i),
                                                is_alpha: Some(*is_alpha),
                                                sum_abs_h: stored_excite.sum_remaining_abs_h,
                                                sum_h_squared: stored_excite.sum_remaining_h_squared,
                                                sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                                sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                                            });
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        } // for det in self.dets


        // Now, convert det_orbs to a screened_sampler
        (out_wf, generate_screened_sampler(eps, det_orbs))
    }


    pub fn approx_matmul_external_no_singles(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> (Wf, ScreenedSampler) {
        // Same as above, but no single excitations in deterministic step (still sets up singles for sampling later)
        // Approximate matrix-vector multiplication
        // Deterministic step uses eps as a cutoff for doubles, but ignores singles
        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
        // ScreenedSampler object is able to sample according to multiple probabilities
        // Note: we can't use the max_sing and max_doub values here, because we have to create the sampler,
        // which requires iterating over exciting electrons even if they will all be screened out deterministically

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // For making screened sampler
        let mut det_orbs: Vec<DetOrbSample> = vec![];

        // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
        let mut out_wf: Wf = Wf::default();
        // for det in &self.dets {
        //     out_wf.push(Det{config: det.config, coeff: det.diag * det.coeff, diag: det.diag});
        // }

        // Off-diagonal component
        for det in &self.dets {
            let mut sum_abs_h_external: f64 = 0.0;
            let mut sum_abs_h_discarded: f64 = 0.0;
            let mut sum_h_sq_external: f64 = 0.0;
            let mut sum_h_sq_discarded: f64 = 0.0;
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            for i in bits(excite_gen.valence & det.config.up) {
                for j in bits(excite_gen.valence & det.config.dn) {
                    for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            // Update the screened sampler, then break
                            det_orbs.push(DetOrbSample {
                                det: det,
                                init: Orbs::Double((i, j)),
                                is_alpha: None,
                                sum_abs_h: stored_excite.sum_remaining_abs_h,
                                sum_h_squared: stored_excite.sum_remaining_h_squared,
                                sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                            });
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None
                        };
                        new_det = det.config.safe_excite_det(&excite);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                }
                                if stored_excite.abs_h < 1e-4 {
                                    sum_abs_h_external += stored_excite.abs_h;
                                    sum_h_sq_external += stored_excite.abs_h * stored_excite.abs_h;
                                }
                            }
                            None => {
                                if stored_excite.abs_h < 1e-4 {
                                    sum_abs_h_discarded += stored_excite.abs_h;
                                    sum_h_sq_discarded += stored_excite.abs_h * stored_excite.abs_h;
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                for (i, j) in bit_pairs(excite_gen.valence & *config) {
                    for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            // Update the screened sampler, then break
                            det_orbs.push(DetOrbSample{
                                det: det,
                                init: Orbs::Double((i, j)),
                                is_alpha: Some(*is_alpha),
                                sum_abs_h: stored_excite.sum_remaining_abs_h,
                                sum_h_squared: stored_excite.sum_remaining_h_squared,
                                sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                            });
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha)
                        };
                        new_det = det.config.safe_excite_det(&excite);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                }
                                if stored_excite.abs_h < 1e-4 {
                                    sum_abs_h_external += stored_excite.abs_h;
                                    sum_h_sq_external += stored_excite.abs_h * stored_excite.abs_h;
                                }
                            }
                            None => {
                                if stored_excite.abs_h < 1e-4 {
                                    sum_abs_h_discarded += stored_excite.abs_h;
                                    sum_h_sq_discarded += stored_excite.abs_h * stored_excite.abs_h;
                                }
                            }
                        }
                    }
                }
            }


            // Single excitations (no deterministic contribution - just set up for sampling later)
            for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                for i in bits(excite_gen.valence & *config) {
                    let mut stored_excite = &excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()[0];
                    det_orbs.push(DetOrbSample{
                        det: det,
                        init: Orbs::Single(i),
                        is_alpha: Some(*is_alpha),
                        sum_abs_h: stored_excite.sum_remaining_abs_h,
                        sum_h_squared: stored_excite.sum_remaining_h_squared,
                        sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                        sum_hc_squared: det.coeff * det.coeff * stored_excite.sum_remaining_h_squared,
                    });
                }
            }
            // println!("Percentage discarded: |H|: {}, H^2: {}", sum_abs_h_discarded / (sum_abs_h_discarded + sum_abs_h_external), sum_h_sq_discarded / (sum_h_sq_discarded + sum_h_sq_external));

        } // for det in self.dets


        // Now, convert det_orbs to a screened_sampler
        (out_wf, generate_screened_sampler(eps, det_orbs))
    }

    pub fn approx_matmul_variational(&self, input_coeffs: &Vec<f64>, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> Vec<f64> {
        // Approximate matrix-vector multiplication within variational space only
        // WARNING: Uses self only to define and access variational dets; uses input_coeffs as the vector to multiply with
        // instead of wf.dets[:].coeff
        // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
        // they meet the cutoff is as expensive as actually calculating the matrix element)
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;
        let mut out: Vec<f64> = vec![0.0f64; self.n];

        // Diagonal component
        for (i_det, det) in enumerate(self.dets.iter()) {
            out[i_det] = det.diag * input_coeffs[i_det];
        }

        // Off-diagonal component

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // only add if it is already in variational wf
        for (i_det, det) in enumerate(self.dets.iter()) {

            local_eps = eps / input_coeffs[i_det].abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out[*ind] += ham.ham_sing(&det.config, &d) * input_coeffs[i_det]
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }

    pub fn find_new_dets(&mut self, ham: &Ham, excite_gen: &ExciteGenerator) -> bool {
        // Get new dets: iterate over all dets; for each, propose all excitations; for each, check if new;
        // if new, add to wf
        // Returns true if no new dets (i.e., returns whether already converged)

        self.eps = self.eps_iter.next().unwrap();

        let new_dets: Wf = self.iterate_excites(ham, excite_gen, self.eps, false);

        // Add all new dets to the wf
        for det in new_dets.dets {
            self.push(det);
        }

        new_dets.n == 0
    }

    fn iterate_excites(&mut self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, matmul: bool) -> Wf {
        // Iterate over excitations using heat-bath cutoff eps
        // Used internally by both approx_matmul and get_new_dets
        // If matmul, then return H*psi; else, return a wf composed of new dets
        println!("Getting new dets with epsilon = {:.1e}", eps);
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;
        // We can't just iterate over dets because we are adding new dets to the same dets data structure
        let mut out: Wf = Wf::default();
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: either add to H*psi or add this det to out
                                    if matmul {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        // out.add_det_with_coeff(det, ham, excite, d,
                                        //                        ham.ham_doub(&det.config, &d) * det.coeff);
                                        todo!()
                                    } else {
                                        // If not already in input or output, compute diagonal element and add to output
                                        if !self.inds.contains_key(&d) {
                                            if !out.inds.contains_key(&d) {
                                                match excite.target {
                                                    Orbs::Double(rs) => {
                                                        out.push(
                                                            Det {
                                                                config: d,
                                                                coeff: 0.0,
                                                                diag: det.new_diag_opp(ham, (i, j), rs)
                                                            }
                                                        );
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if matmul {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        // out.add_det_with_coeff(det, ham, excite, d,
                                        //                       ham.ham_doub(&det.config, &d) * det.coeff);
                                        todo!()
                                    } else {
                                        if !self.inds.contains_key(&d) {
                                            if !out.inds.contains_key(&d) {
                                                match excite.target {
                                                    Orbs::Double(rs) => {
                                                        out.push(
                                                            Det {
                                                                config: d,
                                                                coeff: 0.0,
                                                                diag: det.new_diag_same(ham, (i, j), rs, *is_alpha)
                                                            }
                                                        );
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            // if excite_gen.max_sing >= local_eps {
            //     for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            //         for i in bits(excite_gen.valence & *config) {
            //             for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
            //                 if stored_excite.abs_h < local_eps { break; }
            //                 excite = Excite {
            //                     init: Orbs::Single(i),
            //                     target: stored_excite.target,
            //                     abs_h: stored_excite.abs_h,
            //                     is_alpha: Some(*is_alpha)
            //                 };
            //                 new_det = det.config.safe_excite_det(&excite);
            //                 match new_det {
            //                     Some(d) => {
            //                         if matmul {
            //                             // Compute matrix element and add to H*psi
            //                             // TODO: Do this in a cache efficient way
            //                             // out.add_det_with_coeff(det, ham, excite, d,
            //                             //                       ham.ham_sing(&det.config, &d) * det.coeff);
            //                             todo!()
            //                         } else {
            //                             if !self.inds.contains_key(&d) {
            //                                 if !out.inds.contains_key(&d) {
            //                                     match excite.target {
            //                                         Orbs::Single(r) => {
            //                                             // Compute whether single excitation actually exceeds eps!
            //                                             if ham.ham_sing(&det.config, &d).abs() > local_eps {
            //                                                 out.push(
            //                                                     Det {
            //                                                         config: d,
            //                                                         coeff: 0.0,
            //                                                         diag: det.new_diag_sing(ham, i, r, *is_alpha)
            //                                                     }
            //                                                 );
            //                                             }
            //                                         }
            //                                         _ => {}
            //                                     }
            //                                 }
            //                             }
            //                         }
            //                     }
            //                     None => {}
            //                 }
            //             }
            //         }
            //     }
            // }
        } // for det in self.dets

        out
    }

    pub fn update_n_stored_h(&mut self, n: usize) {
        // Make this setter explicit because we really don't want to accidentally change it
        self.n_stored_h = n;
    }

    pub fn n_stored_h(&self) -> usize {
        self.n_stored_h
    }

    pub fn new_sparse_ham(&mut self) {
        // Create new sparse Hamiltonian, set the diagonal elements to the ones already stored in wf
        // Don't update self.n_stored_h yet, because we haven't computed the off-diagonal elements yet
        self.sparse_ham = SparseMatUpperTri {
            n: self.n,
            diag: vec![0.0; self.n],
            nnz: vec![0; self.n],
            off_diag: vec![Vec::with_capacity(100); self.n],
        };
        for i in 0..self.n {
            self.sparse_ham.diag[i] = self.dets[i].diag;
        }
    }

    pub fn expand_sparse_ham_rows(&mut self) {
        // Just create empty rows to fill up the dimension to new size wf.n
        // Also, fill up all of the new diagonal elements
        // Don't update self.n_stored_h yet, because we haven't computed the off-diagonal elements yet
        println!("Expanding variational H from size {} to size {}", self.sparse_ham.n, self.n);
        self.sparse_ham.diag.append(&mut vec![0.0; self.n - self.sparse_ham.n]);
        self.sparse_ham.nnz.append(&mut vec![0; self.n - self.sparse_ham.n]);
        self.sparse_ham.off_diag.append(&mut vec![Vec::with_capacity(100); self.n - self.sparse_ham.n]);
        for i in self.sparse_ham.n..self.n {
            self.sparse_ham.diag[i] = self.dets[i].diag;
        }
        self.sparse_ham.n = self.n;
    }

}

// Initialize variational wf to the HF det (only needs to be called once)
pub fn init_var_wf(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator) -> Wf {
    let mut wf: Wf = Wf::default();
    wf.n_states = global.n_states;
    wf.converged = false;
    wf.n = 1;
    wf.update_n_stored_h(0); // No stored H yet
    let one: u128 = 1;
    let mut hf = Det {
        config: Config {
            up: ((one << global.nup) - 1),
            dn: ((one << global.ndn) - 1),
        },
        coeff: 1.0,
        diag: 0.0,
    };
    let h: f64 = ham.ham_diag(&hf.config);
    hf.diag = h;
    wf.inds = HashMap::new();
    wf.inds.insert(hf.config, 0);
    wf.dets.push(hf);
    wf.energy = wf.dets[0].diag;
    wf.eps_iter = init_eps(&wf, &global, &excite_gen);
    wf
}
