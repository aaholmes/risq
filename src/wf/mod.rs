//! Variational wavefunction data structure
//! Includes both the variational wf and the variational H
//! Includes functions for initializing, printing, and adding new determinants

pub mod det;
mod eps;

use std::collections::{BinaryHeap, HashMap};

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs};
use crate::stoch::{generate_screened_sampler, DetOrbSample, ScreenedSampler};
use crate::utils::bits::{bit_pairs, bits};
use crate::utils::display::DetByCoeff;
use crate::var::sparse::SparseMatUpperTri;
use det::{Config, Det};
use eps::{init_eps, Eps};
use itertools::enumerate;
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
use rolling_stats::Stats;
use std::cmp::Reverse;
use std::time::Instant;
use crate::excite::iterator::dets_excites_and_excited_dets;
use crate::excite::Orbs::Single;

/// Wavefunction data structure
#[derive(Default)]
pub struct Wf {
    pub n: usize,                     // number of dets
    pub inds: HashMap<Config, usize>, // hashtable : det -> usize for looking up index by det
    pub dets: Vec<Det>,               // for looking up det by index
    pub energy: f64,                  // variational energy
}

/// Variational wavefunction data structure:
/// contains both the wavefunction and the variational Hamiltonian
#[derive(Default)]
pub struct VarWf {
    pub wf: Wf,                        // the variational wf
    pub n_states: i32, // number of states - same as in input file, but want it attached to the wf
    pub converged: bool, // whether variational wf is converged. Update at end of each HCI iteration
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    pub eps: f64,      // current eps value
    n_stored_h: usize, // n for which the variational H has already been stored
    pub sparse_ham: SparseMatUpperTri, // store the sparse variational H off-diagonal elements here
}

impl Wf {
    /// Just adds a new det to the wf
    pub fn push(&mut self, d: Det) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.inds.entry(d.config) {
            e.insert(self.n);
            self.n += 1;
            self.dets.push(d);
        }
    }

    pub fn push_config(&mut self, config: Config) {
        self.push(Det{
            config,
            coeff: 0.0,
            diag: None
        });
    }

    /// Add det with its coefficient
    /// If det already exists in wf, add its coefficient to that det
    /// Also, computes diagonal element if necessary (that's what exciting_det, ham and excite are needed for)
    pub fn add_det_with_coeff(
        &mut self,
        exciting_det: &Det,
        ham: &Ham,
        excite: &Excite,
        new_det: Config,
        coeff: f64,
    ) {
        let ind = self.inds.get(&new_det);
        match ind {
            Some(k) => self.dets[*k].coeff += coeff,
            None => {
                self.inds.insert(new_det, self.n);
                self.n += 1;
                self.dets.push(Det {
                    config: new_det,
                    coeff,
                    diag: Some(exciting_det.new_diag(ham, excite)),
                });
            }
        }
    }

    pub fn approx_matmul_external_dtm_only(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> (Wf, Vec<f64>) {
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
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(
                                        det,
                                        ham,
                                        &excite,
                                        d,
                                        ham.ham_doub(&det.config, &d) * det.coeff,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(
                                        det,
                                        ham,
                                        &excite,
                                        d,
                                        ham.ham_doub(&det.config, &d) * det.coeff,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in
                            excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                sum_remaining_this_det += stored_excite.sum_remaining_h_squared;
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    let sing: f64 = ham.ham_sing(&det.config, &d);
                                    if sing.abs() >= local_eps {
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(
                                            det,
                                            ham,
                                            &excite,
                                            d,
                                            sing * det.coeff,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            out_sum_remaining.push(det.coeff * det.coeff * sum_remaining_this_det);
        } // for det in self.dets

        (out_wf, out_sum_remaining)
    }

    pub fn approx_matmul_external_separate_doubles_and_singles(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> (Wf, ScreenedSampler) {
        // Approximate matrix-vector multiplication
        // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
        // they meet the cutoff is as expensive as actually calculating the matrix element)
        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
        // Screened samplers returned are for doubles that are less than the cutoff and haven't
        // been treated deterministically, as well as singles whose max value is less than the
        // eps cutoff.

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
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample {
                                    det,
                                    init: Orbs::Double((i, j)),
                                    is_alpha: None,
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff
                                        * det.coeff
                                        * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(
                                        det,
                                        ham,
                                        &excite,
                                        d,
                                        ham.ham_doub(&det.config, &d) * det.coeff,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample {
                                    det,
                                    init: Orbs::Double((i, j)),
                                    is_alpha: Some(*is_alpha),
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff
                                        * det.coeff
                                        * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(
                                        det,
                                        ham,
                                        &excite,
                                        d,
                                        ham.ham_doub(&det.config, &d) * det.coeff,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in
                            excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                // Update the screened sampler, then break
                                det_orbs.push(DetOrbSample {
                                    det,
                                    init: Orbs::Single(i),
                                    is_alpha: Some(*is_alpha),
                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                                    sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                    sum_hc_squared: det.coeff
                                        * det.coeff
                                        * stored_excite.sum_remaining_h_squared,
                                });
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if !self.inds.contains_key(&d) {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out_wf.add_det_with_coeff(
                                        det,
                                        ham,
                                        &excite,
                                        d,
                                        ham.ham_sing(&det.config, &d) * det.coeff,
                                    );
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

    pub fn approx_matmul_external_skip_singles(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> (Wf, ScreenedSampler) {
        // For debugging: Skip singles entirely

        // Approximate matrix-vector multiplication
        // Deterministic step uses eps as a cutoff for doubles

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
                    for stored_excite in excite_gen
                        .opp_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None,
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {
                            None => {} // If not a valid excitation, do nothing

                            Some(d) => {
                                if stored_excite.abs_h >= local_eps {
                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(
                                            det,
                                            ham,
                                            &excite,
                                            d,
                                            ham.ham_doub(&det.config, &d) * det.coeff,
                                        );
                                    }
                                } else {
                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: None,
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs()
                                                * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff
                                                * det.coeff
                                                * stored_excite.sum_remaining_h_squared,
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
                    for stored_excite in excite_gen
                        .same_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha),
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {
                            None => {} // If not a valid excitation, do nothing

                            Some(d) => {
                                if stored_excite.abs_h >= local_eps {
                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(
                                            det,
                                            ham,
                                            &excite,
                                            d,
                                            ham.ham_doub(&det.config, &d) * det.coeff,
                                        );
                                    }
                                } else {
                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: Some(*is_alpha),
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs()
                                                * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff
                                                * det.coeff
                                                * stored_excite.sum_remaining_h_squared,
                                        });
                                        break;
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


    pub fn approx_matmul_external_semistoch_singles(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> (Wf, ScreenedSampler) {
        // Approximate matrix-vector multiplication
        // Deterministic step uses eps as a cutoff for both doubles and singles
        // Prepares screened sampler that is intended to sample singles and doubles separately

        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
        // ScreenedSampler object is able to sample according to multiple probabilities
        // Note: we can't use the max_sing and max_doub values here, because we have to create the sampler,
        // which requires iterating over exciting electrons even if they will all be screened out deterministically

        // Finally, since singles are normally oversampled (using max |H| instead of |H|), we compute
        // the amount by which they would be oversampled (for the singles we loop over deterministically),
        // and use that computed oversampling value to reduce the sampling of singles later on

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

        // Keep track of statistics for how much singles would be oversampled
        let mut singles_oversampled_abs_hc: Stats<f64> = Stats::new();
        let mut singles_oversampled_hc_squared: Stats<f64> = Stats::new();

        for det in &self.dets {
            local_eps = eps / det.coeff.abs();

            // Double excitations

            // Opposite spin
            for i in bits(excite_gen.valence & det.config.up) {
                for j in bits(excite_gen.valence & det.config.dn) {
                    for stored_excite in excite_gen
                        .opp_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None,
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {
                            None => {} // If not a valid excitation, do nothing

                            Some(d) => {
                                if stored_excite.abs_h >= local_eps {
                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(
                                            det,
                                            ham,
                                            &excite,
                                            d,
                                            ham.ham_doub(&det.config, &d) * det.coeff,
                                        );
                                    }
                                } else {
                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: None,
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs()
                                                * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff
                                                * det.coeff
                                                * stored_excite.sum_remaining_h_squared,
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
                    for stored_excite in excite_gen
                        .same_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha),
                        };

                        new_det = det.config.safe_excite_det(&excite);

                        match new_det {
                            None => {} // If not a valid excitation, do nothing

                            Some(d) => {
                                if stored_excite.abs_h >= local_eps {
                                    // |H| >= eps: deterministic component
                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(
                                            det,
                                            ham,
                                            &excite,
                                            d,
                                            ham.ham_doub(&det.config, &d) * det.coeff,
                                        );
                                    }
                                } else {
                                    // |H| < eps: no more deterministic pieces allowed

                                    // Update the screened sampler if this excite is valid and points to
                                    // a perturbative det (else, skip until we reach an excite that fits those criteria!)

                                    // First check whether this is a valid excite
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        det_orbs.push(DetOrbSample {
                                            det,
                                            init: Orbs::Double((i, j)),
                                            is_alpha: Some(*is_alpha),
                                            sum_abs_h: stored_excite.sum_remaining_abs_h,
                                            sum_h_squared: stored_excite.sum_remaining_h_squared,
                                            sum_abs_hc: det.coeff.abs()
                                                * stored_excite.sum_remaining_abs_h,
                                            sum_hc_squared: det.coeff
                                                * det.coeff
                                                * stored_excite.sum_remaining_h_squared,
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
            let single_sample_factor = 1e-6; // multiply single probabilities because they can only be estimated
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        let mut stored_sampler: bool = false;
                        for stored_excite in
                            excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
                        {
                            if stored_excite.abs_h >= local_eps {
                                // *max* |H| >= eps: deterministic component
                                excite = Excite {
                                    init: Orbs::Single(i),
                                    target: stored_excite.target,
                                    abs_h: stored_excite.abs_h,
                                    is_alpha: Some(*is_alpha),
                                };
                                // First check whether this is a valid excite
                                new_det = det.config.safe_excite_det(&excite);
                                if let Some(d) = new_det {
                                    // Make sure excite is to a perturbative det (not a variational one)
                                    if !self.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and compare its magnitude to eps
                                        let sing: f64 = ham.ham_sing(&det.config, &d);
                                        // If this excite were sampled, the oversampling would be by a factor of |excite.abs_h / sing| (or that value squared)
                                        singles_oversampled_abs_hc
                                            .update(excite.abs_h / sing.abs());
                                        singles_oversampled_hc_squared
                                            .update(excite.abs_h * excite.abs_h / sing / sing);

                                        // Figure out what's going on with crazy ratios here
                                        // if excite.abs_h / sing.abs() >  1770584291294.0 {
                                        //     println!("Vast overestimation! exciting det = {}, {} -> {}: max|H| = {}, |H| = {}, max|H|/|H| = {}", det.config, excite.init, excite.target, excite.abs_h, sing.abs(), excite.abs_h / sing.abs());
                                        // }
                                        if sing.abs() >= local_eps {
                                            // *exact* |H| >= eps; add to output wf
                                            // TODO: Do this in a cache efficient way
                                            out_wf.add_det_with_coeff(
                                                det,
                                                ham,
                                                &excite,
                                                d,
                                                sing * det.coeff,
                                            );
                                        } else {
                                            // *exact* |H| < eps; store remaining excites in sampler so this valid excite and subsequent ones can be sampled
                                            if !stored_sampler {
                                                // println!("Setup single det_orb sampler when exact |H| < eps, orb = {}", i);
                                                det_orbs.push(DetOrbSample {
                                                    det,
                                                    init: Orbs::Single(i),
                                                    is_alpha: Some(*is_alpha),
                                                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                                                    sum_h_squared: stored_excite
                                                        .sum_remaining_h_squared,
                                                    sum_abs_hc: det.coeff.abs()
                                                        * stored_excite.sum_remaining_abs_h,
                                                    sum_hc_squared: det.coeff
                                                        * det.coeff
                                                        * stored_excite.sum_remaining_h_squared,
                                                });
                                                stored_sampler = true;
                                            }
                                        }
                                    }
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
                                    is_alpha: Some(*is_alpha),
                                };

                                // First check whether this is a valid excite
                                new_det = det.config.safe_excite_det(&excite);
                                match new_det {
                                    None => {}
                                    Some(d) => {
                                        // Make sure excite is to a perturbative det (not a variational one)
                                        if !self.inds.contains_key(&d) {
                                            // println!("Setup single det_orb sampler when max |H| < eps, orb = {}", i);
                                            det_orbs.push(DetOrbSample {
                                                det,
                                                init: Orbs::Single(i),
                                                is_alpha: Some(*is_alpha),
                                                sum_abs_h: stored_excite.sum_remaining_abs_h
                                                    * single_sample_factor,
                                                sum_h_squared: stored_excite
                                                    .sum_remaining_h_squared
                                                    * single_sample_factor,
                                                sum_abs_hc: det.coeff.abs()
                                                    * stored_excite.sum_remaining_abs_h
                                                    * single_sample_factor,
                                                sum_hc_squared: det.coeff
                                                    * det.coeff
                                                    * stored_excite.sum_remaining_h_squared
                                                    * single_sample_factor,
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

        // Print out factor by which singles would have been oversampled
        println!("Singles encountered during deterministic PT and importance sampling setup oversampled by the following factors:");
        println!(
            "p ~ |Hc| : min: {}, max: {}, mean: {}, std_dev: {}",
            singles_oversampled_abs_hc.min,
            singles_oversampled_abs_hc.max,
            singles_oversampled_abs_hc.mean,
            singles_oversampled_abs_hc.std_dev
        );
        println!(
            "p ~ (Hc)^2 : min: {}, max: {}, mean: {}, std_dev: {}",
            singles_oversampled_hc_squared.min,
            singles_oversampled_hc_squared.max,
            singles_oversampled_hc_squared.mean,
            singles_oversampled_hc_squared.std_dev
        );

        // Now, convert det_orbs to a screened_sampler
        (out_wf, generate_screened_sampler(eps, det_orbs))
    }

    // pub fn approx_matmul_external_dtm_singles(
    //     &self,
    //     global: &Global,
    //     ham: &Ham,
    //     excite_gen: &ExciteGenerator,
    //     eps: f64,
    // ) -> (Wf, ScreenedSampler) {
    //     // Approximate matrix-vector multiplication
    //     // Doubles are done semistochastically with full importance sampling (using eps threshold)
    //     // Singles are done semistochastically in a simpler method: excites from the largest-magnitude
    //     // determinants are done deterministically, and excites from smaller-magnitude determinants
    //     // are done fully stochastically ('importance sampling' but under the assumption that all
    //     // single excitation matrix elements are equal, since they are too time consuming to compute)
    //     // Prepares screened sampler for doubles
    //
    //     // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
    //     // ScreenedSampler object is able to sample according to multiple probabilities
    //     // Note: we can't use the max_sing and max_doub values here, because we have to create the sampler,
    //     // which requires iterating over exciting electrons even if they will all be screened out deterministically
    //
    //     // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
    //     // add to output wf
    //     let mut local_eps: f64;
    //     let mut excite: Excite;
    //     let mut new_det: Option<Config>;
    //
    //     // For making screened sampler
    //     let mut det_orbs: Vec<DetOrbSample> = vec![];
    //
    //     // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
    //     let mut out_wf: Wf = Wf::default();
    //
    //     // Off-diagonal component
    //
    //     // Double excitations
    //     let start_doubs: Instant = Instant::now();
    //     let mut num_doub_excites = 0;
    //     for det in &self.dets {
    //         local_eps = eps / det.coeff.abs();
    //
    //         // Opposite spin
    //         for i in bits(excite_gen.valence & det.config.up) {
    //             for j in bits(excite_gen.valence & det.config.dn) {
    //                 for stored_excite in excite_gen
    //                     .opp_doub_sorted_list
    //                     .get(&Orbs::Double((i, j)))
    //                     .unwrap()
    //                 {
    //                     num_doub_excites += 1;
    //                     excite = Excite {
    //                         init: Orbs::Double((i, j)),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: None,
    //                     };
    //
    //                     new_det = det.config.safe_excite_det(&excite);
    //
    //                     match new_det {
    //                         None => {} // If not a valid excitation, do nothing
    //
    //                         Some(d) => {
    //                             if stored_excite.abs_h >= local_eps {
    //                                 // |H| >= eps: deterministic component
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     // Valid excite: add to H*psi
    //                                     // Compute matrix element and compare its magnitude to eps
    //                                     // TODO: Do this in a cache efficient way
    //                                     out_wf.add_det_with_coeff(
    //                                         det,
    //                                         ham,
    //                                         &excite,
    //                                         d,
    //                                         ham.ham_doub(&det.config, &d) * det.coeff,
    //                                     );
    //                                 }
    //                             } else {
    //                                 // |H| < eps: no more deterministic pieces allowed
    //
    //                                 // Update the screened sampler if this excite is valid and points to
    //                                 // a perturbative det (else, skip until we reach an excite that fits those criteria!)
    //
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     det_orbs.push(DetOrbSample {
    //                                         det,
    //                                         init: Orbs::Double((i, j)),
    //                                         is_alpha: None,
    //                                         sum_abs_h: stored_excite.sum_remaining_abs_h,
    //                                         sum_h_squared: stored_excite.sum_remaining_h_squared,
    //                                         sum_abs_hc: det.coeff.abs()
    //                                             * stored_excite.sum_remaining_abs_h,
    //                                         sum_hc_squared: det.coeff
    //                                             * det.coeff
    //                                             * stored_excite.sum_remaining_h_squared,
    //                                     });
    //                                     break;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //
    //         // Same spin
    //         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
    //             for (i, j) in bit_pairs(excite_gen.valence & *config) {
    //                 for stored_excite in excite_gen
    //                     .same_doub_sorted_list
    //                     .get(&Orbs::Double((i, j)))
    //                     .unwrap()
    //                 {
    //                     num_doub_excites += 1;
    //                     excite = Excite {
    //                         init: Orbs::Double((i, j)),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: Some(*is_alpha),
    //                     };
    //
    //                     new_det = det.config.safe_excite_det(&excite);
    //
    //                     match new_det {
    //                         None => {} // If not a valid excitation, do nothing
    //
    //                         Some(d) => {
    //                             if stored_excite.abs_h >= local_eps {
    //                                 // |H| >= eps: deterministic component
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     // Valid excite: add to H*psi
    //                                     // Compute matrix element and compare its magnitude to eps
    //                                     // TODO: Do this in a cache efficient way
    //                                     out_wf.add_det_with_coeff(
    //                                         det,
    //                                         ham,
    //                                         &excite,
    //                                         d,
    //                                         ham.ham_doub(&det.config, &d) * det.coeff,
    //                                     );
    //                                 }
    //                             } else {
    //                                 // |H| < eps: no more deterministic pieces allowed
    //
    //                                 // Update the screened sampler if this excite is valid and points to
    //                                 // a perturbative det (else, skip until we reach an excite that fits those criteria!)
    //
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     det_orbs.push(DetOrbSample {
    //                                         det,
    //                                         init: Orbs::Double((i, j)),
    //                                         is_alpha: Some(*is_alpha),
    //                                         sum_abs_h: stored_excite.sum_remaining_abs_h,
    //                                         sum_h_squared: stored_excite.sum_remaining_h_squared,
    //                                         sum_abs_hc: det.coeff.abs()
    //                                             * stored_excite.sum_remaining_abs_h,
    //                                         sum_hc_squared: det.coeff
    //                                             * det.coeff
    //                                             * stored_excite.sum_remaining_h_squared,
    //                                     });
    //                                     break;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     println!(
    //         "Time for dtm double excitations: {:?}\n",
    //         start_doubs.elapsed()
    //     );
    //
    //     println!(
    //         "Number of deterministic double excitations: {}",
    //         num_doub_excites
    //     );
    //
    //     // Choose a number of single excitations such that the timing is comparable to the timing
    //     // for the double excitation part
    //     let n_dets_dtm_sings = num_doub_excites
    //         / ((4
    //         * (global.nup - global.norb_core)
    //         * (global.norb - global.nup)
    //         * (global.nup - global.norb_core - 1)) as usize);
    //     println!("Performing single excitations deterministically for the {} largest-magnitude determinants", n_dets_dtm_sings);
    //
    //     // Single excitations
    //     let start_sings: Instant = Instant::now();
    //
    //     // Get n_dets_dtm_sings largest-coefficient dets
    //     // Use a min-heap of the n_dets_dtm_sings largest elements
    //     let mut heap = BinaryHeap::with_capacity(n_dets_dtm_sings);
    //     for (ind, det) in self.dets.iter().enumerate() {
    //         if ind < n_dets_dtm_sings {
    //             heap.push(Reverse(DetByCoeff { det }));
    //         } else if det.coeff.abs() > heap.peek().unwrap().0.det.coeff.abs() {
    //             heap.pop();
    //             heap.push(Reverse(DetByCoeff { det }));
    //         }
    //     }
    //
    //     // Iterating over heap in this way produces dets in reverse order. We can easily invert the
    //     // list if needed
    //     println!(
    //         "The magnitude of the {}th largest-magnitude coefficient is {}",
    //         n_dets_dtm_sings,
    //         heap.peek().unwrap().0.det.coeff.abs()
    //     );
    //     while let Some(wrapped_det) = heap.pop() {
    //         let det = wrapped_det.0.det;
    //
    //         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
    //             for i in bits(excite_gen.valence & *config) {
    //                 for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
    //                 {
    //                     excite = Excite {
    //                         init: Orbs::Single(i),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: Some(*is_alpha),
    //                     };
    //                     // First check whether this is a valid excite
    //                     new_det = det.config.safe_excite_det(&excite);
    //                     if let Some(d) = new_det {
    //                         // Make sure excite is to a perturbative det (not a variational one)
    //                         if !self.inds.contains_key(&d) {
    //                             // Valid excite: add to H*psi
    //                             // Compute matrix element
    //                             out_wf.add_det_with_coeff(
    //                                 det,
    //                                 ham,
    //                                 &excite,
    //                                 d,
    //                                 ham.ham_sing(&det.config, &d) * det.coeff,
    //                             );
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     } // for det in self.dets
    //     println!(
    //         "Time for dtm single excitations: {:?}\n",
    //         start_sings.elapsed()
    //     );
    //
    //     // Now, convert the double excitation det_orbs to a screened_sampler, and return
    //     // the number of dets whose singles were treated deterministically already
    //     // (out_wf, generate_screened_sampler(eps, det_orbs), n_dets_dtm_sings)
    //     (out_wf, generate_screened_sampler(eps, det_orbs))
    // }

    // pub fn approx_matmul_external_dtm_singles(
    //     &self,
    //     global: &Global,
    //     ham: &Ham,
    //     excite_gen: &ExciteGenerator,
    //     eps: f64,
    // ) -> (f64, ScreenedSampler) {
    //     // Approximate matrix-vector multiplication
    //     // Doubles are done semistochastically with full importance sampling (using eps threshold)
    //     // Singles are done deterministically
    //     // Deterministic component is just used to compute an approximation to the ENPT2 energy
    //     // Prepares screened sampler for doubles
    //
    //     // Only returns dets that are "external" to self, i.e., dets not in self (variational space)
    //     // ScreenedSampler object is able to sample according to multiple probabilities
    //     // Note: we can't use the max_sing and max_doub values here, because we have to create the sampler,
    //     // which requires iterating over exciting electrons even if they will all be screened out deterministically
    //
    //     // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
    //     // add to output wf
    //     todo!();
    //     let mut excite: Excite;
    //     let mut new_det: Option<Config>;
    //
    //     // For making screened sampler
    //     let mut det_orbs: Vec<DetOrbSample> = vec![];
    //
    //     // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
    //     let mut out_wf: Wf = Wf::default();
    //
    //     // Off-diagonal component
    //
    //     // Double excitations
    //     let start_doubs: Instant = Instant::now();
    //     let mut num_doub_excites = 0;
    //     for det in &self.dets {
    //
    //         // Opposite spin
    //         for i in bits(excite_gen.valence & det.config.up) {
    //             for j in bits(excite_gen.valence & det.config.dn) {
    //                 for stored_excite in excite_gen
    //                     .opp_doub_sorted_list
    //                     .get(&Orbs::Double((i, j)))
    //                     .unwrap()
    //                 {
    //                     num_doub_excites += 1;
    //                     excite = Excite {
    //                         init: Orbs::Double((i, j)),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: None,
    //                     };
    //
    //                     new_det = det.config.safe_excite_det(&excite);
    //
    //                     match new_det {
    //                         None => {} // If not a valid excitation, do nothing
    //
    //                         Some(d) => {
    //                             if stored_excite.abs_h >= local_eps {
    //                                 // |H| >= eps: deterministic component
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     // Valid excite: add to H*psi
    //                                     // Compute matrix element and compare its magnitude to eps
    //                                     // TODO: Do this in a cache efficient way
    //                                     out_wf.add_det_with_coeff(
    //                                         det,
    //                                         ham,
    //                                         &excite,
    //                                         d,
    //                                         ham.ham_doub(&det.config, &d) * det.coeff,
    //                                     );
    //                                 }
    //                             } else {
    //                                 // |H| < eps: no more deterministic pieces allowed
    //
    //                                 // Update the screened sampler if this excite is valid and points to
    //                                 // a perturbative det (else, skip until we reach an excite that fits those criteria!)
    //
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     det_orbs.push(DetOrbSample {
    //                                         det,
    //                                         init: Orbs::Double((i, j)),
    //                                         is_alpha: None,
    //                                         sum_abs_h: stored_excite.sum_remaining_abs_h,
    //                                         sum_h_squared: stored_excite.sum_remaining_h_squared,
    //                                         sum_abs_hc: det.coeff.abs()
    //                                             * stored_excite.sum_remaining_abs_h,
    //                                         sum_hc_squared: det.coeff
    //                                             * det.coeff
    //                                             * stored_excite.sum_remaining_h_squared,
    //                                     });
    //                                     break;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //
    //         // Same spin
    //         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
    //             for (i, j) in bit_pairs(excite_gen.valence & *config) {
    //                 for stored_excite in excite_gen
    //                     .same_doub_sorted_list
    //                     .get(&Orbs::Double((i, j)))
    //                     .unwrap()
    //                 {
    //                     num_doub_excites += 1;
    //                     excite = Excite {
    //                         init: Orbs::Double((i, j)),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: Some(*is_alpha),
    //                     };
    //
    //                     new_det = det.config.safe_excite_det(&excite);
    //
    //                     match new_det {
    //                         None => {} // If not a valid excitation, do nothing
    //
    //                         Some(d) => {
    //                             if stored_excite.abs_h >= local_eps {
    //                                 // |H| >= eps: deterministic component
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     // Valid excite: add to H*psi
    //                                     // Compute matrix element and compare its magnitude to eps
    //                                     // TODO: Do this in a cache efficient way
    //                                     out_wf.add_det_with_coeff(
    //                                         det,
    //                                         ham,
    //                                         &excite,
    //                                         d,
    //                                         ham.ham_doub(&det.config, &d) * det.coeff,
    //                                     );
    //                                 }
    //                             } else {
    //                                 // |H| < eps: no more deterministic pieces allowed
    //
    //                                 // Update the screened sampler if this excite is valid and points to
    //                                 // a perturbative det (else, skip until we reach an excite that fits those criteria!)
    //
    //                                 // First check whether this is a valid excite
    //                                 // Make sure excite is to a perturbative det (not a variational one)
    //                                 if !self.inds.contains_key(&d) {
    //                                     det_orbs.push(DetOrbSample {
    //                                         det,
    //                                         init: Orbs::Double((i, j)),
    //                                         is_alpha: Some(*is_alpha),
    //                                         sum_abs_h: stored_excite.sum_remaining_abs_h,
    //                                         sum_h_squared: stored_excite.sum_remaining_h_squared,
    //                                         sum_abs_hc: det.coeff.abs()
    //                                             * stored_excite.sum_remaining_abs_h,
    //                                         sum_hc_squared: det.coeff
    //                                             * det.coeff
    //                                             * stored_excite.sum_remaining_h_squared,
    //                                     });
    //                                     break;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     println!(
    //         "Time for dtm double excitations: {:?}\n",
    //         start_doubs.elapsed()
    //     );
    //
    //     println!(
    //         "Number of deterministic double excitations: {}",
    //         num_doub_excites
    //     );
    //
    //     // Choose a number of single excitations such that the timing is comparable to the timing
    //     // for the double excitation part
    //     let n_dets_dtm_sings = num_doub_excites
    //         / ((4
    //             * (global.nup - global.norb_core)
    //             * (global.norb - global.nup)
    //             * (global.nup - global.norb_core - 1)) as usize);
    //     println!("Performing single excitations deterministically for the {} largest-magnitude determinants", n_dets_dtm_sings);
    //
    //     // Single excitations
    //     let start_sings: Instant = Instant::now();
    //
    //     // Get n_dets_dtm_sings largest-coefficient dets
    //     // Use a min-heap of the n_dets_dtm_sings largest elements
    //     let mut heap = BinaryHeap::with_capacity(n_dets_dtm_sings);
    //     for (ind, det) in self.dets.iter().enumerate() {
    //         if ind < n_dets_dtm_sings {
    //             heap.push(Reverse(DetByCoeff { det }));
    //         } else if det.coeff.abs() > heap.peek().unwrap().0.det.coeff.abs() {
    //             heap.pop();
    //             heap.push(Reverse(DetByCoeff { det }));
    //         }
    //     }
    //
    //     // Iterating over heap in this way produces dets in reverse order. We can easily invert the
    //     // list if needed
    //     println!(
    //         "The magnitude of the {}th largest-magnitude coefficient is {}",
    //         n_dets_dtm_sings,
    //         heap.peek().unwrap().0.det.coeff.abs()
    //     );
    //     while let Some(wrapped_det) = heap.pop() {
    //         let det = wrapped_det.0.det;
    //
    //         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
    //             for i in bits(excite_gen.valence & *config) {
    //                 for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
    //                 {
    //                     excite = Excite {
    //                         init: Orbs::Single(i),
    //                         target: stored_excite.target,
    //                         abs_h: stored_excite.abs_h,
    //                         is_alpha: Some(*is_alpha),
    //                     };
    //                     // First check whether this is a valid excite
    //                     new_det = det.config.safe_excite_det(&excite);
    //                     if let Some(d) = new_det {
    //                         // Make sure excite is to a perturbative det (not a variational one)
    //                         if !self.inds.contains_key(&d) {
    //                             // Valid excite: add to H*psi
    //                             // Compute matrix element
    //                             out_wf.add_det_with_coeff(
    //                                 det,
    //                                 ham,
    //                                 &excite,
    //                                 d,
    //                                 ham.ham_sing(&det.config, &d) * det.coeff,
    //                             );
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     } // for det in self.dets
    //     println!(
    //         "Time for dtm single excitations: {:?}\n",
    //         start_sings.elapsed()
    //     );
    //
    //     // Now, convert the double excitation det_orbs to a screened_sampler, and return
    //     // the number of dets whose singles were treated deterministically already
    //     // (out_wf, generate_screened_sampler(eps, det_orbs), n_dets_dtm_sings)
    //     (out_wf, generate_screened_sampler(eps, det_orbs))
    // }

    pub fn approx_matmul_external_no_singles(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> (Wf, ScreenedSampler) {
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
            // let mut sum_abs_h_external: f64 = 0.0;
            // let mut sum_abs_h_discarded: f64 = 0.0;
            // let mut sum_h_sq_external: f64 = 0.0;
            // let mut sum_h_sq_discarded: f64 = 0.0;
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            for i in bits(excite_gen.valence & det.config.up) {
                for j in bits(excite_gen.valence & det.config.dn) {
                    for stored_excite in excite_gen
                        .opp_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            // Update the screened sampler, then break
                            det_orbs.push(DetOrbSample {
                                det,
                                init: Orbs::Double((i, j)),
                                is_alpha: None,
                                sum_abs_h: stored_excite.sum_remaining_abs_h,
                                sum_h_squared: stored_excite.sum_remaining_h_squared,
                                sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                sum_hc_squared: det.coeff
                                    * det.coeff
                                    * stored_excite.sum_remaining_h_squared,
                            });
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None,
                        };
                        new_det = det.config.safe_excite_det(&excite);
                        if let Some(d) = new_det {
                            if !self.inds.contains_key(&d) {
                                // Valid excite: add to H*psi
                                // Compute matrix element and add to H*psi
                                // TODO: Do this in a cache efficient way
                                out_wf.add_det_with_coeff(
                                    det,
                                    ham,
                                    &excite,
                                    d,
                                    ham.ham_doub(&det.config, &d) * det.coeff,
                                );
                            }
                        }
                    }
                }
            }

            // Same spin
            for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                for (i, j) in bit_pairs(excite_gen.valence & *config) {
                    for stored_excite in excite_gen
                        .same_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            // Update the screened sampler, then break
                            det_orbs.push(DetOrbSample {
                                det,
                                init: Orbs::Double((i, j)),
                                is_alpha: Some(*is_alpha),
                                sum_abs_h: stored_excite.sum_remaining_abs_h,
                                sum_h_squared: stored_excite.sum_remaining_h_squared,
                                sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                                sum_hc_squared: det.coeff
                                    * det.coeff
                                    * stored_excite.sum_remaining_h_squared,
                            });
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha),
                        };
                        new_det = det.config.safe_excite_det(&excite);
                        if let Some(d) = new_det {
                            if !self.inds.contains_key(&d) {
                                // Valid excite: add to H*psi
                                // Compute matrix element and add to H*psi
                                // TODO: Do this in a cache efficient way
                                out_wf.add_det_with_coeff(
                                    det,
                                    ham,
                                    &excite,
                                    d,
                                    ham.ham_doub(&det.config, &d) * det.coeff,
                                );
                            }
                        }
                    }
                }
            }

            // Single excitations (no deterministic contribution - just set up for sampling later)
            for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                for i in bits(excite_gen.valence & *config) {
                    let stored_excite =
                        &excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()[0];
                    det_orbs.push(DetOrbSample {
                        det,
                        init: Orbs::Single(i),
                        is_alpha: Some(*is_alpha),
                        sum_abs_h: stored_excite.sum_remaining_abs_h,
                        sum_h_squared: stored_excite.sum_remaining_h_squared,
                        sum_abs_hc: det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                        sum_hc_squared: det.coeff
                            * det.coeff
                            * stored_excite.sum_remaining_h_squared,
                    });
                }
            }
            // println!("Percentage discarded: |H|: {}, H^2: {}", sum_abs_h_discarded / (sum_abs_h_discarded + sum_abs_h_external), sum_h_sq_discarded / (sum_h_sq_discarded + sum_h_sq_external));
        } // for det in self.dets

        // Now, convert det_orbs to a screened_sampler
        (out_wf, generate_screened_sampler(eps, det_orbs))
    }

    pub fn approx_matmul_variational(
        &self,
        input_coeffs: &[f64],
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> Vec<f64> {
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
            out[i_det] = det.diag.unwrap() * input_coeffs[i_det];
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
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: add to H*psi
                                if let Some(ind) = self.inds.get(&d) {
                                    out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: add to H*psi
                                if let Some(ind) = self.inds.get(&d) {
                                    out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                }
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(excite_gen.valence & *config) {
                        for stored_excite in
                            excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: add to H*psi
                                if let Some(ind) = self.inds.get(&d) {
                                    out[*ind] += ham.ham_sing(&det.config, &d) * input_coeffs[i_det]
                                }
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }

    pub fn approx_matmul_off_diag_variational_no_singles(
        &self,
        input_coeffs: &Matrix<
            f64,
            Dynamic,
            Const<1_usize>,
            VecStorage<f64, Dynamic, Const<1_usize>>,
        >,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> Vec<f64> {
        // Approximate matrix-vector multiplication within variational space only
        // WARNING: Uses self only to define and access variational dets; uses input_coeffs as the vector to multiply with
        // instead of wf.dets[:].coeff
        // Uses eps as a cutoff for doubles, skips singles

        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;
        let mut out: Vec<f64> = vec![0.0f64; self.n];

        // Off-diagonal component

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // only add if it is already in variational wf
        for (i_det, det) in enumerate(self.dets.iter()) {
            println!("Coeff: {}", input_coeffs[i_det]);

            local_eps = eps / input_coeffs[i_det].abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(excite_gen.valence & det.config.up) {
                    for j in bits(excite_gen.valence & det.config.dn) {
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: add to H*psi
                                if let Some(ind) = self.inds.get(&d) {
                                    out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: add to H*psi
                                if let Some(ind) = self.inds.get(&d) {
                                    out[*ind] += ham.ham_doub(&det.config, &d) * input_coeffs[i_det]
                                }
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }
}

impl VarWf {
    /// Get new dets
    /// Iterate over all dets; for each, propose all excitations; for each, check if new;
    /// if new, add to wf
    /// Returns true if no new dets (i.e., returns whether already converged)
    pub fn find_new_dets(&mut self, ham: &Ham, excite_gen: &ExciteGenerator) -> bool {
        self.eps = self.eps_iter.next().unwrap();

        println!("Getting new dets with epsilon = {:.1e}", self.eps);
        let new_dets: Wf = self.wf.iterate_excites(ham, excite_gen, self.eps, false);

        // Add all new dets to the wf
        for det in new_dets.dets {
            self.wf.push(det);
        }

        new_dets.n == 0
        // let mut n: usize = 0;
        // let mut new_dets: Wf = Wf::default();
        // for (init_det, excite, det) in dets_excites_and_excited_dets(&self.wf, excite_gen, self.eps) {
        //     if let Single(_) = excite.target {
        //         if (ham.ham_off_diag(&init_det.config, &det, &excite) * init_det.coeff).abs() < self.eps {
        //             continue;
        //         }
        //     }
        //     if !new_dets.inds.contains_key(&det) {
        //         new_dets.push(Det {
        //             config: det,
        //             coeff: 0.0,
        //             diag: Some(init_det.new_diag(ham, &excite))
        //         });
        //     }
        // }
        // for det in new_dets.dets {
        //     // println!("Adding new det: {}", det);
        //     self.wf.push(det);
        //     n += 1;
        // }
        // n == 0
    }
}

impl Wf {
    /// Iterate over excitations using heat-bath cutoff eps
    /// Used internally by both approx_matmul and get_new_dets
    /// If matmul, then return H*psi; else, return a wf composed of new dets
    fn iterate_excites(
        &mut self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
        matmul: bool,
    ) -> Wf {
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
                        for stored_excite in excite_gen
                            .opp_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None,
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                // Valid excite: either add to H*psi or add this det to out
                                if matmul {
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    // out.add_det_with_coeff(det, ham, excite, d,
                                    //                        ham.ham_doub(&det.config, &d) * det.coeff);
                                    todo!()
                                } else {
                                    // If not already in input or output, compute diagonal element and add to output
                                    if !self.inds.contains_key(&d) && !out.inds.contains_key(&d) {
                                        if let Orbs::Double(rs) = excite.target {
                                            out.push(Det {
                                                config: d,
                                                coeff: 0.0,
                                                diag: Some(det.new_diag_opp(ham, (i, j), rs)),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(excite_gen.valence & *config) {
                        for stored_excite in excite_gen
                            .same_doub_sorted_list
                            .get(&Orbs::Double((i, j)))
                            .unwrap()
                        {
                            if stored_excite.abs_h < local_eps {
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha),
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            if let Some(d) = new_det {
                                if matmul {
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    // out.add_det_with_coeff(det, ham, excite, d,
                                    //                       ham.ham_doub(&det.config, &d) * det.coeff);
                                    todo!()
                                } else if !self.inds.contains_key(&d) && !out.inds.contains_key(&d)
                                {
                                    if let Orbs::Double(rs) = excite.target {
                                        out.push(Det {
                                            config: d,
                                            coeff: 0.0,
                                            diag: Some(det.new_diag_same(
                                                ham,
                                                (i, j),
                                                rs,
                                                *is_alpha,
                                            )),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Single excitations
            // Since this is expensive, do it only if wf.eps has reached the target value!
            // if self.eps == global.eps_var {
            //     if excite_gen.max_sing >= local_eps {
            //         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            //             for i in bits(excite_gen.valence & *config) {
            //                 for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
            //                     if stored_excite.abs_h < local_eps { break; }
            //                     excite = Excite {
            //                         init: Orbs::Single(i),
            //                         target: stored_excite.target,
            //                         abs_h: stored_excite.abs_h,
            //                         is_alpha: Some(*is_alpha)
            //                     };
            //                     new_det = det.config.safe_excite_det(&excite);
            //                     match new_det {
            //                         Some(d) => {
            //                             if matmul {
            //                                 // Compute matrix element and add to H*psi
            //                                 // TODO: Do this in a cache efficient way
            //                                 // out.add_det_with_coeff(det, ham, excite, d,
            //                                 //                       ham.ham_sing(&det.config, &d) * det.coeff);
            //                                 todo!()
            //                             } else {
            //                                 if !self.inds.contains_key(&d) {
            //                                     if !out.inds.contains_key(&d) {
            //                                         match excite.target {
            //                                             Orbs::Single(r) => {
            //                                                 // Compute whether single excitation actually exceeds eps!
            //                                                 if ham.ham_sing(&det.config, &d).abs() > local_eps {
            //                                                     out.push(
            //                                                         Det {
            //                                                             config: d,
            //                                                             coeff: 0.0,
            //                                                             diag: det.new_diag_sing(ham, i, r, *is_alpha)
            //                                                         }
            //                                                     );
            //                                                 }
            //                                             }
            //                                             _ => {}
            //                                         }
            //                                     }
            //                                 }
            //                             }
            //                         }
            //                         None => {}
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }
        } // for det in self.dets

        out
    }
}

impl VarWf {
    pub fn update_n_stored_h(&mut self, n: usize) {
        // Make this setter explicit because we really don't want to accidentally change it
        self.n_stored_h = n;
    }

    pub fn n_stored_h(&self) -> usize {
        self.n_stored_h
    }

    /// Create new sparse Hamiltonian, set the diagonal elements to the ones already stored in wf
    /// Don't update self.n_stored_h yet, because we haven't computed the off-diagonal elements yet
    pub fn new_sparse_ham(&mut self) {
        self.sparse_ham = SparseMatUpperTri {
            n: self.wf.n,
            diag: vec![0.0; self.wf.n],
            nnz: vec![0; self.wf.n],
            off_diag: vec![Vec::with_capacity(100); self.wf.n],
        };
        for i in 0..self.wf.n {
            self.sparse_ham.diag[i] = self.wf.dets[i].diag.unwrap();
        }
    }

    /// Just create empty rows to fill up the dimension to new size wf.n
    /// Also, fill up all of the new diagonal elements
    /// Don't update self.n_stored_h yet, because we haven't computed the off-diagonal elements yet
    pub fn expand_sparse_ham_rows(&mut self) {
        println!(
            "Expanding variational H from size {} to size {}",
            self.sparse_ham.n, self.wf.n
        );
        self.sparse_ham
            .diag
            .append(&mut vec![0.0; self.wf.n - self.sparse_ham.n]);
        self.sparse_ham
            .nnz
            .append(&mut vec![0; self.wf.n - self.sparse_ham.n]);
        self.sparse_ham.off_diag.append(&mut vec![
            Vec::with_capacity(100);
            self.wf.n - self.sparse_ham.n
        ]);
        for i in self.sparse_ham.n..self.wf.n {
            self.sparse_ham.diag[i] = self.wf.dets[i].diag.unwrap();
        }
        self.sparse_ham.n = self.wf.n;
    }
}

/// Initialize variational wf to the HF det (only needs to be called once)
pub fn init_var_wf(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator) -> VarWf {
    let mut wf: VarWf = VarWf::default();
    wf.n_states = global.n_states;
    wf.converged = false;
    wf.wf.n = 1;
    wf.update_n_stored_h(0); // No stored H yet
    let mut hf = Det {
        config: Config {
            up: ((1u128 << global.nup) - 1),
            dn: ((1u128 << global.ndn) - 1),
        },
        coeff: 1.0,
        diag: None,
    };
    let h: f64 = ham.ham_diag(&hf.config);
    hf.diag = Some(h);
    wf.wf.inds = HashMap::new();
    wf.wf.inds.insert(hf.config, 0);
    wf.wf.dets.push(hf);
    wf.wf.energy = wf.wf.dets[0].diag.unwrap();
    wf.eps_iter = init_eps(&wf.wf, global, excite_gen);
    wf
}
