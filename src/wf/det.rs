// Determinant data structure:
// Includes functions to generate an excited det, and compute its diagonal element
// quickly from the initial det's diagonal element

use crate::ham::Ham;
use crate::utils::bits::{bits, btest, ibset, ibclr, bit_pairs};
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::excite::init::ExciteGenerator;
use crate::wf::Wf;
use crate::stoch::DetOrbSample;
use std::hash::{Hash, Hasher};

// Configuration: up and dn spin occupation bitstrings
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Config {
    pub up: u128,
    pub dn: u128,
}

// Determinant - configuration with coefficient, diagonal H element
#[derive(Clone, Copy, Debug)]
pub struct Det {
    pub config: Config,
    pub coeff: f64,
    pub diag: f64,
}

// These functions are needed for Alias sampling
impl PartialEq for Det {
    // Just compare configs
    fn eq(&self, other: &Self) -> bool {
        self.config.up == other.config.up
            && self.config.dn == other.config.dn
    }
}
impl Eq for Det {}

impl Hash for Det {
    // Hash using only the config, orbs, and is_alpha
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.config.hash(state);
    }
}


// Public functions

impl Config {
    pub fn is_valid(&self, excite: &Excite) -> bool {
        match excite.target {
            Orbs::Double((r, s)) => {
                match excite.is_alpha {
                    None => {
                        if btest(self.up, r) { return false; }
                        !btest(self.dn, s)
                    }
                    Some(is_alpha) => {
                        if is_alpha {
                            if btest(self.up, r) { return false; }
                            !btest(self.up, s)
                        } else {
                            if btest(self.dn, r) { return false; }
                            !btest(self.dn, s)
                        }
                    }
                }
            },
            Orbs::Single(r) => {
                if excite.is_alpha.unwrap() {
                    !btest(self.up, r)
                } else {
                    !btest(self.dn, r)
                }
            }
        }
    }

    pub fn is_valid_stored(&self, excite: &StoredExcite, is_alpha: Option<bool>) -> bool {
        match excite.target {
            Orbs::Double((r, s)) => {
                match is_alpha {
                    None => {
                        if btest(self.up, r) { return false; }
                        !btest(self.dn, s)
                    }
                    Some(is_a) => {
                        if is_a {
                            if btest(self.up, r) { return false; }
                            !btest(self.up, s)
                        } else {
                            if btest(self.dn, r) { return false; }
                            !btest(self.dn, s)
                        }
                    }
                }
            },
            Orbs::Single(r) => {
                if is_alpha.unwrap() {
                    !btest(self.up, r)
                } else {
                    !btest(self.dn, r)
                }
            }
        }
    }

    pub fn excite_det(&self, excite: &Excite) -> Config {
        match (excite.init, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => {
                match excite.is_alpha {
                    None => {
                        Config {
                            up: ibset(ibclr(self.up, p), r),
                            dn: ibset(ibclr(self.dn, q), s)
                        }
                    },
                    Some(is_alpha) => {
                        if is_alpha {
                            Config {
                                up: ibset(ibset(ibclr(ibclr(self.up, p), q), r), s),
                                dn: self.dn
                            }
                        } else {
                            Config {
                                up: self.up,
                                dn: ibset(ibset(ibclr(ibclr(self.dn, p), q), r), s)
                            }
                        }
                    }
                }
            },
            (Orbs::Single(p), Orbs::Single(r)) => {
                if excite.is_alpha.unwrap() {
                    Config {
                        up: ibset(ibclr(self.up, p), r),
                        dn: self.dn
                    }
                } else {
                    Config {
                        up: self.up,
                        dn: ibset(ibclr(self.dn, p), r)
                    }
                }
            },
            _ => Config {
                up: self.up,
                dn: self.dn
            }// Because could be (single, double), etc
        }
    }

    pub fn safe_excite_det(&self, excite: &Excite) -> Option<Config> {
        // Applies excite to det, checking if it's valid first
        if self.is_valid(excite) {
            Some(self.excite_det(excite))
        } else {
            None
        }
    }
}

impl Det {
    pub fn new_diag(&self, ham: &Ham, excite: &Excite) -> f64 {
        match (excite.init, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => {
                match excite.is_alpha {
                    None => {
                        self.new_diag_opp(&ham, (p, q), (r, s))
                    },
                    Some(is_alpha) => {
                        self.new_diag_same(&ham, (p, q), (r, s), is_alpha)
                    }
                }
            },
            (Orbs::Single(p), Orbs::Single(r)) => {
                self.new_diag_sing(&ham, p, r, excite.is_alpha.unwrap())
            },
            _ => 0.0 // Because could be (single, double), etc
        }
    }


// Backend

    pub(crate) fn new_diag_opp(&self, ham: &Ham, init: (i32, i32), target: (i32, i32)) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(target.0, target.0)
            + ham.one_body(target.1, target.1)
            - ham.one_body(init.0, init.0)
            - ham.one_body(init.1, init.1);

        // O(1) Two-body direct part: E += direct(r,s) - direct(p,q)
        new_diag += ham.direct(target.0, target.1, target.0, target.1)
            - ham.direct(init.0, init.1, init.0, init.1);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in (p,q)} direct(i,r) + direct(i,s) - direct(i,p) - direct(i,q)
        for i in bits(self.config.up) {
            if i == init.0 { continue; }
            new_diag += ham.direct_plus_exchange(i, target.0, i, target.0)
                - ham.direct_plus_exchange(i, init.0, i, init.0);
            new_diag += ham.direct(i, target.1, i, target.1)
                - ham.direct(i, init.1, i, init.1);
        }
        for i in bits(self.config.dn) {
            if i == init.1 { continue; }
            new_diag += ham.direct_plus_exchange(i, target.1, i, target.1)
                - ham.direct_plus_exchange(i, init.1, i, init.1);
            new_diag += ham.direct(i, target.0, i, target.0)
                - ham.direct(i, init.0, i, init.0);
        }

        new_diag
    }

    pub(crate) fn new_diag_same(&self, ham: &Ham, init: (i32, i32), target: (i32, i32), is_alpha: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(target.0, target.0)
            + ham.one_body(target.1, target.1)
            - ham.one_body(init.0, init.0)
            - ham.one_body(init.1, init.1);

        // O(1) Two-body direct_and_exchange part: E += direct_and_exchange(r,s) - direct_and_exchange(p,q)
        new_diag += ham.direct_plus_exchange(target.0, target.1, target.0, target.1)
            - ham.direct_plus_exchange(init.0, init.1, init.0, init.1);

        // O(N) Two-body direct_and_exchange part: E += sum_{i in occ. but not in (p,q)} direct_and_exchange(i,r) + direct_and_exchange(i,s) - direct_and_exchange(i,p) - direct_and_exchange(i,q)
        if is_alpha {
            for i in bits(self.config.up) {
                if i == init.0 || i == init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, target.0, i, target.0)
                    + ham.direct_plus_exchange(i, target.1, i, target.1)
                    - ham.direct_plus_exchange(i, init.0, i, init.0)
                    - ham.direct_plus_exchange(i, init.1, i, init.1);
            }
            for i in bits(self.config.dn) {
                new_diag += ham.direct(i, target.0, i, target.0)
                    + ham.direct(i, target.1, i, target.1)
                    - ham.direct(i, init.0, i, init.0)
                    - ham.direct(i, init.1, i, init.1);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == init.0 || i == init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, target.0, i, target.0)
                    + ham.direct_plus_exchange(i, target.1, i, target.1)
                    - ham.direct_plus_exchange(i, init.0, i, init.0)
                    - ham.direct_plus_exchange(i, init.1, i, init.1);
            }
            for i in bits(self.config.up) {
                new_diag += ham.direct(i, target.0, i, target.0)
                    + ham.direct(i, target.1, i, target.1)
                    - ham.direct(i, init.0, i, init.0)
                    - ham.direct(i, init.1, i, init.1);
            }
        }

        new_diag
    }

    pub(crate) fn new_diag_sing(&self, ham: &Ham, init: i32, target: i32, is_alpha: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) - h(p)
        let mut new_diag: f64 = self.diag
            + ham.one_body(target, target)
            - ham.one_body(init, init);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in p} direct(i,r) - direct(i,p)
        if is_alpha {
            for i in bits(self.config.up) {
                if i == init { continue; }
                new_diag += ham.direct_plus_exchange(i, target, i, target)
                    - ham.direct_plus_exchange(i, init, i, init);
            }
            for i in bits(self.config.dn) {
                new_diag += ham.direct(i, target, i, target)
                    - ham.direct(i, init, i, init);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == init { continue; }
                new_diag += ham.direct_plus_exchange(i, target, i, target)
                    - ham.direct_plus_exchange(i, init, i, init);
            }
            for i in bits(self.config.up) {
                new_diag += ham.direct(i, target, i, target)
                    - ham.direct(i, init, i, init);
            }
        }

        new_diag
    }

    pub fn approx_matmul_external_dtm_only_compute_diags(&self, var_wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> Wf {
        // Approximate matrix-vector multiplication
        // Uses eps as a cutoff for both singles and doubles, as in SHCI (but faster of course)
        // Only returns dets that are "external" to self, i.e., dets not in self (variational space)

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        let local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // Diagonal component - none because this is 'external' to the current wf (i.e., perturbative space rather than variational space)
        let mut out_wf: Wf = Wf::default();

        // Off-diagonal component
            local_eps = eps / self.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(self.config.up) {
                    for j in bits(self.config.dn) {
                        for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = self.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !var_wf.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(self, ham, &excite, d, ham.ham_doub(&self.config, &d) * self.coeff);
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
                for (config, is_alpha) in &[(self.config.up, true), (self.config.dn, false)] {
                    for (i, j) in bit_pairs(*config) {
                        for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = self.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !var_wf.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        out_wf.add_det_with_coeff(self, ham, &excite, d, ham.ham_doub(&self.config, &d) * self.coeff);
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
                for (config, is_alpha) in &[(self.config.up, true), (self.config.dn, false)] {
                    for i in bits(*config) {
                        for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps {
                                // No more deterministic excitations will meet the eps cutoff
                                break;
                            }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = self.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if !var_wf.inds.contains_key(&d) {
                                        // Valid excite: add to H*psi
                                        // Compute matrix element and add to H*psi
                                        let sing: f64 = ham.ham_sing(&self.config, &d);
                                        if sing.abs() >= local_eps {
                                            // TODO: Do this in a cache efficient way
                                            out_wf.add_det_with_coeff(self, ham, &excite, d, sing * self.coeff);
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        out_wf
    }

}
