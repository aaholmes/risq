//! Determinant data structure
//! Includes functions to generate an excited det, and compute its diagonal element
//! quickly from the initial det's diagonal element

// Using built-in std functionality instead of unstable feature detection
// use std::detect::__is_feature_detected::popcnt;
use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::ham::Ham;
use crate::utils::bits::{bit_pairs, bits, btest, ibclr, ibset};
use crate::wf::Wf;
use std::hash::{Hash, Hasher};

/// Binary representation of the up- and down-spin orbital occupancies of a Slater determinant
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Config {
    pub up: u128,
    pub dn: u128,
}

/// Full determinant struct. Contains the determinant's `Config`, as well as its coefficient `coeff`
/// and diagonal Hamiltonian matrix element `diag`
#[derive(Clone, Copy, Debug)]
pub struct Det {
    pub config: Config,
    pub coeff: f64,
    pub diag: Option<f64>, // None if it hasn't been computed yet (since computing it is expensive)
}

// These functions are needed for Alias sampling
impl PartialEq for Det {
    // Just compare configs
    fn eq(&self, other: &Self) -> bool {
        self.config.up == other.config.up && self.config.dn == other.config.dn
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
    // Return spin-flipped counterpart
    pub fn flip(&self) -> Config {
        Config{
            up: self.dn,
            dn: self.up
        }
    }

    // Return whether connected to another config by up to a double excitation
    pub fn is_connected(&self, other: &Config) -> bool {
        let mut diff = 0;
        for i in bits(self.up & !other.up) {
            diff += 1;
            if diff > 2 {
                return false;
            }
        }
        for i in bits(self.up & !other.up) {
            diff += 1;
            if diff > 2 {
                return false;
            }
        }
        true
    }

    pub fn is_valid_stored(&self, is_alpha: &Option<bool>, excite: &StoredExcite) -> bool {
        match excite.target {
            Orbs::Double((r, s)) => match is_alpha {
                None => !(btest(self.up, r) || btest(self.dn, s)),
                Some(is_a) => {
                    if *is_a {
                        !(btest(self.up, r) || btest(self.up, s))
                    } else {
                        !(btest(self.dn, r) || btest(self.dn, s))
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

    pub fn is_valid(&self, excite: &Excite) -> bool {
        match excite.target {
            Orbs::Double((r, s)) => match excite.is_alpha {
                None => {
                    if btest(self.up, r) {
                        return false;
                    }
                    !btest(self.dn, s)
                }
                Some(is_a) => {
                    if is_a {
                        if btest(self.up, r) {
                            return false;
                        }
                        !btest(self.up, s)
                    } else {
                        if btest(self.dn, r) {
                            return false;
                        }
                        !btest(self.dn, s)
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

    pub fn excite_det(&self, excite: &Excite) -> Config {
        match (excite.init, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => match excite.is_alpha {
                None => Config {
                    up: ibset(ibclr(self.up, p), r),
                    dn: ibset(ibclr(self.dn, q), s),
                },
                Some(is_alpha) => {
                    if is_alpha {
                        Config {
                            up: ibset(ibset(ibclr(ibclr(self.up, p), q), r), s),
                            dn: self.dn,
                        }
                    } else {
                        Config {
                            up: self.up,
                            dn: ibset(ibset(ibclr(ibclr(self.dn, p), q), r), s),
                        }
                    }
                }
            },
            (Orbs::Single(p), Orbs::Single(r)) => {
                if excite.is_alpha.unwrap() {
                    Config {
                        up: ibset(ibclr(self.up, p), r),
                        dn: self.dn,
                    }
                } else {
                    Config {
                        up: self.up,
                        dn: ibset(ibclr(self.dn, p), r),
                    }
                }
            }
            _ => Config {
                up: self.up,
                dn: self.dn,
            }, // Because could be (single, double), etc
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

    /// Apply a stored excite to a given det, compute new coefficient, but don't compute diagonal
    /// element yet, since it may not be needed
    pub fn apply_excite(
        self,
        is_alpha: Option<bool>,
        init_orbs: &Orbs,
        excite: &StoredExcite,
    ) -> Config {
        match (init_orbs, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => match is_alpha {
                None => Config {
                    up: ibset(ibclr(self.up, *p), r),
                    dn: ibset(ibclr(self.dn, *q), s),
                },
                Some(is_a) => {
                    if is_a {
                        Config {
                            up: ibset(ibset(ibclr(ibclr(self.up, *p), *q), r), s),
                            dn: self.dn,
                        }
                    } else {
                        Config {
                            up: self.up,
                            dn: ibset(ibset(ibclr(ibclr(self.dn, *p), *q), r), s),
                        }
                    }
                }
            },
            (Orbs::Single(p), Orbs::Single(r)) => {
                if is_alpha.unwrap() {
                    Config {
                        up: ibset(ibclr(self.up, *p), r),
                        dn: self.dn,
                    }
                } else {
                    Config {
                        up: self.up,
                        dn: ibset(ibclr(self.dn, *p), r),
                    }
                }
            }
            _ => {
                panic!("Attempted apply_excite with an invalid excite!")
            }
        }
    }
}

impl Det {
    /// Computes the diagonal Hamiltonian element for a determinant generated by applying
    /// a pre-stored excitation (`excite`) to `self`.
    ///
    /// Uses the diagonal element of `self` (`self.diag`) and efficiently updates it
    /// based on the excitation details (`is_alpha`, `init_orbs`, `excite.target`)
    /// by calling the appropriate `new_diag_*` helper function.
    /// Assumes `self.diag` is `Some`.
    pub fn new_diag_stored(
        &self,
        ham: &Ham,
        is_alpha: Option<bool>,
        init_orbs: Orbs,
        excite: &StoredExcite,
    ) -> f64 {
        match (init_orbs, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => match is_alpha {
                None => self.new_diag_opp(&ham, (p, q), (r, s)),
                Some(is_a) => self.new_diag_same(&ham, (p, q), (r, s), is_a),
            },
            (Orbs::Single(p), Orbs::Single(r)) => self.new_diag_sing(&ham, p, r, is_alpha.unwrap()),
            _ => 0.0, // Because could be (single, double), etc
        }
    }

    /// Computes the diagonal Hamiltonian element for a determinant generated by applying
    /// a full excitation (`excite`) to `self`.
    ///
    /// Similar to `new_diag_stored`, but takes a complete `Excite` struct.
    /// Uses the diagonal element of `self` (`self.diag`) and efficiently updates it.
    /// Assumes `self.diag` is `Some`.
    pub fn new_diag(&self, ham: &Ham, excite: &Excite) -> f64 {
        match (excite.init, excite.target) {
            (Orbs::Double((p, q)), Orbs::Double((r, s))) => match excite.is_alpha {
                None => self.new_diag_opp(&ham, (p, q), (r, s)),
                Some(is_alpha) => self.new_diag_same(&ham, (p, q), (r, s), is_alpha),
            },
            (Orbs::Single(p), Orbs::Single(r)) => {
                self.new_diag_sing(&ham, p, r, excite.is_alpha.unwrap())
            }
            _ => 0.0, // Because could be (single, double), etc
        }
    }

    // Backend

    /// Efficiently computes the diagonal element H_kk for a determinant k generated by an
    /// *opposite-spin* double excitation (p_alpha, q_beta -> r_alpha, s_beta) from determinant `self` (i).
    ///
    /// Uses H_ii (stored in `self.diag`) and adds/subtracts the changes in one- and two-body
    /// terms based on the orbitals involved in the excitation (p, q, r, s).
    /// This avoids the full O(N^2) calculation for H_kk. Requires O(N) work due to loops over occupied orbitals.
    /// Assumes `self.diag` is `Some`.
    pub(crate) fn new_diag_opp(&self, ham: &Ham, init: (i32, i32), target: (i32, i32)) -> f64 {

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag = self.one_body(ham, init, target);

        // O(1) Two-body direct part: E += direct(r,s) - direct(p,q)
        new_diag += ham.direct(target.0, target.1, target.0, target.1)
            - ham.direct(init.0, init.1, init.0, init.1);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in (p,q)} direct(i,r) + direct(i,s) - direct(i,p) - direct(i,q)
        for i in bits(self.config.up) {
            if i == init.0 {
                continue;
            }
            new_diag += ham.direct_plus_exchange(i, target.0, i, target.0)
                - ham.direct_plus_exchange(i, init.0, i, init.0);
            new_diag += ham.direct(i, target.1, i, target.1) - ham.direct(i, init.1, i, init.1);
        }
        for i in bits(self.config.dn) {
            if i == init.1 {
                continue;
            }
            new_diag += ham.direct_plus_exchange(i, target.1, i, target.1)
                - ham.direct_plus_exchange(i, init.1, i, init.1);
            new_diag += ham.direct(i, target.0, i, target.0) - ham.direct(i, init.0, i, init.0);
        }

        new_diag
    }

    /// Helper function to calculate the O(1) part of the diagonal energy update for doubles.
    /// Calculates `self.diag + h_rr + h_ss - h_pp - h_qq`. Assumes `self.diag` is `Some`.
    fn one_body(&self, ham: &Ham, init: (i32, i32), target: (i32, i32)) -> f64 {
        self.diag.unwrap() + ham.one_body(target.0, target.0) + ham.one_body(target.1, target.1)
            - ham.one_body(init.0, init.0)
            - ham.one_body(init.1, init.1)
    }

    /// Efficiently computes the diagonal element H_kk for a determinant k generated by a
    /// *same-spin* double excitation (p,q -> r,s, both alpha or both beta) from determinant `self` (i).
    ///
    /// Uses H_ii (stored in `self.diag`) and adds/subtracts the changes in one- and two-body
    /// terms based on the orbitals involved (p, q, r, s) and the spin (`is_alpha`).
    /// Avoids the full O(N^2) calculation. Requires O(N) work.
    /// Assumes `self.diag` is `Some`.
    pub(crate) fn new_diag_same(
        &self,
        ham: &Ham,
        init: (i32, i32),
        target: (i32, i32),
        is_alpha: bool,
    ) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.one_body(ham, init, target);

        // O(1) Two-body direct_and_exchange part: E += direct_and_exchange(r,s) - direct_and_exchange(p,q)
        new_diag += ham.direct_plus_exchange(target.0, target.1, target.0, target.1)
            - ham.direct_plus_exchange(init.0, init.1, init.0, init.1);

        // O(N) Two-body direct_and_exchange part: E += sum_{i in occ. but not in (p,q)} direct_and_exchange(i,r) + direct_and_exchange(i,s) - direct_and_exchange(i,p) - direct_and_exchange(i,q)
        if is_alpha {
            for i in bits(self.config.up) {
                if i == init.0 || i == init.1 {
                    continue;
                }
                new_diag += Self::direct_plus_exchange_this_occ_orb(ham, init, target, i);
            }
            for i in bits(self.config.dn) {
                new_diag += Self::direct_this_occ_orb(ham, init, target, i);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == init.0 || i == init.1 {
                    continue;
                }
                new_diag += Self::direct_plus_exchange_this_occ_orb(ham, init, target, i);
            }
            for i in bits(self.config.up) {
                new_diag += Self::direct_this_occ_orb(ham, init, target, i);
            }
        }

        new_diag
    }

    /// Helper for `new_diag_same`: Calculates change in direct terms for opposite-spin interactions.
    /// Computes `(ir|ir) + (is|is) - (ip|ip) - (iq|iq)`.
    fn direct_this_occ_orb(ham: &Ham, init: (i32, i32), target: (i32, i32), i: i32) -> f64 {
        ham.direct(i, target.0, i, target.0) + ham.direct(i, target.1, i, target.1)
            - ham.direct(i, init.0, i, init.0)
            - ham.direct(i, init.1, i, init.1)
    }

    /// Helper for `new_diag_same`: Calculates change in direct+exchange terms for same-spin interactions.
    /// Computes `[(ir|ir)-(ir|ri)] + [(is|is)-(is|si)] - [(ip|ip)-(ip|pi)] - [(iq|iq)-(iq|qi)]`.
    fn direct_plus_exchange_this_occ_orb(
        ham: &Ham,
        init: (i32, i32),
        target: (i32, i32),
        i: i32,
    ) -> f64 {
        ham.direct_plus_exchange(i, target.0, i, target.0)
            + ham.direct_plus_exchange(i, target.1, i, target.1)
            - ham.direct_plus_exchange(i, init.0, i, init.0)
            - ham.direct_plus_exchange(i, init.1, i, init.1)
    }

    /// Efficiently computes the diagonal element H_kk for a determinant k generated by a
    /// *single* excitation (p -> r) from determinant `self` (i).
    ///
    /// Uses H_ii (stored in `self.diag`) and adds/subtracts the changes in one- and two-body
    /// terms based on the orbitals involved (p, r) and the spin (`is_alpha`).
    /// Avoids the full O(N^2) calculation. Requires O(N) work.
    /// Assumes `self.diag` is `Some`.
    pub(crate) fn new_diag_sing(&self, ham: &Ham, init: i32, target: i32, is_alpha: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) - h(p)
        let mut new_diag: f64 =
            self.diag.unwrap() + ham.one_body(target, target) - ham.one_body(init, init);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in p} direct(i,r) - direct(i,p)
        if is_alpha {
            for i in bits(self.config.up) {
                if i == init {
                    continue;
                }
                new_diag += ham.direct_plus_exchange(i, target, i, target)
                    - ham.direct_plus_exchange(i, init, i, init);
            }
            for i in bits(self.config.dn) {
                new_diag += ham.direct(i, target, i, target) - ham.direct(i, init, i, init);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == init {
                    continue;
                }
                new_diag += ham.direct_plus_exchange(i, target, i, target)
                    - ham.direct_plus_exchange(i, init, i, init);
            }
            for i in bits(self.config.up) {
                new_diag += ham.direct(i, target, i, target) - ham.direct(i, init, i, init);
            }
        }

        new_diag
    }

    /// Computes H*|psi> for a single determinant `self`, focusing on external contributions
    /// and computing diagonal elements for newly generated determinants.
    ///
    /// Similar to `Wf::approx_matmul_external_dtm_only` but operates on a single `Det` (`self`)
    /// instead of a full `Wf`. It iterates through excitations from `self`, checks if the
    /// resulting determinant `d` is external (not in `var_wf`), and if the estimated
    /// contribution `|H_di * c_i|` exceeds `eps`. If so, it adds `d` to `out_wf`,
    /// computing its diagonal element efficiently using `new_diag_stored`.
    pub fn approx_matmul_external_dtm_only_compute_diags(
        &self,
        var_wf: &Wf, // The variational wavefunction (to check if generated dets are external)
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,   // Screening threshold
    ) -> Wf {       // Returns a Wf containing the external determinants generated from self
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
            for i in bits(excite_gen.valence & self.config.up) {
                for j in bits(excite_gen.valence & self.config.dn) {
                    for stored_excite in excite_gen
                        .opp_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: None,
                        };
                        new_det = self.config.safe_excite_det(&excite);
                        self.add_det_if_valid_excite(
                            var_wf,
                            ham,
                            &mut excite,
                            &mut new_det,
                            &mut out_wf,
                        )
                    }
                }
            }
        }

        // Same spin
        if excite_gen.max_same_doub >= local_eps {
            for (config, is_alpha) in &[(self.config.up, true), (self.config.dn, false)] {
                for (i, j) in bit_pairs(excite_gen.valence & *config) {
                    for stored_excite in excite_gen
                        .same_doub_sorted_list
                        .get(&Orbs::Double((i, j)))
                        .unwrap()
                    {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Double((i, j)),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha),
                        };
                        new_det = self.config.safe_excite_det(&excite);
                        self.add_det_if_valid_excite(
                            var_wf,
                            ham,
                            &mut excite,
                            &mut new_det,
                            &mut out_wf,
                        )
                    }
                }
            }
        }

        // Single excitations
        if excite_gen.max_sing >= local_eps {
            for (config, is_alpha) in &[(self.config.up, true), (self.config.dn, false)] {
                for i in bits(excite_gen.valence & *config) {
                    for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap()
                    {
                        if stored_excite.abs_h < local_eps {
                            // No more deterministic excitations will meet the eps cutoff
                            break;
                        }
                        excite = Excite {
                            init: Orbs::Single(i),
                            target: stored_excite.target,
                            abs_h: stored_excite.abs_h,
                            is_alpha: Some(*is_alpha),
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
                                        out_wf.add_det_with_coeff(
                                            self,
                                            ham,
                                            &excite,
                                            d,
                                            sing * self.coeff,
                                        );
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

    fn add_det_if_valid_excite(
        &self,
        var_wf: &Wf,
        ham: &Ham,
        excite: &mut Excite,
        new_det: &Option<Config>,
        out_wf: &mut Wf,
    ) {
        match new_det {
            Some(d) => {
                if !var_wf.inds.contains_key(&d) {
                    // Valid excite: add to H*psi
                    // Compute matrix element and add to H*psi
                    // TODO: Do this in a cache efficient way
                    out_wf.add_det_with_coeff(
                        self,
                        ham,
                        &excite,
                        *d,
                        ham.ham_doub(&self.config, &d) * self.coeff,
                    );
                }
            }
            None => {}
        }
    }
}
