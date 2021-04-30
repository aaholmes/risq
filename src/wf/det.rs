// Determinant data structure:
// Includes functions to generate an excited det, and compute its diagonal element
// quickly from the initial det's diagonal element

use crate::ham::Ham;
use crate::utils::bits::{bits, btest, ibset, ibclr};
use crate::excite::{Doub, Sing, Excite, Orbs};

// Configuration: up and dn spin occupation bitstrings
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Config {
    pub up: u128,
    pub dn: u128,
}

// Determinant - configuration with coefficient, diagonal H element
pub struct Det {
    pub config: Config,
    pub coeff: f64,
    pub diag: f64,
}


// Public functions

impl Config {
    pub fn is_valid(&self, excite: &Excite) -> bool {
        // Returns whether this excitation is a valid excite from this det
        // Assumes init electrons are already filled
        match excite {
            Excite::Double(doub) => {
                match doub.is_alpha {
                    None => self.is_valid_opp(doub),
                    Some(is_alpha) => self.is_valid_same(doub, is_alpha)
                }
            },
            Excite::Single(sing) => {
                self.is_valid_sing(sing, sing.is_alpha.unwrap())
            }
        }
    }

    pub fn excite_det(&self, excite: &Excite) -> Config {
        // Applies excite to det, assuming it's a valid excite
        match excite {
            Excite::Double(doub) => {
                match doub.is_alpha {
                    None => self.excite_det_opp(doub),
                    Some(is_alpha) => self.excite_det_same(doub, is_alpha)
                }
            },
            Excite::Single(sing) => {
                self.excite_det_sing(sing, sing.is_alpha.unwrap())
            }
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
        match excite {
            Excite::Double(doub) => {
                match doub.is_alpha {
                    None => self.new_diag_opp(&ham, doub),
                    Some(is_alpha) => self.new_diag_same(&ham, doub, is_alpha)
                }
            },
            Excite::Single(sing) => {
                self.new_diag_sing(&ham, sing, sing.is_alpha.unwrap())
            }
        }
    }
}


// Backend

impl Config {
    fn is_valid_opp(&self, doub: &Doub) -> bool {
        // Returns whether this double excitation is a valid doub from this det
        // Assumes that init electrons are filled
        if btest(self.up, doub.target.0) { return false; }
        if btest(self.dn, doub.target.1) { return false; }
        true
    }

    fn is_valid_same(&self, doub: &Doub, is_alpha: bool) -> bool {
        if (is_alpha) {
            if btest(self.up, doub.target.0) { return false; }
            if btest(self.up, doub.target.1) { return false; }
        } else {
            if btest(self.dn, doub.target.0) { return false; }
            if btest(self.dn, doub.target.1) { return false; }
        }
        true
    }

    fn is_valid_sing(&self, sing: &Sing, is_alpha: bool) -> bool {
        if (is_alpha) {
            if btest(self.up, &sing.target as i32) { return false; }
        } else {
            if btest(self.dn, &sing.target as i32) { return false; }
        }
        true
    }

    fn excite_det_opp(&self, doub: &Doub) -> Config {
        // Excite det using double excitation
        Config {
            up: ibset(ibclr(self.up, doub.init.0), doub.target.0),
            dn: ibset(ibclr(self.dn, doub.init.1), doub.target.1)
        }
    }

    fn excite_det_same(&self, doub: &Doub, is_up: bool) -> Config {
        // Excite det using double excitation
        // is_up is true if a same-spin up; false if a same-spin dn
        if is_up {
            Config {
                up: ibset(ibset(ibclr(ibclr(self.up, doub.init.0), doub.init.1), doub.target.0), doub.target.1),
                dn: self.dn
            }
        } else {
            Config {
                up: self.up,
                dn: ibset(ibset(ibclr(ibclr(self.dn, doub.init.0), doub.init.1), doub.target.0), doub.target.1)
            }
        }
    }

    fn excite_det_sing(&self, sing: &Sing, is_up: bool) -> Config {
        // Excite det using single excitation
        // is_up is true if a single up; false if a single dn
        if is_up {
            Config {
                up: ibset(ibclr(self.up, sing.init), sing.target),
                dn: self.dn
            }
        } else {
            Config {
                up: self.up,
                dn: ibset(ibclr(self.dn, sing.init), sing.target)
            }
        }
    }
}

impl Det {
    fn new_diag_opp(&self, ham: &Ham, doub: &Doub) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(doub.target.0, doub.target.0)
            + ham.one_body(doub.target.1, doub.target.1)
            - ham.one_body(doub.init.0, doub.init.0)
            - ham.one_body(doub.init.1, doub.init.1);

        // O(1) Two-body direct part: E += direct(r,s) - direct(p,q)
        new_diag += ham.direct(doub.target.0, doub.target.1, doub.target.0, doub.target.1)
            - ham.direct(doub.init.0, doub.init.1, doub.init.0, doub.init.1);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in (p,q)} direct(i,r) + direct(i,s) - direct(i,p) - direct(i,q)
        for i in bits(self.config.up) {
            if i == doub.init.0 { continue; }
            new_diag += ham.direct_plus_exchange(i, doub.target.0, i, doub.target.0)
                - ham.direct_plus_exchange(i, doub.init.0, i, doub.init.0);
            new_diag += ham.direct(i, doub.target.1, i, doub.target.1)
                - ham.direct(i, doub.init.1, i, doub.init.1);
        }
        for i in bits(self.config.dn) {
            if i == doub.init.1 { continue; }
            new_diag += ham.direct_plus_exchange(i, doub.target.1, i, doub.target.1)
                - ham.direct_plus_exchange(i, doub.init.1, i, doub.init.1);
            new_diag += ham.direct(i, doub.target.0, i, doub.target.0)
                - ham.direct(i, doub.init.0, i, doub.init.0);
        }

        new_diag
    }

    fn new_diag_same(&self, ham: &Ham, doub: &Doub, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(doub.target.0, doub.target.0)
            + ham.one_body(doub.target.1, doub.target.1)
            - ham.one_body(doub.init.0, doub.init.0)
            - ham.one_body(doub.init.1, doub.init.1);

        // O(1) Two-body direct_and_exchange part: E += direct_and_exchange(r,s) - direct_and_exchange(p,q)
        new_diag += ham.direct_plus_exchange(doub.target.0, doub.target.1, doub.target.0, doub.target.1)
            - ham.direct_plus_exchange(doub.init.0, doub.init.1, doub.init.0, doub.init.1);

        // O(N) Two-body direct_and_exchange part: E += sum_{i in occ. but not in (p,q)} direct_and_exchange(i,r) + direct_and_exchange(i,s) - direct_and_exchange(i,p) - direct_and_exchange(i,q)
        if is_up {
            for i in bits(self.config.up) {
                if i == doub.init.0 || i == doub.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, doub.target.0, i, doub.target.0)
                    + ham.direct_plus_exchange(i, doub.target.1, i, doub.target.1)
                    - ham.direct_plus_exchange(i, doub.init.0, i, doub.init.0)
                    - ham.direct_plus_exchange(i, doub.init.1, i, doub.init.1);
            }
            for i in bits(self.config.dn) {
                new_diag += ham.direct(i, doub.target.0, i, doub.target.0)
                    + ham.direct(i, doub.target.1, i, doub.target.1)
                    - ham.direct(i, doub.init.0, i, doub.init.0)
                    - ham.direct(i, doub.init.1, i, doub.init.1);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == doub.init.0 || i == doub.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, doub.target.0, i, doub.target.0)
                    + ham.direct_plus_exchange(i, doub.target.1, i, doub.target.1)
                    - ham.direct_plus_exchange(i, doub.init.0, i, doub.init.0)
                    - ham.direct_plus_exchange(i, doub.init.1, i, doub.init.1);
            }
            for i in bits(self.config.up) {
                new_diag += ham.direct(i, doub.target.0, i, doub.target.0)
                    + ham.direct(i, doub.target.1, i, doub.target.1)
                    - ham.direct(i, doub.init.0, i, doub.init.0)
                    - ham.direct(i, doub.init.1, i, doub.init.1);
            }
        }

        new_diag
    }

    fn new_diag_sing(&self, ham: &Ham, sing: &Sing, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) - h(p)
        let mut new_diag: f64 = self.diag
            + ham.one_body(sing.target, sing.target)
            - ham.one_body(sing.init, sing.init);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in p} direct(i,r) - direct(i,p)
        if is_up {
            for i in bits(self.config.up) {
                if i == sing.init { continue; }
                new_diag += ham.direct_plus_exchange(i, sing.target, i, sing.target)
                    - ham.direct_plus_exchange(i, sing.init, i, sing.init);
            }
            for i in bits(self.config.dn) {
                new_diag += ham.direct(i, sing.target, i, sing.target)
                    - ham.direct(i, sing.init, i, sing.init);
            }
        } else {
            for i in bits(self.config.dn) {
                if i == sing.init { continue; }
                new_diag += ham.direct_plus_exchange(i, sing.target, i, sing.target)
                    - ham.direct_plus_exchange(i, sing.init, i, sing.init);
            }
            for i in bits(self.config.up) {
                new_diag += ham.direct(i, sing.target, i, sing.target)
                    - ham.direct(i, sing.init, i, sing.init);
            }
        }

        new_diag
    }
}
