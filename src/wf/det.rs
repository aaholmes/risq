// Determinant data structure:
// Includes functions to generate an excited det, and compute its diagonal element
// quickly from the initial det's diagonal element

use crate::ham::Ham;
use crate::utils::bits::{bits, btest, ibset, ibclr};
use crate::excite::{ Doub, Sing};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Det {
    pub up: u128,
    pub dn: u128,
}

// Augmented determinant - determinant with coefficient, diagonal H element
pub struct AugDet {
    pub det: Det,
    pub coeff: f64,
    pub diag: f64,
}

impl Det {
    pub fn excite_det_opp_doub(&self, doub: &Doub) -> Option<Det> {
        // Excite det using double excitation
        // Returns None if not possible
        if !btest(self.up, doub.init.0) { return None; }
        if !btest(self.dn, doub.init.1) { return None; }
        if btest(self.up, doub.target.0) { return None; }
        if btest(self.dn, doub.target.1) { return None; }
        Some(
            Det {
                up: ibset(ibclr(self.up, doub.init.0), doub.target.0),
                dn: ibset(ibclr(self.dn, doub.init.1), doub.target.1)
            }
        )
    }

    pub fn excite_det_same_doub(&self, doub: &Doub, is_up: bool) -> Option<Det> {
        // Excite det using double excitation
        // is_up is true if a same-spin up; false if a same-spin dn
        // Returns None if not possible
        if is_up {
            if !btest(self.up, doub.init.0) { return None; }
            if !btest(self.up, doub.init.1) { return None; }
            if btest(self.up, doub.target.0) { return None; }
            if btest(self.up, doub.target.1) { return None; }
            Some(
                Det {
                    up: ibset(ibset(ibclr(ibclr(self.up, doub.init.0), doub.init.1), doub.target.0), doub.target.1),
                    dn: self.dn
                }
            )
        } else {
            if !btest(self.dn, doub.init.0) { return None; }
            if !btest(self.dn, doub.init.1) { return None; }
            if btest(self.dn, doub.target.0) { return None; }
            if btest(self.dn, doub.target.1) { return None; }
            Some(
                Det {
                    up: self.up,
                    dn: ibset(ibset(ibclr(ibclr(self.dn, doub.init.0), doub.init.1), doub.target.0), doub.target.1)
                }
            )
        }
    }

    pub fn excite_det_sing(&self, sing: &Sing, is_up: bool) -> Option<Det> {
        // Excite det using single excitation
        // is_up is true if a single up; false if a single dn
        // Returns None if not possible
        if is_up {
            if !btest(self.up, sing.init) { return None; }
            if btest(self.up, sing.target) { return None; }
            Some(
                Det {
                    up: ibset(ibclr(self.up, sing.init), sing.target),
                    dn: self.dn
                }
            )
        } else {
            if !btest(self.up, sing.init) { return None; }
            if btest(self.up, sing.target) { return None; }
            Some(
                Det {
                    up: self.up,
                    dn: ibset(ibclr(self.dn, sing.init), sing.target)
                }
            )
        }
    }
}

impl AugDet {

    pub fn new_diag_opp(&self, ham: &Ham, excite: &Doub) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(excite.target.0, excite.target.0)
            + ham.one_body(excite.target.1, excite.target.1)
            - ham.one_body(excite.init.0, excite.init.0)
            - ham.one_body(excite.init.1, excite.init.1);

        // O(1) Two-body direct part: E += direct(r,s) - direct(p,q)
        new_diag += ham.direct(excite.target.0, excite.target.1, excite.target.0, excite.target.1)
            - ham.direct(excite.init.0, excite.init.1, excite.init.0, excite.init.1);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in (p,q)} direct(i,r) + direct(i,s) - direct(i,p) - direct(i,q)
        for i in bits(self.det.up) {
            if i == excite.init.0 { continue; }
            new_diag += ham.direct_plus_exchange(i, excite.target.0, i, excite.target.0)
                - ham.direct_plus_exchange(i, excite.init.0, i, excite.init.0);
            new_diag += ham.direct(i, excite.target.1, i, excite.target.1)
                - ham.direct(i, excite.init.1, i, excite.init.1);
        }
        for i in bits(self.det.dn) {
            if i == excite.init.1 { continue; }
            new_diag += ham.direct_plus_exchange(i, excite.target.1, i, excite.target.1)
                - ham.direct_plus_exchange(i, excite.init.1, i, excite.init.1);
            new_diag += ham.direct(i, excite.target.0, i, excite.target.0)
                - ham.direct(i, excite.init.0, i, excite.init.0);
        }

        new_diag
    }

    pub fn new_diag_same(&self, ham: &Ham, excite: &Doub, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = self.diag
            + ham.one_body(excite.target.0, excite.target.0)
            + ham.one_body(excite.target.1, excite.target.1)
            - ham.one_body(excite.init.0, excite.init.0)
            - ham.one_body(excite.init.1, excite.init.1);

        // O(1) Two-body direct_and_exchange part: E += direct_and_exchange(r,s) - direct_and_exchange(p,q)
        new_diag += ham.direct_plus_exchange(excite.target.0, excite.target.1, excite.target.0, excite.target.1)
            - ham.direct_plus_exchange(excite.init.0, excite.init.1, excite.init.0, excite.init.1);

        // O(N) Two-body direct_and_exchange part: E += sum_{i in occ. but not in (p,q)} direct_and_exchange(i,r) + direct_and_exchange(i,s) - direct_and_exchange(i,p) - direct_and_exchange(i,q)
        if is_up {
            for i in bits(self.det.up) {
                if i == excite.init.0 || i == excite.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, excite.target.0, i, excite.target.0)
                    + ham.direct_plus_exchange(i, excite.target.1, i, excite.target.1)
                    - ham.direct_plus_exchange(i, excite.init.0, i, excite.init.0)
                    - ham.direct_plus_exchange(i, excite.init.1, i, excite.init.1);
            }
            for i in bits(self.det.dn) {
                new_diag += ham.direct(i, excite.target.0, i, excite.target.0)
                    + ham.direct(i, excite.target.1, i, excite.target.1)
                    - ham.direct(i, excite.init.0, i, excite.init.0)
                    - ham.direct(i, excite.init.1, i, excite.init.1);
            }
        } else {
            for i in bits(self.det.dn) {
                if i == excite.init.0 || i == excite.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, excite.target.0, i, excite.target.0)
                    + ham.direct_plus_exchange(i, excite.target.1, i, excite.target.1)
                    - ham.direct_plus_exchange(i, excite.init.0, i, excite.init.0)
                    - ham.direct_plus_exchange(i, excite.init.1, i, excite.init.1);
            }
            for i in bits(self.det.up) {
                new_diag += ham.direct(i, excite.target.0, i, excite.target.0)
                    + ham.direct(i, excite.target.1, i, excite.target.1)
                    - ham.direct(i, excite.init.0, i, excite.init.0)
                    - ham.direct(i, excite.init.1, i, excite.init.1);
            }
        }

        new_diag
    }

    pub fn new_diag_sing(&self, ham: &Ham, excite: &Sing, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) - h(p)
        let mut new_diag: f64 = self.diag
            + ham.one_body(excite.target, excite.target)
            - ham.one_body(excite.init, excite.init);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in p} direct(i,r) - direct(i,p)
        if is_up {
            for i in bits(self.det.up) {
                if i == excite.init { continue; }
                new_diag += ham.direct_plus_exchange(i, excite.target, i, excite.target)
                    - ham.direct_plus_exchange(i, excite.init, i, excite.init);
            }
            for i in bits(self.det.dn) {
                new_diag += ham.direct(i, excite.target, i, excite.target)
                    - ham.direct(i, excite.init, i, excite.init);
            }
        } else {
            for i in bits(self.det.dn) {
                if i == excite.init { continue; }
                new_diag += ham.direct_plus_exchange(i, excite.target, i, excite.target)
                    - ham.direct_plus_exchange(i, excite.init, i, excite.init);
            }
            for i in bits(self.det.up) {
                new_diag += ham.direct(i, excite.target, i, excite.target)
                    - ham.direct(i, excite.init, i, excite.init);
            }
        }

        new_diag
    }
}
