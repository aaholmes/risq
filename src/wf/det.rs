// Determinant data structure:
// Includes functions to generate an excited det, and compute its diagonal element
// quickly from the initial det's diagonal element

use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::{ExciteGenerator, Doub, OPair, Sing};
use crate::utils::bits::{bits, btest, ibset, ibclr};
use std::cmp::max;
use crate::ham::Ham;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Det {
    pub up: u128,
    pub dn: u128,
}

impl Det {
    pub fn excite_det_opp_doub(&self, doub: Doub) -> Option(Det) {
        // Excite det using double excitation
        // Returns None if not possible
        if !btest(self.up, doub.init.0) { return None; }
        if !btest(self.dn, doub.init.1) { return None; }
        if btest(self.up, doub.target.0) { return None; }
        if btest(self.dn, doub.target.1) { return None; }
        Some(
            Det::new(
                ibset(ibclr(self.up, init.0), target.0),
                ibset(ibclr(self.dn, init.1), target.1)
            )
        )
    }

    pub fn excite_det_same_doub(&self, doub: Doub, is_up: bool) -> Option(Det) {
        // Excite det using double excitation
        // is_up is true if a same-spin up; false if a same-spin dn
        // Returns None if not possible
        if is_up {
            if !btest(self.up, doub.init.0) { return None; }
            if !btest(self.up, doub.init.1) { return None; }
            if btest(self.up, doub.target.0) { return None; }
            if btest(self.up, doub.target.1) { return None; }
            Some(
                Det::new(
                    ibset(ibset(ibclr(ibclr(self.up, init.0), init.1), target.0), target.1),
                    self.dn
                )
            )
        } else {
            if !btest(self.dn, doub.init.0) { return None; }
            if !btest(self.dn, doub.init.1) { return None; }
            if btest(self.dn, doub.target.0) { return None; }
            if btest(self.dn, doub.target.1) { return None; }
            Some(
                Det::new(
                    self.up,
                    ibset(ibset(ibclr(ibclr(self.dn, init.0), init.1), target.0), target.1)
                )
            )
        }
    }

    pub fn excite_det_sing(&self, sing: Sing, is_up: bool) -> Option(Det) {
        // Excite det using single excitation
        // is_up is true if a single up; false if a single dn
        // Returns None if not possible
        if is_up {
            if !btest(self.up, sing.init) { return None; }
            if btest(self.up, sing.target) { return None; }
            Some(
                Det::new(
                    ibset(ibclr(self.up, init.0), target.0),
                    self.dn
                )
            )
        } else {
            if !btest(self.up, sing.init) { return None; }
            if btest(self.up, sing.target) { return None; }
            Some(
                Det::new(
                    self.up,
                    ibset(ibclr(self.dn, init.0), target.0)
                )
            )
        }
    }

    pub fn new_diag_opp(&self, ham: &Ham, old_diag: f64, &excite: Doub) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = old_diag
            + ham.one_body(excite.target.0, excite.target.0)
            + ham.one_body(excite.target.1, excite.target.1)
            - ham.one_body(excite.init.0, excite.init.0)
            - ham.one_body(excite.init.1, excite.init.1);

        // O(1) Two-body direct part: E += direct(r,s) - direct(p,q)
        new_diag += ham.direct(excite.target.0, excite.target.0, excite.target.1, excite.target.1)
            - ham.direct(excite.init.0, excite.init.0, excite.init.1, excite.init.1);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in (p,q)} direct(i,r) + direct(i,s) - direct(i,p) - direct(i,q)
        for i in bits(self.up) {
            if i == excite.init.0 { continue; }
            new_diag += ham.direct(i, i, excite.target.0, excite.target.0)
                - ham.direct(i, i, excite.init.0, excite.init.0)
        }
        for i in bits(self.dn) {
            if i == excite.init.1 { continue; }
            new_diag += ham.direct(i, i, excite.target.1, excite.target.1)
                - ham.direct(i, i, excite.init.1, excite.init.1)
        }

        new_diag
    }

    pub fn new_diag_same(&self, ham: &Ham, old_diag: f64, &excite: Doub, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) + h(s) - h(p) - h(q)
        let mut new_diag: f64 = old_diag
            + ham.one_body(excite.target.0, excite.target.0)
            + ham.one_body(excite.target.1, excite.target.1)
            - ham.one_body(excite.init.0, excite.init.0)
            - ham.one_body(excite.init.1, excite.init.1);

        // O(1) Two-body direct_and_exchange part: E += direct_and_exchange(r,s) - direct_and_exchange(p,q)
        new_diag += ham.direct_plus_exchange(excite.target.0, excite.target.0, excite.target.1, excite.target.1)
            - ham.direct_plus_exchange(excite.init.0, excite.init.0, excite.init.1, excite.init.1);

        // O(N) Two-body direct_and_exchange part: E += sum_{i in occ. but not in (p,q)} direct_and_exchange(i,r) + direct_and_exchange(i,s) - direct_and_exchange(i,p) - direct_and_exchange(i,q)
        if is_up {
            for i in bits(self.up) {
                if i == excite.init.0 || i == excite.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, i, excite.target.0, excite.target.0)
                    + ham.direct_plus_exchange(i, i, excite.target.1, excite.target.1)
                    - ham.direct_plus_exchange(i, i, excite.init.0, excite.init.0)
                    - ham.direct_plus_exchange(i, i, excite.init.1, excite.init.1);
            }
        } else {
            for i in bits(self.dn) {
                if i == excite.init.0 || i == excite.init.1 { continue; }
                new_diag += ham.direct_plus_exchange(i, i, excite.target.0, excite.target.0)
                    + ham.direct_plus_exchange(i, i, excite.target.1, excite.target.1)
                    - ham.direct_plus_exchange(i, i, excite.init.0, excite.init.0)
                    - ham.direct_plus_exchange(i, i, excite.init.1, excite.init.1);
            }
        }

        new_diag
    }

    pub fn new_diag_sing(&self, ham: &Ham, old_diag: f64, &excite: Sing, is_up: bool) -> f64 {
        // Compute new diagonal element given the old one

        // O(1) One-body part: E += h(r) - h(p)
        let mut new_diag: f64 = old_diag
            + ham.one_body(excite.target, excite.target)
            - ham.one_body(excite.init, excite.init);

        // O(N) Two-body direct part: E += sum_{i in occ. but not in p} direct(i,r) - direct(i,p)
        if is_up {
            for i in bits(self.up) {
                if i == excite.init { continue; }
                new_diag += ham.direct(i, i, excite.target, excite.target)
                    - ham.direct(i, i, excite.init, excite.init);
            }
        } else {
            for i in bits(self.dn) {
                if i == excite.init { continue; }
                new_diag += ham.direct(i, i, excite.target, excite.target)
                    - ham.direct(i, i, excite.init, excite.init);
            }
        }

        new_diag
    }
}
