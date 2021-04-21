// Determinant data structure:
// Includes functions to generate an excited det, and compute its diagonal element
// quickly from the initial det's diagonal element

use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::{ExciteGenerator, Doub, OPair, Sing};
use crate::utils::bits::{bits, btest, ibset, ibclr};
use std::cmp::max;

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
                    ibset(
                        ibset(
                            ibclr(
                                ibclr(
                                    self.up, init.0
                                ), init.1
                            ), target.0
                        ), target.1
                    ),
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
                    ibset(
                        ibset(
                            ibclr(
                                ibclr(
                                    self.dn, init.0
                                ), init.1
                            ), target.0
                        ), target.1
                    )
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
                    ibset(
                        ibclr(
                            self.up, init.0
                        ), target.0
                    ),
                    self.dn
                )
            )
        } else {
            if !btest(self.up, sing.init) { return None; }
            if btest(self.up, sing.target) { return None; }
            Some(
                Det::new(
                    self.up,
                    ibset(
                        ibclr(
                            self.dn, init.0
                        ), target.0
                    )
                )
            )
        }
    }

    pub fn new_diag_opp(&self, &excite: Doub) {
        // Compute new diagonal element given the old one
    }

    pub fn new_diag_same(&self, &excite: Doub, is_up: bool) {
        // Compute new diagonal element given the old one
    }

    pub fn new_diag_sing(&self, &excite: Sing, is_up: bool) {
        // Compute new diagonal element given the old one
    }
}
