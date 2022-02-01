//! Functions to compute Hamiltonian matrix elements

pub mod read_ints;

use super::utils::bits::bits;
use super::utils::ints::{combine_2, combine_4, permute, permute_2};
use crate::excite::{Excite, Orbs};
use crate::utils::bits::{bit_pairs, det_bits};
use crate::wf::det::Config;
use read_ints::Ints;

/// Hamiltonian, containing integrals and matrix element computing functions
/// Also contains information about frozen orbitals
#[derive(Default)]
pub struct Ham {
    // diag_computed: bool, // whether a diagonal element has been computed
    ints: Ints,
    pub(crate) core_orbs: Vec<i32>,
    pub(crate) valence_orbs: Vec<i32>,
    pub screen_single_nonzero: Vec<Vec<i128>>,
}

impl Ham {
    /// Get the integral corresponding to pqrs
    fn get_int(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        // incorporates symmetries p-q, r-s, pq-rs
        // Insensitive to whether indices are positive or negative (up or dn spin)
        // NB: get_int starts at index 1 (since that's how FCIDUMP is defined), but
        // all of the ham element functions start at index 0 (since so does Rust)
        self.ints.two_body[combine_4(p, q, r, s)]
    }

    /// Get the one-body energy h_{pq}
    pub fn one_body(&self, p: i32, q: i32) -> f64 {
        self.ints.one_body[combine_2(p + 1, q + 1)]
    }

    /// Get the direct energy corresponding to pq -> rs
    pub fn direct(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        self.get_int(p + 1, r + 1, q + 1, s + 1)
    }

    /// Get the direct plus exchange energy corresponding to pq -> rs
    pub fn direct_plus_exchange(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        self.get_int(p + 1, r + 1, q + 1, s + 1) - self.get_int(p + 1, s + 1, q + 1, r + 1)
    }

    /// Get the diagonal element corresponding to this determinant
    /// Should only be called once
    pub fn ham_diag(&self, det: &Config) -> f64 {
        // if self.diag_computed {
        //     panic!("Computing diagonal element in O(n_elec^2) time (should only happen once!)");
        // }

        // nuclear-nuclear component
        let mut diag: f64 = self.ints.nuc;

        // one-body component
        for i in det_bits(det) {
            diag += self.one_body(i, i);
        }

        // two-body component
        // opposite-spin
        for i in bits(det.up) {
            for j in bits(det.dn) {
                diag += self.direct(i, j, i, j);
            }
        }
        // same-spin, up
        for (i, j) in bit_pairs(det.up) {
            diag += self.direct_plus_exchange(i, j, i, j);
        }
        // same-spin, dn
        for (i, j) in bit_pairs(det.dn) {
            diag += self.direct_plus_exchange(i, j, i, j);
        }

        // self.diag_computed = true;

        diag
    }

    /// Get the single excitation matrix element corresponding to the excitation from det1 to det2
    pub fn ham_sing(&self, det1: &Config, det2: &Config) -> f64 {
        let mut out: f64;
        if det1.dn == det2.dn {
            let i: i32 = (det1.up & !det2.up).trailing_zeros() as i32;
            let j: i32 = (det2.up & !det1.up).trailing_zeros() as i32;

            // One-body term
            out = self.one_body(i, j);
            //
            // // Check whether det1.up & screen_single_nonzero(i, j) is 0; if it is, skip the following for loops!
            // if det1.up & self.screen_single_nonzero(i, j) == 0 {
            //     return permute(det1.up, det2.up) as f64 * out;
            // }

            // Two-body term
            for k in bits(det1.up) {
                out += self.direct_plus_exchange(i, k, j, k);
            }
            for k in bits(det1.dn) {
                out += self.direct(i, k, j, k);
            }
            out *= permute(det1.up, det2.up) as f64;
        } else {
            let i: i32 = (det1.dn & !det2.dn).trailing_zeros() as i32;
            let j: i32 = (det2.dn & !det1.dn).trailing_zeros() as i32;

            // One-body term
            out = self.one_body(i, j);
            //
            // // Check whether det1.dn & screen_single_nonzero(i, j) is 0; if it is, skip the following for loops!
            // if det1.dn & self.screen_single_nonzero(i, j) == 0 {
            //     return permute(det1.dn, det2.dn) as f64 * out;
            // }

            // Two-body term
            for k in bits(det1.dn) {
                out += self.direct_plus_exchange(i, k, j, k);
            }
            for k in bits(det1.up) {
                out += self.direct(i, k, j, k);
            }
            out *= permute(det1.dn, det2.dn) as f64;
        }

        out
    }

    /// Get the double excitation matrix element corresponding to the excitation from det1 to det2
    pub fn ham_doub(&self, det1: &Config, det2: &Config) -> f64 {
        if det1.dn == det2.dn {
            // Same spin, up
            let mut ind: [i32; 4] = [0; 4];
            let mut n = 0;
            for i in bits(det1.up & !det2.up) {
                ind[n] = i;
                n += 1;
            }
            for i in bits(det2.up & !det1.up) {
                ind[n] = i;
                n += 1;
            }
            (permute_2(det1.up, det2.up, ind) as f64)
                * self.direct_plus_exchange(ind[0], ind[1], ind[2], ind[3])
        } else if det1.up == det2.up {
            // Same spin, dn
            let mut ind: [i32; 4] = [0; 4];
            let mut n = 0;
            for i in bits(det1.dn & !det2.dn) {
                ind[n] = i;
                n += 1;
            }
            for i in bits(det2.dn & !det1.dn) {
                ind[n] = i;
                n += 1;
            }
            (permute_2(det1.dn, det2.dn, ind) as f64)
                * self.direct_plus_exchange(ind[0], ind[1], ind[2], ind[3])
        } else {
            // Opposite spin
            let mut ind1: [i32; 2] = [0; 2];
            let mut n = 0;
            for i in bits(det1.up & !det2.up) {
                ind1[n] = i;
                n += 1;
            }
            for i in bits(det2.up & !det1.up) {
                ind1[n] = i;
                n += 1;
            }
            let mut ind2: [i32; 2] = [0; 2];
            let mut n = 0;
            for i in bits(det1.dn & !det2.dn) {
                ind2[n] = i;
                n += 1;
            }
            for i in bits(det2.dn & !det1.dn) {
                ind2[n] = i;
                n += 1;
            }
            ((permute(det1.up, det2.up) * permute(det1.dn, det2.dn)) as f64)
                * self.direct(ind1[0], ind2[0], ind1[1], ind2[1])
        }
    }

    pub fn ham_off_diag(&self, det1: &Config, det2: &Config, excite: &Excite) -> f64 {
        // Compute an off-diagonal matrix element (either single or double),
        // using the information stored in excite
        match excite.init {
            Orbs::Double(_) => self.ham_doub(det1, det2),
            Orbs::Single(_) => self.ham_sing(det1, det2),
        }
    }

    pub fn ham_off_diag_no_excite(&self, det1: &Config, det2: &Config) -> f64 {
        // Compute off-diagonal element (either single or double),
        // without excite information
        match (det1.up == det2.up, det1.dn == det2.dn) {
            (true, true) => 0.0,
            (false, false) => {
                // Opposite spin double
                for det in &[det1.up & !det2.up, det1.dn & !det2.dn] {
                    let mut n = 0;
                    for _ in bits(*det) {
                        n += 1;
                        if n > 1 {
                            return 0.0;
                        }
                    }
                }
                self.ham_doub(det1, det2)
            }
            (true, false) => {
                let mut n = 0;
                for _ in bits(det1.dn & !det2.dn) {
                    n += 1;
                    if n > 2 {
                        return 0.0;
                    }
                }
                if n == 1 {
                    self.ham_sing(det1, det2)
                } else {
                    self.ham_doub(det1, det2)
                }
            }
            (false, true) => {
                let mut n = 0;
                for _ in bits(det1.up & !det2.up) {
                    n += 1;
                    if n > 2 {
                        return 0.0;
                    }
                }
                if n == 1 {
                    self.ham_sing(det1, det2)
                } else {
                    self.ham_doub(det1, det2)
                }
            }
        }
    }
}
