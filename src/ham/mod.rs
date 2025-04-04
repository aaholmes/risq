//! # Hamiltonian Module (`ham`)
//!
//! This module defines the `Ham` struct, which encapsulates the electronic Hamiltonian
//! operator for a molecular system. It stores the one- and two-electron integrals read
//! from an `FCIDUMP` file and provides methods to compute Hamiltonian matrix elements
//! between Slater determinants (`Config` objects) based on the Slater-Condon rules.
//!
//! ## Key Components:
//! *   `Ham`: Struct holding integrals (`Ints`) and orbital information.
//! *   `read_ints`: Submodule responsible for parsing the `FCIDUMP` file.
//! *   Matrix element functions (`ham_diag`, `ham_sing`, `ham_doub`): Compute
//!     `<det1|H|det2>` based on the excitation level between `det1` and `det2`.
//!
//! ## Indexing Convention:
//! Note that the underlying integrals stored (and read from `FCIDUMP`) are typically
//! 1-based, while the determinant representations and orbital indices used within
//! this code are 0-based. The methods in this module handle this conversion internally.

pub mod read_ints;

use super::utils::bits::bits;
use super::utils::ints::{combine_2, combine_4, permute, permute_2};
use crate::excite::{Excite, Orbs};
use crate::utils::bits::{bit_pairs, det_bits};
use crate::wf::det::Config;
use read_ints::Ints;

/// Represents the electronic Hamiltonian operator.
///
/// Stores the one- and two-electron integrals (`ints`) read from an `FCIDUMP` file,
/// along with information about core (frozen) and valence orbitals if specified.
/// Provides methods to compute matrix elements `<det1|H|det2>`.
#[derive(Default, Debug)]
pub struct Ham {
    // diag_computed: bool, // whether a diagonal element has been computed
    ints: Ints,
    pub(crate) core_orbs: Vec<i32>,
    pub(crate) valence_orbs: Vec<i32>,
    pub screen_single_nonzero: Vec<Vec<i128>>,
}

impl Ham {
    /// Retrieves the two-electron integral (pq|rs) using chemist's notation.
    ///
    /// Handles index permutation symmetries (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) =
    /// (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp) and ensures 1-based indexing
    /// consistent with FCIDUMP format is used for lookup.
    /// Orbital indices `p, q, r, s` are expected to be 1-based here.
    fn get_int(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        // incorporates symmetries p-q, r-s, pq-rs
        // Insensitive to whether indices are positive or negative (up or dn spin)
        // NB: get_int starts at index 1 (since that's how FCIDUMP is defined), but
        // all of the ham element functions start at index 0 (since so does Rust)
        self.ints.two_body[combine_4(p, q, r, s)]
    }

    /// Retrieves the one-electron integral h_pq = <p|h|q>.
    ///
    /// Expects 0-based orbital indices `p, q` and converts them to 1-based for lookup.
    pub fn one_body(&self, p: i32, q: i32) -> f64 {
        self.ints.one_body[combine_2(p + 1, q + 1)]
    }

    /// Retrieves the two-electron integral (pr|qs) often used in direct terms.
    ///
    /// Expects 0-based orbital indices `p, q, r, s`.
    pub fn direct(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        self.get_int(p + 1, r + 1, q + 1, s + 1)
    }

    /// Retrieves the two-electron integral combination (pr|qs) - (ps|qr), often used
    /// for same-spin interactions.
    ///
    /// Expects 0-based orbital indices `p, q, r, s`.
    pub fn direct_plus_exchange(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        self.get_int(p + 1, r + 1, q + 1, s + 1) - self.get_int(p + 1, s + 1, q + 1, r + 1)
    }

    /// Computes the diagonal Hamiltonian matrix element <D|H|D> for a given determinant `det`.
    ///
    /// Implements the Slater-Condon rule for diagonal elements:
    /// E_diag = E_nuc + sum_i h_ii + sum_{i<j} [(ii|jj) - (ij|ji)]_{same spin} + sum_{i,j} (ii|jj)_{opposite spin}
    /// where i, j run over occupied spin-orbitals in `det`.
    ///
    /// Note: This calculation scales as O(N_elec^2) and is typically performed only once
    /// per determinant when constructing the Hamiltonian matrix.
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

    /// Computes the off-diagonal Hamiltonian matrix element <D1|H|D2> where `det1` and `det2`
    /// differ by a single spin-orbital excitation (i -> j).
    ///
    /// Implements the Slater-Condon rule for single excitations:
    /// <D1|H|D2> = +/- [ h_ij + sum_k [(ik|jk) - (ik|kj)]_{same spin} + sum_k (ik|jk)_{opposite spin} ]
    /// where k runs over occupied spin-orbitals common to both determinants.
    /// The sign +/- depends on the permutation required to align the determinants.
    ///
    /// Assumes `det1` and `det2` are indeed connected by a single excitation.
    pub fn ham_sing(&self, det1: &Config, det2: &Config) -> f64 {
        let mut out: f64;
        if det1.dn == det2.dn {
            let i: i32 = (det1.up & !det2.up).trailing_zeros() as i32;
            let j: i32 = (det2.up & !det1.up).trailing_zeros() as i32;

            // One-body term
            out = self.one_body(i, j);
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

    /// Computes the off-diagonal Hamiltonian matrix element <D1|H|D2> where `det1` and `det2`
    /// differ by a double spin-orbital excitation (i,j -> k,l).
    ///
    /// Implements the Slater-Condon rule for double excitations:
    /// <D1|H|D2> = +/- [(ik|jl) - (il|jk)]   (if i,j,k,l have same spin)
    /// <D1|H|D2> = +/- [(ik|jl)]             (if i,k have one spin, j,l have the other)
    /// The sign +/- depends on the permutation required to align the determinants.
    ///
    /// Assumes `det1` and `det2` are indeed connected by a double excitation.
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

    /// Computes an off-diagonal matrix element <det1|H|det2> using pre-computed excitation info.
    ///
    /// This function dispatches to either `ham_sing` or `ham_doub` based on the
    /// information contained within the `excite` struct, which presumably stores
    /// the excitation level and involved orbitals. This avoids re-calculating the
    /// excitation level.
    pub fn ham_off_diag(&self, det1: &Config, det2: &Config, excite: &Excite) -> f64 {
        // Compute an off-diagonal matrix element (either single or double),
        // using the information stored in excite
        match excite.init {
            Orbs::Double(_) => self.ham_doub(det1, det2),
            Orbs::Single(_) => self.ham_sing(det1, det2),
        }
    }

    /// Computes an off-diagonal matrix element <det1|H|det2> without pre-computed excitation info.
    ///
    /// Determines the excitation level (single, double, or higher/zero) by comparing
    /// the bit representations of `det1` and `det2`. It then calls the appropriate
    /// `ham_sing` or `ham_doub` function, or returns 0.0 if the determinants differ
    /// by more than two spin-orbitals or are identical.
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
