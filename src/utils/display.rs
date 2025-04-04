//! # Display Formatting Utilities (`utils::display`)
//!
//! This module provides implementations of the `std::fmt::Display` trait for various
//! custom data structures used in the `risq` crate, allowing them to be easily printed
//! in a human-readable format (e.g., using `println!("{}", my_struct)`).
//!
//! It also includes helper functions for formatting specific types (like bitstrings)
//! and methods attached to `VarWf` for printing wavefunction information.

use crate::excite::{Excite, Orbs};
use crate::stoch::DetOrbSample;
use crate::wf::det::{Config, Det};
use crate::wf::VarWf;
use std::cmp::Ordering::Equal;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;

/// Formats a `u128` bitstring representing orbital occupancy for display.
/// Shows the binary representation, replacing '0' with '_' for readability.
pub(crate) fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

/// Implements `Display` for `Config` (determinant configuration).
/// Shows the formatted `up` and `dn` bitstrings.
impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "up: {}, dn: {}", fmt_det(self.up), fmt_det(self.dn))
    }
}

/// Implements `Display` for `Det` (determinant with coefficient and diagonal).
/// Shows the configuration, coefficient, and diagonal energy (or "N/A" if not computed).
impl fmt::Display for Det {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} : coeff = {}, diag = {}",
            self.config,
            self.coeff,
            if let Some(_diag) = self.diag {
                self.diag.unwrap().to_string()
            } else {
                "N/A".to_string()
            }
        )
    }
}

/// Implements `Display` for `Orbs` (single or double orbital indices).
/// Shows `(p, q)` for `Double` and `p` for `Single`.
impl fmt::Display for Orbs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Orbs::Double((p, q)) => {
                write!(f, "({}, {})", p, q)
            }
            Orbs::Single(p) => {
                write!(f, "{}", p)
            }
        }
    }
}

/// Implements `Display` for `Excite` (full excitation information).
/// Describes the excitation type (Single/Double), spin channel (Up/Down/Opp),
/// initial and target orbitals, and the estimated |H| value.
impl fmt::Display for Excite {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(a) = self.is_alpha {
            match self.init {
                Orbs::Double(_) => {
                    if a {
                        write!(f, "Double, Up: {} -> {}, |H| = {}", self.init, self.target, self.abs_h)
                    } else {
                        write!(f, "Double, Down: {} -> {}, |H| = {}", self.init, self.target, self.abs_h)
                    }
                }
                Orbs::Single(_) => {
                    if a {
                        write!(f, "Single, Up: {} -> {}, |H| = {}", self.init, self.target, self.abs_h)
                    } else {
                        write!(f, "Single, Down: {} -> {}, |H| = {}", self.init, self.target, self.abs_h)
                    }
                }
            }
        } else {
            write!(f, "Double, Opp: {} -> {}, |H| = {}", self.init, self.target, self.abs_h)
        }
    }
}

/// Implements `Display` for `DetOrbSample` (source det/orbs for sampling).
/// Shows the source determinant, initial orbitals, spin channel, and the
/// cumulative sums used for importance sampling.
impl fmt::Display for DetOrbSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.is_alpha {
            None => {
                write!(
                    f,
                    "{}, orbs = {}, opposite-spin double, sum_abs_hc = {}, sum_hc_squared = {}",
                    self.det, self.init, self.sum_abs_hc, self.sum_hc_squared
                )
            }
            Some(is_alpha) => {
                if is_alpha {
                    write!(
                        f,
                        "{}, orbs = {}, spin = up, sum_abs_hc = {}, sum_hc_squared = {}",
                        self.det, self.init, self.sum_abs_hc, self.sum_hc_squared
                    )
                } else {
                    write!(
                        f,
                        "{}, orbs = {}, spin = dn, sum_abs_hc = {}, sum_hc_squared = {}",
                        self.det, self.init, self.sum_abs_hc, self.sum_hc_squared
                    )
                }
            }
        }
    }
}

impl VarWf {
    /// Prints the entire variational wavefunction (`self.wf`) to standard output.
    /// Lists each determinant's coefficient, configuration (up/dn bitstrings), and diagonal energy.
    pub fn print(&self) {
        println!(
            "\nWavefunction has {} dets with energy {:.4}",
            self.wf.n, self.wf.energy
        );
        println!("Coeff     Det_up     Det_dn    <D|H|D>");
        for d in self.wf.dets.iter() {
            println!(
                "{:.4}   {}   {}   {:.3}",
                d.coeff,
                fmt_det(d.config.up),
                fmt_det(d.config.dn),
                d.diag.unwrap()
            );
        }
        println!("\n");
    }

    /// Prints the `k` determinants with the largest absolute coefficients in the wavefunction.
    /// Uses a min-heap to find the top `k` elements efficiently in O(N log k) time,
    /// where N is the total number of determinants.
    pub fn print_largest(&self, k: usize) {
        // Prints the k dets with largest abs coeff
        // O(N + k log k) time

        // Use a min-heap of the k largest elements
        let mut heap = BinaryHeap::with_capacity(k);

        for (ind, det) in self.wf.dets.iter().enumerate() {
            if ind < k {
                heap.push(Reverse(DetByCoeff { det }));
            } else if det.coeff.abs() > heap.peek().unwrap().0.det.coeff.abs() {
                heap.pop();
                heap.push(Reverse(DetByCoeff { det }));
            }
        }

        let mut top_k = heap.into_vec();
        top_k.sort_by(|a, b| {
            b.0.det
                .coeff
                .abs()
                .partial_cmp(&a.0.det.coeff.abs())
                .unwrap_or(Equal)
        });

        println!(
            "\nWavefunction has {} dets with energy {:.4}",
            self.wf.n, self.wf.energy
        );
        println!("Coeff     Det_up     Det_dn    <D|H|D>");
        for d in top_k {
            println!(
                "{:.4}   {}   {}   {:.3}",
                d.0.det.coeff,
                fmt_det(d.0.det.config.up),
                fmt_det(d.0.det.config.dn),
                d.0.det.diag.unwrap()
            );
        }
        println!("\n");
    }
}

// Wrapper for Det that enables sorting by coeff (usually sort by config), needed by print_largest
/// A wrapper around a `Det` reference that implements `Ord` and `PartialOrd`
/// based on the absolute value of the determinant's coefficient.
/// Used by `print_largest` to sort determinants by coefficient magnitude using a heap.
pub struct DetByCoeff<'a> {
    /// Reference to the determinant.
    pub(crate) det: &'a Det,
}

impl Ord for DetByCoeff<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.det
            .coeff
            .abs()
            .partial_cmp(&other.det.coeff.abs())
            .unwrap_or(Equal)
    }
}

impl PartialOrd for DetByCoeff<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DetByCoeff<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.det.coeff.abs() == other.det.coeff.abs()
    }
}

impl Eq for DetByCoeff<'_> {}
