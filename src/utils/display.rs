//! Display definitions for custom types

use crate::excite::Orbs;
use crate::stoch::DetOrbSample;
use crate::wf::det::{Config, Det};
use crate::wf::VarWf;
use std::cmp::Ordering::Equal;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;

pub(crate) fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "up: {}, dn: {}", fmt_det(self.up), fmt_det(self.dn))
    }
}

impl fmt::Display for Det {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} : coeff = {}, diag = {}",
            self.config, self.coeff, self.diag
        )
    }
}

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
                d.diag
            );
        }
        println!("\n");
    }

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
                d.0.det.diag
            );
        }
        println!("\n");
    }
}

// Wrapper for Det that enables sorting by coeff (usually sort by config), needed by print_largest
pub struct DetByCoeff<'a> {
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
