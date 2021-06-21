// Display definitions for custom types

use std::fmt;
use crate::wf::det::{Config, Det};
use crate::excite::Orbs;
use crate::stoch::DetOrbSample;
use crate::wf::Wf;

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
        write!(f, "{} : coeff = {}, diag = {}", self.config, self.coeff, self.diag)
    }
}

impl fmt::Display for Orbs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Orbs::Double((p, q)) => {
                write!(f, "({}, {})", p, q)
            },
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
                write!(f, "{}, orbs = {}, opposite-spin double, sum_abs_hc = {}", self.det, self.init, self.sum_abs_hc)
            },
            Some(is_alpha) => {
                if is_alpha {
                    write!(f, "{}, orbs = {}, spin = up, sum_abs_hc = {}", self.det, self.init, self.sum_abs_hc)
                } else {
                    write!(f, "{}, orbs = {}, spin = dn, sum_abs_hc = {}", self.det, self.init, self.sum_abs_hc)
                }
            }
        }
    }
}

impl Wf {
    pub fn print(&self) {
        println!("\nWavefunction has {} dets with energy {:.4}", self.n, self.energy);
        println!("Coeff     Det_up     Det_dn    <D|H|D>");
        for d in self.dets.iter() {
            println!("{:.4}   {}   {}   {:.3}", d.coeff, fmt_det(d.config.up), fmt_det(d.config.dn), d.diag);
        }
        println!("\n");
    }
}
