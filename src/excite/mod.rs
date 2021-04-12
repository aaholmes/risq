// Excitation generation module (includes usual heat-bath routines)

use std::collections::HashMap;

use super::utils::bits::bits;
use super::utils::ints::combine_2;
use super::utils::read_input::Global;

use super::ham::Ham;

// Orbital pair
pub struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    target: OPair,
    abs_h: f64,
}

// Heat-bath excitation generator:
// each electron pair points to a sorted vector of double excitations
pub struct ExciteGenerator {
    same_spin_doub_generator: HashMap<usize, Vec<Doub>>,
    opp_spin_doub_generator: HashMap<OPair, Vec<Doub>>,
}

pub fn init_excite_generator(global: &Global, ham: &Ham) -> ExciteGenerator {
    // Initialize by sorting double excitation element for all pairs
    let mut excite_gen: ExciteGenerator;
    let mut v: Vec<Doub>;

    // Opposite spin
    // Assume p/r are up, q/s are dn
    for p in 0..global.norb {
        for q in 0..global.norb {
            v = vec![];
            for r in 0..global.norb {
                if p == r  { continue; };
                for s in 0..global.norb {
                    if  q == s { continue; };
                    // Compute H elem
                    v.push(
                        Doub{
                            target: OPair(r, s),
                            abs_h: (ham.get_int(p + 1, q + 1, r + 1, s + 1)).abs()
                        }
                    );
                }
            }
            // Sort v
            v.sort_by(|a, b| b.abs_h.cmp(a.abs_h));
            println!("Exciting orbitals: {} {}", p, q);
            for elem in v {
                println!("{} {} {}", elem.target.0, elem.target.1, elem.abs_h);
            }
            excite_gen.opp_spin_doub_generator.insert(OPair(p, q), v);
        }
    }

    // Same spin
    for p in 0..global.norb {
        for q in p+1..global.norb {
            v = vec![];
            for r in 0..global.norb {
                if p == r || q == r { continue; };
                for s in r+1..global.norb {
                    if p == s || q == s { continue; };
                    // Compute H elem
                    v.push(
                        Doub{
                            target: OPair(r, s),
                            abs_h: (ham.get_int(p + 1, q + 1, r + 1, s + 1) - ham.get_int(p + 1, q + 1, s + 1, r + 1)).abs()
                        }
                    );
                }
            }
            // Sort v
            v.sort_by(|a, b| b.abs_h.cmp(a.abs_h));
            println!("Exciting orbitals: {} {}", p, q);
            for elem in v {
                println!("{} {} {}", elem.target.0, elem.target.1, elem.abs_h);
            }
            excite_gen.same_spin_doub_generator.insert(combine_2(p, q), v);
        }
    }

    excite_gen

}


impl ExciteGenerator {

    pub fn double_excite(&self, det: Det, eps: f64) {
        // Generate all double excitations from the given determinant
        // larger in magnitude than eps

        // Opposite spin doubles
        for i in bits(det.up) {
            for j in bits(det.up) {
                for excite in self.opp_spin_doub_generator[OPair(i, j)] {
                    if (excite.abs_h < eps) {
                        break;
                    }
                    // Add this excitation here
                }
            }
        }

        // Same spin doubles
        for i in bits(det.up) {
            for j in bits(det.up) {
                for excite in self.same_spin_doub_generator[combine_2(i, j)] {
                    if (excite.abs_h < eps) {
                        break;
                    }
                    // Add this excitation here
                }
            }
        }
        for i in bits(det.up) {
            for j in bits(det.up) {
                for excite in self.same_spin_doub_generator[combine_2(i, j)] {
                    if (excite.abs_h < eps) {
                        break;
                    }
                    // Add this excitation here
                }
            }
        }
    }
}