// Excitation generation module (includes usual heat-bath routines)

use std::collections::HashMap;

use super::utils::bits::bits;
use super::utils::ints::combine_2;
use super::utils::read_input::Global;

use super::ham::Ham;

use super::wf::Det;

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
    // TODO: Add singles generator
}

pub fn init_excite_generator(global: &Global, ham: &Ham) -> ExciteGenerator {
    // Initialize by sorting double excitation element for all pairs
    let mut excite_gen: ExciteGenerator = ExciteGenerator { same_spin_doub_generator: Default::default(), opp_spin_doub_generator: Default::default() };
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

    pub fn iter(&self, det: Det) -> Vec<dyn Iterator<Type=Doub>> {
        // Return a vector of iterators for this det's double excitations
        // Can stay with the det, so future HCI iterations don't replicate
        // excite generation effort (at least for the highest-weight dets,
        // since storing it for all of them is probably too much)

        let mut excite_vec: Vec<dyn Iterator<Type=Doub>> = vec![];

        // Opposite spin doubles
        for i in bits(det.up) {
            for j in bits(det.dn) {
                excite_vec.append(self.opp_spin_doub_generator[&OPair(i, j)].to_iter());
            }
        }

        // Same spin doubles
        for i in bits(det.up) {
            for j in bits(det.up) {
                excite_vec.append(self.same_spin_doub_generator[&combine_2(i, j)].to_iter());
            }
        }
        for i in bits(det.dn) {
            for j in bits(det.dn) {
                excite_vec.append(self.same_spin_doub_generator[&combine_2(i, j)].to_iter());
            }
        }

        // TODO: Single excitations

        excite_vec
    }

}