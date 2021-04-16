// Excitation generation module (includes usual heat-bath routines)

use std::collections::HashMap;

use super::utils::bits::bits;
use super::utils::read_input::Global;

use super::ham::Ham;

use super::wf::Det;
use std::cmp;
use std::cmp::max;
use std::cmp::Ordering::Equal;
use std::hash::{Hash, Hasher};

// macro_rules! impl_comparisons {
//     ($ty:ty, $($field:tt), *) => {
//         impl Ord for $ty {
//             fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//                 $(
//                     let cmp = self.$field.cmp(&other.$field);
//                     if cmp != std::cmp::Ordering::Equal {
//                         return cmp;
//                     }
//                 )*
//                 std::cmp::Ordering::Equal
//             }
//         }
//
//         impl PartialOrd for $ty {
//             fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//                 Some(self.cmp(other))
//             }
//         }
//
//         impl PartialEq for $ty {
//             fn eq(&self, other: &Self) -> bool {
//                 self.cmp(other) == std::cmp::Ordering::Equal
//             }
//         }
//
//         impl Eq for $ty {}
//     };
// }

// Orbital pair
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    target: OPair,
    abs_h: f64,
}

// Single excitation doublet (r, max |H|)
pub struct Sing {
    target: i32,
    max_abs_h: f64,
}

// Heat-bath excitation generator
pub struct ExciteGenerator {

    // Doubles:
    // max_doub is the global largest-magnitude double
    // each orbital pair maps onto a sorted list of target orbital pairs
    pub(crate) max_doub: f64,
    same_spin_doub_generator: HashMap<OPair, Vec<Doub>>,
    opp_spin_doub_generator: HashMap<OPair, Vec<Doub>>,

    // Singles:
    // max_sing is the global largest-magnitude single,
    // max_sing_list(p) is the largest-magnitude single coming out of orbital p
    // sing_generator is a hashmap of single excitations along with corresponding
    // max possible values; unlike doubles, magnitudes must be
    // re-computed because these depend on other occupied orbitals
    max_sing: f64,
    max_sing_list: Vec<f64>,
    sing_generator: Vec<Vec<Sing>>,

}

pub fn init_excite_generator(global: &Global, ham: &Ham) -> ExciteGenerator {
    // Initialize by sorting double excitation element for all pairs
    let mut excite_gen: ExciteGenerator = ExciteGenerator { max_doub: 0.0, same_spin_doub_generator: Default::default(), opp_spin_doub_generator: Default::default(), max_sing: 0.0, max_sing_list: vec![], sing_generator: vec![] };

    let mut v: Vec<Doub>;
    let mut h: f64;

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
                    h = (ham.get_int(p + 1, r + 1, q + 1, s + 1)).abs();
                    if h > excite_gen.max_doub { excite_gen.max_doub = h; }
                    v.push(
                        Doub{
                            target: OPair(r, s),
                            abs_h: h
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));
            println!("Opposite spin: Exciting orbitals: {} {}", p, q);
            for elem in &v {
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
                    // prqs - psqr
                    h = (ham.get_int(p + 1, r + 1, q + 1, s + 1)
                        - ham.get_int(p + 1, s + 1, q + 1, r + 1)).abs();
                    if h > excite_gen.max_doub { excite_gen.max_doub = h; }
                    v.push(
                        Doub{
                            target: OPair(r, s),
                            abs_h: h
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));
            println!("Same spin: Exciting orbitals: {} {}", p, q);
            for elem in &v {
                println!("{} {} {}", elem.target.0, elem.target.1, elem.abs_h);
            }
            excite_gen.same_spin_doub_generator.insert(OPair(p, q), v);
        }
    }

    // Single excitations
    // Loop over all p, r:
    // For each, loop over all remaining q (of either spin), get all matrix elements
    // (with signs)
    // Then, compute the max excitation from p to r as follows:
    // max(|f_pr + sum_q g_pqqr|, |f_pr + sum_q g_pqqr|),
    // where the sums on q are over the N-1 other orbitals
    // that are either the largest or the smallest (true value, not abs)
    // Compute v_same and v_opp vectors for each pr, which are the same-spin
    // and opposite-spin components: (g_prqq - g_pqqr) and g_prqq, respectively
    // Assumes that same number of up and dn spin electrons for now (easy to fix later,
    // but for now just does the up spin part)
    let mut v_sing: Vec<Sing>;
    let mut v_same: Vec<f64>;
    let mut v_opp: Vec<f64>;
    let mut max1: f64 = 0.0;
    let mut max2: f64 = 0.0;
    for p in 0..global.norb {
        v_sing = vec![];
        for r in 0..global.norb {
            if p == r { continue; }
            v_same = vec![];
            v_opp = vec![];
            for q in 0..global.norb {
                if p==q || q==r { continue; }
                v_same.push(
                    ham.get_int(p + 1, r + 1, q + 1, q + 1)
                        - ham.get_int(p + 1, q + 1, q + 1, r + 1)
                );
                v_opp.push(
                    ham.get_int(p + 1, r + 1, q + 1, q + 1)
                );
            }
            v_same.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            v_opp.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            max1 = ham.get_int(p + 1, r + 1, 0, 0)
                + v_same[ .. (global.nup - 1) as usize ].iter().sum::<f64>()
                + v_opp[ .. global.ndn as usize ].iter().sum::<f64>();
            max2 = ham.get_int(p + 1, r + 1, 0, 0)
                + v_same[ -(global.nup - 1) as usize .. ].iter().sum::<f64>()
                + v_opp[ -global.ndn as usize .. ].iter().sum::<f64>();
            v_sing.push(
                Sing{
                    target: r,
                    max_abs_h: {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } }
                }
            );
        }
        // Sort the max excites coming from this p in decreasing order by magnitude
        v_sing.sort_by(|a, b| b.max_abs_h.partial_cmp(&a.max_abs_h).unwrap_or(Equal));
        // Finally, add this sorted vector to sing_generator
        excite_gen.sing_generator.push(v_sing);
    }

    // Now, for each p, get its largest-magnitude excite among all p->r excites from above
    // (The first element in sing_generator[p] since it's already sorted in decreasing order)
    for p in 0..global.norb {
        excite_gen.max_sing_list.push(excite_gen.sing_generator[p as usize][0].max_abs_h);
    }

    // Finally, get the global max_sing by taking max_p over the above
    //excite_gen.max_sing = *excite_gen.max_sing_list.iter().max().unwrap();
    excite_gen.max_sing = excite_gen.max_sing_list.iter().cloned().fold(0./0., f64::max);

    excite_gen
}


// impl ExciteGenerator {
//
//     pub fn iter(&self, det: Det) -> Vec<dyn Iterator<Type = Doub>> {
//         // Return a vector of iterators for this det's double excitations
//         // Can stay with the det, so future HCI iterations don't replicate
//         // excite generation effort (at least for the highest-weight dets,
//         // since storing it for all of them is probably too much)
//
//         let mut excite_vec: Vec<dyn Iterator<Type=Doub>> = vec![];
//
//         // Opposite spin doubles
//         for i in bits(det.up) {
//             for j in bits(det.dn) {
//                 excite_vec.append(self.opp_spin_doub_generator[&OPair(i, j)].to_iter());
//             }
//         }
//
//         // Same spin doubles
//         for i in bits(det.up) {
//             for j in bits(det.up) {
//                 excite_vec.append(self.same_spin_doub_generator[&OPair(i, j)].to_iter());
//             }
//         }
//         for i in bits(det.dn) {
//             for j in bits(det.dn) {
//                 excite_vec.append(self.same_spin_doub_generator[&OPair(i, j)].to_iter());
//             }
//         }
//
//         // TODO: Single excitations
//
//         excite_vec
//     }
//
// }
