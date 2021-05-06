// Initialize sorted excitation arrays

use core::cmp::Ordering::Equal;
use core::default::Default;
use std::collections::HashMap;

use crate::excite::{StoredExcite, Orbs};
use crate::utils::read_input::Global;
use crate::ham::Ham;

// Heat-bath excitation generator
pub struct ExciteGenerator {

    // Doubles:
    // max_(same/opp)_spin_doub is the global largest-magnitude double
    // each orbital pair maps onto a sorted list of target orbital pairs
    pub max_opp_doub: f64,
    pub opp_doub_generator: HashMap<Orbs, Vec<StoredExcite>>,
    pub max_same_doub: f64,
    pub same_doub_generator: HashMap<Orbs, Vec<StoredExcite>>,

    // Singles:
    // max_sing is the global largest-magnitude single,
    // sing_generator is a vector of single excitations along with corresponding
    // target orbs and max possible values; unlike doubles, magnitudes must be
    // re-computed because these depend on other occupied orbitals
    pub max_sing: f64,
    pub sing_generator: HashMap<Orbs, Vec<StoredExcite>>,

}

pub fn init_excite_generator(global: &Global, ham: &Ham) -> ExciteGenerator {
    // Initialize by sorting double excitation element for all pairs
    let mut excite_gen: ExciteGenerator = ExciteGenerator {
        max_same_doub: 0.0, max_opp_doub: 0.0, same_doub_generator: Default::default(),
        opp_doub_generator: Default::default(), max_sing: 0.0, sing_generator: Default:: default()
    };

    let mut v: Vec<StoredExcite>;
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
                    h = (ham.direct(p, q, r, s)).abs();
                    if h > excite_gen.max_opp_doub { excite_gen.max_opp_doub = h; }
                    v.push(
                        StoredExcite {
                            target: Orbs::Double((r, s)),
                            abs_h: h,
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));
            excite_gen.opp_doub_generator.insert(Orbs::Double((p, q)), v);
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
                    h = (ham.direct_plus_exchange(p, q, r, s)).abs();
                    if h > excite_gen.max_same_doub { excite_gen.max_same_doub = h; }
                    v.push(
                        StoredExcite {
                            target: Orbs::Double((r, s)),
                            abs_h: h,
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));
            excite_gen.same_doub_generator.insert(Orbs::Double((p, q)), v);
        }
    }

    // Single excitations
    // Loop over all p, r:
    // For each, loop over all remaining q (of either spin), get all matrix elements
    // (with signs)
    // Then, compute the max excitation from p to r as follows:
    // max(|f_pr + sum_{q in A} g_pqqr|, |f_pr + sum_{q in B} g_pqqr|),
    // where the sums on q are over the N-1 other orbitals
    // that are either the largest (A) or the smallest (B) - true value, not abs value
    // Compute v_same and v_opp vectors for each pr, which are the same-spin
    // and opposite-spin components: (g_prqq - g_pqqr) and g_prqq, respectively
    // Assumes that same number of up and dn spin electrons for now (easy to fix later,
    // but for now just does the up spin part)
    let mut max_sing_list: Vec<f64> = vec![];
    let mut v_sing: Vec<StoredExcite>;
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
                v_same.push(ham.direct_plus_exchange(p, r, q, q));
                v_opp.push(ham.direct(p, r, q, q));
            }
            v_same.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            v_opp.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            max1 = ham.one_body(p, r)
                + v_same[ .. (global.nup - 1) as usize ].iter().sum::<f64>()
                + v_opp[ .. global.ndn as usize ].iter().sum::<f64>();
            max2 = ham.one_body(p, r)
                + v_same[ v_same.len() - (global.nup - 1) as usize .. ].iter().sum::<f64>()
                + v_opp[ v_same.len() - global.ndn as usize .. ].iter().sum::<f64>();
            v_sing.push(
                StoredExcite {
                    target: Orbs::Single(r),
                    abs_h: {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } },
                }
            );
        }
        // Sort the max excites coming from this p in decreasing order by magnitude
        v_sing.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));
        // Finally, add this sorted vector to sing_generator
        excite_gen.sing_generator.insert(Orbs::Single(p), v_sing);
    }

    // Now, for each p, get its largest-magnitude excite among all p->r excites from above
    // (The first element in sing_generator[p] since it's already sorted in decreasing order)
    for p in 0..global.norb {
        max_sing_list.push(excite_gen.sing_generator[p as usize][0].abs_h);
    }

    // Finally, get the global max_sing by taking max_p over the above
    excite_gen.max_sing = max_sing_list.iter().cloned().fold(0./0., f64::max);

    excite_gen
}
