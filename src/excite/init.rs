// Initialize sorted excitation arrays

use core::cmp::Ordering::Equal;
use core::default::Default;
use std::collections::HashMap;

use crate::excite::{StoredExcite, Orbs, Excite};
use crate::utils::read_input::Global;
use crate::ham::Ham;
use crate::wf::det::Config;
use crate::stoch::ImpSampleDist;
use crate::stoch::utils::sample_cdf;
use crate::utils::bits::{bits, bit_pairs, ibset};

// Heat-bath excitation generator
// Contains sorted lists of excitations, for efficient deterministic and importance sampled treatment
// Also contains alias samplers of stored excitations, for fully stochastic importance sampling (TODO)
pub struct ExciteGenerator {

    // Doubles:
    // max_(same/opp)_spin_doub is the global largest-magnitude double
    // each orbital pair maps onto a sorted list of target orbital pairs
    pub max_opp_doub: f64,
    pub opp_doub_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    pub max_same_doub: f64,
    pub same_doub_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    // Singles:
    // max_sing is the global largest-magnitude single,
    // sing_sorted_list is a vector of single excitations along with corresponding
    // target orbs and max possible values; unlike doubles, magnitudes must be
    // re-computed because these depend on other occupied orbitals
    pub max_sing: f64,
    pub sing_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    // Valence orbital screen
    pub valence: u128,
}

pub fn init_excite_generator(global: &Global, ham: &Ham) -> ExciteGenerator {
    // Initialize by sorting double excitation element for all pairs
    let mut excite_gen: ExciteGenerator = ExciteGenerator {
        max_same_doub: 0.0, max_opp_doub: 0.0, same_doub_sorted_list: Default::default(),
        opp_doub_sorted_list: Default::default(), max_sing: 0.0, sing_sorted_list: Default:: default(),
        valence: 0
    };

    for i in &ham.valence_orbs {
        excite_gen.valence = ibset(excite_gen.valence, *i);
    }

    let mut v: Vec<StoredExcite>;
    let mut h: f64;

    // Opposite spin
    // Assume p/r are up, q/s are dn
    for p in &ham.valence_orbs {
        for q in &ham.valence_orbs {
            v = vec![];
            for r in &ham.valence_orbs {
                if p == r  { continue; };
                for s in &ham.valence_orbs {
                    if  q == s { continue; };
                    // Compute H elem
                    h = (ham.direct(*p, *q, *r, *s)).abs();
                    if h > excite_gen.max_opp_doub { excite_gen.max_opp_doub = h; }
                    v.push(
                        StoredExcite {
                            target: Orbs::Double((*r, *s)),
                            abs_h: h,
                            sum_remaining_abs_h: h,
                            sum_remaining_h_squared: h * h,
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

            // Finally, compute sum_remaining_abs_h for all of these
            for i in (0 .. v.len() - 1).rev() {
                v[i].sum_remaining_abs_h = v[i + 1].sum_remaining_abs_h + v[i].abs_h;
                v[i].sum_remaining_h_squared = v[i + 1].sum_remaining_h_squared + v[i].abs_h * v[i].abs_h;
            }

            excite_gen.opp_doub_sorted_list.insert(Orbs::Double((*p, *q)), v);
        }
    }

    // Same spin
    for p in &ham.valence_orbs {
        for q in &ham.valence_orbs {
            if p >= q { continue; }
            v = vec![];
            for r in &ham.valence_orbs {
                if p == r || q == r { continue; };
                for s in &ham.valence_orbs {
                    if r >= s { continue; }
                    if p == s || q == s { continue; };
                    // Compute H elem
                    // prqs - psqr
                    h = (ham.direct_plus_exchange(*p, *q, *r, *s)).abs();
                    if h > excite_gen.max_same_doub { excite_gen.max_same_doub = h; }
                    v.push(
                        StoredExcite {
                            target: Orbs::Double((*r, *s)),
                            abs_h: h,
                            sum_remaining_abs_h: h,
                            sum_remaining_h_squared: h * h,
                        }
                    );
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

            // Finally, compute sum_remaining_abs_h for all of these
            for i in (0 .. v.len() - 1).rev() {
                v[i].sum_remaining_abs_h = v[i + 1].sum_remaining_abs_h + v[i].abs_h;
                v[i].sum_remaining_h_squared = v[i + 1].sum_remaining_h_squared + v[i].abs_h * v[i].abs_h;
            }

            excite_gen.same_doub_sorted_list.insert(Orbs::Double((*p, *q)), v);
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
    let mut max1: f64;
    let mut max2: f64;
    for p in &ham.valence_orbs {
        v_sing = vec![];
        for r in &ham.valence_orbs {
            if p == r { continue; }
            v_same = vec![];
            v_opp = vec![];
            for q in 0..global.norb {
                if *p == q || q == *r { continue; }
                v_same.push(ham.direct_plus_exchange(*p, q, *r, q));
                v_opp.push(ham.direct(*p, q, *r, q));
            }
            v_same.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            v_opp.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
            max1 = ham.one_body(*p, *r)
                + v_same[ .. (global.nup - 1) as usize ].iter().sum::<f64>()
                + v_opp[ .. global.ndn as usize ].iter().sum::<f64>();
            max2 = ham.one_body(*p, *r)
                + v_same[ v_same.len() - (global.nup - 1) as usize .. ].iter().sum::<f64>()
                + v_opp[ v_same.len() - global.ndn as usize .. ].iter().sum::<f64>();
            // println!("One body = {}, max1 = {}, max2 = {}, max value = {}", ham.one_body(*p, *r), max1, max2, {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } });
            v_sing.push(
                StoredExcite {
                    target: Orbs::Single(*r),
                    abs_h: {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } },
                    sum_remaining_abs_h: {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } },
                    sum_remaining_h_squared: {if max1.abs() > max2.abs() { max1 * max1 } else { max2 * max2 } },
                }
            );
        }
        // Sort the max excites coming from this p in decreasing order by magnitude
        v_sing.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

        // Finally, compute sum_remaining_abs_h for all of these
        for i in (0 .. v_sing.len() - 1).rev() {
            v_sing[i].sum_remaining_abs_h = v_sing[i + 1].sum_remaining_abs_h + v_sing[i].abs_h;
            v_sing[i].sum_remaining_h_squared = v_sing[i + 1].sum_remaining_h_squared + v_sing[i].abs_h * v_sing[i].abs_h;
        }

        excite_gen.sing_sorted_list.insert(Orbs::Single(*p), v_sing);
    }

    // Now, for each p, get its largest-magnitude excite among all p->r excites from above
    // (The first element in sing_generator[p] since it's already sorted in decreasing order)
    for p in &ham.valence_orbs {
        max_sing_list.push(excite_gen.sing_sorted_list.get(&Orbs::Single(*p)).unwrap()[0].abs_h);
    }

    // Finally, get the global max_sing by taking max_p over the above
    excite_gen.max_sing = max_sing_list.iter().cloned().fold(0./0., f64::max);

    println!("Largest magnitude opposite-spin double excitation in H: {:.4}", excite_gen.max_opp_doub);
    println!("Largest magnitude same-spin double excitation in H: {:.4}", excite_gen.max_same_doub);

    excite_gen
}


// Sample excitations with probability |H| (for the cross term in ENPT2)
// Currently uses CDF searching, but can replace with Alias sampling later
impl ExciteGenerator {
    pub fn sample_excite(&self, init: Orbs, is_alpha: Option<bool>) -> Option<(Excite, f64)> {
        // Sample an excitation from the selected orbs with probability proportional to |H|
        // Returns an excite and the sample probability
        // Can sample an invalid excitation
        let sample: Option<(&StoredExcite, f64)>;
        match is_alpha {
            None => {
                // Opposite-spin double
                sample = sample_cdf(&self.opp_doub_sorted_list.get(&init).unwrap(), &ImpSampleDist::AbsHc, None);
                // Some((Excite { init, target: sample.0.target, abs_h: sample.0.abs_h, is_alpha }, sample.1))
            }
            Some(_) => {
                // Same-spin single or double
                match init {
                    Orbs::Double(_) => {
                        // Same-spin double
                        sample = sample_cdf(&self.same_doub_sorted_list.get(&init).unwrap(), &ImpSampleDist::AbsHc, None);
                        // Some((Excite { init, target: sample.0.target, abs_h: sample.0.abs_h, is_alpha }, sample.1))
                    },
                    Orbs::Single(_) => {
                        // Same-spin single
                        // println!("Sing sorted list: {}", self.sing_sorted_list.get(&init).unwrap()[0].sum_remaining_abs_h);
                        sample = sample_cdf(&self.sing_sorted_list.get(&init).unwrap(), &ImpSampleDist::AbsHc, None);
                        // Some((Excite { init, target: sample.0.target, abs_h: sample.0.abs_h, is_alpha }, sample.1))
                    }
                }
            }
        }
        match sample {
            None => None,
            Some(s) => Some((Excite { init, target: s.0.target, abs_h: s.0.abs_h, is_alpha }, s.1))
        }
    }

    pub fn sample_excites_from_all_pairs(&self, det: Config) -> Vec<(Excite, f64)> {
        // Sample an excitation from each electron pair in the occupied determinant
        // Returns a vector of (Excite, sampling probability) pairs
        // Some of which may be invalid (excitations to already-occupied orbitals)
        let mut out: Vec<(Excite, f64)> = vec![];
        // Opposite-spin double
        for i in bits(self.valence & det.up) {
            for j in bits(self.valence & det.dn) {
                match self.sample_excite(Orbs::Double((i, j)), None) {
                    None => {},
                    Some(v) => out.push(v)
                }
            }
        }
        // Same-spin double
        for (config, is_alpha) in &[(det.up, true), (det.dn, false)] {
            for (i, j) in bit_pairs(self.valence & *config) {
                match self.sample_excite(Orbs::Double((i, j)), Some(*is_alpha)) {
                    None => {},
                    Some(v) => out.push(v)
                }
            }
        }
        // Single excitations
        for (config, is_alpha) in &[(det.up, true), (det.dn, false)] {
            for i in bits(self.valence & *config) {
                match self.sample_excite(Orbs::Single(i), Some(*is_alpha)) {
                    None => {},
                    Some(v) => out.push(v)
                }
            }
        }
        out
    }
}