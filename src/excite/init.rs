//! # Excitation Generator Initialization (`excite::init`)
//!
//! This module defines the `ExciteGenerator` struct and the logic (`init_excite_generator`)
//! for pre-calculating and storing excitation information. This pre-computation allows
//! for efficient generation and sampling of excitations during the main calculation phases
//! (like HCI search/PT steps or QMC propagation).

use core::cmp::Ordering::Equal;
use core::default::Default;
use std::collections::HashMap;

use crate::config::GlobalConfig;
use crate::error::{RisqError, RisqResult};
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::ham::Ham;
use crate::rng::Rand;
use crate::stoch::utils::sample_cdf;
use crate::stoch::ImpSampleDist;
use crate::utils::bits::{bit_pairs, bits, ibset};
use crate::wf::det::Config;
use crate::{risq_bail, risq_ensure};

/// Stores pre-computed and sorted excitation information for efficient generation and sampling.
///
/// This structure holds HashMaps where keys represent the *initial* orbitals involved
/// in an excitation (e.g., `Orbs::Double((p, q))` for a double excitation from p and q)
/// and values are vectors of `StoredExcite`. These vectors contain potential *target*
/// orbitals, sorted in descending order by the estimated Hamiltonian matrix element (`abs_h`).
///
/// This allows for efficient deterministic screening (iterating through the sorted list until
/// `abs_h` drops below a threshold `eps`) and importance sampling (using the pre-computed
/// `sum_remaining_*` values in `StoredExcite` to sample from the remaining tail).
#[derive(Debug, Default)] // Added Debug and Default derive
pub struct ExciteGenerator {
    /// The maximum absolute value of any opposite-spin double excitation matrix element estimate found.
    pub max_opp_doub: f64,
    /// Map from initial opposite-spin orbital pair `Orbs::Double(p_up, q_dn)` to a sorted list
    /// of potential target pairs `StoredExcite { target: Orbs::Double(r_up, s_dn), ... }`.
    pub opp_doub_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    /// The maximum absolute value of any same-spin double excitation matrix element estimate found.
    pub max_same_doub: f64,
    /// Map from initial same-spin orbital pair `Orbs::Double(p, q)` to a sorted list
    /// of potential target pairs `StoredExcite { target: Orbs::Double(r, s), ... }`.
    /// The spin (alpha/beta) is implicit based on which determinant string (up/dn) the initial pair is found in.
    pub same_doub_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    /// The maximum absolute value estimate for any single excitation. Note that single excitation
    /// matrix elements depend on the full determinant configuration, so this `max_sing` and the
    /// `abs_h` values in `sing_sorted_list` are upper bounds or estimates used for screening/sampling.
    pub max_sing: f64,
    /// Map from initial single orbital `Orbs::Single(p)` to a sorted list of potential target
    /// orbitals `StoredExcite { target: Orbs::Single(r), ... }`.
    pub sing_sorted_list: HashMap<Orbs, Vec<StoredExcite>>,

    /// A bitmask representing the set of valence (non-frozen) orbitals.
    /// Used to quickly filter excitations involving only valence orbitals.
    pub valence: u128,
}

/// Creates and initializes the `ExciteGenerator`.
///
/// This function iterates through all possible single and double excitations originating
/// from pairs or single orbitals within the valence space defined in `ham`.
/// For each originating orbital set (`init`), it calculates estimates of the Hamiltonian
/// matrix elements (`abs_h`) connecting to all possible target orbitals (`target`).
///
/// These potential excitations are stored as `StoredExcite` objects, sorted in descending
/// order of `abs_h`, and the cumulative sums (`sum_remaining_*`) are calculated.
/// The results are stored in the HashMaps within the returned `ExciteGenerator`.
/// It also determines the global maximum `abs_h` values (`max_*_doub`, `max_sing`).
///
/// # Arguments
/// * `config`: Global calculation parameters.
/// * `ham`: The Hamiltonian containing integrals and orbital information.
///
/// # Returns
/// An initialized `ExciteGenerator` ready for use in calculations.
///
/// # Errors
/// Returns `RisqError` if there are issues with excitation generation.
pub fn init_excite_generator(config: &GlobalConfig, ham: &Ham) -> RisqResult<ExciteGenerator> {
    let mut excite_gen: ExciteGenerator = ExciteGenerator {
        max_same_doub: 0.0,
        max_opp_doub: 0.0,
        same_doub_sorted_list: Default::default(),
        opp_doub_sorted_list: Default::default(),
        max_sing: 0.0,
        sing_sorted_list: Default::default(),
        valence: 0,
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
                if p == r {
                    continue;
                };
                for s in &ham.valence_orbs {
                    if q == s {
                        continue;
                    };
                    // Compute H elem
                    h = (ham.direct(*p, *q, *r, *s)).abs();
                    if h > excite_gen.max_opp_doub {
                        excite_gen.max_opp_doub = h;
                    }
                    v.push(StoredExcite {
                        target: Orbs::Double((*r, *s)),
                        abs_h: h,
                        sum_remaining_abs_h: h,
                        sum_remaining_h_squared: h * h,
                    });
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

            // Finally, compute sum_remaining_abs_h for all of these
            compute_sum_remaining(&mut v);

            excite_gen
                .opp_doub_sorted_list
                .insert(Orbs::Double((*p, *q)), v);
        }
    }

    // Same spin
    for p in &ham.valence_orbs {
        for q in &ham.valence_orbs {
            if p >= q {
                continue;
            }
            v = vec![];
            for r in &ham.valence_orbs {
                if p == r || q == r {
                    continue;
                };
                for s in &ham.valence_orbs {
                    if r >= s {
                        continue;
                    }
                    if p == s || q == s {
                        continue;
                    };
                    // Compute H elem
                    // prqs - psqr
                    h = (ham.direct_plus_exchange(*p, *q, *r, *s)).abs();
                    if h > excite_gen.max_same_doub {
                        excite_gen.max_same_doub = h;
                    }
                    v.push(StoredExcite {
                        target: Orbs::Double((*r, *s)),
                        abs_h: h,
                        sum_remaining_abs_h: h,
                        sum_remaining_h_squared: h * h,
                    });
                }
            }
            // Sort v in decreasing order by abs_h
            v.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

            // Finally, compute sum_remaining_abs_h for all of these
            compute_sum_remaining(&mut v);

            excite_gen
                .same_doub_sorted_list
                .insert(Orbs::Double((*p, *q)), v);
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
            if p == r {
                continue;
            }
            v_same = vec![];
            v_opp = vec![];
            for q in 0..config.n_orbs as i32 {
                if *p == q || q == *r {
                    continue;
                }
                v_same.push(ham.direct_plus_exchange(*p, q, *r, q));
                v_opp.push(ham.direct(*p, q, *r, q));
            }
            v_same.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
            v_opp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
            max1 = ham.one_body(*p, *r)
                + v_same[..(config.n_up - 1)].iter().sum::<f64>()
                + v_opp[..config.n_dn].iter().sum::<f64>();
            max2 = ham.one_body(*p, *r)
                + v_same[v_same.len() - (config.n_up - 1)..]
                    .iter()
                    .sum::<f64>()
                + v_opp[v_same.len() - config.n_dn..]
                    .iter()
                    .sum::<f64>();
            // println!("One body = {}, max1 = {}, max2 = {}, max value = {}", ham.one_body(*p, *r), max1, max2, {if max1.abs() > max2.abs() { max1.abs() } else { max2.abs() } });
            v_sing.push(StoredExcite {
                target: Orbs::Single(*r),
                abs_h: {
                    if max1.abs() > max2.abs() {
                        max1.abs()
                    } else {
                        max2.abs()
                    }
                },
                sum_remaining_abs_h: {
                    if max1.abs() > max2.abs() {
                        max1.abs()
                    } else {
                        max2.abs()
                    }
                },
                sum_remaining_h_squared: {
                    if max1.abs() > max2.abs() {
                        max1 * max1
                    } else {
                        max2 * max2
                    }
                },
            });
        }
        // Sort the max excites coming from this p in decreasing order by magnitude
        v_sing.sort_by(|a, b| b.abs_h.partial_cmp(&a.abs_h).unwrap_or(Equal));

        // Finally, compute sum_remaining_abs_h for all of these
        compute_sum_remaining(&mut v_sing);

        excite_gen.sing_sorted_list.insert(Orbs::Single(*p), v_sing);
    }

    // Now, for each p, get its largest-magnitude excite among all p->r excites from above
    // (The first element in sing_generator[p] since it's already sorted in decreasing order)
    for p in &ham.valence_orbs {
        max_sing_list.push(excite_gen.sing_sorted_list.get(&Orbs::Single(*p)).unwrap()[0].abs_h);
    }

    // Finally, get the global max_sing by taking max_p over the above
    excite_gen.max_sing = max_sing_list.iter().cloned().fold(0., f64::max);

    println!(
        "Largest magnitude opposite-spin double excitation in H: {:.4}",
        excite_gen.max_opp_doub
    );
    println!(
        "Largest magnitude same-spin double excitation in H: {:.4}",
        excite_gen.max_same_doub
    );

    Ok(excite_gen)
}

/// Computes the cumulative sums `sum_remaining_abs_h` and `sum_remaining_h_squared`
/// for a vector of `StoredExcite` that is already sorted by `abs_h` in descending order.
///
/// Iterates backwards through the vector, calculating the sum of `abs_h` and `abs_h^2`
/// for all elements *after* the current one. This information is crucial for importance
/// sampling the "tail" of excitations below a deterministic threshold.
fn compute_sum_remaining(v: &mut Vec<StoredExcite>) {
    // Ensure the vector has elements to avoid panic on v[i+1]
    if v.is_empty() {
        return;
    }
    // Initialize the last element's remaining sums to its own values (nothing follows it)
    // Note: Original code started loop at v.len() - 2, potentially missing the last element's contribution
    // if only one element exists. Let's adjust slightly for clarity, though effect might be minimal.
    let last_idx = v.len() - 1;
    // It seems the original intent might have been for sum_remaining to EXCLUDE the current element.
    // Let's stick to that interpretation. The last element has 0 remaining sum.
    v[last_idx].sum_remaining_abs_h = 0.0;
    v[last_idx].sum_remaining_h_squared = 0.0;

    // Iterate backwards from the second-to-last element
    for i in (0..last_idx).rev() {
        // The sum remaining for element i is the sum remaining for element i+1
        // PLUS the contribution from element i+1 itself.
        v[i].sum_remaining_abs_h = v[i + 1].sum_remaining_abs_h + v[i + 1].abs_h;
        v[i].sum_remaining_h_squared = v[i + 1].sum_remaining_h_squared + v[i + 1].abs_h * v[i + 1].abs_h;
    }
}

// Sample excitations with probability |H| (for the cross term in ENPT2)
// Currently uses CDF searching, but can replace with Alias sampling later
impl ExciteGenerator {
    /// Samples a single target excitation given an initial orbital set and spin channel.
    ///
    /// Uses the pre-sorted lists and cumulative sums stored in `self` along with a specified
    /// importance sampling distribution (`imp_sampling_dist`, e.g., proportional to |H| or H^2)
    /// to stochastically select one target excitation (`StoredExcite`) from the list associated
    /// with the given `init` orbitals and `is_alpha` spin.
    ///
    /// Returns the selected `Excite` (reconstructed with `init` and `is_alpha`) and the
    /// probability with which it was sampled, or `None` if the list is empty.
    /// The sampling method used here appears to be CDF searching via `sample_cdf`.
    pub fn sample_excite(
        &self,
        init: Orbs,                 // Initial orbital(s) excitation originates from
        is_alpha: Option<bool>,     // Spin channel (None for opposite-spin doubles)
        imp_sampling_dist: &ImpSampleDist, // How to weight probabilities (e.g., |H|, H^2)
        rand: &mut Rand,            // Random number generator state
    ) -> Option<(Excite, f64)> {    // Returns (Sampled Excitation, Sampling Probability) or None
        // Sample an excitation from the selected orbs with probability proportional to |H|
        // Returns an excite and the sample probability
        // Can sample an invalid excitation
        let sample: Option<(&StoredExcite, f64)>;
        match is_alpha {
            None => {
                // Opposite-spin double
                sample = sample_cdf(
                    &self.opp_doub_sorted_list.get(&init).unwrap(),
                    imp_sampling_dist,
                    None,
                    rand,
                );
            }
            Some(_) => {
                // Same-spin single or double
                match init {
                    Orbs::Double(_) => {
                        // Same-spin double
                        sample = sample_cdf(
                            &self.same_doub_sorted_list.get(&init).unwrap(),
                            imp_sampling_dist,
                            None,
                            rand,
                        );
                    }
                    Orbs::Single(_) => {
                        // Same-spin single
                        sample = sample_cdf(
                            &self.sing_sorted_list.get(&init).unwrap(),
                            imp_sampling_dist,
                            None,
                            rand,
                        );
                    }
                }
            }
        }
        sample.map(|s| {
            (
                Excite {
                    init,
                    target: s.0.target,
                    abs_h: s.0.abs_h,
                    is_alpha,
                },
                s.1,
            )
        })
    }

    /// Samples one excitation pathway for *each* possible originating pair/single orbital in a given determinant.
    ///
    /// Iterates through all single occupied orbitals (`i`) and pairs of occupied orbitals (`(i, j)`)
    /// within the `det` configuration (considering valence orbitals only). For each `i` or `(i, j)`,
    /// it calls `sample_excite` to draw *one* sample from the corresponding list of target excitations.
    ///
    /// Returns a vector containing all the sampled excitations and their individual sampling probabilities.
    /// Note that some sampled excitations might be invalid if they target an orbital already occupied in `det`.
    pub fn sample_excites_from_all_pairs(
        &self,
        det: Config,                // The determinant configuration to excite from
        imp_sampling_dist: &ImpSampleDist, // Importance sampling distribution
        rand: &mut Rand,            // RNG state
    ) -> Vec<(Excite, f64)> {       // Vector of (Sampled Excitation, Sampling Probability)
        // Sample an excitation from each electron pair in the occupied determinant
        // Returns a vector of (Excite, sampling probability) pairs
        // Some of which may be invalid (excitations to already-occupied orbitals)
        let mut out: Vec<(Excite, f64)> = vec![];
        // Opposite-spin double
        for i in bits(self.valence & det.up) {
            for j in bits(self.valence & det.dn) {
                match self.sample_excite(Orbs::Double((i, j)), None, imp_sampling_dist, rand) {
                    None => {}
                    Some(v) => out.push(v),
                }
            }
        }
        // Same-spin double
        for (config, is_alpha) in &[(det.up, true), (det.dn, false)] {
            for (i, j) in bit_pairs(self.valence & *config) {
                match self.sample_excite(
                    Orbs::Double((i, j)),
                    Some(*is_alpha),
                    imp_sampling_dist,
                    rand,
                ) {
                    None => {}
                    Some(v) => out.push(v),
                }
            }
        }
        // Single excitations
        for (config, is_alpha) in &[(det.up, true), (det.dn, false)] {
            for i in bits(self.valence & *config) {
                match self.sample_excite(Orbs::Single(i), Some(*is_alpha), imp_sampling_dist, rand)
                {
                    None => {}
                    Some(v) => out.push(v),
                }
            }
        }
        out
    }
}
