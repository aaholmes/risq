//! # Variational Hamiltonian Generation (`var::ham_gen`)
//!
//! This module implements algorithms for constructing the sparse Hamiltonian matrix
//! within the variational space used in HCI. It appears to contain several different
//! strategies (`opp_algo`, `same_algo` parameters in `Global`) developed over time
//! to optimize the generation of off-diagonal matrix elements, particularly for
//! opposite-spin and same-spin double excitations.
//!
//! The core function `gen_sparse_ham_fast` orchestrates this process.

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::utils::read_input::Global;
use crate::var::off_diag::{add_el, add_el_and_spin_flipped};
use crate::var::utils::{remove_1e, remove_2e};
use crate::wf::det::Config;
use crate::wf::VarWf;
use std::cmp::Ordering::Equal;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;
// Removed commented-out thread pool imports

/// Helper function to insert a value into a vector stored within a HashMap.
/// If the key exists, appends the value to the existing vector.
/// If the key does not exist, creates a new vector with the value and inserts it.
fn insert_into_hashmap_of_vectors<T, U>(hashmap: &mut HashMap<T, Vec<U>>, key: T, value: U)
where
    T: Eq + Hash,
{
    match hashmap.get_mut(&key) {
        None => {
            hashmap.insert(key, vec![value]);
        }
        Some(v) => {
            v.push(value);
        }
    }
}

// Removed commented-out function `gen_doubles`

/// Generates the off-diagonal elements of the sparse variational Hamiltonian matrix.
///
/// This function implements potentially complex algorithms (selected by `global.opp_algo`
/// and `global.same_algo`) to efficiently find pairs of determinants (i, j) within the
/// current variational space (`wf.wf`) that are connected by single or double excitations
/// and computes the corresponding Hamiltonian matrix element H_ij.
///
/// It updates the `wf.sparse_ham` structure by adding the newly computed elements.
/// It focuses on generating elements involving *new* determinants added since the last
/// iteration (those with index >= `wf.n_stored_h()`), connecting them to all other
/// determinants in the space.
///
/// # Arguments
/// * `global`: Global calculation parameters, including algorithm selectors.
/// * `wf`: Mutable reference to the variational wavefunction structure. `wf.sparse_ham` is updated.
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator 
pub fn gen_sparse_ham_fast(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
) {
    // Generate Ham as a sparse matrix using my 2019 notes when I was working pro bono
    // For now, assumes that nup == ndn
    // Updates wf.sparse_ham if it already exists from the previous iteration

    // Setup
    let start_setup: Instant = Instant::now();

    // 1. Find all unique up dets, and create a map from each to the (indices, dn dets) of its corresponding determinants
    // TODO: make this a wf member variable so we don't have to regenerate the whole thing from scratch
    let mut unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    let mut new_unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    for (config, ind) in &wf.wf.inds {
        insert_into_hashmap_of_vectors(&mut unique_up_dict, config.up, (*ind, config.dn));
        if *ind >= wf.n_stored_h() {
            match new_unique_up_dict.get_mut(&config.up) {
                None => {
                    new_unique_up_dict.insert(config.up, vec![(*ind, config.dn)]);
                }
                Some(v) => {
                    v.push((*ind, config.dn));
                }
            }
        }
    }
    // if verbose {
    //     println!("Unique_up_dict:");
    //     for (key, val) in &unique_up_dict {
    //         println!("{}: ({}, {})", key, val[0].0, val[0].1);
    //     }
    // }

    // 2. Sort the unique up dets in decreasing order by corresponding number of determinants.
    let mut unique_ups_sorted: Vec<Unique> = Vec::with_capacity(unique_up_dict.len());
    for (up, dets) in &unique_up_dict {
        unique_ups_sorted.push(Unique {
            up: *up,
            n_dets: dets.len(),
            n_dets_remaining: 0,
        });
    }
    unique_ups_sorted.sort_by(|a, b| b.n_dets.partial_cmp(&a.n_dets).unwrap_or(Equal));
    let mut new_unique_ups_sorted: Vec<Unique> = Vec::with_capacity(new_unique_up_dict.len());
    for (up, dets) in &new_unique_up_dict {
        new_unique_ups_sorted.push(Unique {
            up: *up,
            n_dets: dets.len(),
            n_dets_remaining: 0,
        });
    }
    new_unique_ups_sorted.sort_by(|a, b| b.n_dets.partial_cmp(&a.n_dets).unwrap_or(Equal));

    // 3. Compute the cumulative number of determinants left for each unique det in sorted order.
    unique_ups_sorted.iter_mut().rev().fold(0, |acc, x| {
        x.n_dets_remaining = acc;
        acc + x.n_dets
    });
    new_unique_ups_sorted.iter_mut().rev().fold(0, |acc, x| {
        x.n_dets_remaining = acc;
        acc + x.n_dets
    });
    // if verbose { println!("Unique_ups_sorted:"); }
    // for val in &unique_ups_sorted {
    //     if verbose { println!("{}, {}, {}", val.up, val.n_dets, val.n_dets_remaining); }
    // }

    // 4. Generate a map, upSingles, from each unique up det to all following ones that are a single excite away
    // Note: Since we are adding onto the previously stored H, we excite from *new* dets to *all* dets
    let mut up_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
    for (ind, unique) in unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            insert_into_hashmap_of_vectors(&mut up_single_excite_constructor, up_r1, ind);
        }
    }
    // up_singles is *new* to *all*
    let mut up_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for unique in new_unique_ups_sorted.iter() {
        for up_r1 in remove_1e(unique.up) {
            if let Some(connected_inds) = up_single_excite_constructor.get(&up_r1) {
                for connected_ind in connected_inds {
                    // Skip over the diagonal excitations (there will be nup of these)
                    if unique.up == unique_ups_sorted[*connected_ind].up {
                        continue;
                    }
                    match up_singles.get_mut(&unique.up) {
                        None => {
                            up_singles
                                .insert(unique.up, vec![unique_ups_sorted[*connected_ind].up]);
                        }
                        Some(v) => {
                            v.push(unique_ups_sorted[*connected_ind].up);
                        }
                    }
                }
            }
        }
    }
    // if verbose { println!("Up_singles:"); }
    // for (key, val) in &up_singles {
    //     if verbose { println!("{}: {}", key, val[0]); }
    // }

    // 5. Find the unique dn dets, a map from each dn det to its index in the unique dn det list, and a map dnSingles
    // (analogous to upSingles, except want all excitations, not just to later unique dets)
    let mut unique_dns_set: HashSet<u128> = HashSet::default();
    let mut new_unique_dns_set: HashSet<u128> = HashSet::default();
    for (ind, det) in wf.wf.dets.iter().enumerate() {
        unique_dns_set.insert(det.config.dn);
        if ind >= wf.n_stored_h() {
            new_unique_dns_set.insert(det.config.dn);
        }
    }
    let mut unique_dns_vec: Vec<u128> = Vec::with_capacity(unique_dns_set.len());
    let mut new_unique_dns_vec: Vec<u128> = Vec::with_capacity(new_unique_dns_set.len());
    // Problem: following results in a randomly ordered vector!
    for dn in unique_dns_set {
        unique_dns_vec.push(dn);
    }
    for dn in new_unique_dns_set {
        new_unique_dns_vec.push(dn);
    }

    let mut dn_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
    for (ind, dn) in new_unique_dns_vec.iter().enumerate() {
        for dn_r1 in remove_1e(*dn) {
            insert_into_hashmap_of_vectors(&mut dn_single_excite_constructor, dn_r1, ind);
        }
    }
    // dn_singles is *all* to *new*
    let mut dn_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for dn in &unique_dns_vec {
        for dn_r1 in remove_1e(*dn) {
            if let Some(connected_inds) = dn_single_excite_constructor.get_mut(&dn_r1) {
                for connected_ind in connected_inds {
                    // can't compare *connected_ind to wf.n_stored_h() because connected_ind
                    // is the index in the vector of unique dns, not in the wf!
                    // Skip over the diagonal excitations (there will be ndn of these)
                    if *dn == unique_dns_vec[*connected_ind] {
                        continue;
                    }
                    match dn_singles.get_mut(&dn) {
                        None => {
                            dn_singles.insert(*dn, vec![unique_dns_vec[*connected_ind]]);
                        }
                        Some(v) => {
                            v.push(unique_dns_vec[*connected_ind]);
                        }
                    }
                }
            }
        }
    }
    // 6. Dn excite constructor: Look up (ind, dn) lists using keys (up, dn_r1)
    let mut dn_single_constructor: HashMap<Config, Vec<(usize, u128)>> = HashMap::default();
    if global.opp_algo == 4 {
        for (ind, det) in wf.wf.dets.iter().enumerate() {
            for dn_r1 in remove_1e(excite_gen.valence & det.config.dn) {
                let key = Config {
                    up: det.config.up,
                    dn: dn_r1,
                };
                // println!("Assigning key: {}", key);
                match dn_single_constructor.get_mut(&key) {
                    None => {
                        dn_single_constructor.insert(key, vec![(ind, det.config.dn)]);
                    }
                    Some(v) => {
                        v.push((ind, det.config.dn));
                    }
                }
            }
        }
    }
    // println!("dn_single_constructor:");
    // for (key, val) in dn_single_constructor.iter() {
    //     println!("Key: {}, val has length {}", key, val.len());
    // }
    println!("Time for H gen setup: {:?}", start_setup.elapsed());

    if wf.n_stored_h() == 0 {
        // Create new off-diagonal elements holder
        wf.new_sparse_ham();
    } else {
        // Expand old off-diagonal elements holder to hold new rows
        wf.expand_sparse_ham_rows();
    }

    // Removed commented-out thread pool code
    // Parameter for choosing which of the two same-spin algorithms to use
    // let max_n_dets_double_loop = global.ndn as usize; // ((global.ndn * (global.ndn - 1)) / 2) as usize;
    // });

    // Parameter for choosing which of the two same-spin algorithms to use
    // let max_n_dets_double_loop = global.ndn as usize; // ((global.ndn * (global.ndn - 1)) / 2) as usize;

    let start_opp: Instant = Instant::now();
    all_opposite_spin_excites(
        global,
        wf,
        ham,
        excite_gen,
        &unique_up_dict,
        &new_unique_up_dict,
        &new_unique_ups_sorted,
        &up_singles,
        &dn_singles,
        &dn_single_constructor,
    );
    println!("Time for opposite-spin: {:?}", start_opp.elapsed());

    let start_same: Instant = Instant::now();
    all_same_spin_excites(
        global,
        wf,
        ham,
        &unique_up_dict,
        &new_unique_up_dict,
        &mut unique_ups_sorted,
        // max_n_dets_double_loop,
    );
    println!("Time for same-spin: {:?}", start_same.elapsed());

    // Sort and remove duplicates
    wf.sparse_ham.sort_remove_duplicates();

    // Update wf.n_stored_h
    wf.update_n_stored_h(wf.wf.n);
}

/// Orchestrates the calculation of opposite-spin double excitation contributions to the sparse Hamiltonian.
/// Iterates through new unique alpha-spin strings and calls `opposite_spin_excites`.
/// Internal helper function for `gen_sparse_ham_fast`.
fn all_opposite_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator, // Passed down, potentially unused
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, // Map: up_string -> [(det_idx, dn_string)]
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, // Map for *new* dets
    new_unique_ups_sorted: &Vec<Unique>, // Sorted list of new unique up_strings + counts
    up_singles: &HashMap<u128, Vec<u128>>, // Map: up_string -> [connected_up_strings_via_single]
    dn_singles: &HashMap<u128, Vec<u128>>, // Map: dn_string -> [connected_dn_strings_via_single]
    dn_single_constructor: &HashMap<Config, Vec<(usize, u128)>>, // Map: (up, dn_r1) -> [(idx, dn)]
) {
    // Loop over new dets only
    for unique in new_unique_ups_sorted {
        // Opposite-spin excitations
        opposite_spin_excites(
            global,
            wf,
            ham,
            excite_gen,
            unique_up_dict,
            new_unique_up_dict,
            &up_singles,
            &dn_singles,
            dn_single_constructor,
            unique,
        );
    }
}

/// Orchestrates the calculation of same-spin double excitation contributions to the sparse Hamiltonian.
/// Iterates through all unique alpha-spin strings and calls `same_spin_excites`.
/// Internal helper function for `gen_sparse_ham_fast`.
fn all_same_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    unique_ups_sorted: &Vec<Unique>, // Changed to immutable borrow
    // max_n_dets_double_loop: usize, // Parameter seems unused now
) {
    for unique in unique_ups_sorted {
        // Same-spin excitations
        same_spin_excites(
            global,
            wf,
            ham,
            unique_up_dict,
            new_unique_up_dict,
            // max_n_dets_double_loop,
            unique,
        );
    }
}

/// Helper struct used in sorting unique alpha-spin strings.
#[derive(Debug)] // Added Debug derive
struct Unique {
    /// The unique alpha-spin bitstring.
    up: u128,
    /// The number of determinants in the wavefunction sharing this `up` string.
    n_dets: usize,
    /// The cumulative number of determinants associated with `up` strings *after* this one
    /// in a sorted list (used for algorithmic choices).
    n_dets_remaining: usize,
}

/// Calculates and adds opposite-spin double excitation matrix elements involving a specific `unique` up-string.
///
/// This function implements several different algorithms (selected by `global.opp_algo`)
/// for finding pairs of determinants `(i, j)` where `i` has the `unique.up` alpha string
/// and `j` has an alpha string connected to `unique.up` by a single excitation, AND
/// where the beta strings of `i` and `j` are also connected by a single excitation.
/// For each such pair `(i, j)`, it computes `H_ij` and adds it to `wf.sparse_ham`.
/// Focuses on connections where at least one determinant index is new (`>= wf.n_stored_h()`).
/// Internal helper function.
fn opposite_spin_excites( // Made private
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator, // Potentially unused
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    up_singles: &HashMap<u128, Vec<u128>>,
    dn_singles: &HashMap<u128, Vec<u128>>,
    dn_single_constructor: &HashMap<Config, Vec<(usize, u128)>>,
    unique: &Unique,
) {
    // Current status:
    // For frozen-core F2 in VTZ at equilibrium with eps_var = 3e-4, the full variational stage takes:
    // Algo 1: 63 s
    // Algo 2: 77 s
    // Algo 3: 132 s
    // Algo 4: 88 s
    // Algo 5: 287 s
    // but on the last iteration, algo 4 is 3.5 times faster than algo 2, so algo 4 may become faster for larger variational spaces

    // let start_this_opp: Instant = Instant::now();
    // let first_algo_complexity = unique.n_dets * (global.ndn * global.ndn) as usize + unique.n_dets_remaining; // Term in parentheses is an estimate
    // let second_algo_complexity = global.ndn as usize * (unique.n_dets + unique.n_dets_remaining);
    // let third_algo_complexity = unique.n_dets * unique.n_dets_remaining;
    // let mut first_algo_fastest = false;
    // let mut second_algo_fastest = true;
    // if first_algo_complexity <= second_algo_complexity {
    //     if first_algo_complexity <= third_algo_complexity { first_algo_fastest = true; }
    // } else {
    //     if second_algo_complexity <= third_algo_complexity { second_algo_fastest = true; }
    // }
    if global.opp_algo == 1 {
        // if verbose { println!("First algo fastest"); }
        // Use a single for-loop version of the double for-loop algorithm from "Fast SHCI"
        // up_singles is *new* to *all*
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                let mut dn_candidates: HashMap<u128, Vec<usize>> = HashMap::default();
                for dn in &new_unique_up_dict[&unique.up] {
                    if let Some(dns) = up_singles.get(&dn.1) {
                        for dn2 in dns {
                            match dn_candidates.get_mut(&dn2) {
                                None => {
                                    dn_candidates.insert(*dn2, vec![dn.0]);
                                }
                                Some(v) => {
                                    v.push(dn.0);
                                }
                            }
                        }
                    }
                }
                for up in ups {
                    for dn in &unique_up_dict[&up] {
                        match dn_candidates.get(&dn.1) {
                            None => {}
                            Some(dn_connections) => {
                                // We need to do this one in terms of up config rather than ind
                                let ind1 = wf.wf.inds[&Config { up: *up, dn: dn.1 }];
                                for dn_connection_ind in dn_connections {
                                    // Check that the index is new (because even if up and dn are independently new, (up, dn) can be old)
                                    if *dn_connection_ind >= wf.n_stored_h() {
                                        add_el(wf, ham, ind1, *dn_connection_ind, None);
                                        // only adds if elem != 0
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if global.opp_algo == 2 {
        // if verbose { println!("Second algo fastest"); }
        // Loop over ways to remove an electron to get all connected dets

        // up_singles are *new* to *all*
        // up_singles[&unique.up] can be empty, so...
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                let mut dn_single_excite_constructor: HashMap<u128, Vec<usize>> =
                    HashMap::default();
                for dn in &new_unique_up_dict[&unique.up] {
                    for dn_r1 in remove_1e(dn.1) {
                        match dn_single_excite_constructor.get_mut(&dn_r1) {
                            None => {
                                dn_single_excite_constructor.insert(dn_r1, vec![dn.0]);
                            }
                            Some(v) => {
                                v.push(dn.0);
                            }
                        }
                    }
                }
                for up in ups {
                    for dn in &unique_up_dict[&up] {
                        for dn_r1 in remove_1e(dn.1) {
                            match dn_single_excite_constructor.get(&dn_r1) {
                                None => {}
                                Some(connected_inds) => {
                                    for connected_ind in connected_inds {
                                        add_el(wf, ham, dn.0, *connected_ind, None);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if global.opp_algo == 3 {
        // if verbose { println!("Third algo fastest"); }
        // Loop over all pairs of dn dets, one from each of the two unique up configs that are linked
        // if verbose { println!("unique.up = {}", unique.up); }
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                for up in ups {
                    // if verbose { println!("up: {}", up); }
                    for ind1 in &new_unique_up_dict[&unique.up] {
                        // if verbose { println!("ind1= ({}, {})", ind1.0, ind1.1); }
                        for ind2 in &unique_up_dict[&up] {
                            // if verbose { println!("Found excitation: {} {}", ind1.0, ind2.0); }
                            add_el(wf, ham, ind1.0, ind2.0, None); // only adds if elem != 0
                        }
                    }
                }
            }
        }
    } else if global.opp_algo == 4 {
        // Algorithm 4: Loop over up-spin singles (new to all); for each, loop over ways of removing 1
        // dn electron, use that to look up all opposite spin double excites
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                // Loop over all dn dets in new_unique_up_dict[unique.up]
                for (new_ind, new_dn) in &new_unique_up_dict[&unique.up] {
                    // Loop over ways to remove 1 electron from this det
                    for new_dn_r1 in remove_1e(excite_gen.valence & *new_dn) {
                        // Look up this dn_r1 in the previously stored data structure that contains all dets it could be connected to
                        for all_up in ups {
                            // println!("Key: {}", Config{ up: *all_up, dn: new_dn_r1 });
                            if let Some(v) = dn_single_constructor.get(&Config {
                                up: *all_up,
                                dn: new_dn_r1,
                            }) {
                                for (all_ind, all_dn) in v {
                                    // Skip dn1 == dn2 (single excites)
                                    if all_dn == new_dn {
                                        continue;
                                    }
                                    // Ensure ind2 > ind1 (no double counting)
                                    if new_ind > all_ind {
                                        // Found excite new_ind, all_ind!
                                        add_el(wf, ham, *all_ind, *new_ind, None);
                                        // only adds if elem != 0
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if global.opp_algo == 5 {
        // Algorithm 5: Loop over all new dets: loop over up_singles (new to all) and dn_singles (new to all),
        // check whether resulting det exists in wf
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                for dn in &new_unique_up_dict[&unique.up] {
                    match up_singles.get(&dn.1) {
                        None => {}
                        Some(dns) => {
                            for all_up in ups {
                                for all_dn in dns {
                                    match wf.wf.inds.get(&Config {
                                        up: *all_up,
                                        dn: *all_dn,
                                    }) {
                                        None => {}
                                        Some(ind) => add_el(wf, ham, *ind, dn.0, None),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Algo 6: like algo 1 but without the dn_candidates data structure
        match up_singles.get(&unique.up) {
            None => {}
            Some(ups) => {
                for all_up in ups {
                    for all_dn in &unique_up_dict[&all_up] {
                        match dn_singles.get(&all_dn.1) {
                            None => {}
                            Some(dn_connections) => {
                                // We need to do this one in terms of up config rather than ind
                                let ind1 = wf.wf.inds[&Config {
                                    up: *all_up,
                                    dn: all_dn.1,
                                }];
                                for dn_new in dn_connections {
                                    if let Some(ind2) = wf.wf.inds.get(&Config {
                                        up: unique.up,
                                        dn: *dn_new,
                                    }) {
                                        // Check that the index is new (because even if up and dn are independently new, (up, dn) can be old)
                                        if *ind2 >= wf.n_stored_h() {
                                            add_el(wf, ham, ind1, *ind2, None); // only adds if elem != 0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // println!("Time for opposite-spin excites for up config {}: {:?}", unique.up, start_this_opp.elapsed());
}

/// Calculates and adds same-spin double excitation matrix elements involving a specific `unique` up-string.
///
/// Finds pairs of determinants `(i, j)` that share the same `unique.up` alpha string,
/// but whose beta strings (`dn_i`, `dn_j`) differ by a double excitation. It also handles
/// the symmetric case where alpha strings differ by a double excitation and beta strings are identical.
/// Computes `H_ij` for these pairs and adds them to `wf.sparse_ham`.
/// Focuses on connections where at least one determinant index is new (`>= wf.n_stored_h()`).
/// Contains different algorithmic approaches selected by `global.same_algo`.
/// Internal helper function.
fn same_spin_excites( // Made private
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    // max_n_dets_double_loop: usize, // Parameter seems unused now
    unique: &Unique,
) {
    if global.same_algo == 1 {
        //unique.n_dets <= max_n_dets_double_loop {
        // if verbose { println!("Same-spin first algo fastest"); }
        // Use the double for-loop algorithm from "Fast SHCI"
        for ind1 in unique_up_dict[&unique.up].iter() {
            // for ind2 in unique_up_dict[&unique.up][i_ind + 1..].iter() {
            //     add_el_and_spin_flipped(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
            // }
            if let Some(new_dn_inds) = new_unique_up_dict.get(&unique.up) {
                for ind2 in new_dn_inds {
                    if ind2.0 > ind1.0 {
                        // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                        add_el_and_spin_flipped(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
                    }
                }
            }
        }
    } else {
        // if verbose { println!("Same-spin second algo fastest"); }
        // Loop over all pairs of electrons to get all connections (asymptotically optimal in Full CI limit)
        let mut double_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
        match new_unique_up_dict.get(&unique.up) {
            None => {}
            Some(dns) => {
                for dn in dns {
                    for dn_r2 in remove_2e(dn.1) {
                        insert_into_hashmap_of_vectors(&mut double_excite_constructor, dn_r2, dn.0);
                    }
                }
            }
        }
        for dn in &unique_up_dict[&unique.up] {
            for dn_r2 in remove_2e(dn.1) {
                if let Some(dns) = double_excite_constructor.get(&dn_r2) {
                    for dn2 in dns {
                        if dn.0 != *dn2 {
                            // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                            add_el_and_spin_flipped(wf, ham, dn.0, *dn2); // only adds if elem != 0
                        }
                    }
                }
            }
        }
    }
}
