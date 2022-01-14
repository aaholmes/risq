//! Variational Hamiltonian generation algorithms

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
// extern crate threads_pool;
// use threads_pool::ThreadPool;
// use std::thread;
// use std::time::Duration;

/// Insert a (key, value) pair into a hashmap of vectors, i.e., append value to hashmap(key)
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

// pub fn gen_sparse_ham_doubles<'a>(wf: &'a Wf, ham: &'a Ham, excite_gen: &ExciteGenerator) -> SparseMatDoubles<'a> {
//     // Generate the sparse H where the doubles are done in O(N^2 N_det log N_det) time
//     // and O(N^2 N_det) space
//
//     let n = wf.n;
//
//     // Diagonal
//     let mut diag = DVector::from(
//         {
//             let mut d = Vec::with_capacity(n);
//             for det in &wf.dets {
//                 d.push(det.diag);
//             }
//             d
//         }
//     );
//
//     // Singles
//     let singles = gen_singles(wf, ham);
//
//     // Doubles
//     let doubles = gen_doubles(wf, ham, excite_gen);
//
//     // println!("Doubles:");
//     // for (k, v) in doubles.iter() {
//     //     println!("({} {}): {:?}", k.0, k.1, v);
//     // }
//
//     SparseMatDoubles{n, diag, singles, doubles, ham, wf}
//
// }

// pub fn gen_singles(wf: &Wf, ham: &Ham) -> CsMat<f64> {
//     // Loop over all electrons to get all connections (asymptotically optimal in Full CI limit)
//
//     let mut off_diag_elems = OffDiagElems::new(wf.n);
//     let start_gen_singles: Instant = Instant::now();
//     let mut unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
//     for (config, ind) in &wf.inds {
//         match unique_up_dict.get_mut(&config.up) {
//             None => { unique_up_dict.insert(config.up, vec![(*ind, config.dn)]); },
//             Some(mut v) => { v.push((*ind, config.dn)); }
//         }
//     }
//     for dns_this_up in unique_up_dict.values() {
//         let mut double_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
//         for dn in dns_this_up {
//             for dn_r1 in remove_1e(dn.1) {
//                 match double_excite_constructor.get_mut(&dn_r1) {
//                     None => { double_excite_constructor.insert(dn_r1, vec![dn.0]); },
//                     Some(mut v) => { v.push(dn.0); }
//                 }
//             }
//         }
//         for dn in dns_this_up {
//             for dn_r1 in remove_1e(dn.1) {
//                 for dn2 in &double_excite_constructor[&dn_r1] {
//                     if dn.0 != *dn2 {
//                         // Found dn-spin excitations (and their spin-flipped up-spin excitations):
//                         off_diag_elems.add_el_and_spin_flipped(wf, ham, dn.0, *dn2); // only adds if elem != 0
//                     }
//                 }
//             }
//         }
//     }
//     println!("Time to convert stored singles indices to sparse H: {:?}", start_gen_singles.elapsed());
//
//     // Put the matrix elements into a sparse matrix
//     let start_create_sparse: Instant = Instant::now();
//     let n = wf.n;
//     let shape = (n, n);
//     let indptr: Vec<usize> = off_diag_elems.nnz
//         .iter()
//         .scan(0, |acc, &x| {
//             *acc = *acc + x;
//             Some(*acc)
//         })
//         .collect();
//     let indices: Vec<usize> = off_diag_elems.indices.clone().into_iter().flatten().collect();
//     let data: Vec<f64> = off_diag_elems.values.clone().into_iter().flatten().collect();
//     println!("Variational Hamiltonian has {} nonzero single excitation elements", indices.len());
//
//     let mat = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
//     println!("Time to convert stored singles indices to sparse H: {:?}", start_create_sparse.elapsed());
//
//     match mat {
//         Err(_) => { panic!("Error in constructing CsMat"); }
//         Ok(res) => { return res; }
//     }
// }

#[cfg(test)]
pub fn gen_doubles(
    wf: &Wf,
    excite_gen: &ExciteGenerator,
) -> HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>> {
    // Output data structure:
    // key: orb1, orb2, is_alpha
    // value: vector of (config with orb1/2 removed, index of this config in wf) tuples, sorted by
    // the first element in the tuple

    // To find all pq->rs excites:
    // Look up the pq and rs vectors
    // Loop over the overlap in linear time
    // Each tuple in intersection contains the indices of this matrix element
    // This algorithm takes anywhere from O(N^4 N_det) time to O(N^6/M^2 N_det) time

    let mut doub = HashMap::default();

    for (det_ind, det) in wf.dets.iter().enumerate() {
        // Opposite spin
        for i in bits(excite_gen.valence & det.config.up) {
            for j in bits(excite_gen.valence & det.config.dn) {
                let det_r2 = Config {
                    up: ibclr(det.config.up, i),
                    dn: ibclr(det.config.dn, j),
                };
                let key = (i, j, None);
                insert_into_hashmap_of_vectors(&mut doub, key, (det_r2, det_ind))
            }
        }

        // Same spin
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for (i, j) in bit_pairs(excite_gen.valence & *config) {
                let det_r2 = {
                    if *is_alpha {
                        Config {
                            up: ibclr(ibclr(det.config.up, i), j),
                            dn: det.config.dn,
                        }
                    } else {
                        Config {
                            up: det.config.up,
                            dn: ibclr(ibclr(det.config.dn, i), j),
                        }
                    }
                };
                let key = (i, j, Some(*is_alpha));
                insert_into_hashmap_of_vectors(&mut doub, key, (det_r2, det_ind))
            }
        }
    }

    // Sort each vector in place by det_r2
    for vec in doub.values_mut() {
        vec.sort_by_key(|(det, _)| (det.up, det.dn));
    }

    doub
}

// pub fn gen_dense_ham_connections(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> DMatrix<f64> {
//     // Generate Ham as a dense matrix by using all connections to each variational determinant
//     // Simplest algorithm, very slow
//
//     let mut excite: Excite;
//     let mut new_det: Option<Config>;
//
//     // Generate Ham
//     let mut ham_matrix = generate_random_sparse_symmetric(wf.n, wf.n, 0.0); //DMatrix::<f64>::zeros(wf.n, wf.n);
//     for (i_det, det) in wf.dets.iter().enumerate() {
//
//         // Diagonal element
//         ham_matrix[(i_det, i_det)] = det.diag;
//
//         // Double excitations
//         // Opposite spin
//         for i in bits(excite_gen.valence & det.config.up) {
//             for j in bits(excite_gen.valence & det.config.dn) {
//                 for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
//                     excite = Excite {
//                         init: Orbs::Double((i, j)),
//                         target: stored_excite.target,
//                         abs_h: stored_excite.abs_h,
//                         is_alpha: None
//                     };
//                     new_det = det.config.safe_excite_det(&excite);
//                     match new_det {
//                         Some(d) => {
//                             // Valid excite: add to H*psi
//                             match wf.inds.get(&d) {
//                                 // TODO: Do this in a cache efficient way
//                                 Some(ind) => {
//                                     if *ind > i_det as usize {
//                                         ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
//                                         ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
//                                     }
//                                 },
//                                 _ => {}
//                             }
//                         }
//                         None => {}
//                     }
//                 }
//             }
//         }
//
//         // Same spin
//         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
//             for (i, j) in bit_pairs(excite_gen.valence & *config) {
//                 for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
//                     excite = Excite {
//                         init: Orbs::Double((i, j)),
//                         target: stored_excite.target,
//                         abs_h: stored_excite.abs_h,
//                         is_alpha: Some(*is_alpha)
//                     };
//                     new_det = det.config.safe_excite_det(&excite);
//                     match new_det {
//                         Some(d) => {
//                             // Valid excite: add to H*psi
//                             match wf.inds.get(&d) {
//                                 // TODO: Do this in a cache efficient way
//                                 Some(ind) => {
//                                     if *ind > i_det as usize {
//                                         ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
//                                         ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
//                                     }
//                                 },
//                                 _ => {}
//                             }
//                         }
//                         None => {}
//                     }
//                 }
//             }
//         }
//
//         // Single excitations
//         for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
//             for i in bits(excite_gen.valence & *config) {
//                 for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
//                     excite = Excite {
//                         init: Orbs::Single(i),
//                         target: stored_excite.target,
//                         abs_h: stored_excite.abs_h,
//                         is_alpha: Some(*is_alpha)
//                     };
//                     new_det = det.config.safe_excite_det(&excite);
//                     match new_det {
//                         Some(d) => {
//                             // Valid excite: add to H*psi
//                             match wf.inds.get(&d) {
//                                 // Compute matrix element and add to H*psi
//                                 // TODO: Do this in a cache efficient way
//                                 Some(ind) => {
//                                     if *ind > i_det as usize {
//                                         ham_matrix[(i_det as usize, *ind)] = ham.ham_sing(&det.config, &d);
//                                         ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
//                                     }
//                                 },
//                                 _ => {}
//                             }
//                         }
//                         None => {}
//                     }
//                 }
//             }
//         }
//     }
//     ham_matrix
// }

pub fn gen_sparse_ham_fast(global: &Global, wf: &mut VarWf, ham: &Ham, excite_gen: &ExciteGenerator) {
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

    // Thread pool
    // let mut pool = ThreadPool::new(4);
    // pool.execute(|| {
    //     println!("Initiating threadpool");
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

fn all_opposite_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_ups_sorted: &Vec<Unique>,
    up_singles: &HashMap<u128, Vec<u128>>,
    dn_singles: &HashMap<u128, Vec<u128>>,
    dn_single_constructor: &HashMap<Config, Vec<(usize, u128)>>,
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

fn all_same_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    unique_ups_sorted: &mut Vec<Unique>,
    // max_n_dets_double_loop: usize,
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

pub struct Unique {
    up: u128,
    n_dets: usize,
    n_dets_remaining: usize,
}

pub fn opposite_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
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

pub fn same_spin_excites(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
    unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>,
    // max_n_dets_double_loop: usize,
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
