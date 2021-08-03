// Variational Hamiltonian generation algorithms

use crate::excite::init::ExciteGenerator;
use crate::wf::Wf;
use crate::ham::Ham;
use nalgebra::DMatrix;
use crate::excite::{Excite, Orbs};
use crate::wf::det::Config;
use eigenvalues::utils::generate_random_sparse_symmetric;
use crate::utils::bits::{bit_pairs, bits};
use crate::var::sparse::SparseMat;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering::Equal;
use crate::var::utils::{remove_1e, remove_2e};
use crate::utils::read_input::Global;
use sprs::CsMat;
use nalgebra::base::DVector;
use std::time::Instant;
use itertools::Itertools;
// extern crate threads_pool;
// use threads_pool::ThreadPool;
// use std::thread;
// use std::time::Duration;


pub fn gen_sparse_ham_doubles(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>> {
    // Generate the sparse H as a Doubles data structure, in O(N^2 N_det log N_det) time
    // and O(N^2 N_det) space

    // Output data structure:
    // key: orb1, orb2, is_alpha
    // value: vector of (config with orb1/2 removed, index of this config in wf) tuples, sorted by
    // the first element in the tuple

    // To find all pq->rs excites:
    // Look up the pq and rs vectors
    // Loop over the overlap in linear time
    // Each tuple in intersection contains the indices of this matrix element
    // This algorithm takes anywhere from O(N^4 N_det) time to O(N^6/M^2 N_det) time

    let mut doub: HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>>::default() = ();

    for (det_ind, det) in wf.dets.iter().enumerate() {

        // Opposite spin
        for i in bits(excite_gen.valence & det.config.up) {
            for j in bits(excite_gen.valence & det.config.dn) {
                let key = (i, j, None);
                match doub.get(&key) {
                    None => { doub.insert(key, vec![(det_r2, det_ind)]); },
                    Some(mut v) => { v.push((det_r2, det_ind)); }
                }
            }
        }

        // Same spin
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for (i, j) in bit_pairs(excite_gen.valence & *config) {
                let key = (i, j, Some(*is_alpha));
                match doub.get(&key) {
                    None => { doub.insert(key, vec![(det_r2, det_ind)]); },
                    Some(mut v) => { v.push((det_r2, det_ind)); }
                }
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


pub fn gen_sparse_ham_fast(global: &Global, wf: &Wf, ham: &Ham, verbose: bool) -> SparseMat {
    // Generate Ham as a sparse matrix using my 2019 notes when I was working pro bono
    // For now, assumes that nup == ndn

    // Setup
    let start_setup: Instant = Instant::now();

    // 1. Find all unique up dets, and create a map from each to the (indices, dn dets) of its corresponding determinants
    let mut unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    for (config, ind) in &wf.inds {
        match unique_up_dict.get(&config.up) {
            None => {
                unique_up_dict.insert(config.up, vec![(*ind, config.dn)]);
            },
            Some(_) => {
                unique_up_dict.get_mut(&config.up).unwrap().push((*ind, config.dn));
            }
        }
    }
    if verbose {
        println!("Unique_up_dict:");
        for (key, val) in &unique_up_dict {
            println!("{}: ({}, {})", key, val[0].0, val[0].1);
        }
    }

    // 2. Sort the unique up dets in decreasing order by corresponding number of determinants.
    let mut unique_ups_sorted: Vec<Unique> = Vec::with_capacity(unique_up_dict.len());
    for (up, dets) in &unique_up_dict {
        unique_ups_sorted.push(Unique{up: *up, n_dets: dets.len(), n_dets_remaining: 0});
    }
    unique_ups_sorted.sort_by(|a, b| b.n_dets.partial_cmp(&a.n_dets).unwrap_or(Equal));

    // 3. Compute the cumulative number of determinants left for each unique det in sorted order.
    unique_ups_sorted.iter_mut().rev().fold(0, |acc, x| {
        x.n_dets_remaining = acc;
        acc + x.n_dets
    });
    if verbose { println!("Unique_ups_sorted:"); }
    for val in &unique_ups_sorted {
        if verbose { println!("{}, {}, {}", val.up, val.n_dets, val.n_dets_remaining); }
    }

    // 4. Generate a map, upSingles, from each unique up det to all following ones that are a single excite away
    let mut up_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
    for (ind, unique) in unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            match up_single_excite_constructor.get(&up_r1) {
                None => { up_single_excite_constructor.insert(up_r1, vec![ind]); },
                Some(_) => { up_single_excite_constructor.get_mut(&up_r1).unwrap().push(ind); }
            }
        }
    }
    let mut up_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for (ind, unique) in unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            for connected_ind in &up_single_excite_constructor[&up_r1] {
                // Skip over the diagonal excitations (there will be nup of these)
                if unique.up == unique_ups_sorted[*connected_ind].up { continue; }
                if *connected_ind > ind {
                    match up_singles.get(&unique.up) {
                        None => { up_singles.insert(unique.up, vec![unique_ups_sorted[*connected_ind].up]); },
                        Some(_) => { up_singles.get_mut(&unique.up).unwrap().push(unique_ups_sorted[*connected_ind].up); }
                    }
                }
            }
        }
    }
    if verbose { println!("Up_singles:"); }
    for (key, val) in &up_singles {
        if verbose { println!("{}: {}", key, val[0]); }
    }

    // 5. Find the unique dn dets, a map from each dn det to its index in the unique dn det list, and a map dnSingles
    // (analogous to upSingles, except want all excitations, not just to later unique dets)
    let mut unique_dns_set: HashSet<u128> = HashSet::default();
    for config in wf.inds.keys() {
        unique_dns_set.insert(config.dn);
    }
    let mut unique_dns_vec: Vec<u128> = Vec::with_capacity(unique_dns_set.len());
    // Problem: following results in a randomly ordered vector!
    for dn in unique_dns_set {
        unique_dns_vec.push(dn);
    }

    let mut dn_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
    for (ind, dn) in unique_dns_vec.iter().enumerate() {
        for dn_r1 in remove_1e(*dn) {
            match dn_single_excite_constructor.get(&dn_r1) {
                None => { dn_single_excite_constructor.insert(dn_r1, vec![ind]); },
                Some(_) => { dn_single_excite_constructor.get_mut(&dn_r1).unwrap().push(ind); }
            }
        }
    }
    let mut dn_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for dn in &unique_dns_vec {
        for dn_r1 in remove_1e(*dn) {
            for connected_ind in &dn_single_excite_constructor[&dn_r1] {
                // Skip over the diagonal excitations (there will be ndn of these)
                if *dn == unique_dns_vec[*connected_ind] { continue; }
                match dn_singles.get(&dn) {
                    None => { dn_singles.insert(*dn, vec![unique_dns_vec[*connected_ind]]); },
                    Some(_) => { dn_singles.get_mut(dn).unwrap().push(unique_dns_vec[*connected_ind]); }
                }
            }
        }
    }

    // Accumulator of off-diagonal elements
    // key is the (row, col) where row < col, and value is the element
    // Will put them in the SparseMat data structure later
    let mut off_diag_elems = OffDiagElems::new(wf.n);
    println!("Time for H gen setup: {:?}", start_setup.elapsed());

    // Thread pool
    // let mut pool = ThreadPool::new(4);
    // pool.execute(|| {
    //     println!("Initiating threadpool");
    // });

    // Parameter for choosing which of the two same-spin algorithms to use
    let max_n_dets_double_loop = global.ndn as usize; // ((global.ndn * (global.ndn - 1)) / 2) as usize;

    let start_opp: Instant = Instant::now();
    all_opposite_spin_excites(global, wf, ham, &mut unique_up_dict, &mut unique_ups_sorted, &mut up_singles, &mut unique_dns_vec, &mut dn_singles, &mut off_diag_elems);
    println!("Time for opposite-spin: {:?}", start_opp.elapsed());

    let start_same: Instant = Instant::now();
    all_same_spin_excites(wf, ham, &mut unique_up_dict, &mut unique_ups_sorted, &mut off_diag_elems, max_n_dets_double_loop);
    println!("Time for same-spin: {:?}", start_same.elapsed());

    // Finally, put collected off-diag elems into a sparse matrix
    off_diag_elems.to_sparse(wf, verbose)
}

fn all_opposite_spin_excites(global: &Global, wf: &Wf, ham: &Ham, mut unique_up_dict: &mut HashMap<u128, Vec<(usize, u128)>>, unique_ups_sorted: &mut Vec<Unique>, mut up_singles: &mut HashMap<u128, Vec<u128>>, mut unique_dns_vec: &mut Vec<u128>, mut dn_singles: &mut HashMap<u128, Vec<u128>>, mut off_diag_elems: &mut OffDiagElems) {
    for unique in unique_ups_sorted {
        // Opposite-spin excitations
        opposite_spin_excites(global, wf, ham, &mut unique_up_dict, &mut up_singles, &mut unique_dns_vec, &mut dn_singles, &mut off_diag_elems, unique);
    }
}

fn all_same_spin_excites(wf: &Wf, ham: &Ham, mut unique_up_dict: &mut HashMap<u128, Vec<(usize, u128)>>, unique_ups_sorted: &mut Vec<Unique>, mut off_diag_elems: &mut OffDiagElems, max_n_dets_double_loop: usize) {
    for unique in unique_ups_sorted {
        // Same-spin excitations
        same_spin_excites(wf, ham, &mut unique_up_dict, max_n_dets_double_loop, &mut off_diag_elems, unique);
    }
}


pub struct Unique {
    up: u128,
    n_dets: usize,
    n_dets_remaining: usize
}


pub fn opposite_spin_excites(global: &Global, wf: &Wf, ham: &Ham, unique_up_dict: &mut HashMap<u128, Vec<(usize, u128)>>, up_singles: &mut HashMap<u128, Vec<u128>>, unique_dns_vec: &mut Vec<u128>, dn_singles: &mut HashMap<u128, Vec<u128>>, off_diag_elems: &mut OffDiagElems, unique: &Unique) {
    // Current status:
    // For the later iterations of C2 vdz, eps=1-4, first algo is ~10x faster than third algo
    // Second algo not yet implemented
    // So, for now, first algo is always on
    // let start_this_opp: Instant = Instant::now();
    let first_algo_complexity = unique.n_dets * (global.ndn * global.ndn) as usize + unique.n_dets_remaining; // Term in parentheses is an estimate
    let second_algo_complexity = global.ndn as usize * (unique.n_dets + unique.n_dets_remaining);
    let third_algo_complexity = unique.n_dets * unique.n_dets_remaining;
    let mut first_algo_fastest = false;
    let mut second_algo_fastest = true;
    // if first_algo_complexity <= second_algo_complexity {
    //     if first_algo_complexity <= third_algo_complexity { first_algo_fastest = true; }
    // } else {
    //     if second_algo_complexity <= third_algo_complexity { second_algo_fastest = true; }
    // }
    if first_algo_fastest {
        // if verbose { println!("First algo fastest"); }
        // Use a single for-loop version of the double for-loop algorithm from "Fast SHCI"
        let mut dn_candidates: HashMap<u128, Vec<usize>> = HashMap::default();
        for dn in &unique_up_dict[&unique.up] {
            for dn2 in &dn_singles[&dn.1] {
                match dn_candidates.get(&dn2) {
                    None => { dn_candidates.insert(*dn2, vec![dn.0]); },
                    Some(_) => { dn_candidates.get_mut(dn2).unwrap().push(dn.0); }
                }
            }
        }
        match up_singles.get(&unique.up) {
            None => {},
            Some(ups) => {
                for up in ups {
                    for dn in &unique_up_dict[&up] {
                        match dn_candidates.get(&dn.1) {
                            None => {},
                            Some(dn_connections) => {
                                // We need to do this one in terms of up config rather than ind
                                let ind1 = wf.inds[&Config { up: *up, dn: dn.1 }];
                                for dn_connection_ind in dn_connections {
                                    off_diag_elems.add_el(wf, ham, ind1, *dn_connection_ind); // only adds if elem != 0
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if second_algo_fastest {
        // if verbose { println!("Second algo fastest"); }
        // Loop over ways to remove an electron to get all connected dets
        let mut dn_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
        for dn in &unique_up_dict[&unique.up] {
            for dn_r1 in remove_1e(dn.1) {
                match dn_single_excite_constructor.get(&dn_r1) {
                    None => { dn_single_excite_constructor.insert(dn_r1, vec![dn.0]); },
                    Some(_) => { dn_single_excite_constructor.get_mut(&dn_r1).unwrap().push(dn.0); }
                }
            }
        }
        // up_singles[&unique.up] can be empty, so
        match up_singles.get(&unique.up) {
            None => {},
            Some(ups) => {
                for up in ups {
                    for dn in &unique_up_dict[&up] {
                        for dn_r1 in remove_1e(dn.1) {
                            match dn_single_excite_constructor.get(&dn_r1) {
                                None => {},
                                Some(connected_inds) => {
                                    for connected_ind in connected_inds {
                                        off_diag_elems.add_el(wf, ham, dn.0, *connected_ind)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // if verbose { println!("Third algo fastest"); }
        // Loop over all pairs of dn dets, one from each of the two unique up configs that are linked
        // if verbose { println!("unique.up = {}", unique.up); }
        match up_singles.get(&unique.up) {
            None => {},
            Some(ups) => {
                for up in ups {
                    // if verbose { println!("up: {}", up); }
                    for ind1 in &unique_up_dict[&unique.up] {
                        // if verbose { println!("ind1= ({}, {})", ind1.0, ind1.1); }
                        for ind2 in &unique_up_dict[&up] {
                            // if verbose { println!("Found excitation: {} {}", ind1.0, ind2.0); }
                            off_diag_elems.add_el(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
                        }
                    }
                }
            }
        }
    }
    // println!("Time for opposite-spin excites for up config {}: {:?}", unique.up, start_this_opp.elapsed());
}


pub fn same_spin_excites(wf: &Wf, ham: &Ham, unique_up_dict: &mut HashMap<u128, Vec<(usize, u128)>>, max_n_dets_double_loop: usize, off_diag_elems: &mut OffDiagElems, unique: &Unique) {
    if unique.n_dets <= max_n_dets_double_loop {
        // if verbose { println!("Same-spin first algo fastest"); }
        // Use the double for-loop algorithm from "Fast SHCI"
        for (i_ind, ind1) in unique_up_dict[&unique.up].iter().enumerate() {
            for ind2 in unique_up_dict[&unique.up][i_ind + 1..].iter() {
                // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                off_diag_elems.add_el_and_spin_flipped(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
            }
        }
    } else {
        // if verbose { println!("Same-spin second algo fastest"); }
        // Loop over all pairs of electrons to get all connections (asymptotically optimal in Full CI limit)
        let mut double_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
        for dn in &unique_up_dict[&unique.up] {
            for dn_r2 in remove_2e(dn.1) {
                match double_excite_constructor.get(&dn_r2) {
                    None => { double_excite_constructor.insert(dn_r2, vec![dn.0]); },
                    Some(_) => { double_excite_constructor.get_mut(&dn_r2).unwrap().push(dn.0); }
                }
            }
        }
        for dn in &unique_up_dict[&unique.up] {
            for dn_r2 in remove_2e(dn.1) {
                for dn2 in &double_excite_constructor[&dn_r2] {
                    if dn.0 != *dn2 {
                        // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                        off_diag_elems.add_el_and_spin_flipped(wf, ham, dn.0, *dn2); // only adds if elem != 0
                    }
                }
            }
        }
    }
}

//
// // Off-diag elems data structure
// pub struct OffDiagElems {
//     rows: Vec<usize>,
//     indices: Vec<usize>, // vector of indices for each row
//     values: Vec<f64>, // vector of values for each row
// }
//
//
// impl OffDiagElems {
//     pub fn new(n: usize) -> Self {
//         Self{
//             rows: Vec::with_capacity(100 * n),
//             indices: Vec::with_capacity(100 * n),
//             values: Vec::with_capacity(100 * n)
//         }
//     }
//
//     pub fn add_el(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij} to off_diag_elems
//         // i and j can be of any order
//         // For now, assumes no repeats
//         // TODO: fix this if we start using the 2nd algorithm for same-spin excites!
//
//         if i == j { return; }
//         if i < j {
//             self.rows.push(i);
//             self.indices.push(j);
//         } else {
//             self.rows.push(j);
//             self.indices.push(i);
//         }
//         let elem = ham.ham_off_diag_no_excite(&wf.dets[i].config, &wf.dets[j].config);
//         if elem != 0.0 {
//             self.values.push(elem);
//         }
//     }
//
//     pub fn add_el_and_spin_flipped(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
//         // i and j can be of any order
//         // TODO: Can speed up by skipping the second part if the first part is skipped
//
//         // Element
//         self.add_el(wf, ham, i, j);
//
//         // Spin-flipped element
//         let i_spin_flipped = {
//             let config = wf.dets[i].config;
//             wf.inds[&Config { up: config.dn, dn: config.up }]
//         };
//         let j_spin_flipped = {
//             let config = wf.dets[j].config;
//             wf.inds[&Config { up: config.dn, dn: config.up }]
//         };
//         self.add_el(wf, ham, i_spin_flipped, j_spin_flipped);
//     }
//
//     pub fn to_sparse(&self, wf: &Wf, verbose: bool) -> SparseMat {
//         // Put the matrix elements into a sparse matrix
//         let n = wf.n;
//
//         // Diagonal component
//         let mut diag = Vec::with_capacity(n);
//         for det in &wf.dets {
//             diag.push(det.diag);
//         }
//
//         // Off-diagonal component
//         let shape = (n, n);
//
//         // Sort by row then by index, remove duplicates, duplicate each one, then sort again
//
//
//
//
//         let indptr: Vec<usize> = self.nnz
//             .iter()
//             .scan(0, |acc, &x| {
//                 *acc = *acc + x;
//                 Some(*acc)
//             })
//             .collect();
//         if verbose { println!("indptr:"); }
//         for i in &indptr {
//             if verbose { println!("{}", i); }
//         }
//
//         let indices: Vec<usize> = self.indices.clone().into_iter().flatten().collect();
//         let data: Vec<f64> = self.values.clone().into_iter().flatten().collect();
//
//         if verbose { println!("indices, data:"); }
//         for (i, j) in indices.iter().zip(data.iter()) {
//             if verbose { println!("{}, {}", i, j); }
//         }
//
//         // Off-diagonal component
//         let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
//         match off_diag_component {
//             Err(_) => { panic!("Error in constructing CsMat"); }
//             Ok(off_diag) => { return SparseMat{n, diag: DVector::from(diag), off_diag}; }
//         }
//     }
// }



// Off-diag elems data structure
pub struct OffDiagElems {
    nonzero_inds: HashSet<(usize, usize)>, // number of nonzero elements in the upper triangular half of the matrix
    nnz: Vec<usize>, // number of nonzero values for each row
    indices: Vec<Vec<usize>>, // vector of indices for each row
    values: Vec<Vec<f64>>, // vector of values for each row
}


impl OffDiagElems {
    pub fn new(n: usize) -> Self {
        Self{
            nonzero_inds: HashSet::default(),
            nnz: vec![0; n + 1],
            indices: vec![vec![]; n],
            values: vec![vec![]; n]
        }
    }

    pub fn add_el(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
        // Add an off-diagonal element H_{ij} to off_diag_elems
        // i and j can be of any order

        if i == j { return; }
        let key = {
            if i < j {
                (i, j)
            } else {
                (j, i)
            }
        };
        match self.nonzero_inds.get(&key) {
            None => {
                let elem = ham.ham_off_diag_no_excite(&wf.dets[i].config, &wf.dets[j].config);
                if elem != 0.0 {
                    self.insert_nonzero_inds(key); //nonzero_inds.insert(key);

                    self.nnz[i + 1] += 1;
                    // self.indices[i].push(j);
                    // self.values[i].push(elem);

                    self.nnz[j + 1] += 1;
                    // self.indices[j].push(i);
                    // self.values[j].push(elem);

                    self.push_indices_values(i, j, elem);
                }
            },
            Some(_) => {}
        }
    }

    fn insert_nonzero_inds(&mut self, key: (usize, usize)) {
        self.nonzero_inds.insert(key);
    }

    fn push_indices_values(&mut self, i: usize, j: usize, elem: f64) {
        self.indices[i].push(j);
        self.values[i].push(elem);
        self.indices[j].push(i);
        self.values[j].push(elem);
    }

    pub fn add_el_and_spin_flipped(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
        // Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
        // i and j can be of any order
        // TODO: Can speed up by skipping the second part if the first part is skipped

        // Element
        self.add_el(wf, ham, i, j);

        // Spin-flipped element
        let i_spin_flipped = {
            let config = wf.dets[i].config;
            wf.inds[&Config { up: config.dn, dn: config.up }]
        };
        let j_spin_flipped = {
            let config = wf.dets[j].config;
            wf.inds[&Config { up: config.dn, dn: config.up }]
        };
        self.add_el(wf, ham, i_spin_flipped, j_spin_flipped);
    }

    pub fn to_sparse(&self, wf: &Wf, verbose: bool) -> SparseMat {
        let start_to_sparse: Instant = Instant::now();
        // Put the matrix elements into a sparse matrix
        let n = wf.n;

        // Diagonal component
        let mut diag = Vec::with_capacity(n);
        for det in &wf.dets {
            diag.push(det.diag);
        }
        if verbose { println!("Ready to make sparse mat: n = {}", n); }
        if verbose { println!("nnz:"); }
        for i in &self.nnz {
            if verbose { println!("{}", i); }
        }
        let shape = (n, n);
        let indptr: Vec<usize> = self.nnz
            .iter()
            .scan(0, |acc, &x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect();
        if verbose { println!("indptr:"); }
        for i in &indptr {
            if verbose { println!("{}", i); }
        }

        let indices: Vec<usize> = self.indices.clone().into_iter().flatten().collect();
        let data: Vec<f64> = self.values.clone().into_iter().flatten().collect();

        println!("Variational Hamiltonian has {} nonzero off-diagonal elements", indices.len());

        if verbose { println!("indices, data:"); }
        for (i, j) in indices.iter().zip(data.iter()) {
            if verbose { println!("{}, {}", i, j); }
        }

        // Off-diagonal component
        let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
        println!("Time for converting stored nonzero indices to sparse H: {:?}", start_to_sparse.elapsed());
        match off_diag_component {
            Err(_) => { panic!("Error in constructing CsMat"); }
            Ok(off_diag) => { return SparseMat{n, diag: DVector::from(diag), off_diag}; }
        }
    }
}


// // Faster (?) off-diag elems data structure - fewer hash lookups, vectors of vectors, etc
// // Just add elements to a vector of (ind1, ind2) where ind1 < ind2
// // After they're all added, merge sort, then compute matrix elements
// struct FastOffDiagElems {
//     nonzero_inds: Vec<(usize, usize)>
// }
//
// impl FastOffDiagElems {
//     pub fn new(n: usize) -> Self {
//         Self{
//             nonzero_inds: Vec::with_capacity(100 * n),
//         }
//     }
//
//     pub fn add_el(&mut self, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij} to off_diag_elems
//         // i and j can be of any order
//
//         if i < j {
//             self.nonzero_inds.push((i, j));
//         } else {
//             self.nonzero_inds.push((j, i));
//         }
//     }
//
//     pub fn add_el_and_spin_flipped(&mut self, wf: &Wf, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
//         // i and j can be of any order
//
//         // Element
//         self.add_el(i, j);
//
//         // Spin-flipped element
//         let i_spin_flipped = {
//             let config = wf.dets[i].config;
//             wf.inds[&Config { up: config.dn, dn: config.up }]
//         };
//         let j_spin_flipped = {
//             let config = wf.dets[j].config;
//             wf.inds[&Config { up: config.dn, dn: config.up }]
//         };
//         self.add_el(i_spin_flipped, j_spin_flipped);
//     }
//
//     pub fn to_sparse(&self, wf: &Wf, verbose: bool) -> SparseMat {
//         // Put the matrix elements into a sparse matrix
//
//         // To do: Sort elements to find redundancies, then compute matrix elements, then
//         // clone to lower triangular component, then sort again
//
//         let n = wf.n;
//
//         // Diagonal component
//         let mut diag = Vec::with_capacity(n);
//         for det in &wf.dets {
//             diag.push(det.diag);
//         }
//         if verbose { println!("Ready to make sparse mat: n = {}", n); }
//         if verbose { println!("nnz:"); }
//         for i in &self.nnz {
//             if verbose { println!("{}", i); }
//         }
//
//         // Off-diagonal component
//         let shape = (n, n);
//
//         // Sort indices by row, then by column
//
//         // Remove duplicates
//
//         // For each index: compute matrix element and
//
//
//
//
//         let indptr: Vec<usize> = self.nnz
//             .iter()
//             .scan(0, |acc, &x| {
//                 *acc = *acc + x;
//                 Some(*acc)
//             })
//             .collect();
//         if verbose { println!("indptr:"); }
//         for i in &indptr {
//             if verbose { println!("{}", i); }
//         }
//
//         let indices: Vec<usize> = self.indices.clone().into_iter().flatten().collect();
//         let data: Vec<f64> = self.values.clone().into_iter().flatten().collect();
//
//         if verbose { println!("indices, data:"); }
//         for (i, j) in indices.iter().zip(data.iter()) {
//             if verbose { println!("{}, {}", i, j); }
//         }
//
//         // Off-diagonal component
//         let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
//         match off_diag_component {
//             Err(_) => { panic!("Error in constructing CsMat"); }
//             Ok(off_diag) => {
//                 return SparseMat{
//                     n,
//                     diag: DVector::from(diag),
//                     off_diag
//                 };
//             }
//         }
//     }
//
// }

