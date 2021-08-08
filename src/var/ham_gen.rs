// Variational Hamiltonian generation algorithms

use crate::excite::init::ExciteGenerator;
use crate::wf::Wf;
use crate::ham::Ham;
use nalgebra::DMatrix;
use crate::excite::{Excite, Orbs};
use crate::wf::det::Config;
use eigenvalues::utils::generate_random_sparse_symmetric;
use crate::utils::bits::{bit_pairs, bits, ibclr};
use crate::var::sparse::{SparseMat, SparseMatDoubles};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering::Equal;
use crate::var::utils::{remove_1e, remove_2e};
use crate::utils::read_input::Global;
use sprs::CsMat;
use std::time::Instant;
use itertools::Itertools;
use crate::var::off_diag::{OffDiagElemsNoHash, add_el, add_el_and_spin_flipped, create_sparse};
// extern crate threads_pool;
// use threads_pool::ThreadPool;
// use std::thread;
// use std::time::Duration;


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


pub fn gen_doubles(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>> {
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
                let det_r2 = Config{ up: ibclr(det.config.up, i), dn: ibclr(det.config.dn, j) };
                let key = (i, j, None);
                match doub.get_mut(&key) {
                    None => { doub.insert(key, vec![(det_r2, det_ind)]); },
                    Some(v) => { v.push((det_r2, det_ind)); }
                }
            }
        }

        // Same spin
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for (i, j) in bit_pairs(excite_gen.valence & *config) {
                let det_r2 = { if *is_alpha { Config{ up: ibclr(ibclr(det.config.up, i), j), dn: det.config.dn } }
                    else { Config{ up: det.config.up, dn: ibclr(ibclr(det.config.dn, i), j) } } };
                let key = (i, j, Some(*is_alpha));
                match doub.get_mut(&key) {
                    None => { doub.insert(key, vec![(det_r2, det_ind)]); },
                    Some(v) => { v.push((det_r2, det_ind)); }
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


pub fn gen_sparse_ham_fast(global: &Global, wf: &mut Wf, ham: &Ham, verbose: bool) -> SparseMat {
    // Generate Ham as a sparse matrix using my 2019 notes when I was working pro bono
    // For now, assumes that nup == ndn
    // Updates wf.sparse_ham if it already exists from the previous iteration

    // Setup
    let start_setup: Instant = Instant::now();

    // 1. Find all unique up dets, and create a map from each to the (indices, dn dets) of its corresponding determinants
    // TODO: make this a wf member variable so we don't have to regenerate the whole thing from scratch
    let mut unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    let mut new_unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    for (config, ind) in &wf.inds {
        match unique_up_dict.get_mut(&config.up) {
            None => { unique_up_dict.insert(config.up, vec![(*ind, config.dn)]); },
            Some(v) => { v.push((*ind, config.dn)); }
        }
        if *ind >= wf.n_stored_h() {
            match new_unique_up_dict.get_mut(&config.up) {
                None => { new_unique_up_dict.insert(config.up, vec![(*ind, config.dn)]); },
                Some(v) => { v.push((*ind, config.dn)); }
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
        unique_ups_sorted.push(Unique{up: *up, n_dets: dets.len(), n_dets_remaining: 0});
    }
    unique_ups_sorted.sort_by(|a, b| b.n_dets.partial_cmp(&a.n_dets).unwrap_or(Equal));
    let mut new_unique_ups_sorted: Vec<Unique> = Vec::with_capacity(new_unique_up_dict.len());
    for (up, dets) in &new_unique_up_dict {
        new_unique_ups_sorted.push(Unique{up: *up, n_dets: dets.len(), n_dets_remaining: 0});
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
            match up_single_excite_constructor.get_mut(&up_r1) {
                None => { up_single_excite_constructor.insert(up_r1, vec![ind]); },
                Some(v) => { v.push(ind); }
            }
        }
    }
    // up_singles is *new* to *all*
    let mut up_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for (ind, unique) in new_unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            if let Some(connected_inds) = up_single_excite_constructor.get(&up_r1) {
                for connected_ind in connected_inds {
                    // Skip over the diagonal excitations (there will be nup of these)
                    if unique.up == unique_ups_sorted[*connected_ind].up { continue; }
                    match up_singles.get_mut(&unique.up) {
                        None => { up_singles.insert(unique.up, vec![unique_ups_sorted[*connected_ind].up]); },
                        Some(v) => { v.push(unique_ups_sorted[*connected_ind].up); }
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
            match dn_single_excite_constructor.get_mut(&dn_r1) {
                None => { dn_single_excite_constructor.insert(dn_r1, vec![ind]); },
                Some(v) => { v.push(ind); }
            }
        }
    }
    let mut dn_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for dn in &unique_dns_vec {
        for dn_r1 in remove_1e(*dn) {
            for connected_ind in &dn_single_excite_constructor[&dn_r1] {
                // Skip over the diagonal excitations (there will be ndn of these)
                if *dn == unique_dns_vec[*connected_ind] { continue; }
                match dn_singles.get_mut(&dn) {
                    None => { dn_singles.insert(*dn, vec![unique_dns_vec[*connected_ind]]); },
                    Some(v) => { v.push(unique_dns_vec[*connected_ind]); }
                }
            }
        }
    }
    println!("Time for H gen setup: {:?}", start_setup.elapsed());

    // Accumulator of new off-diagonal elements
    // Will put them in the SparseMat data structure later
    let n = wf.n;
    if let Some(off_diag) = &mut wf.off_diag_h_elems {
        // Expand old off-diagonal elements holder to hold new rows
        off_diag.expand_rows(n);
    } else {
        // Create new off-diagonal elements holder
        wf.off_diag_h_elems = Some(OffDiagElemsNoHash::new(n));
    }
    // match &mut wf.off_diag_h_elems {
    //     None => {
    //         // Create new off-diagonal elements holder
    //         wf.off_diag_h_elems = Some(OffDiagElemsNoHash::new(n));
    //     },
    //     Some(mut off_diag) => {
    //         // Expand old off-diagonal elements holder to hold new rows
    //         off_diag.expand_rows(n);
    //     }
    // }

    // Thread pool
    // let mut pool = ThreadPool::new(4);
    // pool.execute(|| {
    //     println!("Initiating threadpool");
    // });

    // Parameter for choosing which of the two same-spin algorithms to use
    let max_n_dets_double_loop = global.ndn as usize; // ((global.ndn * (global.ndn - 1)) / 2) as usize;

    let start_opp: Instant = Instant::now();
    all_opposite_spin_excites(global, wf, ham, &unique_up_dict, &new_unique_up_dict, &new_unique_ups_sorted, &up_singles, &mut unique_dns_vec, &mut dn_singles);
    println!("Time for opposite-spin: {:?}", start_opp.elapsed());

    let start_same: Instant = Instant::now();
    all_same_spin_excites(wf, ham, &unique_up_dict, &new_unique_up_dict, &mut unique_ups_sorted, max_n_dets_double_loop);
    println!("Time for same-spin: {:?}", start_same.elapsed());

    // Finally, put collected off-diag elems into a sparse matrix
    create_sparse(wf)
}

fn all_opposite_spin_excites(global: &Global, wf: &mut Wf, ham: &Ham, unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, new_unique_ups_sorted: &Vec<Unique>, up_singles: &HashMap<u128, Vec<u128>>, mut unique_dns_vec: &mut Vec<u128>, mut dn_singles: &mut HashMap<u128, Vec<u128>>) {
    // Loop over new dets only
    for unique in new_unique_ups_sorted {
        // Opposite-spin excitations
        opposite_spin_excites(global, wf, ham, unique_up_dict, new_unique_up_dict, &up_singles, &mut unique_dns_vec, &mut dn_singles, unique);
    }
}

fn all_same_spin_excites(wf: &mut Wf, ham: &Ham, unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, unique_ups_sorted: &mut Vec<Unique>, max_n_dets_double_loop: usize) {
    for unique in unique_ups_sorted {
        // Same-spin excitations
        same_spin_excites(wf, ham, unique_up_dict, new_unique_up_dict, max_n_dets_double_loop, unique);
    }
}


pub struct Unique {
    up: u128,
    n_dets: usize,
    n_dets_remaining: usize
}


pub fn opposite_spin_excites(global: &Global, wf: &mut Wf, ham: &Ham, unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, up_singles: &HashMap<u128, Vec<u128>>, unique_dns_vec: &mut Vec<u128>, dn_singles: &mut HashMap<u128, Vec<u128>>, unique: &Unique) {
    // Current status:
    // For the later iterations of C2 vdz, eps=1-4, first algo is ~10x faster than third algo
    // Second even faster
    // So, for now, second algo always on

    // let start_this_opp: Instant = Instant::now();
    let first_algo_complexity = unique.n_dets * (global.ndn * global.ndn) as usize + unique.n_dets_remaining; // Term in parentheses is an estimate
    let second_algo_complexity = global.ndn as usize * (unique.n_dets + unique.n_dets_remaining);
    let third_algo_complexity = unique.n_dets * unique.n_dets_remaining;
    let mut first_algo_fastest = true;
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
        for dn in &new_unique_up_dict[&unique.up] {
            for dn2 in &dn_singles[&dn.1] {
                match dn_candidates.get_mut(&dn2) {
                    None => { dn_candidates.insert(*dn2, vec![dn.0]); },
                    Some(v) => { v.push(dn.0); }
                }
            }
        }
        // up_singles is *new* to *all*
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
                                    add_el(wf, ham, ind1, *dn_connection_ind, None); // only adds if elem != 0
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
                match dn_single_excite_constructor.get_mut(&dn_r1) {
                    None => { dn_single_excite_constructor.insert(dn_r1, vec![dn.0]); },
                    Some(v) => { v.push(dn.0); }
                }
            }
        }
        // up_singles[&unique.up] can be empty, so
        match up_singles.get(&unique.up) {
            None => {},
            Some(ups) => {
                for up in ups {
                    for dn in &new_unique_up_dict[&up] {
                        for dn_r1 in remove_1e(dn.1) {
                            match dn_single_excite_constructor.get(&dn_r1) {
                                None => {},
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
                            add_el(wf, ham, ind1.0, ind2.0, None); // only adds if elem != 0
                        }
                    }
                }
            }
        }
    }
    // println!("Time for opposite-spin excites for up config {}: {:?}", unique.up, start_this_opp.elapsed());
}


pub fn same_spin_excites(wf: &mut Wf, ham: &Ham, unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, new_unique_up_dict: &HashMap<u128, Vec<(usize, u128)>>, max_n_dets_double_loop: usize, unique: &Unique) {
    if true { //unique.n_dets <= max_n_dets_double_loop {
        // if verbose { println!("Same-spin first algo fastest"); }
        // Use the double for-loop algorithm from "Fast SHCI"
        for (i_ind, ind1) in unique_up_dict[&unique.up].iter().enumerate() {
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
            None => {},
            Some(dns) => {
                for dn in dns {
                    for dn_r2 in remove_2e(dn.1) {
                        match double_excite_constructor.get_mut(&dn_r2) {
                            None => { double_excite_constructor.insert(dn_r2, vec![dn.0]); },
                            Some(v) => { v.push(dn.0); }
                        }
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


