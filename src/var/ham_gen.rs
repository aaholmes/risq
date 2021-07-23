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
use std::collections::HashMap;
use std::cmp::Ordering::Equal;
// use std::ptr::Unique;
use crate::var::utils::{remove_1e, remove_2e};
use crate::utils::read_input::Global;
use std::iter::Repeat;

pub fn gen_dense_ham_connections(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> DMatrix<f64> {
    // Generate Ham as a dense matrix by using all connections to each variational determinant
    // Simplest algorithm, very slow

    let mut excite: Excite;
    let mut new_det: Option<Config>;

    // Generate Ham
    let mut ham_matrix = generate_random_sparse_symmetric(wf.n, wf.n, 0.0); //DMatrix::<f64>::zeros(wf.n, wf.n);
    for (i_det, det) in wf.dets.iter().enumerate() {

        // Diagonal element
        ham_matrix[(i_det, i_det)] = det.diag;

        // Double excitations
        // Opposite spin
        for i in bits(det.config.up) {
            for j in bits(det.config.dn) {
                for stored_excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                    excite = Excite {
                        init: Orbs::Double((i, j)),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: None
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        // Same spin
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for (i, j) in bit_pairs(*config) {
                for stored_excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                    excite = Excite {
                        init: Orbs::Double((i, j)),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: Some(*is_alpha)
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        // Single excitations
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for i in bits(*config) {
                for stored_excite in excite_gen.sing_sorted_list.get(&Orbs::Single(i)).unwrap() {
                    excite = Excite {
                        init: Orbs::Single(i),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: Some(*is_alpha)
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // Compute matrix element and add to H*psi
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_sing(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }
    ham_matrix
}


pub fn gen_sparse_ham_fast(global: &Global, wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> SparseMat {
    // Generate Ham as a sparse matrix using my 2019 notes when I was working pro bono
    // For now, assumes that nup == ndn

    // Setup

    // 1. Find all unique up dets, and create a map from each to the indices of its corresponding determinants
    let mut unique_up_dict: HashMap<u128, Vec<(usize, u128)>> = HashMap::default();
    for (config, ind) in wf.inds {
        match unique_up_dict.get(&config.up) {
            None => {
                unique_up_dict.insert(config.up, vec![(ind, config.dn)]);
            },
            Some(_) => {
                unique_up_dict[&config.up].push((ind, config.dn));
            }
        }
    }

    // 2. Sort the unique up dets in decreasing order by corresponding number of determinants.
    struct UniqueUp {
        up: u128,
        n_dets: usize,
        n_dets_remaining: usize
    }
    let mut unique_ups_sorted: Vec<UniqueUp> = Vec::with_capacity(unique_up_dict.len());
    for (up, dets) in unique_up_dict {
        unique_ups_sorted.push(UniqueUp{up, n_dets: dets.len(), n_dets_remaining: 0});
    }
    unique_ups_sorted.sort_by(|a, b| b.n_dets.partial_cmp(&a.n_dets).unwrap_or(Equal));

    // 3. Compute the cumulative number of determinants left for each unique det in sorted order.
    unique_ups_sorted.iter_mut().rev().fold(0, |mut acc, x| {
        *x.n_dets_remaining = acc;
        acc += *x.n_dets;
        x
    });

    // 4. Generate a map, upSingles, from each unique up det to all following ones that are a single excite away
    let mut up_single_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
    for (ind, unique) in unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            match up_single_excite_constructor.get(&up_r1) {
                None => up_single_excite_constructor.insert(up_r1, vec![ind]),
                Some(_) => up_single_excite_constructor[&up_r1].push(ind)
            }
        }
    }
    let mut up_singles: HashMap<u128, Vec<u128>> = HashMap::default();
    for (ind, unique) in unique_ups_sorted.iter().enumerate() {
        for up_r1 in remove_1e(unique.up) {
            for connected_ind in up_single_excite_constructor[&up_r1] {
                if connected_ind > ind {
                    match up_singles.get(&unique.up) {
                        None => up_singles.insert(unique.up, vec![unique_ups_sorted[connected_ind].up]),
                        Some(_) => up_singles[&unique.up].push(unique_ups_sorted[connected_ind].up)
                    }
                }
            }
        }
    }

    // 5. Find the unique dn dets, a map from each dn det to its index in the unique dn det list, and a map dnSingles
    // (analogous to upSingles, except want all excitations, not just to later unique dets)

    // TODO


    // Accumulator of off-diagonal elements
    // key is the (row, col) where row < col, and value is the element
    // Will put them in the SparseMat data structure later
    let mut off_diag_elems: HashMap<(usize, usize), f64> = HashMap::default();

    // Opposite-spin excitations

    for unique in unique_ups_sorted {
        if (first algorithm fastest) {
            // Use a single for-loop version of the double for-loop algorithm from "Fast SHCI"
            let mut dn_candidates: HashMap<u128, Vec<usize>> = HashMap::default();
            for dn in unique_up_dict[&unique.up] {
                for dn2 in dn_singles[&dn.1] {
                    match dn_candidates.get(&dn.1) {
                        None => dn_candidates.insert(dn.1, vec![ind]),
                        Some(_) => dn_candidates[&dn.1].push(ind)
                    }
                }
            }
        }
    }




    // Same-spin excitations

    // Parameter for choosing which of the two algorithms to use
    let max_n_dets_double_loop = (global.ndn * (global.ndn - 1)) as usize;

    for unique in unique_ups_sorted {
        if unique.n_dets <= max_n_dets_double_loop {
            // Use the double for-loop algorithm from "Fast SHCI"
            for (i_ind, ind1) in unique_up_dict[&unique.up].iter().enumerate() {
                for ind2 in unique_up_dict[&unique.up][i_ind + 1 ..].iter() {
                    // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                    add_el_and_spin_flipped(wf, ham, &mut off_diag_elems, *ind1.0, *ind2.0); // only adds if elem != 0
                }
            }
        } else {
            // Loop over all pairs of electrons to get all connections (asymptotically optimal in Full CI limit)
            let mut double_excite_constructor: HashMap<u128, Vec<usize>> = HashMap::default();
            for dn in unique_up_dict[&unique.up] {
                for dn_r2 in remove_2e(dn.1) {
                    match double_excite_constructor.get(&dn_r2) {
                        None => double_excite_constructor.insert(dn_r2, vec![dn.0]),
                        Some(_) => double_excite_constructor[&dn_r2].push(dn.0)
                    }
                }
            }
            for dn in unique_up_dict[&unique.up] {
                for dn_r2 in remove_2e(dn.1) {
                    for dn2 in double_excite_constructor[&dn_r2] {
                        if dn.0 != dn2 {
                            // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                            add_el_and_spin_flipped(wf, ham, &mut off_diag_elems, *dn.0, *dn2); // only adds if elem != 0
                        }
                    }
                }
            }
        }
    }

    todo!()
}


fn add_el(wf: &Wf, ham: &Ham, off_diag_elems: &mut HashMap<(usize, usize), f64>, i: usize, j: usize) {
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
    match off_diag_elems.get(&key) {
        None => {
            let elem = ham.ham_off_diag_no_excite(wf.dets[&i].config, wf.dets[&j].config);
            if elem != 0.0 {
                off_diag_elems.insert(key, elem);
            }
        },
        Some(_) => {}
    }
}

fn add_el_and_spin_flipped(wf: &Wf, ham: &Ham, off_diag_elems: &mut HashMap<(usize, usize), f64>, i: usize, j: usize) {
    // Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
    // i and j can be of any order
    // TODO: Can speed up by skipping the second part if the first part is skipped

    // Element
    add_el(wf, ham, off_diag_elems, i, j);

    // Spin-flipped element
    let i_spin_flipped = wf.inds[&wf.dets[i].config];
    let j_spin_flipped = wf.inds[&wf.dets[j].config];
    add_el(wf, ham, off_diag_elems, i_spin_flipped, j_spin_flipped);
}