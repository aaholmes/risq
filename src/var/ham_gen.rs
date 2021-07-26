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


pub fn gen_sparse_ham_fast(global: &Global, wf: &Wf, ham: &Ham, verbose: bool) -> SparseMat {
    // Generate Ham as a sparse matrix using my 2019 notes when I was working pro bono
    // For now, assumes that nup == ndn

    // Setup

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
    if verbose { println!("Unique_up_dict:"); }
    for (key, val) in &unique_up_dict {
        if verbose { println!("{}: ({}, {})", key, val[0].0, val[0].1); }
    }

    // 2. Sort the unique up dets in decreasing order by corresponding number of determinants.
    struct Unique {
        up: u128,
        n_dets: usize,
        n_dets_remaining: usize
    }
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


    // Opposite-spin excitations

    for unique in &unique_ups_sorted {
        let first_algo_complexity = unique.n_dets * (global.ndn * global.ndn) as usize + unique.n_dets_remaining; // Term in parentheses is an estimate
        let second_algo_complexity = global.ndn as usize * (unique.n_dets + unique.n_dets_remaining);
        let third_algo_complexity = unique.n_dets * unique.n_dets_remaining;
        let mut first_algo_fastest = false;
        let mut second_algo_fastest = false;
        // if first_algo_complexity <= second_algo_complexity {
        //     if first_algo_complexity <= third_algo_complexity { first_algo_fastest = true; }
        // } else {
        //     if second_algo_complexity <= third_algo_complexity { second_algo_fastest = true; }
        // }
        if first_algo_fastest {
            if verbose { println!("First algo fastest"); }
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
            for up in &up_singles[&unique.up] {
                for dn in &unique_up_dict[&up] {
                    match dn_candidates.get(&dn.1) {
                        None => {},
                        Some(dn_connections) => {
                            // We need to do this one in terms of up config rather than ind
                            let ind1 = wf.inds[&Config { up: *up, dn: dn.1 }];
                            for dn_connection_ind in dn_connections {
                                let ind2 = wf.inds[&Config { up: *up, dn: unique_dns_vec[*dn_connection_ind] }];
                                off_diag_elems.add_el(wf, ham, ind1, ind2); // only adds if elem != 0
                            }
                        }
                    }
                }
            }
        } else if second_algo_fastest {
            if verbose { println!("Second algo fastest"); }
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
                                for connected_ind in &dn_single_excite_constructor[&dn_r1] {
                                    off_diag_elems.add_el(wf, ham, dn.0, *connected_ind)
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if verbose { println!("Third algo fastest"); }
            // Loop over all pairs of dn dets
            if verbose { println!("unique.up = {}", unique.up); }
            match up_singles.get(&unique.up) {
                None => {},
                Some(ups) => {
                    for up in ups {
                        if verbose { println!("up: {}", up); }
                        for ind1 in &unique_up_dict[&unique.up] {
                            if verbose { println!("ind1= ({}, {})", ind1.0, ind1.1); }
                            for ind2 in &unique_up_dict[&up] {
                                if verbose { println!("Found excitation: {} {}", ind1.0, ind2.0); }
                                off_diag_elems.add_el(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
                            }
                        }
                    }
                }
            }
        }
    }


    // Same-spin excitations

    // Parameter for choosing which of the two algorithms to use
    let max_n_dets_double_loop = (global.ndn * (global.ndn - 1)) as usize;

    for unique in &unique_ups_sorted {
        if true { // unique.n_dets <= max_n_dets_double_loop {
            if verbose { println!("Same-spin first algo fastest"); }
            // Use the double for-loop algorithm from "Fast SHCI"
            for (i_ind, ind1) in unique_up_dict[&unique.up].iter().enumerate() {
                for ind2 in unique_up_dict[&unique.up][i_ind + 1 ..].iter() {
                    // Found dn-spin excitations (and their spin-flipped up-spin excitations):
                    off_diag_elems.add_el_and_spin_flipped(wf, ham, ind1.0, ind2.0); // only adds if elem != 0
                }
            }
        } else {
            if verbose { println!("Same-spin second algo fastest"); }
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


    // Finally, put collected off-diag elems into a sparse matrix
    off_diag_elems.to_sparse(wf, verbose)
}


// Off-diag elems data structure
struct OffDiagElems {
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
                    self.nonzero_inds.insert(key);

                    self.nnz[i + 1] += 1;
                    self.indices[i].push(j);
                    self.values[i].push(elem);

                    self.nnz[j + 1] += 1;
                    self.indices[j].push(i);
                    self.values[j].push(elem);
                }
            },
            Some(_) => {}
        }
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

        if verbose { println!("indices, data:"); }
        for (i, j) in indices.iter().zip(data.iter()) {
            if verbose { println!("{}, {}", i, j); }
        }

        // Off-diagonal component
        let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
        match off_diag_component {
            Err(_) => { panic!("Error in constructing CsMat"); }
            Ok(off_diag) => { return SparseMat{n, diag: DVector::from(diag), off_diag}; }
        }
    }
}
