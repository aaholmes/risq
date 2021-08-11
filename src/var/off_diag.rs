// Data structure for holding off-diagonal Hamiltonian matrix elements
// and putting them into the SparseMat data structure for Davidson

use sprs::CsMat;
use std::time::Instant;
use nalgebra::base::DVector;

use crate::ham::Ham;
use crate::var::sparse::{SparseMat, SparseMatUpperTri};
use crate::wf::Wf;
use crate::wf::det::Config;

// Off-diag elems data structure
#[derive(Default)]
pub struct OffDiagElemsNoHash {
    n: usize, // number of rows
    nonzero: Vec<Vec<(usize, f64)>>, // vector of (ind, val) pairs for each row
    nnz: Vec<usize>, // number of nonzero elements (only update when generating SparseMat, only used to determine what has been added onto the end)
}

impl OffDiagElemsNoHash {
    pub fn new(n: usize) -> Self {
        Self {
            n: n,
            nonzero: vec![Vec::with_capacity(100); n],
            nnz: vec![0; n + 1], // 0th element is always 0; nnz[i + 1] corresponds to row i
        }
    }

    pub fn expand_rows(&mut self, n: usize) {
        // Just create empty rows to fill up the dimension to new size n
        println!("Expanding variational H from size {} to size {}", self.n, n);
        self.nonzero.append(&mut vec![Vec::with_capacity(100); n - self.n]);
        self.nnz.append(&mut vec![0; n - self.n]);
        self.n = n;
    }
}

pub fn add_el(wf: &mut Wf, ham: &Ham, i: usize, j: usize, elem: Option<f64>) {
    // Add an off-diagonal element H_{ij} to off_diag_elems
    // i and j can be of any order

    match elem {
        None => {
            if i == j { return; }
            let elem = ham.ham_off_diag_no_excite(&wf.dets[i].config, &wf.dets[j].config);
            if elem != 0.0 {
                if i < j {
                    wf.sparse_ham.off_diag[i].push((j, elem));
                    // if j >= wf.n_stored_h() { wf.sparse_ham.off_diag[i].push((j, elem)); }
                } else {
                    wf.sparse_ham.off_diag[j].push((i, elem));
                    // if i >= wf.n_stored_h() { wf.sparse_ham.off_diag[j].push((i, elem)); }
                }
            }
        },
        Some(elem) => {
            if i < j {
                wf.sparse_ham.off_diag[i].push((j, elem));
                // if j >= wf.n_stored_h() { wf.sparse_ham.off_diag[i].push((j, elem)); }
            } else {
                wf.sparse_ham.off_diag[j].push((i, elem)) ;
                // if i >= wf.n_stored_h() { wf.sparse_ham.off_diag[j].push((i, elem)); }
            }
        }
    }
}

pub fn add_el_and_spin_flipped(wf: &mut Wf, ham: &Ham, i: usize, j: usize) {
    // Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
    // i and j can be of any order

    if i == j { return; }
    let elem = ham.ham_off_diag_no_excite(&wf.dets[i].config, &wf.dets[j].config);
    if elem != 0.0 {

        // Element
        add_el(wf, ham, i, j, Some(elem));

        // Spin-flipped element
        let i_spin_flipped = {
            let config = wf.dets[i].config;
            wf.inds[&Config { up: config.dn, dn: config.up }]
        };
        let j_spin_flipped = {
            let config = wf.dets[j].config;
            wf.inds[&Config { up: config.dn, dn: config.up }]
        };
        add_el(wf, ham, i_spin_flipped, j_spin_flipped, Some(elem));
    }
}

// pub fn create_sparse(wf: &mut Wf) -> SparseMat {
//     // Creates sparse matrix from the off-diag elements
//     // At the end of this routine, update wf.n_last to wf.n (these two variables only needed to be different for determining which part of H is new)
//     let start_create_sparse: Instant = Instant::now();
//
//     // Put the matrix elements into a sparse matrix
//     let n = wf.n;
//
//     // Diagonal component
//     let mut diag = Vec::with_capacity(n);
//     for det in &wf.dets {
//         diag.push(det.diag);
//     }
//
//     // Off-diagonal component
//
//     let shape = (n, n);
//
//     // Sort each vec, remove duplicates
//     // (updating self.nnz in the process)
//
//     let res;
//
//     if let Some(stored_off_diag) = &mut wf.off_diag_h_elems {
//
//         for (i, v) in stored_off_diag.nonzero.iter_mut().enumerate() {
//             // Sort by index
//             v[stored_off_diag.nnz[i + 1]..].sort_by_key(|k| k.0);
//             // Call dedup on everything because I can't figure out how to do dedup on a slice
//             // (but should be fast enough anyway)
//             v.dedup_by_key(|k| k.0);
//             stored_off_diag.nnz[i + 1] = v.len();
//         }
//
//         let indptr: Vec<usize> = stored_off_diag.nnz
//             .iter()
//             .scan(0, |acc, &x| {
//                 *acc = *acc + x;
//                 Some(*acc)
//             })
//             .collect();
//
//         let indices: Vec<usize> = stored_off_diag.nonzero.iter()
//             .flatten()
//             .map(|x| x.0)
//             .collect();
//         let data: Vec<f64> = stored_off_diag.nonzero.iter()
//             .flatten()
//             .map(|x| x.1)
//             .collect();
//
//         println!("Variational Hamiltonian has {} nonzero off-diagonal elements", indices.len());
//
//         // Off-diagonal component
//         let off_diag = CsMat::<f64>::new(shape, indptr, indices, data);
//         println!("Time for converting stored nonzero indices to sparse H: {:?}", start_create_sparse.elapsed());
//
//         res = SparseMat { n, diag: DVector::from(diag), off_diag };
//
//     } else {
//
//         panic!("Failed to access wf's stored off-diagonal elements!")
//
//     }
//
//     // Update n_stored_h to be equal to n because we have now stored the whole variational H
//     wf.update_n_stored_h(wf.n);
//
//     res
// }

// Off-diag elems data structure
// pub struct OffDiagElems {
//     nonzero_inds: HashSet<(usize, usize)>, // number of nonzero elements in the upper triangular half of the matrix
//     nnz: Vec<usize>, // number of nonzero values for each row
//     indices: Vec<Vec<usize>>, // vector of indices for each row
//     values: Vec<Vec<f64>>, // vector of values for each row
// }
//
//
// impl OffDiagElems {
//     pub fn new(n: usize) -> Self {
//         Self{
//             nonzero_inds: HashSet::default(),
//             nnz: vec![0; n + 1],
//             indices: vec![vec![]; n],
//             values: vec![vec![]; n]
//         }
//     }
//
//     pub fn add_el(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij} to off_diag_elems
//         // i and j can be of any order
//
//         if i == j { return; }
//         let key = {
//             if i < j {
//                 (i, j)
//             } else {
//                 (j, i)
//             }
//         };
//         match self.nonzero_inds.get(&key) {
//             None => {
//                 let elem = ham.ham_off_diag_no_excite(&wf.dets[i].config, &wf.dets[j].config);
//                 if elem != 0.0 {
//                     self.insert_nonzero_inds(key); //nonzero_inds.insert(key);
//
//                     self.nnz[i + 1] += 1;
//                     // self.indices[i].push(j);
//                     // self.values[i].push(elem);
//
//                     self.nnz[j + 1] += 1;
//                     // self.indices[j].push(i);
//                     // self.values[j].push(elem);
//
//                     self.push_indices_values(i, j, elem);
//                 }
//             },
//             Some(_) => {}
//         }
//     }
//
//     fn insert_nonzero_inds(&mut self, key: (usize, usize)) {
//         self.nonzero_inds.insert(key);
//     }
//
//     fn push_indices_values(&mut self, i: usize, j: usize, elem: f64) {
//         self.indices[i].push(j);
//         self.values[i].push(elem);
//         self.indices[j].push(i);
//         self.values[j].push(elem);
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
//     pub fn create_sparse(&self, wf: &Wf) -> SparseMat {
//         let start_create_sparse: Instant = Instant::now();
//         // Put the matrix elements into a sparse matrix
//         let n = wf.n;
//
//         // Diagonal component
//         let mut diag = Vec::with_capacity(n);
//         for det in &wf.dets {
//             diag.push(det.diag);
//         }
//         let shape = (n, n);
//         let indptr: Vec<usize> = self.nnz
//             .iter()
//             .scan(0, |acc, &x| {
//                 *acc = *acc + x;
//                 Some(*acc)
//             })
//             .collect();
//
//         let indices: Vec<usize> = self.indices.clone().into_iter().flatten().collect();
//         let data: Vec<f64> = self.values.clone().into_iter().flatten().collect();
//
//         println!("Variational Hamiltonian has {} nonzero off-diagonal elements", indices.len());
//
//         // Off-diagonal component
//         let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
//         println!("Time for converting stored nonzero indices to sparse H: {:?}", start_create_sparse.elapsed());
//         match off_diag_component {
//             Err(_) => { panic!("Error in constructing CsMat"); }
//             Ok(off_diag) => { return SparseMat{n, diag: DVector::from(diag), off_diag}; }
//         }
//     }
//
//     pub fn create_sparse_verbose(&self, wf: &Wf, verbose: bool) -> SparseMat {
//         let start_create_sparse: Instant = Instant::now();
//         // Put the matrix elements into a sparse matrix
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
//         let shape = (n, n);
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
//         println!("Variational Hamiltonian has {} nonzero off-diagonal elements", indices.len());
//
//         if verbose { println!("indices, data:"); }
//         for (i, j) in indices.iter().zip(data.iter()) {
//             if verbose { println!("{}, {}", i, j); }
//         }
//
//         // Off-diagonal component
//         let off_diag_component = CsMat::<f64>::new_from_unsorted(shape, indptr, indices, data);
//         println!("Time for converting stored nonzero indices to sparse H: {:?}", start_create_sparse.elapsed());
//         match off_diag_component {
//             Err(_) => { panic!("Error in constructing CsMat"); }
//             Ok(off_diag) => { return SparseMat{n, diag: DVector::from(diag), off_diag}; }
//         }
//     }
// }

// Off-diag elems data structure
// pub struct OffDiagElemsNoHash2 {
//     indices: Vec<(usize, usize)>, // vector of (row, col) indices
// }
//
//
// impl OffDiagElemsNoHash2 {
//     pub fn new(n: usize) -> Self {
//         Self {
//             indices: Vec::with_capacity(100 * n),
//         }
//     }
//
//     pub fn add_el(&mut self, wf: &Wf, ham: &Ham, i: usize, j: usize) {
//         // Add an off-diagonal element H_{ij} to off_diag_elems
//         // i and j can be of any order
//         if i == j { return; }
//         self.indices.push((i, j));
//         self.indices.push((j, i));
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
//     pub fn create_sparse(&mut self, ham: &Ham, wf: &Wf) -> SparseMat {
//         let start_create_sparse: Instant = Instant::now();
//         // Put the matrix elements into a sparse matrix
//         let n = wf.n;
//
//         // Diagonal component
//         let mut diag = Vec::with_capacity(n);
//         for det in &wf.dets {
//             diag.push(det.diag);
//         }
//         let shape = (n, n);
//
//         // Off-diagonal component
//
//         // Sort indices
//         self.indices.sort_by_key(|k| (k.0, k.1));
//
//         // Remove duplicates
//         self.indices.dedup();
//
//         // Compute indptr, reorganize indices, compute matrix elements
//         let mut nnz: Vec<usize> = vec![0; n + 1];
//         let mut indices: Vec<usize> = vec![0; self.indices.len()];
//         let mut data: Vec<f64> = vec![0.0; self.indices.len()];
//         for (i, ind) in self.indices.iter().enumerate() {
//             nnz[ind.0 + 1] += 1;
//             indices[i] = ind.1;
//             data[i] = ham.ham_off_diag_no_excite(&wf.dets[ind.0].config, &wf.dets[ind.1].config);
//         }
//         let indptr: Vec<usize> = nnz
//             .iter()
//             .scan(0, |acc, &x| {
//                 *acc = *acc + x;
//                 Some(*acc)
//             })
//             .collect();
//
//         println!("Variational Hamiltonian has {} nonzero off-diagonal elements", indices.len());
//
//         // Off-diagonal component
//         let off_diag = CsMat::<f64>::new(shape, indptr, indices, data);
//         println!("Time for converting stored nonzero indices to sparse H: {:?}", start_create_sparse.elapsed());
//         return SparseMat { n, diag: DVector::from(diag), off_diag };
//     }
// }
