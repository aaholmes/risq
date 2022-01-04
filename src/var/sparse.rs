// Sparse matrix datatype that works with eigenvalues module

extern crate nalgebra;
extern crate sprs;
use sprs::CsMat;
use nalgebra::base::{DVector, DMatrixSlice, DVectorSlice};
use nalgebra::DMatrix;
use std::collections::HashMap;
// use rayon::prelude::into_par_iter;
// use rayon::iter::IntoParallelIterator;

use crate::wf::det::Config;
use crate::var::utils::intersection;
use crate::ham::Ham;
use crate::wf::Wf;
use crate::var::eigenvalues::matrix_operations::MatrixOperations;


// Upper triangular sparse matrix
// nonzero off-diagonal elements are at (ind1, ind2) where ind1 < ind2
#[derive(Default)]
pub struct SparseMatUpperTri{
    pub(crate) n: usize,
    pub(crate) diag: Vec<f64>,
    pub(crate) nnz: Vec<usize>, // Number of nonzero off-diagonal elements in each row
    pub off_diag: Vec<Vec<(usize, f64)>>, // Make our own CsMat because we need it to be temporarily unsorted
}

impl SparseMatUpperTri {
    pub fn sort_remove_duplicates(&mut self) {
        let mut n_upper_t = 0;
        for (i, v) in self.off_diag.iter_mut().enumerate() {
            // println!("Before sort and dedup: Row {} has {} elements:", i, self.nnz[i]);
            // println!("{:?}", v.iter().scan(0, |_, (ind, val)| Some(ind)));
            // Sort by index
            v[self.nnz[i]..].sort_by_key(|k| k.0);
            // Call dedup on everything because I can't figure out how to do dedup on a slice
            // (but should be fast enough anyway)
            v.dedup_by_key(|k| k.0);
            self.nnz[i] = v.len();
            // println!("Row {} has {} elements:", i, self.nnz[i]);
            // println!("{:?}", v.iter().scan(0, |_, (ind, val)| Some(ind)));
            n_upper_t += self.nnz[i];
        }
        println!("Variational Hamiltonian has {} nonzero off-diagonal elements in upper triangle, {} total", n_upper_t, 2 * n_upper_t);
    }
}

impl MatrixOperations for SparseMatUpperTri {
    fn matrix_vector_prod(&self, vs: DVectorSlice<'_, f64>) -> DVector<f64> {
        let mut res: DVector<f64> = DVector::from(vec![0.0; self.n]);

        for row_ind in (0..self.n).into_iter() {
            // Diagonal component
            res[row_ind] += self.diag[row_ind] * vs[row_ind];

            // Off-diagonal component
            for (ind, val) in self.off_diag[row_ind].iter() {
                res[row_ind] += val * vs[*ind];
                res[*ind] += val * vs[row_ind];
            }
        }
        res
    }

    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    fn diagonal(&self) -> DVector<f64> {
        DVector::from(self.diag.clone())
    }

    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        for i in 0..self.n {
            self.diag[i] = diag[i];
        }
    }

    fn ncols(&self) -> usize { self.n }

    fn nrows(&self) -> usize { self.n }
}


pub struct SparseMat{
    pub n: usize,
    pub diag: DVector<f64>,
    pub off_diag: CsMat<f64>,
}

impl SparseMat {
    pub fn from_dense(mtx: DMatrix<f64>) -> Self {
        // Convert a dense matrix to a sparse matrix
        // (for testing only)

        let n = mtx.ncols();
        println!("n = {}", n);

        // Diag part
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            diag.push(mtx[(i, i)]);
        }

        // Off-diag part
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);
        let mut nnz = 0;

        for i in 0..n {
            indptr.push(nnz);
            for j in 0..n {
                if i != j {
                    if mtx[(i, j)] != 0.0 {
                        indices.push(j);
                        data.push(mtx[(i, j)]);
                        nnz += 1;
                    }
                }
            }
        }
        indptr.push(nnz);

        println!("Constructing {} x {} matrix with {} nonzero elements", n, n, nnz);
        SparseMat{
            n,
            diag: DVector::from(diag),
            off_diag: sprs::CsMat::new((n, n), indptr, indices, data)
        }
    }
}

impl MatrixOperations for SparseMat {
    fn matrix_vector_prod(&self, vs: DVectorSlice<'_, f64>) -> DVector<f64> {
        let mut res: DVector<f64> = vs.clone().into();
        let mut dprod: f64;

        for (row_ind, lvec) in self.off_diag.outer_iterator().enumerate() {
            // Diagonal component
            dprod = self.diag[row_ind] * vs[row_ind];

            // Off-diagonal component
            for (ind, val) in lvec.iter() {
                // println!("Excite: ({} {}) = {}", row_ind, ind, val);
                dprod += val * vs[ind];
            }
            res[row_ind] = dprod;
        }
        res
    }

    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    fn diagonal(&self) -> DVector<f64> {
        self.diag.clone()
    }

    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        self.diag = diag.clone();
    }

    fn ncols(&self) -> usize { self.n }

    fn nrows(&self) -> usize { self.n }
}


pub struct SparseMatDoubles<'a> {
    // Sparse mat with doubles stored as a lookup data structure that is smaller than the actual H
    pub n: usize,
    pub diag: DVector<f64>,
    pub singles: CsMat<f64>,
    pub doubles: HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>>,
    pub ham: &'a Ham, // point to ham, so we can compute matrix elements as needed
    pub wf: &'a Wf, // point to wf, so we can look up configs as needed
}

impl MatrixOperations for SparseMatDoubles<'_> {
    fn matrix_vector_prod(&self, vs: DVectorSlice<'_, f64>) -> DVector<f64> {
        let mut res: DVector<f64> = vs.clone().into();
        let mut dprod: f64;

        for (row_ind, lvec) in self.singles.outer_iterator().enumerate() {
            // Diagonal component
            dprod = self.diag[row_ind] * vs[row_ind];

            // Singles component
            for (ind, val) in lvec.iter() {
                // println!("Single excite: ({} {}) = {}", row_ind, ind, val);
                dprod += val * vs[ind];
            }
            res[row_ind] = dprod;
        }

        // Doubles
        for (k, v_k) in self.doubles.iter() {
            for l in self.doubles.keys() {
                if k.2 == l.2 { // Same total spin
                    if if k.2 == None { l.0 > k.0 } else { l.0 > k.0 || (l.0 == k.0 && l.1 > k.1) } { // No double counting
                        // Check that these are 4 distinct spin-orbitals
                        if {
                            if k.2 == None {
                                // Up must be distinct, dn must be distinct
                                k.0 != l.0 && k.1 != l.1
                            } else {
                                // All 4 must be distinct orbitals
                                k.0 != l.0 && k.0 != l.1 && k.1 != l.0 && k.1 != l.1
                            }
                        } {
                            let h_kl = {
                                if k.2 == None { self.ham.direct(k.0, k.1, l.0, l.1) } else { self.ham.direct_plus_exchange(k.0, k.1, l.0, l.1) }
                            };
                            if h_kl != 0.0 { // Skip zero elements
                                // Loop over intersection of v_k.0 and v_l.0
                                // Read off the corresponding pairs of indices: these are connected with this type of double exctie matrix element!
                                for (i, j) in intersection(v_k, self.doubles.get(&l).unwrap()) {
                                    let h_ij = self.ham.ham_doub(&self.wf.dets[i].config, &self.wf.dets[j].config);
                                    // println!("({} {}) {}, ({} {}) {}: H = {}",
                                    //          k.0, k.1, { match k.2 { None => { "opp spin" }, Some(b) => if b { "spin up" } else { "spin dn" } } },
                                    //          l.0, l.1, { match l.2 { None => { "opp spin" }, Some(b) => if b { "spin up" } else { "spin dn" } } },
                                    //          h_kl);
                                    // println!("Updating out vec! with H({}, {}) = {}", i, j, h_ij);
                                    res[i] += h_ij * vs[j];
                                    res[j] += h_ij * vs[i];
                                }
                            }
                        }
                    }
                }
            }

        }

        res
    }

    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    fn diagonal(&self) -> DVector<f64> { 
        self.diag.clone()
    }

    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        self.diag = diag.clone();
    }

    fn ncols(&self) -> usize { self.n }

    fn nrows(&self) -> usize { self.n }
}


// #[cfg(test)]
// mod tests {
//     use eigenvalues::davidson::Davidson;
//     use eigenvalues::{DavidsonCorrection, SpectrumTarget};
//     use super::*;
//
//     #[test]
//     fn test_sparse_davidson() {
//         let sparse_ham: DMatrix<f64> = eigenvalues::utils::generate_diagonal_dominant(20, 0.005);
//
//         let tolerance = 1e-4;
//
//         // Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
//         let eig = Davidson::new(sparse_ham, 2, DavidsonCorrection::DPR, SpectrumTarget::Lowest, tolerance).unwrap();
//         println!("eigenvalues:{}", eig.eigenvalues[0]);
//         println!("eigenvectors:{}", eig.eigenvectors);
//         assert_eq!(1, 1);
//     }
//
//     #[test]
//     fn test_sparse_ham_davidson() {
//         let sparse_ham = SparseMat { ham: CsMat::new((3, 3),
//                         vec![0, 2, 4, 5],
//                         vec![0, 1, 0, 2, 2],
//                         vec![1., 2., 3., 4., 5.])};
// //        let mut sparse_ham = SparseMat { ham: CsMat::eye(2)};
//
// //        sparse_ham.ham.set(0, 1, 0.005);
// //        sparse_ham.ham.set(1, 0, 0.005);
//
//         let tolerance = 1e-9;
//
//         // Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
//         let eig = Davidson::new(sparse_ham, 1, DavidsonCorrection::DPR, SpectrumTarget::Lowest, tolerance).unwrap();
//         println!("eigenvalues:{}", eig.eigenvalues[0]);
//         println!("eigenvectors:{}", eig.eigenvectors);
//         assert_eq!(1, 1);
//     }
// }