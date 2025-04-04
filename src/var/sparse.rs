//! # Custom Sparse Matrix Representations (`var::sparse`)
//!
//! This module defines custom data structures for representing the sparse Hamiltonian matrix
//! used in the variational HCI calculation. It provides implementations of the
//! `MatrixOperations` trait required by the Davidson eigenvalue solver.
//!
//! Different structures might represent different storage strategies or optimizations.

extern crate nalgebra;
extern crate sprs;
use nalgebra::base::{DMatrixSlice, DVector, DVectorSlice};
use nalgebra::DMatrix;
use sprs::CsMat;
use std::collections::HashMap;
// Removed commented-out Rayon imports

use crate::ham::Ham;
use crate::var::eigenvalues::matrix_operations::MatrixOperations;
use crate::var::utils::intersection;
use crate::wf::det::Config;
use crate::wf::Wf;

/// Represents a symmetric sparse matrix storing only the upper triangle.
///
/// This structure stores the diagonal elements explicitly and the off-diagonal
/// elements as a vector of vectors (adjacency list format). `off_diag[i]` contains
/// a list of tuples `(j, H_ij)` where `j > i`.
/// This format allows for efficient addition of elements during Hamiltonian generation,
/// as sorting is only needed once before diagonalization.
#[derive(Default, Debug)] // Added Debug derive
pub struct SparseMatUpperTri {
    /// Dimension of the square matrix (N x N).
    pub(crate) n: usize,
    /// Vector storing the diagonal elements `H_ii`.
    pub(crate) diag: Vec<f64>,
    /// Tracks the number of non-zero off-diagonal elements currently stored *per row*
    /// *before* deduplication. Used internally during construction.
    pub(crate) nnz: Vec<usize>,
    /// Stores the non-zero upper triangular off-diagonal elements.
    /// `off_diag[i]` is a `Vec<(usize, f64)>` containing pairs `(j, H_ij)` where `j > i`.
    /// Elements might be unsorted or contain duplicates during construction.
    pub off_diag: Vec<Vec<(usize, f64)>>,
}

impl SparseMatUpperTri {
    /// Sorts the off-diagonal elements for each row by column index and removes duplicates.
    ///
    /// This should be called after generating all Hamiltonian elements and before using
    /// the matrix in operations like matrix-vector products (e.g., within Davidson).
    /// Updates the internal `nnz` count after deduplication.
    pub fn sort_remove_duplicates(&mut self) {
        let mut n_upper_t = 0;
        for (i, v) in self.off_diag.iter_mut().enumerate() {
            // Sort by index
            v[self.nnz[i]..].sort_by_key(|k| k.0);
            // Call dedup on everything because I can't figure out how to do dedup on a slice
            // (but should be fast enough anyway)
            v.dedup_by_key(|k| k.0);
            self.nnz[i] = v.len();
            n_upper_t += self.nnz[i];
        }
        println!("Variational Hamiltonian has {} nonzero off-diagonal elements in upper triangle, {} total", n_upper_t, 2 * n_upper_t);
    }
}

impl MatrixOperations for SparseMatUpperTri {
    /// Computes the matrix-vector product `y = A*x` where `A` is `self` and `x` is `vs`.
    /// Exploits the symmetry and upper triangular storage:
    /// `y_i = A_ii*x_i + sum_{j>i} A_ij*x_j + sum_{j<i} A_ji*x_j`
    /// where `A_ji = A_ij` is retrieved from `self.off_diag[j]`.
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

    /// Computes the matrix-matrix product `C = A*B` where `A` is `self` and `B` is `mtx`.
    /// Performs the product column by column using `matrix_vector_prod`.
    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    /// Returns a copy of the diagonal elements as a `DVector`.
    fn diagonal(&self) -> DVector<f64> {
        DVector::from(self.diag.clone())
    }

    /// Sets the diagonal elements from a `DVector`.
    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        for i in 0..self.n {
            self.diag[i] = diag[i];
        }
    }

    /// Returns the number of columns (matrix dimension).
    fn ncols(&self) -> usize {
        self.n
    }

    /// Returns the number of rows (matrix dimension).
    fn nrows(&self) -> usize {
        self.n
    }
}

/// Represents a sparse matrix using `sprs::CsMat` for off-diagonal elements.
///
/// This structure stores the diagonal separately and uses the Compressed Sparse Row (CSR)
/// format from the `sprs` crate for the off-diagonal part. Likely used for testing or
/// comparison, as `SparseMatUpperTri` seems to be the primary format for HCI.
#[derive(Debug)] // Added Debug derive
pub struct SparseMat {
    /// Dimension of the square matrix (N x N).
    pub n: usize,
    /// Vector storing the diagonal elements `H_ii`.
    pub diag: DVector<f64>,
    /// Off-diagonal elements stored in Compressed Sparse Row format.
    pub off_diag: CsMat<f64>,
}

impl SparseMat {
    /// Creates a `SparseMat` from a dense `nalgebra::DMatrix`.
    /// Primarily intended for testing purposes.
    #[cfg(test)]
    pub fn from_dense(mtx: DMatrix<f64>) -> Self {
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
                if i != j && mtx[(i, j)] != 0.0 {
                    indices.push(j);
                    data.push(mtx[(i, j)]);
                    nnz += 1;
                }
            }
        }
        indptr.push(nnz);

        println!(
            "Constructing {} x {} matrix with {} nonzero elements",
            n, n, nnz
        );
        SparseMat {
            n,
            diag: DVector::from(diag),
            off_diag: sprs::CsMat::new((n, n), indptr, indices, data),
        }
    }
}

impl MatrixOperations for SparseMat {
    /// Computes the matrix-vector product `y = A*x`.
    /// Combines the diagonal contribution and the off-diagonal contribution computed
    /// using the `sprs::CsMat` multiplication.
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

    /// Computes the matrix-matrix product `C = A*B`.
    /// Performs the product column by column using `matrix_vector_prod`.
    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    /// Returns a copy of the diagonal elements.
    fn diagonal(&self) -> DVector<f64> {
        self.diag.clone()
    }

    /// Sets the diagonal elements.
    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        self.diag = diag.clone();
    }

    /// Returns the number of columns.
    fn ncols(&self) -> usize {
        self.n
    }

    /// Returns the number of rows.
    fn nrows(&self) -> usize {
        self.n
    }
}

/// Represents a sparse matrix where single and double excitations are handled differently.
///
/// This structure seems designed for specific algorithms (perhaps related to `ham_gen` variants)
/// where single excitation contributions are stored explicitly in a `CsMat`, but double
/// excitation contributions are computed on-the-fly during matrix-vector products using
/// pre-computed lookup tables (`doubles`) and the `Ham` object.
/// This might be an optimization if the number of double excitation *types* is much smaller
/// than the number of actual non-zero double excitation matrix elements.
#[derive(Debug)] // Added Debug derive
pub struct SparseMatDoubles<'a> {
    /// Matrix dimension.
    pub n: usize,
    /// Diagonal elements.
    pub diag: DVector<f64>,
    /// Off-diagonal elements arising from *single* excitations, stored in CSR format.
    pub singles: CsMat<f64>,
    /// Lookup table for double excitations. Maps an initial orbital pair `(p, q)` and spin `is_alpha`
    /// to a list of `(Config, index)` tuples representing determinants that can be formed by
    /// removing `p` and `q` from some determinant in the basis. Used to find connected pairs for doubles.
    pub doubles: HashMap<(i32, i32, Option<bool>), Vec<(Config, usize)>>,
    /// Reference to the Hamiltonian to compute double excitation matrix elements on the fly.
    pub ham: &'a Ham,
    /// Reference to the wavefunction to look up determinant configurations by index.
    pub wf: &'a Wf,
}

impl MatrixOperations for SparseMatDoubles<'_> {
    /// Computes the matrix-vector product `y = A*x`.
    ///
    /// Calculates contributions from:
    /// 1. Diagonal elements (`self.diag`).
    /// 2. Single excitations (using `self.singles` CSR matrix).
    /// 3. Double excitations: Iterates through the `doubles` lookup table, finds connected
    ///    pairs `(i, j)` using the `intersection` utility, computes `H_ij` using `self.ham.ham_doub`,
    ///    and adds the contributions `H_ij * x_j` to `y_i` and `H_ij * x_i` to `y_j`.
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
                if k.2 == l.2 {
                    // Same total spin
                    if if k.2 == None {
                        l.0 > k.0
                    } else {
                        l.0 > k.0 || (l.0 == k.0 && l.1 > k.1)
                    } {
                        // No double counting
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
                                if k.2 == None {
                                    self.ham.direct(k.0, k.1, l.0, l.1)
                                } else {
                                    self.ham.direct_plus_exchange(k.0, k.1, l.0, l.1)
                                }
                            };
                            if h_kl != 0.0 {
                                // Skip zero elements
                                // Loop over intersection of v_k.0 and v_l.0
                                // Read off the corresponding pairs of indices: these are connected with this type of double exctie matrix element!
                                for (i, j) in intersection(v_k, self.doubles.get(&l).unwrap()) {
                                    let h_ij = self
                                        .ham
                                        .ham_doub(&self.wf.dets[i].config, &self.wf.dets[j].config);
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

    /// Computes the matrix-matrix product `C = A*B`.
    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, mut out_col) in mtx.column_iter().zip(res.column_iter_mut()) {
            out_col.copy_from(&self.matrix_vector_prod(in_col));
        }
        res
    }

    /// Returns a copy of the diagonal elements.
    fn diagonal(&self) -> DVector<f64> {
        self.diag.clone()
    }

    /// Sets the diagonal elements.
    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        self.diag = diag.clone();
    }

    /// Returns the number of columns.
    fn ncols(&self) -> usize {
        self.n
    }

    /// Returns the number of rows.
    fn nrows(&self) -> usize {
        self.n
    }
}

// Removed commented-out test module `tests`
