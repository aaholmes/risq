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

use crate::var::eigenvalues::matrix_operations::MatrixOperations;

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


// Removed commented-out test module `tests`
