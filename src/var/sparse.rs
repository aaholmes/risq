// Sparse matrix datatype that works with eigenvalues module

extern crate eigenvalues;
extern crate nalgebra;
extern crate sprs;
use sprs::CsMat;
use eigenvalues::matrix_operations::MatrixOperations;
use nalgebra::base::{DVector, DMatrixSlice, DVectorSlice};
use nalgebra::DMatrix;

#[derive(Clone)]
pub struct SparseMat{
    n: usize,
    diag: DVector<f64>,
    off_diag: CsMat<f64>,
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