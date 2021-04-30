// Module for just performing sparse Davidson

extern crate eigenvalues;
extern crate nalgebra;
extern crate sprs;
use sprs::CsMat;
use eigenvalues::matrix_operations::MatrixOperations;
use nalgebra::base::{DVector, DMatrixSlice, DVectorSlice};
use nalgebra::DMatrix;

// We have to wrap the type CsMat in our own struct to avoid breaking backwards compatibility
#[derive(Clone)]
pub struct SparseMat{
    ham: CsMat<f64>,
}

impl MatrixOperations for SparseMat {
    fn matrix_vector_prod(&self, vs: DVectorSlice<'_, f64>) -> DVector<f64> {
        let mut res: DVector<f64> = vs.clone().into();
        let mut dprod: f64;
        for (row_ind, lvec) in self.ham.outer_iterator().enumerate() {
            dprod = 0.0;
            for (ind, val) in lvec.iter() {
                dprod += val * vs[ind];
            }
            res[row_ind] = dprod;
        }
        res
    }

    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        let mut res: DMatrix<f64> = mtx.clone().into();
        for (in_col, out_col) in mtx.iter().zip(res.iter()) {
            out_col = self.matrix_vector_prod(in_col);
        }
        res
    }

    fn diagonal(&self) -> DVector<f64> { 
        let mut diag: DVector<f64> = DVector::zeros(self.ham.rows());
        for (row_ind, lvec) in self.ham.outer_iterator().enumerate() {
            for (ind, val) in lvec.iter() {
                if ind == row_ind {
                    diag[ind] = *val;
                    break;
                }
            }
        }
        diag
     }

    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        for (row_ind, lvec) in self.ham.outer_iterator().enumerate() {
            for (ind, _) in lvec.iter() {
                if ind == row_ind {
                    // TODO: Fix this!
                    // self.diagonal()[ind] = *diag[ind];
                    break;
                }
            }
        }
    }

    fn ncols(&self) -> usize { self.ham.cols() }

    fn nrows(&self) -> usize { self.ham.rows() }
}

#[cfg(test)]
mod tests {
    use eigenvalues::davidson::Davidson;
    use eigenvalues::{DavidsonCorrection, SpectrumTarget};
    use super::*;

    #[test]
    fn test_sparse_davidson() {
        let sparse_ham: DMatrix<f64> = eigenvalues::utils::generate_diagonal_dominant(20, 0.005);

        let tolerance = 1e-4;

        // Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
        let eig = Davidson::new(sparse_ham, 2, DavidsonCorrection::DPR, SpectrumTarget::Lowest, tolerance).unwrap();
        println!("eigenvalues:{}", eig.eigenvalues[0]);
        println!("eigenvectors:{}", eig.eigenvectors);
        assert_eq!(1, 1);
    }

    #[test]
    fn test_sparse_ham_davidson() {
        let sparse_ham = SparseMat { ham: CsMat::new((3, 3),
                        vec![0, 2, 4, 5],
                        vec![0, 1, 0, 2, 2],
                        vec![1., 2., 3., 4., 5.])};
//        let mut sparse_ham = SparseMat { ham: CsMat::eye(2)};

//        sparse_ham.ham.set(0, 1, 0.005);
//        sparse_ham.ham.set(1, 0, 0.005);

        let tolerance = 1e-9;

        // Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
        let eig = Davidson::new(sparse_ham, 1, DavidsonCorrection::DPR, SpectrumTarget::Lowest, tolerance).unwrap();
        println!("eigenvalues:{}", eig.eigenvalues[0]);
        println!("eigenvectors:{}", eig.eigenvectors);
        assert_eq!(1, 1);
    }
}