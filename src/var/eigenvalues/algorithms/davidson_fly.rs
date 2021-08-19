/*!

# On-the-fly Davidson Diagonalization

The Davidson method is suitable for diagonal-dominant symmetric matrices,
that are quite common in certain scientific problems like [electronic
structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson
method could be not practical for other kind of symmetric matrices.

The current implementation uses a general davidson algorithm, meaning
that it compute all the requested eigenvalues simultaneusly using a variable
size block approach. The family of Davidson algorithm only differ in the way
that the correction vector is computed.

Available correction methods are:
 * **DPR**: Diagonal-Preconditioned-Residue
 * **GJD**: Generalized Jacobi Davidson

*/

use super::{DavidsonCorrection, SpectrumTarget};
use crate::matrix_operations::MatrixOperations;
use crate::utils;
use crate::MGS;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector, Dynamic};
use std::error;
use std::f64;
use std::fmt;

/// Structure containing the initial configuration data
struct Config {
    method: DavidsonCorrection,
    spectrum_target: SpectrumTarget,
    tolerance: f64,
    max_iters: usize,
    max_search_space: usize,
    init_dim: usize,   // Initial dimension of the subpace
    update_dim: usize, // number of vector to add to the search space
}
impl Config {
    /// Choose sensible default values for the davidson algorithm, where:
    /// * `nvalues` - Number of eigenvalue/eigenvector pairs to compute
    /// * `dim` - dimension of the matrix to diagonalize
    /// * `method` - Either DPR or GJD
    /// * `target` Lowest, highest or somewhere in the middle portion of the spectrum
    /// * `tolerance` Numerical tolerance to reach convergence
    fn new(
        nvalues: usize,
        dim: usize,
        method: DavidsonCorrection,
        target: SpectrumTarget,
        tolerance: f64,
    ) -> Self {
        let max_search_space = if nvalues * 10 < dim {
            nvalues
        } else {
            dim
        };
        Config {
            method,
            spectrum_target: target,
            tolerance,
            max_iters: 50,
            max_search_space,
            init_dim: nvalues,
            update_dim: nvalues,
        }
    }
}
#[derive(Debug, PartialEq)]
pub struct DavidsonError;

impl fmt::Display for DavidsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Davidson Algorithm did not converge!")
    }
}

impl error::Error for DavidsonError {}

/// Structure with the configuration data
pub struct Davidson {
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
}

impl Davidson {
    /// The new static method takes the following arguments:
    /// * `wf` - A wavefunction defined in the space of determinants being diagonalized (needed for
    ///          its associated function approx_matmul_variational, instead of storing H)
    /// * `nvalues` - the number of eigenvalues/eigenvectors pair to compute
    /// * `method` Either DPR or GJD
    /// * `spectrum_target` Lowest or Highest part of the spectrum
    /// * `tolerance` numerical tolerance.
    pub fn new(
        wf: Wf,
        nvalues: usize,
        method: DavidsonCorrection,
        spectrum_target: SpectrumTarget,
        tolerance: f64,
    ) -> Result<Self, DavidsonError> {
        // Initial configuration
        let conf = Config::new(nvalues, wf.n, method, spectrum_target, tolerance);

        // Initial subpace
        let mut dim_sub = conf.init_dim;
        // 1.1 Select the initial ortogonal subspace
        let mut basis = Self::generate_subspace(&wf, &conf); // only uses diagonal for sorting

        // 1.2 Select the correction to use
        let corrector = CorrectionMethod::new(&wf, conf.method);

        // 2. Generate subpace matrix problem by projecting into the basis
        let first_subspace = basis.columns(0, dim_sub);
        let mut matrix_subspace = h.matrix_matrix_prod(first_subspace);

        // Krylov-projected H
        let mut matrix_proj = first_subspace.transpose() * &matrix_subspace;

        // Outer loop block Davidson schema
        let mut result = Err(DavidsonError);
        for i in 0..conf.max_iters {
            let ord_sort = !matches!(conf.spectrum_target, SpectrumTarget::Highest);

            let eig = utils::sort_eigenpairs(SymmetricEigen::new(matrix_proj.clone()), ord_sort);

            // 4. Check for convergence
            // 4.1 Compute the residues
            let ritz_vectors = basis.columns(0, dim_sub) * eig.eigenvectors.columns(0, dim_sub);
            let residues = Self::compute_residues(&ritz_vectors, &matrix_subspace, &eig);

            // 4.2 Check Converge for each pair eigenvalue/eigenvector
            let errors = DVector::<f64>::from_iterator(
                nvalues,
                residues
                    .columns(0, nvalues)
                    .column_iter()
                    .map(|col| col.norm()),
            );
            // 4.3 Check if all eigenvalues/eigenvectors have converged
            if errors.iter().all(|&x| x < conf.tolerance) {
                result = Ok(Self::create_results(
                    &eig.eigenvalues,
                    &ritz_vectors,
                    nvalues,
                ));
                break;
            }
            // 5. Update subspace basis set
            // 5.1 Add the correction vectors to the current basis
            if dim_sub + conf.update_dim <= conf.max_search_space {
                let correction =
                    corrector.compute_correction(residues, &eig.eigenvalues, &ritz_vectors);
                update_subspace(&mut basis, correction, (dim_sub, dim_sub + conf.update_dim));

                // 6. Orthogonalize the subspace
                MGS::orthonormalize(&mut basis, dim_sub, dim_sub + conf.update_dim);

                // Update projected matrix
                matrix_subspace = {
                    let mut tmp = matrix_subspace.insert_columns(dim_sub, conf.update_dim, 0.0);
                    let new_block = h.matrix_matrix_prod(basis.columns(dim_sub, conf.update_dim));
                    let mut slice = tmp.columns_mut(dim_sub, conf.update_dim);
                    slice.copy_from(&new_block);
                    tmp
                };

                matrix_proj = {
                    let new_dim = dim_sub + conf.update_dim;
                    let new_subspace = basis.columns(0, new_dim);
                    let mut tmp = DMatrix::<f64>::zeros(new_dim, new_dim);
                    let mut slice = tmp.index_mut((..dim_sub, ..dim_sub));
                    slice.copy_from(&matrix_proj);
                    let new_block = new_subspace.transpose()
                        * matrix_subspace.columns(dim_sub, conf.update_dim);
                    let mut slice = tmp.index_mut((.., dim_sub..));
                    slice.copy_from(&new_block);
                    let mut slice = tmp.index_mut((dim_sub.., ..));
                    slice.copy_from(&new_block.transpose());
                    tmp
                };
                // update counter
                dim_sub += conf.update_dim;

            // 5.2 Otherwise reduce the basis of the subspace to the current
            // correction
            } else {
                dim_sub = conf.init_dim;
                basis.fill(0.0);
                update_subspace(&mut basis, ritz_vectors, (0, dim_sub));
                // Update projected matrix
                matrix_subspace = h.matrix_matrix_prod(basis.columns(0, dim_sub));
                matrix_proj = basis.columns(0, dim_sub).transpose() * &matrix_subspace;
            }
            // Check number of iterations
            if i > conf.max_iters {
                break;
            }
        }
        result
    }

    /// Extract the requested eigenvalues/eigenvectors pairs
    fn create_results(
        subspace_eigenvalues: &DVector<f64>,
        ritz_vectors: &DMatrix<f64>,
        nvalues: usize,
    ) -> Davidson {
        let eigenvectors = DMatrix::<f64>::from_iterator(
            ritz_vectors.nrows(),
            nvalues,
            ritz_vectors.columns(0, nvalues).iter().cloned(),
        );
        let eigenvalues = DVector::<f64>::from_iterator(
            nvalues,
            subspace_eigenvalues.rows(0, nvalues).iter().cloned(),
        );
        Davidson {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Residue vectors
    fn compute_residues(
        ritz_vectors: &DMatrix<f64>,
        matrix_subspace: &DMatrix<f64>,
        eig: &SymmetricEigen<f64, Dynamic>,
    ) -> DMatrix<f64> {
        let dim_sub = eig.eigenvalues.nrows();
        let lambda = {
            let mut tmp = DMatrix::<f64>::zeros(dim_sub, dim_sub);
            tmp.set_diagonal(&eig.eigenvalues);
            tmp
        };
        let vs = matrix_subspace * &eig.eigenvectors;
        let guess = ritz_vectors * lambda;
        vs - guess
    }

    /// Generate initial orthonormal subspace
    fn generate_subspace(wf: &Wf, conf: &Config) -> DMatrix<f64> {
        if conf.max_search_space > 1 {
            // More than one state, so initialize using spectrum for now
            let mut diags: Vec<f64> = vec![];
            for d in wf.dets {
                diags.push(d.diag)
            }
            let xs = diags.as_slice().to_vec();
            let mut rs = xs.clone();

            // update the matrix according to the spectrumtarget
            sort_diagonal(&mut rs, &conf);
            let mut mtx = DMatrix::<f64>::zeros(wf.n, conf.max_search_space);
            for i in 0..conf.max_search_space {
                let index = rs
                    .iter()
                    .position(|&x| (x - xs[i]).abs() < f64::EPSILON)
                    .unwrap();
                mtx[(i, index)] = 1.0;
            }
            mtx
        } else {
            // Use current wf's coeffs
            let  mut mtx = DMatrix::<f64>::zeros(diag.nrows(), conf.max_search_space);
            for (i, det) in wf.dets.iter().enumerate() {
                mtx[(0, i)] = det.coeff;
            }
            mtx
        }
    }
}

/// Structure containing the correction methods
struct CorrectionMethod<'a>
{
    /// The initial target matrix - here we use the wf instead because it has the ability to apply H
    /// on the fly
    target: &'a Wf,
    /// Method used to compute the correction
    method: DavidsonCorrection,
}

impl<'a> CorrectionMethod<'a>
{
    fn new(target: &'a Wf, method: DavidsonCorrection) -> CorrectionMethod<'a> {
        CorrectionMethod { target, method }
    }

    /// compute the correction vectors using either DPR or GJD
    fn compute_correction(
        &self,
        residues: DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        ritz_vectors: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        match self.method {
            DavidsonCorrection::DPR => self.compute_dpr_correction(residues, eigenvalues),
            DavidsonCorrection::GJD => {
                self.compute_gjd_correction(residues, eigenvalues, ritz_vectors)
            }
        }
    }

    /// Use the Diagonal-Preconditioned-Residue (DPR) method to compute the correction
    fn compute_dpr_correction(
        &self,
        residues: DMatrix<f64>,
        eigenvalues: &DVector<f64>,
    ) -> DMatrix<f64> {
        let d = self.target.diagonal();
        let mut correction = DMatrix::<f64>::zeros(self.target.n, residues.ncols());
        for (k, lambda) in eigenvalues.iter().enumerate() {
            let tmp = DVector::<f64>::repeat(self.target.n, *lambda) - &d;
            let rs = residues.column(k).component_div(&tmp);
            correction.set_column(k, &rs);
        }
        correction
    }

    /// Use the Generalized Jacobi Davidson (GJD) to compute the correction
    fn compute_gjd_correction(
        &self,
        residues: DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        ritz_vectors: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let dimx = self.target.n;
        let dimy = residues.ncols();
        let id = DMatrix::<f64>::identity(dimx, dimx);
        let ones = DVector::<f64>::repeat(dimx, 1.0);
        let mut correction = DMatrix::<f64>::zeros(dimx, dimy);
        let diag = self.target.diagonal();
        for (k, r) in ritz_vectors.column_iter().enumerate() {
            // Create the components of the linear system
            let t1 = &id - r * r.transpose();
            let mut t2 = self.target.clone();
            let val = &diag - &(eigenvalues[k] * &ones);
            t2.set_diagonal(&val);
            let arr = &t1 * &t2.matrix_matrix_prod(t1.rows(0, dimx));
            // Solve the linear system
            let decomp = arr.lu();
            let mut b = -residues.column(k);
            decomp.solve_mut(&mut b);
            correction.set_column(k, &b);
        }
        correction
    }
}

/// Update the subpace with new vectors
fn update_subspace(basis: &mut DMatrix<f64>, vectors: DMatrix<f64>, range: (usize, usize)) {
    let (start, end) = range;
    let mut slice = basis.index_mut((.., start..end));
    slice.copy_from(&vectors.columns(0, end - start));
}

fn sort_diagonal(rs: &mut Vec<f64>, conf: &Config) {
    match conf.spectrum_target {
        SpectrumTarget::Lowest => utils::sort_vector(rs, true),
        SpectrumTarget::Highest => utils::sort_vector(rs, false),
        _ => panic!("Not implemented error!"),
    }
}

/// Check if a vector is sorted in ascending order
fn is_sorted(xs: &DVector<f64>) -> bool {
    for k in 1..xs.len() {
        if xs[k] < xs[k - 1] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod test {
    use nalgebra::DMatrix;

    #[test]
    fn test_update_subspace() {
        let mut arr = DMatrix::<f64>::repeat(3, 3, 1.);
        let brr = DMatrix::<f64>::zeros(3, 2);
        super::update_subspace(&mut arr, brr, (0, 2));
        assert_eq!(arr.column(1).sum(), 0.);
        assert_eq!(arr.column(2).sum(), 3.);
    }
}
