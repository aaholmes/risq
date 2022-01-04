/*!

# Davidson Diagonalization

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
use crate::var::eigenvalues::matrix_operations::MatrixOperations;
use crate::var::eigenvalues::modified_gram_schmidt::MGS;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector, Dynamic};
use std::error;
use std::f64;
use std::fmt;
use crate::var::eigenvalues::utils::{sort_vector, sort_eigenpairs};
use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::wf::Wf;

/// Structure containing the initial configuration data
struct Config {
    method: DavidsonCorrection,
    spectrum_target: SpectrumTarget,
    tolerance: f64,
    max_iters: usize,
    max_search_space: usize,
    init_dim: usize,   // Initial dimension of the subpace
    update_dim: usize, // number of vector to add to the search space
    reset_dim: usize, // dimension to reset to once max_search_space is reached
    energy_tolerance: f64,
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
        init_dim: usize,
        method: DavidsonCorrection,
        target: SpectrumTarget,
        tolerance: f64,
        energy_tolerance: f64
    ) -> Self {
        let mut max_search_space = if nvalues + 10 < dim {
            nvalues + 10
        } else {
            dim
        };
        let update_dim = nvalues;
        Config {
            method,
            spectrum_target: target,
            tolerance,
            energy_tolerance,
            max_iters: 50,
            max_search_space,
            init_dim,
            update_dim,
            reset_dim: nvalues,
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
    /// * `h` - A highly diagonal symmetric matrix
    /// * `nvalues` - the number of eigenvalues/eigenvectors pair to compute
    /// * `init` - the initial vector (optional; can be None)
    /// * `method` Either DPR or GJD
    /// * `spectrum_target` Lowest or Highest part of the spectrum
    /// * `tolerance` numerical tolerance.
    pub fn new<M: MatrixOperations>(
        h: &M,
        nvalues: usize,
        init: Option<DMatrix<f64>>,
        method: DavidsonCorrection,
        spectrum_target: SpectrumTarget,
        tolerance: f64,
        energy_tolerance: f64,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        wf: &Wf
    ) -> Result<Self, DavidsonError> {
        // Initial configuration
        let mut init_dim;
        match init {
            None => init_dim = 2, // If no input wf, start from a subspace of size 2
            Some(_) => init_dim = 1, // Else, just start from the input wf
        }
        let conf = Config::new(nvalues, h.nrows(), init_dim, method, spectrum_target, tolerance, energy_tolerance);

        // Initial subpace
        let mut dim_sub = conf.init_dim;
        // 1.1 Select the initial orthogonal subspace
        // Uses diagonal to find the dim_sub basis states with the lowest diagonal elements
        let mut basis = Self::generate_subspace(&h.diagonal(), &conf, init);

        // 1.2 Select the correction to use
        let corrector = CorrectionMethod::<M>::new(&h, conf.method);

        // 2. Generate subpace matrix problem by projecting into the basis
        let first_subspace = basis.columns(0, dim_sub);
        let mut matrix_subspace = h.matrix_matrix_prod(first_subspace);
        let mut matrix_proj = first_subspace.transpose() * &matrix_subspace;
        let mut last_eig: Vec<f64> = vec![0.0f64; nvalues];

        // Outer loop block Davidson schema
        let mut result = Err(DavidsonError);
        for i in 0..conf.max_iters {
            let ord_sort = !matches!(conf.spectrum_target, SpectrumTarget::Highest);

            let eig = sort_eigenpairs(SymmetricEigen::new(matrix_proj.clone()), ord_sort);

            if i > 0 { println!("Davidson Iteration {}: energy = {:.6}", i, eig.eigenvalues[0]); }

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
            // println!("Errors: {}", errors);
            // Check whether all coefficients and energies have converged:
            if h.nrows() <= 2 || (
                errors.iter().all(|&x| x < conf.tolerance) &&
                eig.eigenvalues.iter().zip(last_eig.iter()).all(|(&x, &y)| (x - y).abs() < conf.energy_tolerance)
            ) {
                result = Ok(Self::create_results(
                    &eig.eigenvalues,
                    &ritz_vectors,
                    nvalues,
                ));
                break;
            }
            for i in 0..nvalues {
                last_eig[i] = eig.eigenvalues[i];
            }
            // 5. Update subspace basis set
            // 5.1 Add the correction vectors to the current basis
            if dim_sub + conf.update_dim <= conf.max_search_space {
                let correction =
                    corrector.compute_correction(&residues, &eig.eigenvalues, &ritz_vectors, conf.update_dim, ham, excite_gen, wf);
                // correction vector can be close to zero!
                if correction.norm() < 1e-9 {
                    println!("Davidson converged because correction is too small!");
                    result = Ok(Self::create_results(
                        &eig.eigenvalues,
                        &ritz_vectors,
                        nvalues,
                    ));
                    break;
                }
                update_subspace(&mut basis, correction, (dim_sub, dim_sub + conf.update_dim));

                // 6. Orthogonalize the subspace
                if !MGS::orthonormalize(&mut basis, dim_sub, dim_sub + conf.update_dim) {
                    // New direction is not independent of previous ones
                    println!("Davidson converged because new correction vector is redundant!");
                    result = Ok(Self::create_results(
                        &eig.eigenvalues,
                        &ritz_vectors,
                        nvalues,
                    ));
                    break;
                }

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
                println!("Reducing the Krylov basis");
                dim_sub = conf.reset_dim;
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
    fn generate_subspace(diag: &DVector<f64>, conf: &Config, init: Option<DMatrix<f64>>) -> DMatrix<f64> {
        match init {
            None => {
                // If no input vector, start with the two basis states with lowest diagonal element
                DMatrix::<f64>::identity(diag.nrows(), conf.max_search_space)
            },
            Some(v) => {
                // If input vector exists, start with it (and other low-energy basis states if needed)
                let mut basis = DMatrix::<f64>::identity(diag.nrows(), conf.max_search_space);
                update_subspace(&mut basis, v, (conf.init_dim - 1, conf.init_dim));
                if conf.init_dim > 1 { MGS::orthonormalize(&mut basis, conf.init_dim - 1, conf.init_dim); }
                basis
            }
        }
    }
}

/// Structure containing the correction methods
struct CorrectionMethod<'a, M>
where
    M: MatrixOperations,
{
    /// The initial target matrix
    target: &'a M,
    /// Method used to compute the correction
    method: DavidsonCorrection,
}

impl<'a, M> CorrectionMethod<'a, M>
where
    M: MatrixOperations,
{
    fn new(target: &'a M, method: DavidsonCorrection) -> CorrectionMethod<'a, M> {
        CorrectionMethod { target, method }
    }

    /// compute the correction vectors using either DPR or GJD
    fn compute_correction(
        &self,
        residues: &DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        ritz_vectors: &DMatrix<f64>,
        update_dim: usize,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        wf: &Wf
    ) -> DMatrix<f64> {
        match self.method {
            DavidsonCorrection::DPR => self.compute_dpr_correction(residues, eigenvalues, update_dim),
            DavidsonCorrection::GJD => self.compute_gjd_correction(residues, eigenvalues, ritz_vectors),
            DavidsonCorrection::HPR(eps) => self.compute_hpr_correction(residues, eigenvalues, update_dim, ham, excite_gen, wf, eps),
        }
    }

    /// Use the Diagonal-Preconditioned-Residue (DPR) method to compute the correction
    fn compute_dpr_correction(
        &self,
        residues: &DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        update_dim: usize,
    ) -> DMatrix<f64> {
        let d = self.target.diagonal();
        let mut correction = DMatrix::<f64>::zeros(self.target.nrows(), residues.ncols());
        for (k, lambda) in eigenvalues.iter().enumerate() {
            if k == update_dim { break; }
            let tmp = DVector::<f64>::repeat(self.target.nrows(), *lambda) - &d;
            let rs = residues.column(k).component_div(&tmp);
            correction.set_column(k, &rs);
        }
        correction
    }

    /// Use the Heatbath-Preconditioned-Residue (HPR) method to compute the correction:
    /// Recall that the preconditioner (M - E) ^ {-1} multiplies the residual, and
    /// Davidson converges faster if M is close to H. Of course, we can't have M = H because
    /// inverting H is too expensive. Diagonal preconditioning chooses M to be the diagonal of H.
    /// Here, we use Heat-bath to approximate (H - E) ^ {-1}:
    /// Let H = D + O
    /// (H - E) ^ {-1} = (D - E) ^ {-1} * (1 + O / (D - E)) ^ {-1}
    /// ~ (D - E) ^ {-1} * (1 - O / (D - E)) + O(O^2)
    /// = (D - E) ^ {-1} - (D - E) ^ {-1} * O * (D - E) ^ {-1}
    /// = (1 - (D - E) ^ {-1} * O) * (D - E) ^ {-1}
    /// So, we compute the usual Davidson preconditioner, then act on it with an approximation
    /// to the second term in parentheses above
    fn compute_hpr_correction(
        &self,
        residues: &DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        update_dim: usize,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        wf: &Wf,
        eps: f64
    ) -> DMatrix<f64> {
        let d = self.target.diagonal();
        let mut correction = DMatrix::<f64>::zeros(self.target.nrows(), residues.ncols());
        for (k, lambda) in eigenvalues.iter().enumerate() {
            if k == update_dim { break; }
            let mut tmp = DVector::<f64>::repeat(self.target.nrows(), *lambda) - &d;
            let rs = residues.column(k).component_div(&tmp);
            // rs is (D - E) ^ {-1} * residual
            println!("DPR Correctioin: {:?}", rs);
            println!("Performing HB...");
            let scaled_eps = eps; // * (rs[0] / wf.dets[0].coeff).abs();
            println!("Wf eps = {}, but using scaled eps = {} instead", eps, scaled_eps);
            let mut hb_off_times_rs = DVector::from(wf.approx_matmul_off_diag_variational_no_singles(&rs, ham, excite_gen, scaled_eps)); // Perform off-diagonal heat-bath here
            println!("Done performing HB...");
            println!("{:?}", hb_off_times_rs);
            for i in 0..5 {
                hb_off_times_rs[i] = 0.0;
            }
            let rs2 = hb_off_times_rs.component_div(&tmp);
            println!("HPR Correction: {:?}", &rs - &rs2);
            correction.set_column(k, &(rs - rs2)); // Add the DPR correction to the heat-bath off-diagonal correction to get the full correction
        }
        correction
    }

    /// Use the Generalized Jacobi Davidson (GJD) to compute the correction
    fn compute_gjd_correction(
        &self,
        residues: &DMatrix<f64>,
        eigenvalues: &DVector<f64>,
        ritz_vectors: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        // Commenting out because this clones the entire matrix, but we don't use this routine anyway
        // let dimx = self.target.nrows();
        // let dimy = residues.ncols();
        // let id = DMatrix::<f64>::identity(dimx, dimx);
        // let ones = DVector::<f64>::repeat(dimx, 1.0);
        // let mut correction = DMatrix::<f64>::zeros(dimx, dimy);
        // let diag = self.target.diagonal();
        // for (k, r) in ritz_vectors.column_iter().enumerate() {
        //     // Create the components of the linear system
        //     let t1 = &id - r * r.transpose();
        //     let mut t2 = self.target.clone();
        //     let val = &diag - &(eigenvalues[k] * &ones);
        //     t2.set_diagonal(&val);
        //     let arr = &t1 * &t2.matrix_matrix_prod(t1.rows(0, dimx));
        //     // Solve the linear system
        //     let decomp = arr.lu();
        //     let mut b = -residues.column(k);
        //     decomp.solve_mut(&mut b);
        //     correction.set_column(k, &b);
        // }
        // correction
        todo!()
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
        SpectrumTarget::Lowest => sort_vector(rs, true),
        SpectrumTarget::Highest => sort_vector(rs, false),
        _ => panic!("Not implemented error!"),
    }
}

/// Check if a vector is sorted in ascending order
// fn is_sorted(xs: &DVector<f64>) -> bool {
//     for k in 1..xs.len() {
//         if xs[k] < xs[k - 1] {
//             return false;
//         }
//     }
//     true
// }

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
