pub mod algorithms;
pub mod matrix_operations;
pub mod modified_gram_schmidt;
pub mod utils;

// use nalgebra as na;

// use algorithms::lanczos::HermitianLanczos;
// use algorithms::SpectrumTarget;
// use utils::{generate_random_sparse_symmetric, sort_eigenpairs};

// fn main() {
//     let matrix = generate_random_sparse_symmetric(200, 5, 0.5);
//     let eig = sort_eigenpairs(na::linalg::SymmetricEigen::new(matrix.clone()), false);
//     let spectrum_target = SpectrumTarget::Highest;
//     let lanczos = HermitianLanczos::new(matrix, 50, spectrum_target).unwrap();
//     println!("Computed eigenvalues:\n{}", lanczos.eigenvalues.rows(0, 3));
//     println!("Expected eigenvalues:\n{}", eig.eigenvalues.rows(0, 3));
//     let x = eig.eigenvectors.column(0);
//     let y = lanczos.eigenvectors.column(0);
//     println!("parallel:{}", x.dot(&y));
// }
