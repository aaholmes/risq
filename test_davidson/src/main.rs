extern crate eigenvalues;
extern crate nalgebra as na;

// Use the Davidson method
use eigenvalues::davidson::Davidson;
use eigenvalues::SpectrumTarget;
use na::{DMatrix, DVector};

fn main() {
    // Generate random symmetric matrix
    let matrix = eigenvalues::utils::generate_diagonal_dominant(100, 0.005);
    let tolerance = 1e-4;

    // Compute the first 2 lowest eigenvalues/eigenvectors using the DPR method
    let eig = Davidson::new(matrix.clone(), 2, "DPR", SpectrumTarget::Lowest, tolerance).unwrap();
    println!("eigenvalues:{}", eig.eigenvalues);
    println!("eigenvectors:{}", eig.eigenvectors);
}
