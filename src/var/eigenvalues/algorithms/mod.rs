/*!

## Algorithms to compute (some) eigenvalues/eigenvectors for symmetric matrices.

*/
pub mod davidson;
pub mod lanczos;
// pub mod davidson_fly;

/// Option to compute the lowest, highest or somewhere in the middle part of the
/// spectrum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrumTarget {
    Lowest,
    Highest,
    Target(f64),
}

/// Correction method for the Davidson algorithm
#[derive(Debug, Copy, Clone)]
pub enum DavidsonCorrection {
    DPR,
    GJD,
    HPR(f64), // Heat-bath preconditioned residue
}
