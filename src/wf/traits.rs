//! # Wavefunction Traits Module
//!
//! This module defines traits for abstracting matrix-vector multiplication operations
//! and eigenvalue solving algorithms used throughout the HCI implementation.

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::stoch::ScreenedSampler;
use crate::wf::{Wf, VarWf};
use nalgebra::{Const, Dynamic, Matrix, VecStorage};

/// Strategy for handling single excitations in matrix-vector products
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SinglesStrategy {
    /// Include all singles deterministically
    Include,
    /// Skip singles entirely
    Skip,
    /// Handle singles stochastically (semi-stochastic approach)
    Semistochastic,
    /// No singles processing
    None,
}

/// Strategy for handling double excitations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoublesStrategy {
    /// Separate processing for doubles and singles
    Separate,
    /// Standard double excitation processing
    Standard,
    /// Deterministic-only processing
    DeterministicOnly,
}

/// Configuration for matrix-vector product operations
#[derive(Debug, Clone)]
pub struct MatVecConfig {
    /// Screening threshold (epsilon)
    pub eps: f64,
    /// Strategy for handling single excitations
    pub singles_strategy: SinglesStrategy,
    /// Strategy for handling double excitations
    pub doubles_strategy: DoublesStrategy,
    /// Whether to compute diagonal elements
    pub compute_diagonals: bool,
    /// Input coefficients for variational operations
    pub input_coeffs: Option<Vec<f64>>,
}

/// Result of matrix-vector multiplication operations
pub enum MatVecResult {
    /// External space result: new wavefunction + matrix elements vector
    ExternalWithElements(Wf, Vec<f64>),
    /// External space result: new wavefunction + screened sampler (boxed to avoid lifetime issues)
    ExternalWithSampler(Wf, Box<dyn std::fmt::Debug>),
    /// Variational space result: coefficient vector
    Variational(Vec<f64>),
    /// Diagonal computation result: wavefunction with computed diagonals
    WithDiagonals(Wf),
}

/// Trait for matrix-vector multiplication operations on wavefunctions
///
/// This trait unifies the various approx_matmul_* functions by providing
/// a common interface for different matrix-vector product strategies.
pub trait MatrixVectorProduct {
    /// Apply the Hamiltonian to the wavefunction using specified configuration
    ///
    /// # Arguments
    /// * `ham` - The Hamiltonian operator
    /// * `excite_gen` - Excitation generator for determinant connections
    /// * `config` - Configuration specifying the operation parameters
    ///
    /// # Returns
    /// Result of the matrix-vector multiplication based on the configuration
    fn apply_hamiltonian(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        config: &MatVecConfig,
    ) -> MatVecResult;

    /// Convenience method for external perturbative space operations
    fn apply_external(
        &self,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
        singles_strategy: SinglesStrategy,
    ) -> MatVecResult {
        let config = MatVecConfig {
            eps,
            singles_strategy,
            doubles_strategy: DoublesStrategy::Standard,
            compute_diagonals: false,
            input_coeffs: None,
        };
        self.apply_hamiltonian(ham, excite_gen, &config)
    }

    /// Convenience method for variational space operations
    fn apply_variational(
        &self,
        input_coeffs: &[f64],
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> Vec<f64> {
        let config = MatVecConfig {
            eps,
            singles_strategy: SinglesStrategy::Include,
            doubles_strategy: DoublesStrategy::Standard,
            compute_diagonals: false,
            input_coeffs: Some(input_coeffs.to_vec()),
        };
        
        match self.apply_hamiltonian(ham, excite_gen, &config) {
            MatVecResult::Variational(coeffs) => coeffs,
            _ => panic!("Expected variational result from apply_variational"),
        }
    }
}

/// Trait for eigenvalue solver algorithms
///
/// This trait abstracts different eigenvalue solving approaches like Davidson,
/// Lanczos, or other iterative methods used in quantum chemistry.
pub trait EigenSolver {
    /// Error type for eigenvalue solver operations
    type Error;
    
    /// Result type containing eigenvalues and eigenvectors
    type Solution;

    /// Solve the eigenvalue problem for the given system
    ///
    /// # Arguments
    /// * `wf` - The variational wavefunction containing the system
    /// * `ham` - The Hamiltonian operator
    /// * `excite_gen` - Excitation generator
    /// * `n_states` - Number of eigenvalues/eigenvectors to compute
    /// * `tolerance` - Convergence tolerance
    /// * `max_iterations` - Maximum number of iterations
    ///
    /// # Returns
    /// Solution containing eigenvalues and eigenvectors, or error
    fn solve(
        &mut self,
        wf: &mut VarWf,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        n_states: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self::Solution, Self::Error>;

    /// Check if the solver has converged
    fn is_converged(&self) -> bool;

    /// Get the number of iterations performed
    fn iterations(&self) -> usize;
}

/// Davidson eigenvalue solver implementation
#[derive(Debug)]
pub struct DavidsonSolver {
    /// Current iteration count
    iterations: usize,
    /// Convergence status
    converged: bool,
    /// Energy convergence threshold
    energy_eps: f64,
    /// Coefficient convergence threshold
    coeff_eps: f64,
}

/// Solution from Davidson eigenvalue solver
#[derive(Debug)]
pub struct DavidsonSolution {
    /// Computed eigenvalues
    pub eigenvalues: Vec<f64>,
    /// Computed eigenvectors
    pub eigenvectors: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    /// Number of iterations to convergence
    pub iterations: usize,
}

/// Error types for Davidson solver
#[derive(Debug, thiserror::Error)]
pub enum DavidsonError {
    #[error("Failed to converge after {0} iterations")]
    ConvergenceFailure(usize),
    #[error("Invalid matrix dimensions: {message}")]
    InvalidDimensions { message: String },
    #[error("Numerical instability detected")]
    NumericalInstability,
}

impl DavidsonSolver {
    /// Create a new Davidson solver with specified tolerances
    pub fn new(energy_eps: f64, coeff_eps: f64) -> Self {
        Self {
            iterations: 0,
            converged: false,
            energy_eps,
            coeff_eps,
        }
    }
}

impl EigenSolver for DavidsonSolver {
    type Error = DavidsonError;
    type Solution = DavidsonSolution;

    fn solve(
        &mut self,
        wf: &mut VarWf,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        n_states: usize,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self::Solution, Self::Error> {
        // This will delegate to the existing Davidson implementation
        // For now, return a placeholder to establish the interface
        todo!("Implement Davidson solver using existing var::davidson::sparse_optimize")
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn iterations(&self) -> usize {
        self.iterations
    }
}