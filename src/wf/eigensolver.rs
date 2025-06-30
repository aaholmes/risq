//! # Eigenvalue Solver Abstractions
//!
//! This module provides unified traits and implementations for eigenvalue solving
//! algorithms used in quantum chemistry, particularly Davidson and Lanczos methods.

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::var::eigenvalues::algorithms::davidson::{Davidson, DavidsonError};
use crate::var::eigenvalues::algorithms::{DavidsonCorrection, SpectrumTarget};
use crate::var::sparse::SparseMatUpperTri;
use crate::wf::{VarWf, Wf};
use nalgebra::{DMatrix, DVector};
use std::fmt;

/// Configuration for eigenvalue solver operations
#[derive(Debug, Clone)]
pub struct EigenConfig {
    /// Number of eigenvalues/eigenvectors to compute
    pub n_states: usize,
    /// Convergence tolerance for eigenvectors
    pub tolerance: f64,
    /// Convergence tolerance for eigenvalues
    pub energy_tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Initial guess vectors (optional)
    pub initial_guess: Option<DMatrix<f64>>,
    /// Spectrum target (lowest, highest, etc.)
    pub spectrum_target: SpectrumTarget,
    /// Correction method for Davidson
    pub correction_method: DavidsonCorrection,
}

impl Default for EigenConfig {
    fn default() -> Self {
        Self {
            n_states: 1,
            tolerance: 1e-8,
            energy_tolerance: 1e-10,
            max_iterations: 100,
            initial_guess: None,
            spectrum_target: SpectrumTarget::Lowest,
            correction_method: DavidsonCorrection::DPR,
        }
    }
}

/// Result from eigenvalue solver containing eigenvalues and eigenvectors
#[derive(Debug, Clone)]
pub struct EigenSolution {
    /// Computed eigenvalues
    pub eigenvalues: DVector<f64>,
    /// Computed eigenvectors
    pub eigenvectors: DMatrix<f64>,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
}

/// Error types for eigenvalue solvers
#[derive(Debug, Clone)]
pub enum EigenSolverError {
    /// Davidson algorithm failed to converge
    DavidsonConvergenceFailure {
        iterations: usize,
        tolerance: f64,
    },
    /// Invalid matrix dimensions
    InvalidDimensions {
        message: String,
    },
    /// Numerical instability detected
    NumericalInstability,
    /// Configuration error
    ConfigError {
        message: String,
    },
}

impl fmt::Display for EigenSolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenSolverError::DavidsonConvergenceFailure { iterations, tolerance } => {
                write!(f, "Davidson failed to converge after {} iterations (tolerance: {:.2e})", iterations, tolerance)
            }
            EigenSolverError::InvalidDimensions { message } => {
                write!(f, "Invalid matrix dimensions: {}", message)
            }
            EigenSolverError::NumericalInstability => {
                write!(f, "Numerical instability detected during eigenvalue solving")
            }
            EigenSolverError::ConfigError { message } => {
                write!(f, "Configuration error: {}", message)
            }
        }
    }
}

impl std::error::Error for EigenSolverError {}

impl From<DavidsonError> for EigenSolverError {
    fn from(_: DavidsonError) -> Self {
        EigenSolverError::DavidsonConvergenceFailure {
            iterations: 0,
            tolerance: 0.0,
        }
    }
}

/// Trait for eigenvalue solver algorithms
///
/// This trait abstracts different eigenvalue solving approaches like Davidson,
/// Lanczos, or other iterative methods used in quantum chemistry.
pub trait EigenSolver {
    /// Solve the eigenvalue problem for the given system
    ///
    /// # Arguments
    /// * `wf` - The variational wavefunction containing the system
    /// * `ham` - The Hamiltonian operator
    /// * `excite_gen` - Excitation generator
    /// * `config` - Solver configuration
    ///
    /// # Returns
    /// Solution containing eigenvalues and eigenvectors, or error
    fn solve(
        &mut self,
        wf: &mut VarWf,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        config: &EigenConfig,
    ) -> Result<EigenSolution, EigenSolverError>;

    /// Check if the solver has converged
    fn is_converged(&self) -> bool;

    /// Get the number of iterations performed
    fn iterations(&self) -> usize;

    /// Get solver name for debugging/logging
    fn name(&self) -> &'static str;
}

/// Davidson eigenvalue solver implementation
#[derive(Debug)]
pub struct DavidsonSolver {
    /// Current iteration count
    iterations: usize,
    /// Convergence status
    converged: bool,
    /// Last computed solution
    last_solution: Option<EigenSolution>,
}

impl DavidsonSolver {
    /// Create a new Davidson solver
    pub fn new() -> Self {
        Self {
            iterations: 0,
            converged: false,
            last_solution: None,
        }
    }
}

impl Default for DavidsonSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EigenSolver for DavidsonSolver {
    fn solve(
        &mut self,
        wf: &mut VarWf,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        config: &EigenConfig,
    ) -> Result<EigenSolution, EigenSolverError> {
        // Validate configuration
        if config.n_states == 0 {
            return Err(EigenSolverError::ConfigError {
                message: "Number of states must be positive".to_string(),
            });
        }

        if wf.wf.n == 0 {
            return Err(EigenSolverError::InvalidDimensions {
                message: "Wavefunction has no determinants".to_string(),
            });
        }

        // Use the existing Davidson implementation
        let davidson_result = Davidson::new(
            &wf.sparse_ham,
            config.n_states,
            config.initial_guess.clone(),
            config.correction_method,
            config.spectrum_target,
            config.tolerance,
            config.energy_tolerance,
            ham,
            excite_gen,
            &wf.wf,
        );

        match davidson_result {
            Ok(davidson) => {
                self.converged = true;
                self.iterations += 1; // This would need to be extracted from Davidson implementation
                
                let solution = EigenSolution {
                    eigenvalues: davidson.eigenvalues,
                    eigenvectors: davidson.eigenvectors,
                    iterations: self.iterations,
                    converged: true,
                };
                
                self.last_solution = Some(solution.clone());
                Ok(solution)
            }
            Err(err) => {
                self.converged = false;
                Err(EigenSolverError::from(err))
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn name(&self) -> &'static str {
        "Davidson"
    }
}

/// Lanczos eigenvalue solver implementation (placeholder for future implementation)
#[derive(Debug)]
pub struct LanczosSolver {
    iterations: usize,
    converged: bool,
}

impl LanczosSolver {
    /// Create a new Lanczos solver
    pub fn new() -> Self {
        Self {
            iterations: 0,
            converged: false,
        }
    }
}

impl Default for LanczosSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EigenSolver for LanczosSolver {
    fn solve(
        &mut self,
        _wf: &mut VarWf,
        _ham: &Ham,
        _excite_gen: &ExciteGenerator,
        _config: &EigenConfig,
    ) -> Result<EigenSolution, EigenSolverError> {
        // Placeholder implementation
        Err(EigenSolverError::ConfigError {
            message: "Lanczos solver not yet implemented".to_string(),
        })
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn name(&self) -> &'static str {
        "Lanczos"
    }
}

/// Factory for creating eigenvalue solvers
pub struct EigenSolverFactory;

impl EigenSolverFactory {
    /// Create a Davidson solver with default configuration
    pub fn davidson() -> DavidsonSolver {
        DavidsonSolver::new()
    }

    /// Create a Lanczos solver with default configuration
    pub fn lanczos() -> LanczosSolver {
        LanczosSolver::new()
    }

    /// Create the most appropriate solver for the given problem size
    pub fn auto(problem_size: usize) -> Box<dyn EigenSolver> {
        if problem_size < 1000 {
            Box::new(DavidsonSolver::new())
        } else {
            // For larger problems, Davidson is still preferred in quantum chemistry
            Box::new(DavidsonSolver::new())
        }
    }
}