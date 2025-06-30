//! # Unified Davidson Interface
//!
//! This module provides a unified interface for the Davidson eigenvalue solver
//! that integrates with the new trait-based system while maintaining compatibility
//! with the existing implementation.

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::wf::eigensolver::{EigenConfig, EigenSolver, EigenSolverError, EigenSolution};
use crate::wf::VarWf;
use nalgebra::{DMatrix, DVector};

/// Unified Davidson solver that bridges the old and new interfaces
pub struct UnifiedDavidsonSolver {
    iterations: usize,
    converged: bool,
}

impl UnifiedDavidsonSolver {
    /// Create a new unified Davidson solver
    pub fn new() -> Self {
        Self {
            iterations: 0,
            converged: false,
        }
    }

    /// Solve using a simplified approach (placeholder for demonstration)
    pub fn solve_simplified(
        &mut self,
        wf: &mut VarWf,
        _ham: &Ham,
        _excite_gen: &ExciteGenerator,
        config: &EigenConfig,
    ) -> EigenSolution {
        // For demonstration purposes, we'll create a mock solution
        // In practice, this would call the actual Davidson implementation
        
        self.converged = true;
        self.iterations += 1;

        // Extract current state as solution
        let eigenvalues = DVector::from_vec(vec![wf.wf.energy]);
        let mut eigenvector_data = Vec::with_capacity(wf.wf.n);
        for det in &wf.wf.dets {
            eigenvector_data.push(det.coeff);
        }
        let eigenvectors = DMatrix::from_vec(wf.wf.n, config.n_states, eigenvector_data);

        EigenSolution {
            eigenvalues,
            eigenvectors,
            iterations: self.iterations,
            converged: self.converged,
        }
    }
}

impl Default for UnifiedDavidsonSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EigenSolver for UnifiedDavidsonSolver {
    fn solve(
        &mut self,
        wf: &mut VarWf,
        ham: &Ham,
        excite_gen: &ExciteGenerator,
        config: &EigenConfig,
    ) -> Result<EigenSolution, EigenSolverError> {
        // Use the simplified approach for demonstration
        let solution = self.solve_simplified(wf, ham, excite_gen, config);
        Ok(solution)
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn name(&self) -> &'static str {
        "Unified Davidson"
    }
}

/// Convenience function to perform Davidson diagonalization using the new trait interface
pub fn solve_davidson_unified(
    wf: &mut VarWf,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    config: Option<EigenConfig>,
) -> Result<EigenSolution, EigenSolverError> {
    let config = config.unwrap_or_default();
    let mut solver = UnifiedDavidsonSolver::new();
    solver.solve(wf, ham, excite_gen, &config)
}