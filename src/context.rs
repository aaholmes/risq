//! # Calculation Context (`context`)
//!
//! Central context struct that replaces all global state and provides
//! dependency injection for the entire calculation pipeline.

use crate::config::{AlgorithmParameters, GlobalConfig};
use crate::error::{RisqError, RisqResult};
use crate::excite::init::{init_excite_generator, ExciteGenerator};
use crate::ham::{read_ints::read_ints, Ham};
use crate::rng::{init_rand, Rand};
use std::path::Path;

/// Central context struct that replaces all global state
#[derive(Debug)]
pub struct RisqContext {
    /// Molecular system configuration
    pub config: GlobalConfig,
    
    /// Algorithm parameters and thresholds
    pub algorithm_params: AlgorithmParameters,
    
    /// Electronic integrals and Hamiltonian
    pub hamiltonian: Ham,
    
    /// Pre-computed excitation generator
    pub excitation_generator: ExciteGenerator,
    
    /// Random number generator state (initialized on demand)
    pub rng: Option<Rand>,
}

impl RisqContext {
    /// Create context from file paths
    pub fn from_files<P1, P2>(config_path: P1, fcidump_path: P2) -> RisqResult<Self>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        // Read and validate configuration
        let (config, algorithm_params) = GlobalConfig::from_file(config_path)?;
        
        // Read molecular integrals
        let hamiltonian = read_ints(&config, fcidump_path)?;
        
        // Initialize excitation generator
        let excitation_generator = init_excite_generator(&config, &hamiltonian)?;
        
        Ok(Self {
            config,
            algorithm_params,
            hamiltonian,
            excitation_generator,
            rng: None,
        })
    }
    
    /// Create context with explicit components
    pub fn new(
        config: GlobalConfig,
        algorithm_params: AlgorithmParameters,
        hamiltonian: Ham,
        excitation_generator: ExciteGenerator,
    ) -> Self {
        Self {
            config,
            algorithm_params,
            hamiltonian,
            excitation_generator,
            rng: None,
        }
    }
    
    /// Initialize random number generator (lazy initialization)
    pub fn init_rng(&mut self) -> RisqResult<()> {
        if self.rng.is_none() {
            self.rng = Some(init_rand()?);
        }
        Ok(())
    }
    
    /// Get random number generator, initializing if needed
    pub fn get_rng(&mut self) -> RisqResult<&mut Rand> {
        if self.rng.is_none() {
            self.init_rng()?;
        }
        Ok(self.rng.as_mut().unwrap())
    }
    
    /// Get a reference to the configuration as the old Global struct
    /// TODO: Remove this compatibility method after full refactoring
    pub fn as_global(&self) -> crate::utils::read_input::Global {
        crate::utils::read_input::Global {
            norb: self.config.n_orbs as i32,
            norb_core: self.config.n_core as i32,
            nup: self.config.n_up as i32,
            ndn: self.config.n_dn as i32,
            z_sym: self.config.z_sym,
            n_states: self.config.n_states as i32,
            eps_var: self.algorithm_params.eps_var,
            eps_pt_dtm: self.algorithm_params.eps_pt_dtm,
            opp_algo: self.algorithm_params.opp_algo as i32,
            same_algo: self.algorithm_params.same_algo as i32,
            target_uncertainty: self.algorithm_params.target_uncertainty,
            n_samples_per_batch: self.algorithm_params.n_samples_per_batch as i32,
            n_batches: self.algorithm_params.n_batches as i32,
            n_cross_term_samples: self.algorithm_params.n_cross_term_samples as i32,
            use_new_semistoch: self.algorithm_params.use_new_semistoch,
        }
    }
    
    /// Check if random number generator is initialized
    pub fn has_rng(&self) -> bool {
        self.rng.is_some()
    }
    
    /// Get number of active orbitals
    pub fn n_orb_active(&self) -> usize {
        self.config.n_orb_active()
    }
    
    /// Get number of active electrons
    pub fn n_elec_active(&self) -> usize {
        self.config.n_elec_active()
    }
    
    /// Print context summary
    pub fn print_summary(&self) {
        println!("System Configuration:");
        println!("  Total orbitals:      {}", self.config.n_orbs);
        println!("  Core orbitals:       {}", self.config.n_core);
        println!("  Active orbitals:     {}", self.n_orb_active());
        println!("  Alpha electrons:     {}", self.config.n_up);
        println!("  Beta electrons:      {}", self.config.n_dn);
        println!("  Total electrons:     {}", self.config.n_elec_total());
        println!("  Active electrons:    {}", self.n_elec_active());
        println!("  Target states:       {}", self.config.n_states);
        println!();
        println!("Algorithm Parameters:");
        println!("  Variational threshold:  {:.2e}", self.algorithm_params.eps_var);
        println!("  PT threshold:           {:.2e}", self.algorithm_params.eps_pt_dtm);
        println!("  Target uncertainty:     {:.2e}", self.algorithm_params.target_uncertainty);
        println!("  Max var iterations:     {}", self.algorithm_params.max_iter_var);
        println!("  Max PT iterations:      {}", self.algorithm_params.max_iter_pt);
        println!("  Samples per batch:      {}", self.algorithm_params.n_samples_per_batch);
        println!("  Max batches:            {}", self.algorithm_params.n_batches);
        println!("  Use new semistochastic: {}", self.algorithm_params.use_new_semistoch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_context_error_handling() {
        // Test with nonexistent files
        let result = RisqContext::from_files("nonexistent.json", "nonexistent.fcidump");
        assert!(result.is_err());
        
        if let Err(e) = result {
            assert!(e.to_string().contains("nonexistent.json"));
        }
    }
    
    #[test]
    fn test_config_validation() {
        // Create config with invalid values (negative orbitals)
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"{{
            "norb": -1,
            "norb_core": 0,
            "nup": 1,
            "ndn": 1,
            "eps_var": 1e-4,
            "eps_pt_dtm": 1e-6,
            "target_uncertainty": 1e-4,
            "n_samples_per_batch": 1000,
            "n_batches": 100
        }}"#).unwrap();
        
        let result = RisqContext::from_files(temp_file.path(), "nonexistent");
        match result {
            Err(RisqError::InvalidConfig { message }) => {
                assert!(message.contains("norb must be positive"));
            },
            _ => panic!("Expected invalid config error"),
        }
    }
}