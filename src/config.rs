//! # Configuration Management (`config`)
//!
//! Handles reading and validating configuration parameters from JSON files,
//! replacing the old Global struct with proper validation and error handling.

use crate::error::{RisqError, RisqResult};
use crate::{risq_bail, risq_ensure};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Core system configuration (cleaned up version of old Global struct)
#[derive(Debug, Clone)]
pub struct GlobalConfig {
    /// Total number of spatial orbitals
    pub n_orbs: usize,
    /// Number of core orbitals to freeze
    pub n_core: usize,
    /// Number of alpha electrons
    pub n_up: usize,
    /// Number of beta electrons  
    pub n_dn: usize,
    /// Number of electronic states to target
    pub n_states: usize,
    /// Target spin symmetry
    pub z_sym: i32,
}

/// Algorithm-specific parameters
#[derive(Debug, Clone)]
pub struct AlgorithmParameters {
    /// Variational screening threshold (epsilon_1) for HCI
    pub eps_var: f64,
    /// Screening threshold dividing deterministic and stochastic parts in PT
    pub eps_pt_dtm: f64,
    /// Target standard error for stochastic PT energy component
    pub target_uncertainty: f64,
    
    /// Maximum iterations for variational stage
    pub max_iter_var: usize,
    /// Maximum iterations for perturbative stage
    pub max_iter_pt: usize,
    
    /// Number of stochastic samples per batch
    pub n_samples_per_batch: usize,
    /// Maximum number of batches for stochastic PT
    pub n_batches: usize,
    /// Number of samples for cross terms
    pub n_cross_term_samples: usize,
    
    /// Algorithm variant selectors
    pub opp_algo: AlgorithmVariant,
    pub same_algo: AlgorithmVariant,
    /// Flag to select between semistochastic PT implementations
    pub use_new_semistoch: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum AlgorithmVariant {
    Method1 = 1,
    Method2 = 2,
}

/// Raw configuration struct for JSON deserialization
#[derive(Deserialize, Debug)]
struct GlobalConfigRaw {
    norb: i32,
    norb_core: i32,
    nup: i32,
    ndn: i32,
    #[serde(default = "default_z_sym")]
    z_sym: i32,
    #[serde(default = "default_n_states")]
    n_states: i32,
    eps_var: f64,
    eps_pt_dtm: f64,
    #[serde(default = "default_opp_algo")]
    opp_algo: i32,
    #[serde(default = "default_same_algo")]
    same_algo: i32,
    target_uncertainty: f64,
    n_samples_per_batch: i32,
    n_batches: i32,
    #[serde(default)]
    n_cross_term_samples: i32,
    #[serde(default)]
    use_new_semistoch: bool,
    #[serde(default = "default_max_iter_var")]
    max_iter_var: Option<i32>,
    #[serde(default = "default_max_iter_pt")]
    max_iter_pt: Option<i32>,
}

// Default value functions for serde
fn default_z_sym() -> i32 { 1 }
fn default_n_states() -> i32 { 1 }
fn default_opp_algo() -> i32 { 1 }
fn default_same_algo() -> i32 { 1 }
fn default_max_iter_var() -> Option<i32> { Some(50) }
fn default_max_iter_pt() -> Option<i32> { Some(100) }

impl GlobalConfig {
    /// Create configuration from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> RisqResult<(Self, AlgorithmParameters)> {
        let path = path.as_ref();
        
        let file = File::open(path)
            .map_err(|e| RisqError::io_error(path, e))?;
        
        let reader = BufReader::new(file);
        let raw: GlobalConfigRaw = serde_json::from_reader(reader)
            .map_err(|e| RisqError::json_error(path, e))?;
        
        let (config, params) = Self::from_raw(raw)?;
        Ok((config, params))
    }
    
    /// Convert and validate raw configuration
    fn from_raw(raw: GlobalConfigRaw) -> RisqResult<(Self, AlgorithmParameters)> {
        // Validate basic parameters
        risq_ensure!(raw.norb > 0, InvalidConfig { 
            message: format!("norb must be positive, got {}", raw.norb) 
        });
        
        risq_ensure!(raw.norb_core >= 0, InvalidConfig { 
            message: format!("norb_core must be non-negative, got {}", raw.norb_core) 
        });
        
        risq_ensure!(raw.nup >= 0 && raw.ndn >= 0, InvalidConfig { 
            message: format!("electron counts must be non-negative, got nup={}, ndn={}", raw.nup, raw.ndn) 
        });
        
        risq_ensure!(raw.norb_core < raw.norb, InvalidConfig { 
            message: format!("norb_core ({}) must be less than norb ({})", raw.norb_core, raw.norb) 
        });
        
        // Validate electron count vs orbital count
        let n_elec_total = raw.nup + raw.ndn;
        let n_core_elec = 2 * raw.norb_core;
        let n_elec_active = n_elec_total - n_core_elec;
        let n_orb_active = raw.norb - raw.norb_core;
        
        risq_ensure!(n_elec_active >= 0, InvalidConfig { 
            message: format!("Not enough electrons ({}) for {} core orbitals (need {} electrons)", n_elec_total, raw.norb_core, n_core_elec) 
        });
        
        risq_ensure!(n_elec_active <= 2 * n_orb_active, InvalidConfig { 
            message: format!("Too many active electrons ({}) for {} active orbitals (max {} electrons)", n_elec_active, n_orb_active, 2 * n_orb_active) 
        });
        
        // Validate thresholds
        risq_ensure!(raw.eps_var > 0.0, InvalidConfig {
            message: format!("eps_var must be positive, got {}", raw.eps_var)
        });
        
        risq_ensure!(raw.eps_pt_dtm > 0.0, InvalidConfig {
            message: format!("eps_pt_dtm must be positive, got {}", raw.eps_pt_dtm)
        });
        
        risq_ensure!(raw.target_uncertainty > 0.0, InvalidConfig {
            message: format!("target_uncertainty must be positive, got {}", raw.target_uncertainty)
        });
        
        // Validate sampling parameters
        risq_ensure!(raw.n_samples_per_batch > 0, InvalidConfig {
            message: format!("n_samples_per_batch must be positive, got {}", raw.n_samples_per_batch)
        });
        
        risq_ensure!(raw.n_batches > 0, InvalidConfig {
            message: format!("n_batches must be positive, got {}", raw.n_batches)
        });
        
        // Create validated configuration
        let config = Self {
            n_orbs: raw.norb as usize,
            n_core: raw.norb_core as usize,
            n_up: raw.nup as usize,
            n_dn: raw.ndn as usize,
            n_states: raw.n_states as usize,
            z_sym: raw.z_sym,
        };
        
        let params = AlgorithmParameters {
            eps_var: raw.eps_var,
            eps_pt_dtm: raw.eps_pt_dtm,
            target_uncertainty: raw.target_uncertainty,
            max_iter_var: raw.max_iter_var.unwrap_or(50) as usize,
            max_iter_pt: raw.max_iter_pt.unwrap_or(100) as usize,
            n_samples_per_batch: raw.n_samples_per_batch as usize,
            n_batches: raw.n_batches as usize,
            n_cross_term_samples: raw.n_cross_term_samples as usize,
            opp_algo: AlgorithmVariant::from_i32(raw.opp_algo)?,
            same_algo: AlgorithmVariant::from_i32(raw.same_algo)?,
            use_new_semistoch: raw.use_new_semistoch,
        };
        
        Ok((config, params))
    }
    
    /// Get number of active orbitals
    pub fn n_orb_active(&self) -> usize {
        self.n_orbs - self.n_core
    }
    
    /// Get number of active electrons
    pub fn n_elec_active(&self) -> usize {
        (self.n_up + self.n_dn) - (2 * self.n_core)
    }
    
    /// Get total number of electrons
    pub fn n_elec_total(&self) -> usize {
        self.n_up + self.n_dn
    }
    
    /// Get number of core electrons
    pub fn n_elec_core(&self) -> usize {
        2 * self.n_core
    }
}

impl AlgorithmVariant {
    fn from_i32(value: i32) -> RisqResult<Self> {
        match value {
            1 => Ok(Self::Method1),
            2 => Ok(Self::Method2),
            _ => risq_bail!(InvalidConfig {
                message: format!("Invalid algorithm variant: {}, must be 1 or 2", value)
            }),
        }
    }
}

impl From<AlgorithmVariant> for i32 {
    fn from(variant: AlgorithmVariant) -> Self {
        variant as i32
    }
}

impl Default for AlgorithmParameters {
    fn default() -> Self {
        Self {
            eps_var: 1e-4,
            eps_pt_dtm: 1e-6,
            target_uncertainty: 1e-4,
            max_iter_var: 50,
            max_iter_pt: 100,
            n_samples_per_batch: 1000,
            n_batches: 100,
            n_cross_term_samples: 0,
            opp_algo: AlgorithmVariant::Method1,
            same_algo: AlgorithmVariant::Method1,
            use_new_semistoch: true,
        }
    }
}