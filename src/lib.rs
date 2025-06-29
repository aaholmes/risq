//! # Rust Implementation of Semistochastic Quantum chemistry (RISQ)
//!
//! A library for performing electronic structure calculations using advanced
//! configuration interaction methods including Heat-bath CI (HCI) and
//! Semistochastic HCI (SHCI).
//!
//! ## Features
//!
//! - Heat-bath Configuration Interaction (HCI)
//! - Semistochastic Epstein-Nesbet Perturbation Theory  
//! - Dynamic Semistochastic Full CI Quantum Monte Carlo (DS-FCIQMC)
//! - Efficient determinant representation using bitstrings
//! - Davidson eigenvalue solver with preconditioning
//! - Alias sampling for importance sampling
//!
//! ## Usage
//!
//! ```rust,no_run
//! use risq::context::RisqContext;
//! use risq::wf::init_var_wf;
//! use risq::var::variational;
//! use risq::pt::perturbative;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize calculation context
//! let mut context = RisqContext::from_files("in.json", "FCIDUMP")?;
//!
//! // Set up wavefunction
//! let mut wavefunction = init_var_wf(&context)?;
//!
//! // Run variational calculation
//! let var_energy = variational(&mut context, &mut wavefunction)?;
//!
//! // Run perturbative correction
//! let pt2_energy = perturbative(&context, &wavefunction.wf)?;
//!
//! println!("Total energy: {:.10}", var_energy + pt2_energy);
//! # Ok(())
//! # }
//! ```

#![doc(html_root_url = "https://aaholmes.github.io/risq/")]

// Re-export key modules for public API
pub mod context;
pub mod config;
pub mod error;

// Internal modules
pub mod excite;
pub mod ham;
pub mod pt;
pub mod rng;
pub mod semistoch;
pub mod stoch;
pub mod utils;
pub mod var;
pub mod wf;

// Temporary module for compatibility during refactoring
pub mod temporary_wrappers;

// Re-export common types
pub use context::RisqContext;
pub use error::{RisqError, RisqResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");