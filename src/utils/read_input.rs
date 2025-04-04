//! # Input File Reading (`utils::read_input`)
//!
//! This module defines the `Global` struct which holds all calculation parameters
//! read from the input file (typically `in.json`), and the `read_input` function
//! which performs the file reading and deserialization using `serde_json`.

extern crate serde;
extern crate serde_json;

use serde::Deserialize;
use serde_json::{from_reader, Result};

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Holds global configuration parameters read from the input file (e.g., `in.json`).
/// Uses `serde` to deserialize from JSON.
#[derive(Deserialize, Debug)]
pub struct Global {
    /// Total number of spatial orbitals.
    pub norb: i32,
    /// Number of core orbitals to freeze (lowest energy based on h_ii).
    pub norb_core: i32,
    /// Number of alpha electrons.
    pub nup: i32,
    /// Number of beta electrons.
    pub ndn: i32,
    /// Target spin symmetry (+1 for singlet, -1 for triplet, etc. - affects initial epsilon).
    pub z_sym: i32,
    /// Number of electronic states to target
    pub n_states: i32,
    /// Variational screening threshold (epsilon_1) for HCI.
    pub eps_var: f64,
    /// Screening threshold dividing deterministic and stochastic parts in semistochastic PT.
    pub eps_pt_dtm: f64,
    /// Selector for opposite-spin algorithm variant (internal use).
    pub opp_algo: i32,
    /// Selector for same-spin algorithm variant (internal use).
    pub same_algo: i32,
    /// Target standard error for the stochastic PT energy component.
    pub target_uncertainty: f64,
    /// Number of stochastic samples to collect in each PT batch.
    pub n_samples_per_batch: i32,
    /// Maximum number of batches for stochastic PT (may terminate early if uncertainty target met).
    pub n_batches: i32,
    /// Number of samples for cross terms
    pub n_cross_term_samples: i32,
    /// Flag to select between different semistochastic PT implementations.
    pub use_new_semistoch: bool,
}

/// Reads calculation parameters from a JSON file into a `Global` struct.
///
/// # Arguments
/// * `path`: The path to the JSON input file (e.g., "in.json").
///
/// # Returns
/// A `serde_json::Result<Global>` which is `Ok(Global)` on successful parsing,
/// or `Err` if the file cannot be opened or the JSON is malformed/doesn't match
/// the `Global` struct definition.
///
/// Also prints the parsed parameters to standard output.
pub fn read_input<P: AsRef<Path>>(path: P) -> Result<Global> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let global: Global = from_reader(reader)?;

    println!("\nInput:");
    println!("  norb: {}", global.norb);
    println!("  norb_core: {}", global.norb_core);
    println!("  nup: {}", global.nup);
    println!("  ndn: {}", global.ndn);
    println!("  z_sym: {}", global.z_sym);
    println!("  n_states: {}", global.n_states);
    println!("  eps_var: {}", global.eps_var);
    println!("  eps_pt_dtm: {}", global.eps_pt_dtm);
    println!("  opp_algo: {}", global.opp_algo);
    println!("  same_algo: {}", global.same_algo);
    println!("  target_uncertainty: {}", global.target_uncertainty);
    println!("  n_samples_per_batch: {}", global.n_samples_per_batch);
    println!("  n_batches: {}", global.n_batches);
    println!("  n_cross_term_samples: {}", global.n_cross_term_samples);
    println!("  use_new_semistoch: {}", global.use_new_semistoch);
    println!("\n");

    // Removed commented-out validation checks. Consider re-adding if needed.

    Ok(global)
}
