#![crate_name = "risq"]
#![crate_type = "bin"]
#![doc(html_root_url = "https://aaholmes.github.io/risq/")]

//! # Rust Implementation of Semistochastic Quantum chemistry (RISQ)
//!
//! This crate provides a command-line application for performing electronic structure
//! calculations using methods like Heat-bath Configuration Interaction (HCI) and
//! Semistochastic HCI (SHCI), which combines HCI with Epstein-Nesbet perturbation theory.
//!
//! ## Architecture Notes
//!
//! *   **Binary Crate:** This is structured as a binary (`bin`) crate, meaning it compiles
//!     to an executable (`risq`) rather than a reusable library.
//! *   **Fixed Workflow:** The `main` function executes a predetermined sequence:
//!     1.  Setup (Read input, integrals, initialize structures)
//!     2.  Variational HCI stage
//!     3.  Perturbative correction stage (Epstein-Nesbet PT2)
//! *   **Global State & Hardcoded Files:** Core data structures (`GLOBAL` parameters,
//!     `HAM` Hamiltonian, `EXCITE_GEN` excitation generator) are initialized once using
//!     `lazy_static!` and read from hardcoded filenames (`in.json`, `FCIDUMP`) in the
//!     current working directory. This simplifies the current execution flow but makes
//!     the application inflexible for running different calculations sequentially or
//!     using it as a library. A refactor to pass configuration explicitly would be needed
//!     for greater flexibility.
#[macro_use]
extern crate lazy_static;
extern crate alloc;
use std::time::Instant;
mod ham;
use ham::read_ints::read_ints;
use ham::Ham;
mod excite;
use excite::init::{init_excite_generator, ExciteGenerator};
pub mod wf;
use wf::{init_var_wf, VarWf};
mod pt;
mod rng;
mod semistoch;
mod stoch;
mod utils;
mod var;
// mod projector;

use crate::pt::perturbative;
use crate::var::variational;
use utils::read_input::{read_input, Global};

fn main() {
    let start: Instant = Instant::now();

    println!(" //==================================================================\\\\");
    println!("//   Rust Implementation of Semistochastic Quantum chemistry (RISQ)   \\\\");
    println!("\\\\                        Adam A Holmes, 2021                         //");
    println!(" \\\\==================================================================//");

    // --- Setup Stage ---
    println!("\n\n=====\nSetup\n=====\n");
    let start_setup: Instant = Instant::now();

    // Read global parameters from hardcoded 'in.json' using lazy_static.
    // Panics if 'in.json' is not found or invalid.
    println!("Reading input file (in.json)");
    lazy_static! {
        static ref GLOBAL: Global = read_input("in.json").unwrap();
    }

    // Read integrals from hardcoded 'FCIDUMP' using lazy_static.
    // Depends on GLOBAL being initialized first. Panics if 'FCIDUMP' is not found or invalid.
    println!("Reading integrals (FCIDUMP)");
    lazy_static! {
        static ref HAM: Ham = read_ints(&GLOBAL, "FCIDUMP");
    }

    // Initialize the excitation generator using lazy_static.
    // Depends on GLOBAL and HAM being initialized.
    println!("Initializing excitation generator");
    lazy_static! {
        static ref EXCITE_GEN: ExciteGenerator = init_excite_generator(&GLOBAL, &HAM);
    }

    // Initialize the variational wavefunction structure.
    println!("Initializing wavefunction");
    let mut wf: VarWf = init_var_wf(&GLOBAL, &HAM, &EXCITE_GEN);
    wf.print(); // Print initial state (likely just the reference determinant)
    println!("Time for setup: {:?}", start_setup.elapsed());

    // --- Variational Stage (HCI) ---
    println!("\n\n=================\nVariational stage\n=================\n");
    let start_var: Instant = Instant::now();
    // Perform the Heat-bath Configuration Interaction calculation.
    // This iteratively selects important determinants and solves the CI eigenvalue problem.
    // Modifies `wf` in place.
    variational(&GLOBAL, &HAM, &EXCITE_GEN, &mut wf);
    println!("Time for variational stage: {:?}", start_var.elapsed());

    // --- Perturbative Stage (Epstein-Nesbet PT2) ---
    println!("\n\n==================\nPerturbative stage\n==================\n");
    let start_enpt2: Instant = Instant::now();
    // Calculate the second-order Epstein-Nesbet perturbative correction
    // using the variational wavefunction `wf` as the reference.
    perturbative(&GLOBAL, &HAM, &EXCITE_GEN, &wf.wf);
    println!("Time for perturbative stage: {:?}", start_enpt2.elapsed());

    println!("Total Time: {:?}", start.elapsed());
}
