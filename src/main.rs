//! # RISQ Binary
//!
//! Command-line interface for the RISQ quantum chemistry library.
//! Now refactored to use the library with proper error handling.

use risq::context::RisqContext;
use risq::error::{RisqError, RisqResult};
use risq::temporary_wrappers::{init_var_wf, variational, perturbative};

use std::env;
use std::time::Instant;

fn main() -> RisqResult<()> {
    let start = Instant::now();
    
    println!(" //==================================================================\\\\");
    println!("//   Rust Implementation of Semistochastic Quantum chemistry (RISQ)   \\\\");
    println!("\\\\                        Adam A Holmes, 2021                         //");
    println!(" \\\\==================================================================//");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (config_path, fcidump_path) = parse_args(&args)?;
    
    // Run calculation
    run_calculation(&config_path, &fcidump_path)?;
    
    println!("Total Time: {:?}", start.elapsed());
    Ok(())
}

fn parse_args(args: &[String]) -> RisqResult<(String, String)> {
    match args.len() {
        1 => {
            // Default behavior: look for files in current directory
            Ok(("in.json".to_string(), "FCIDUMP".to_string()))
        },
        3 => {
            // Custom file paths provided
            Ok((args[1].clone(), args[2].clone()))
        },
        _ => {
            Err(RisqError::invalid_config(
                "Usage: risq [config.json] [FCIDUMP]"
            ))
        }
    }
}

fn run_calculation(config_path: &str, fcidump_path: &str) -> RisqResult<()> {
    println!("\n\n=====\nSetup\n=====\n");
    let setup_start = Instant::now();

    // Initialize context (replaces all lazy_static globals)
    println!("Reading input file ({})", config_path);
    println!("Reading integrals ({})", fcidump_path);
    println!("Initializing calculation context");
    
    let mut context = RisqContext::from_files(config_path, fcidump_path)?;
    context.print_summary();
    
    // Initialize wavefunction
    println!("Initializing wavefunction");
    let mut wf = init_var_wf(&context)?;
    wf.print();
    
    println!("Time for setup: {:?}", setup_start.elapsed());

    // Variational stage
    println!("\n\n=================\nVariational stage\n=================\n");
    let var_start = Instant::now();
    
    let var_energy = variational(&mut context, &mut wf)?;
    
    println!("Time for variational stage: {:?}", var_start.elapsed());

    // Perturbative stage
    println!("\n\n==================\nPerturbative stage\n==================\n");
    let pt_start = Instant::now();
    
    let pt2_energy = perturbative(&context, &wf.wf)?;
    
    println!("Final energies:");
    println!("  Variational: {:.10}", var_energy);
    println!("  PT2:         {:.10}", pt2_energy);
    println!("  Total:       {:.10}", var_energy + pt2_energy);
    
    println!("Time for perturbative stage: {:?}", pt_start.elapsed());
    
    Ok(())
}
