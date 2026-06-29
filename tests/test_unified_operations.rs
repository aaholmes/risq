//! Test for unified matrix-vector operations
//!
//! This test demonstrates how the new operations module provides a unified
//! interface for all the approx_matmul_* functions.

use risq::context::RisqContext;
use risq::error::RisqResult;
use risq::wf::operations::{MatVecOperations, MatVecConfig, MatVecResult, SinglesStrategy, DoublesStrategy};
use risq::temporary_wrappers::init_var_wf;
use std::path::Path;

#[test]
fn test_unified_variational_operations() -> RisqResult<()> {
    // Use the Beryllium system for testing
    let config_path = Path::new("examples/be/in.json");
    let fcidump_path = Path::new("examples/be/FCIDUMP");
    
    // Skip test if files don't exist
    if !config_path.exists() || !fcidump_path.exists() {
        println!("Skipping test - example files not found");
        return Ok(());
    }
    
    let context = RisqContext::from_files(config_path, fcidump_path)?;
    let wf = init_var_wf(&context)?;
    
    // Test the unified variational interface
    let input_coeffs: Vec<f64> = wf.wf.dets.iter().map(|det| det.coeff).collect();
    let eps = 1e-8;
    
    // Use the new unified interface
    let result = MatVecOperations::apply_variational(
        &wf.wf,
        &input_coeffs,
        &context.hamiltonian,
        &context.excitation_generator,
        eps,
    );
    
    // Verify we get a reasonable result
    assert_eq!(result.len(), wf.wf.n);
    assert!(result.iter().any(|&x| x.abs() > 1e-10), "Should have non-zero matrix elements");
    
    println!("✅ Unified variational operations test passed");
    println!("   Wavefunction size: {}", wf.wf.n);
    println!("   Result vector length: {}", result.len());
    println!("   Max coefficient: {:.6e}", result.iter().fold(0.0f64, |a, &b| a.max(b.abs())));
    
    Ok(())
}

#[test]
fn test_unified_external_operations() -> RisqResult<()> {
    // Use the Beryllium system for testing
    let config_path = Path::new("examples/be/in.json");
    let fcidump_path = Path::new("examples/be/FCIDUMP");
    
    // Skip test if files don't exist
    if !config_path.exists() || !fcidump_path.exists() {
        println!("Skipping test - example files not found");
        return Ok(());
    }
    
    let context = RisqContext::from_files(config_path, fcidump_path)?;
    let wf = init_var_wf(&context)?;
    
    let eps = 1e-6;
    
    // Test different external strategies using the unified interface
    let strategies = vec![
        SinglesStrategy::Include,
        SinglesStrategy::Skip,
        SinglesStrategy::Semistochastic,
        SinglesStrategy::None,
    ];
    
    for strategy in strategies {
        println!("Testing strategy: {:?}", strategy);
        
        let result = MatVecOperations::apply_external(
            &wf.wf,
            &context.hamiltonian,
            &context.excitation_generator,
            eps,
            strategy,
        );
        
        match result {
            MatVecResult::ExternalWithSampler(new_wf, _sampler) => {
                println!("  Got ExternalWithSampler result");
                println!("  New wavefunction size: {}", new_wf.n);
                assert!(new_wf.n > 0, "Should generate some external determinants");
            }
            MatVecResult::ExternalWithElements(new_wf, elements) => {
                println!("  Got ExternalWithElements result");
                println!("  New wavefunction size: {}", new_wf.n);
                println!("  Elements vector length: {}", elements.len());
                assert_eq!(new_wf.n, elements.len(), "Wavefunction and elements should match");
            }
            _ => panic!("Expected external result"),
        }
    }
    
    println!("✅ Unified external operations test passed");
    Ok(())
}

#[test]
fn test_unified_config_based_operations() -> RisqResult<()> {
    // Use the Beryllium system for testing
    let config_path = Path::new("examples/be/in.json");
    let fcidump_path = Path::new("examples/be/FCIDUMP");
    
    // Skip test if files don't exist
    if !config_path.exists() || !fcidump_path.exists() {
        println!("Skipping test - example files not found");
        return Ok(());
    }
    
    let context = RisqContext::from_files(config_path, fcidump_path)?;
    let wf = init_var_wf(&context)?;
    
    // Test using the full configuration interface
    let config = MatVecConfig {
        eps: 1e-6,
        singles_strategy: SinglesStrategy::Include,
        doubles_strategy: DoublesStrategy::Separate,
        compute_diagonals: false,
        input_coeffs: None,
    };
    
    let result = MatVecOperations::apply_hamiltonian(
        &wf.wf,
        &context.hamiltonian,
        &context.excitation_generator,
        &config,
    );
    
    match result {
        MatVecResult::ExternalWithSampler(new_wf, _) => {
            println!("✅ Configuration-based operation successful");
            println!("   Strategy: {:?} + {:?}", config.singles_strategy, config.doubles_strategy);
            println!("   Generated {} external determinants", new_wf.n);
            assert!(new_wf.n > 0);
        }
        _ => panic!("Expected external sampler result"),
    }
    
    Ok(())
}

#[test]
fn test_consolidation_consistency() -> RisqResult<()> {
    // Test that the new unified interface produces the same results as the old functions
    let config_path = Path::new("examples/be/in.json");
    let fcidump_path = Path::new("examples/be/FCIDUMP");
    
    // Skip test if files don't exist
    if !config_path.exists() || !fcidump_path.exists() {
        println!("Skipping test - example files not found");
        return Ok(());
    }
    
    let context = RisqContext::from_files(config_path, fcidump_path)?;
    let wf = init_var_wf(&context)?;
    
    let input_coeffs: Vec<f64> = wf.wf.dets.iter().map(|det| det.coeff).collect();
    let eps = 1e-8;
    
    // Compare new unified interface with old direct call
    let new_result = MatVecOperations::apply_variational(
        &wf.wf,
        &input_coeffs,
        &context.hamiltonian,
        &context.excitation_generator,
        eps,
    );
    
    let old_result = wf.wf.approx_matmul_variational(
        &input_coeffs,
        &context.hamiltonian,
        &context.excitation_generator,
        eps,
    );
    
    // Results should be identical
    assert_eq!(new_result.len(), old_result.len(), "Result lengths should match");
    
    for (i, (&new_val, &old_val)) in new_result.iter().zip(old_result.iter()).enumerate() {
        let diff = (new_val - old_val).abs();
        assert!(diff < 1e-12, "Values should be identical at index {}: {} vs {} (diff: {})", i, new_val, old_val, diff);
    }
    
    println!("✅ Consolidation consistency test passed");
    println!("   Verified {} coefficients match exactly", new_result.len());
    
    Ok(())
}