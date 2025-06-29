//! Comprehensive Integration Test Suite - Scientific Validation
//!
//! This test suite validates that the refactored codebase produces scientifically
//! correct results by running complete HCI calculations on multiple molecular systems
//! and comparing against reference values from the working implementation.

use risq::context::RisqContext;
use risq::error::RisqResult;
use risq::temporary_wrappers::{init_var_wf, variational};
use std::path::Path;

/// Tolerance for energy comparison (1e-6 Hartree for robust testing)
const ENERGY_TOLERANCE: f64 = 1e-6;

/// Test case structure for molecular systems
struct TestCase {
    name: &'static str,
    config_path: &'static str,
    fcidump_path: &'static str,
    expected_var_energy: f64,
    expected_n_orbs: usize,
    expected_n_up: usize,
    expected_n_dn: usize,
    expected_n_core: usize,
}

/// Test cases covering diverse molecular systems
const TEST_CASES: &[TestCase] = &[
    TestCase {
        name: "Beryllium",
        config_path: "examples/be/in.json",
        fcidump_path: "examples/be/FCIDUMP",
        expected_var_energy: -14.6168425934,  // From working calculation
        expected_n_orbs: 14,
        expected_n_up: 2,
        expected_n_dn: 2,
        expected_n_core: 1,
    },
    TestCase {
        name: "He2",
        config_path: "examples/he2/in.json",
        fcidump_path: "examples/he2/FCIDUMP",
        expected_var_energy: -5.5565366249,  // From working calculation
        expected_n_orbs: 4,
        expected_n_up: 2,
        expected_n_dn: 2,
        expected_n_core: 0,
    },
    TestCase {
        name: "LiH",
        config_path: "examples/lih/in.json",
        fcidump_path: "examples/lih/FCIDUMP",
        expected_var_energy: -7.9152839074,  // From working calculation
        expected_n_orbs: 19,
        expected_n_up: 2,
        expected_n_dn: 2,
        expected_n_core: 1,
    },
    TestCase {
        name: "Li2",
        config_path: "examples/li2/in.json",
        fcidump_path: "examples/li2/FCIDUMP",
        expected_var_energy: -14.6198472544,  // From working calculation
        expected_n_orbs: 18,
        expected_n_up: 3,
        expected_n_dn: 3,
        expected_n_core: 2,
    },
    TestCase {
        name: "Be2",
        config_path: "examples/be2/in.json",
        fcidump_path: "examples/be2/FCIDUMP",
        expected_var_energy: -28.6608360148,  // From working calculation
        expected_n_orbs: 18,
        expected_n_up: 4,
        expected_n_dn: 4,
        expected_n_core: 2,
    },
    TestCase {
        name: "F2",
        config_path: "examples/f2/dz/in.json",
        fcidump_path: "examples/f2/dz/FCIDUMP",
        expected_var_energy: -199.0913503137,  // From working calculation
        expected_n_orbs: 28,
        expected_n_up: 9,
        expected_n_dn: 9,
        expected_n_core: 2,
    },
];

/// Generic test function for any molecular system
fn test_molecular_system(test_case: &TestCase) -> RisqResult<f64> {
    println!("🧪 Testing {} System", test_case.name);
    println!("{}", "=".repeat(50));
    
    // Get paths to test data files  
    let config_path = Path::new(test_case.config_path);
    let fcidump_path = Path::new(test_case.fcidump_path);
    
    // Verify test files exist
    assert!(config_path.exists(), "{} config file not found: {:?}", test_case.name, config_path);
    assert!(fcidump_path.exists(), "{} FCIDUMP file not found: {:?}", test_case.name, fcidump_path);
    
    println!("📁 Loading {} system files...", test_case.name);
    println!("   Config: {:?}", config_path);
    println!("   FCIDUMP: {:?}", fcidump_path);
    
    // Initialize calculation context
    let mut context = RisqContext::from_files(config_path, fcidump_path)?;
    
    println!("\n🔧 System Configuration:");
    context.print_summary();
    
    // Validate system parameters
    assert_eq!(context.config.n_orbs, test_case.expected_n_orbs, "Orbital count mismatch");
    assert_eq!(context.config.n_up, test_case.expected_n_up, "Alpha electron count mismatch");
    assert_eq!(context.config.n_dn, test_case.expected_n_dn, "Beta electron count mismatch");
    assert_eq!(context.config.n_core, test_case.expected_n_core, "Core orbital count mismatch");
    
    // Initialize wavefunction
    println!("\n🌊 Initializing wavefunction...");
    let mut wavefunction = init_var_wf(&context)?;
    println!("   Initial determinants: {}", wavefunction.wf.n);
    
    // Run variational HCI calculation
    println!("\n🎯 Running Variational HCI Calculation...");
    let var_energy = variational(&mut context, &mut wavefunction)?;
    
    println!("   Final determinants: {}", wavefunction.wf.n);
    println!("   Variational energy: {:.10} Hartree", var_energy);
    
    // Validate energy if reference is available (non-zero)
    if test_case.expected_var_energy != 0.0 {
        println!("\n✅ Validating Results Against Reference Values:");
        println!("   Tolerance: {:.2e} Hartree", ENERGY_TOLERANCE);
        
        let var_error = (var_energy - test_case.expected_var_energy).abs();
        println!("   Variational: {:.10} vs {:.10} (Δ = {:.2e})", 
                 var_energy, test_case.expected_var_energy, var_error);
        
        assert!(
            var_error < ENERGY_TOLERANCE,
            "Variational energy mismatch for {}: expected {:.10}, got {:.10}, error = {:.2e} > {:.2e}",
            test_case.name, test_case.expected_var_energy, var_energy, var_error, ENERGY_TOLERANCE
        );
        
        println!("   ✅ {} variational energy validated", test_case.name);
    } else {
        println!("\n📝 Recording energy for future validation:");
        println!("   {} variational energy: {:.10} Hartree", test_case.name, var_energy);
    }
    
    println!("\n🔬 {} calculation completed successfully!", test_case.name);
    Ok(var_energy)
}

#[test]
fn test_beryllium_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[0])?;
    println!("\n🎉 Beryllium test PASSED!");
    Ok(())
}

#[test]
fn test_he2_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[1])?;
    println!("\n🎉 He2 test PASSED!");
    Ok(())
}

#[test]
fn test_lih_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[2])?;
    println!("\n🎉 LiH test PASSED!");
    Ok(())
}

#[test]
fn test_li2_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[3])?;
    println!("\n🎉 Li2 test PASSED!");
    Ok(())
}

#[test]
fn test_be2_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[4])?;
    println!("\n🎉 Be2 test PASSED!");
    Ok(())
}

#[test]
fn test_f2_hci_energy() -> RisqResult<()> {
    test_molecular_system(&TEST_CASES[5])?;
    println!("\n🎉 F2 test PASSED!");
    Ok(())
}

#[test]
fn test_all_systems_suite() -> RisqResult<()> {
    println!("🚀 Running Comprehensive Integration Test Suite");
    println!("{}", "=".repeat(80));
    
    let mut results = Vec::new();
    
    // Test only the three systems with reference energies
    for (i, test_case) in TEST_CASES.iter().enumerate() {
        println!("\n[{}/{}] Testing {}", i+1, TEST_CASES.len(), test_case.name);
        let var_energy = test_molecular_system(test_case)?;
        results.push((test_case.name, var_energy));
    }
    
    // Summary report
    println!("\n🏆 COMPREHENSIVE TEST SUITE RESULTS");
    println!("{}", "=".repeat(80));
    println!("System        | Variational Energy (Hartree)  | Status");
    println!("{}", "-".repeat(80));
    
    for (name, energy) in &results {
        println!("{:12} | {:28.10} | ✅ PASS", name, energy);
    }
    
    println!("{}", "=".repeat(80));
    println!("🎉 ALL {} MOLECULAR SYSTEMS VALIDATED SUCCESSFULLY!", results.len());
    println!("🔬 Scientific correctness of refactored code CONFIRMED across diverse systems!");
    
    Ok(())
}

#[test]
fn test_system_properties_validation() -> RisqResult<()> {
    println!("🔍 Testing System Properties Validation");
    
    for test_case in TEST_CASES {
        let config_path = Path::new(test_case.config_path);
        let fcidump_path = Path::new(test_case.fcidump_path);
        
        let context = RisqContext::from_files(config_path, fcidump_path)?;
        
        // Verify system parameters match expected values
        assert_eq!(context.config.n_orbs, test_case.expected_n_orbs, 
                  "{}: orbital count mismatch", test_case.name);
        assert_eq!(context.config.n_up, test_case.expected_n_up, 
                  "{}: alpha electron count mismatch", test_case.name);
        assert_eq!(context.config.n_dn, test_case.expected_n_dn, 
                  "{}: beta electron count mismatch", test_case.name);
        assert_eq!(context.config.n_core, test_case.expected_n_core, 
                  "{}: core orbital count mismatch", test_case.name);
        
        println!("✅ {} system properties validated", test_case.name);
    }
    
    println!("🎉 All system properties validation PASSED!");
    Ok(())
}

#[test]
fn test_reproducibility_suite() -> RisqResult<()> {
    println!("🔄 Testing Calculation Reproducibility Across Systems");
    
    // Test reproducibility for first two systems (to keep test time reasonable)
    for test_case in &TEST_CASES[0..2] {
        println!("\n🔄 Testing {} reproducibility", test_case.name);
        
        let config_path = Path::new(test_case.config_path);
        let fcidump_path = Path::new(test_case.fcidump_path);
        
        // Run calculation twice
        let mut context1 = RisqContext::from_files(config_path, fcidump_path)?;
        let mut wf1 = init_var_wf(&context1)?;
        let energy1 = variational(&mut context1, &mut wf1)?;
        
        let mut context2 = RisqContext::from_files(config_path, fcidump_path)?;
        let mut wf2 = init_var_wf(&context2)?;
        let energy2 = variational(&mut context2, &mut wf2)?;
        
        // Results should be identical (deterministic)
        assert_eq!(energy1, energy2, "{}: calculations should be reproducible", test_case.name);
        assert_eq!(wf1.wf.n, wf2.wf.n, "{}: wavefunction sizes should be identical", test_case.name);
        
        println!("✅ {} calculations are reproducible", test_case.name);
    }
    
    println!("🎉 Reproducibility test PASSED!");
    Ok(())
}