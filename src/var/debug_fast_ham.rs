//! Debug tool to compare fast vs original Hamiltonian construction on small systems

use crate::var::fast_ham::{VariationalSpace, gen_sparse_ham_fast_lookup};
use crate::var::ham_gen::gen_sparse_ham_fast;
use crate::wf::det::Config;
use crate::wf::{VarWf, Wf};
use crate::ham::Ham;
use crate::utils::read_input::Global;
use crate::excite::init::ExciteGenerator;
use crate::utils::bits::ibset;
use std::collections::{HashMap, HashSet};

/// Create a simple 4-electron, 4-orbital test system
pub fn create_test_system() -> (Ham, VarWf, ExciteGenerator, Global) {
    use crate::config::GlobalConfig;
    use crate::ham::read_ints::read_ints;
    use std::io::Write;
    use std::fs::File;
    
    // Create a minimal FCIDUMP file for 4 electrons in 4 orbitals
    let fcidump_content = "&FCI NORB=4,NELEC=4,MS2=0,\n  ORBSYM=1,1,1,1,\n  ISYM=1,\n&END\n0.500000000000000E+00   1   1   1   1\n0.200000000000000E+00   1   1   2   2\n0.200000000000000E+00   2   2   1   1\n0.200000000000000E+00   2   2   2   2\n0.100000000000000E+00   1   2   1   2\n0.100000000000000E+00   1   2   2   1\n0.100000000000000E+00   2   1   1   2\n0.100000000000000E+00   2   1   2   1\n0.200000000000000E+00   3   3   3   3\n0.100000000000000E+00   3   3   4   4\n0.100000000000000E+00   4   4   3   3\n0.200000000000000E+00   4   4   4   4\n0.050000000000000E+00   3   4   3   4\n0.050000000000000E+00   3   4   4   3\n0.050000000000000E+00   4   3   3   4\n0.050000000000000E+00   4   3   4   3\n-1.000000000000000E+00   1   1   0   0\n-0.800000000000000E+00   2   2   0   0\n-0.600000000000000E+00   3   3   0   0\n-0.400000000000000E+00   4   4   0   0\n0.100000000000000E+00   1   2   0   0\n0.100000000000000E+00   2   1   0   0\n0.050000000000000E+00   3   4   0   0\n0.050000000000000E+00   4   3   0   0\n0.000000000000000E+00   0   0   0   0\n";
    
    let temp_path = "/tmp/debug_fcidump.txt";
    let mut file = File::create(temp_path).expect("Failed to create temp FCIDUMP file");
    file.write_all(fcidump_content.as_bytes()).expect("Failed to write FCIDUMP");
    
    let config = GlobalConfig {
        n_orbs: 4,
        n_core: 0,
        n_up: 2,
        n_dn: 2,
        n_states: 1,
        z_sym: 1,
    };
    
    let ham = read_ints(&config, temp_path).expect("Failed to read FCIDUMP");
    
    // Create a small set of determinants to test
    let determinants = vec![
        create_config(&[0, 1], &[0, 1]),  // HF reference: |01⟩|01⟩
        create_config(&[0, 2], &[0, 1]),  // Single excitation: α(1→2)
        create_config(&[0, 1], &[0, 2]),  // Single excitation: β(1→2)
        create_config(&[0, 2], &[0, 2]),  // Double excitation: both α(1→2), β(1→2)
        create_config(&[1, 2], &[0, 1]),  // Single excitation: α(0→2)
        create_config(&[0, 1], &[1, 2]),  // Single excitation: β(0→2)
    ];
    
    // Create wavefunction
    let mut dets = Vec::new();
    let mut inds = HashMap::new();
    for (i, config) in determinants.iter().enumerate() {
        // Compute diagonal element for this determinant
        let diagonal = ham.ham_diag(config);
        dets.push(crate::wf::det::Det {
            config: *config,
            coeff: if i == 0 { 1.0 } else { 0.0 },
            diag: Some(diagonal),
        });
        inds.insert(*config, i);
    }
    
    let wf = Wf {
        n: determinants.len(),
        dets,
        inds,
        energy: -1.0,
    };
    
    let mut var_wf = VarWf::default();
    var_wf.wf = wf;
    var_wf.new_sparse_ham();
    
    // Create minimal global and excite_gen
    let global = Global {
        norb: 4,
        norb_core: 0,
        nup: 2,
        ndn: 2,
        z_sym: 1,
        n_states: 1,
        eps_var: 1e-4,
        eps_pt_dtm: 1e-6,
        opp_algo: 1,
        same_algo: 1,
        target_uncertainty: 1e-5,
        n_samples_per_batch: 1000,
        n_batches: 100,
        n_cross_term_samples: 1000,
        use_new_semistoch: true,
    };
    let excite_gen = ExciteGenerator::default();
    
    (ham, var_wf, excite_gen, global)
}

fn create_config(up_orbs: &[i32], dn_orbs: &[i32]) -> Config {
    let mut up = 0u128;
    let mut dn = 0u128;
    
    for &orb in up_orbs {
        up = ibset(up, orb);
    }
    for &orb in dn_orbs {
        dn = ibset(dn, orb);
    }
    
    Config { up, dn }
}

/// Compare Hamiltonian matrices from both algorithms
pub fn debug_hamiltonian_differences() {
    println!("=== Debugging Fast vs Original Hamiltonian Construction ===");
    
    let (ham, mut var_wf, excite_gen, global) = create_test_system();
    
    println!("Test system: {} determinants", var_wf.wf.n);
    for (i, det) in var_wf.wf.dets.iter().enumerate() {
        println!("  Det {}: α={:04b} β={:04b}", i, det.config.up as u8, det.config.dn as u8);
    }
    
    // Create two separate wavefunction instances
    let (_, mut wf_original, _, _) = create_test_system();
    let (_, mut wf_fast, _, _) = create_test_system();
    
    // Run original algorithm
    println!("\n--- Running Original Algorithm ---");
    gen_sparse_ham_fast(&global, &mut wf_original, &ham, &excite_gen);
    
    let original_elements = extract_matrix_elements(&wf_original);
    println!("Original algorithm found {} off-diagonal elements", original_elements.len());
    
    // Run fast algorithm
    println!("\n--- Running Fast Algorithm ---");
    gen_sparse_ham_fast_lookup(&global, &mut wf_fast, &ham);
    
    let fast_elements = extract_matrix_elements(&wf_fast);
    println!("Fast algorithm found {} off-diagonal elements", fast_elements.len());
    
    // Compare results
    println!("\n--- Comparing Results ---");
    compare_matrix_elements(&original_elements, &fast_elements, &wf_original);
    
    // Print full matrices for detailed analysis
    println!("\n--- Original Algorithm Matrix Elements ---");
    print_matrix_elements(&original_elements, &wf_original);
    
    println!("\n--- Fast Algorithm Matrix Elements ---");
    print_matrix_elements(&fast_elements, &wf_fast);
}

fn extract_matrix_elements(wf: &VarWf) -> Vec<(usize, usize, f64)> {
    let mut elements = Vec::new();
    
    for i in 0..wf.sparse_ham.n {
        for &(j, val) in &wf.sparse_ham.off_diag[i] {
            elements.push((i, j, val));
        }
    }
    
    elements.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    elements
}

fn compare_matrix_elements(
    original: &[(usize, usize, f64)], 
    fast: &[(usize, usize, f64)],
    wf: &VarWf
) {
    let original_set: HashSet<(usize, usize)> = original.iter().map(|(i, j, _)| (*i, *j)).collect();
    let fast_set: HashSet<(usize, usize)> = fast.iter().map(|(i, j, _)| (*i, *j)).collect();
    
    let missing_in_fast: Vec<_> = original_set.difference(&fast_set).collect();
    let extra_in_fast: Vec<_> = fast_set.difference(&original_set).collect();
    
    println!("Missing from fast algorithm: {} elements", missing_in_fast.len());
    for &(i, j) in missing_in_fast.iter().take(10) {
        let det_i = &wf.wf.dets[*i].config;
        let det_j = &wf.wf.dets[*j].config;
        let original_val = original.iter().find(|(ii, jj, _)| ii == i && jj == j).map(|(_, _, v)| *v).unwrap_or(0.0);
        println!("  ({}, {}): {:04b}|{:04b} <-> {:04b}|{:04b} = {:.6}", 
            i, j, 
            det_i.up as u8, det_i.dn as u8, 
            det_j.up as u8, det_j.dn as u8,
            original_val
        );
        
        // Analyze excitation type
        let excitation_type = analyze_excitation_type(det_i, det_j);
        println!("      Excitation type: {}", excitation_type);
    }
    
    println!("Extra in fast algorithm: {} elements", extra_in_fast.len());
    for &(i, j) in extra_in_fast.iter().take(10) {
        let det_i = &wf.wf.dets[*i].config;
        let det_j = &wf.wf.dets[*j].config;
        let fast_val = fast.iter().find(|(ii, jj, _)| ii == i && jj == j).map(|(_, _, v)| *v).unwrap_or(0.0);
        println!("  ({}, {}): {:04b}|{:04b} <-> {:04b}|{:04b} = {:.6}", 
            i, j, 
            det_i.up as u8, det_i.dn as u8, 
            det_j.up as u8, det_j.dn as u8,
            fast_val
        );
        
        let excitation_type = analyze_excitation_type(det_i, det_j);
        println!("      Excitation type: {}", excitation_type);
    }
    
    // Check for value differences in common elements
    let mut value_differences = 0;
    for (i, j, original_val) in original {
        if let Some((_, _, fast_val)) = fast.iter().find(|(ii, jj, _)| ii == i && jj == j) {
            if (original_val - fast_val).abs() > 1e-12 {
                value_differences += 1;
                println!("Value difference at ({}, {}): original={:.10}, fast={:.10}", 
                    i, j, original_val, fast_val);
            }
        }
    }
    
    if value_differences == 0 {
        println!("✅ All common matrix elements have identical values");
    } else {
        println!("❌ {} matrix elements have different values", value_differences);
    }
}

fn analyze_excitation_type(det1: &Config, det2: &Config) -> String {
    let up_diff = (det1.up ^ det2.up).count_ones();
    let dn_diff = (det1.dn ^ det2.dn).count_ones();
    
    match (up_diff, dn_diff) {
        (0, 0) => "identical".to_string(),
        (2, 0) => "alpha single".to_string(),
        (0, 2) => "beta single".to_string(),
        (4, 0) => "alpha double".to_string(),
        (0, 4) => "beta double".to_string(),
        (2, 2) => "opposite-spin double".to_string(),
        _ => format!("higher order ({} up, {} dn)", up_diff, dn_diff),
    }
}

fn print_matrix_elements(elements: &[(usize, usize, f64)], wf: &VarWf) {
    for (i, j, val) in elements {
        let det_i = &wf.wf.dets[*i].config;
        let det_j = &wf.wf.dets[*j].config;
        let excitation_type = analyze_excitation_type(det_i, det_j);
        println!("  H[{},{}] = {:8.6} | {:04b}|{:04b} <-> {:04b}|{:04b} | {}", 
            i, j, val,
            det_i.up as u8, det_i.dn as u8, 
            det_j.up as u8, det_j.dn as u8,
            excitation_type
        );
    }
}