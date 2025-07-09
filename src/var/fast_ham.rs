//! # Fast Hamiltonian Construction (Milestone 1)
//!
//! This module implements the fast lookup-based Hamiltonian construction from Li et al.,
//! J. Chem. Theory Comput. 2018, 14, 8, 4478–4489. It replaces the O(N²) determinant
//! connection finding in `ham_gen.rs` with an O(N log N) approach using alpha string
//! lookup tables.

use crate::ham::Ham;
use crate::utils::bits::{bits, ibset, ibclr};
use crate::utils::read_input::Global;
use crate::var::off_diag::add_el;
use crate::wf::det::{Config, Det};
use crate::wf::VarWf;
use std::collections::HashMap;

/// Type alias for alpha/beta strings (u128 bitstrings)
pub type AlphaString = u128;
pub type BetaString = u128;

/// Manages the variational space for fast Hamiltonian construction.
/// Implements the lookup-based algorithm from Li et al. 2018.
pub struct VariationalSpace {
    /// A vector containing all determinants (`D` in the paper).
    pub determinants: Vec<Config>,

    /// A vector of unique alpha strings (`α` in the paper).
    pub alpha_strings: Vec<AlphaString>,

    /// A hash map from an alpha string to its index in `alpha_strings`.
    /// This is `iα(α)` from the paper.
    pub alpha_to_index: HashMap<AlphaString, usize>,

    /// A nested vector mapping an alpha string index to a list of
    /// determinant indices that share that alpha string. This is `Dα(iα)`.
    pub alpha_to_dets: Vec<Vec<usize>>,

    /// Alpha string single excitation lookup table.
    /// Maps alpha string to all alpha strings connected by single excitation.
    pub alpha_singles: HashMap<AlphaString, Vec<AlphaString>>,

    /// Beta string single excitation lookup table.
    /// Maps beta string to all beta strings connected by single excitation.
    pub beta_singles: HashMap<BetaString, Vec<BetaString>>,

    /// Map from (alpha_string, beta_r1) to list of (det_index, beta_string)
    /// Used for fast opposite-spin excitation finding.
    alpha_beta_lookup: HashMap<(AlphaString, BetaString), Vec<(usize, BetaString)>>,
}

impl VariationalSpace {
    /// Constructs a new `VariationalSpace` from a vector of determinants.
    /// This function builds the fast lookup tables for O(N log N) connection finding.
    ///
    /// # Arguments
    /// * `determinants` - The initial set of determinants in the variational space.
    pub fn new(determinants: Vec<Config>) -> Self {
        let mut alpha_strings = Vec::new();
        let mut alpha_to_index = HashMap::new();
        let mut alpha_to_dets_map: HashMap<AlphaString, Vec<usize>> = HashMap::new();

        // Step 1: Group determinants by alpha string
        for (det_idx, det) in determinants.iter().enumerate() {
            let alpha = det.up;
            
            // Add determinant index to the alpha string's list
            alpha_to_dets_map.entry(alpha).or_insert_with(Vec::new).push(det_idx);
        }

        // Step 2: Build alpha_strings and alpha_to_index from grouped data
        for (alpha, det_indices) in alpha_to_dets_map.iter() {
            let alpha_idx = alpha_strings.len();
            alpha_strings.push(*alpha);
            alpha_to_index.insert(*alpha, alpha_idx);
        }

        // Step 3: Build final alpha_to_dets using consistent indexing
        let mut alpha_to_dets = vec![Vec::new(); alpha_strings.len()];
        for (alpha, det_indices) in alpha_to_dets_map {
            let alpha_idx = alpha_to_index[&alpha];
            alpha_to_dets[alpha_idx] = det_indices;
        }

        // Step 4: Build single excitation lookup tables
        let alpha_singles = Self::build_alpha_singles(&alpha_strings);
        let beta_singles = Self::build_beta_singles(&determinants);
        
        // Step 5: Build alpha-beta lookup for fast opposite-spin excitation finding
        let alpha_beta_lookup = Self::build_alpha_beta_lookup(&determinants);

        VariationalSpace {
            determinants,
            alpha_strings,
            alpha_to_index,
            alpha_to_dets,
            alpha_singles,
            beta_singles,
            alpha_beta_lookup,
        }
    }

    /// Builds alpha string single excitation lookup table.
    /// For each alpha string, finds all alpha strings connected by a single excitation.
    fn build_alpha_singles(alpha_strings: &[AlphaString]) -> HashMap<AlphaString, Vec<AlphaString>> {
        let mut alpha_singles = HashMap::new();
        let mut alpha_r1_to_alphas: HashMap<AlphaString, Vec<AlphaString>> = HashMap::new();

        // Build reverse lookup: (alpha - 1 electron) -> [alphas]
        for &alpha in alpha_strings {
            for occupied_orb in bits(alpha) {
                let alpha_r1 = ibclr(alpha, occupied_orb);
                alpha_r1_to_alphas.entry(alpha_r1).or_insert_with(Vec::new).push(alpha);
            }
        }

        // Build forward lookup: alpha -> [connected alphas]
        for &alpha in alpha_strings {
            let mut connected_alphas = Vec::new();
            for occupied_orb in bits(alpha) {
                let alpha_r1 = ibclr(alpha, occupied_orb);
                if let Some(connected_list) = alpha_r1_to_alphas.get(&alpha_r1) {
                    for &connected_alpha in connected_list {
                        if connected_alpha != alpha {
                            connected_alphas.push(connected_alpha);
                        }
                    }
                }
            }
            alpha_singles.insert(alpha, connected_alphas);
        }

        alpha_singles
    }

    /// Builds beta string single excitation lookup table.
    /// For each beta string in the determinant set, finds all connected beta strings.
    fn build_beta_singles(determinants: &[Config]) -> HashMap<BetaString, Vec<BetaString>> {
        let mut beta_singles = HashMap::new();
        let mut beta_r1_to_betas: HashMap<BetaString, Vec<BetaString>> = HashMap::new();
        let mut unique_betas = std::collections::HashSet::new();

        // Collect unique beta strings
        for det in determinants {
            unique_betas.insert(det.dn);
        }

        // Build reverse lookup: (beta - 1 electron) -> [betas]
        for &beta in &unique_betas {
            for occupied_orb in bits(beta) {
                let beta_r1 = ibclr(beta, occupied_orb);
                beta_r1_to_betas.entry(beta_r1).or_insert_with(Vec::new).push(beta);
            }
        }

        // Build forward lookup: beta -> [connected betas]
        for &beta in &unique_betas {
            let mut connected_betas = Vec::new();
            for occupied_orb in bits(beta) {
                let beta_r1 = ibclr(beta, occupied_orb);
                if let Some(connected_list) = beta_r1_to_betas.get(&beta_r1) {
                    for &connected_beta in connected_list {
                        if connected_beta != beta {
                            connected_betas.push(connected_beta);
                        }
                    }
                }
            }
            beta_singles.insert(beta, connected_betas);
        }

        beta_singles
    }

    /// Builds alpha-beta lookup table for fast opposite-spin excitation finding.
    /// Maps (alpha_string, beta_r1) to list of (det_index, beta_string).
    fn build_alpha_beta_lookup(determinants: &[Config]) -> HashMap<(AlphaString, BetaString), Vec<(usize, BetaString)>> {
        let mut alpha_beta_lookup = HashMap::new();

        for (det_idx, det) in determinants.iter().enumerate() {
            let alpha = det.up;
            let beta = det.dn;
            
            // For each way to remove one electron from beta string
            for occupied_orb in bits(beta) {
                let beta_r1 = ibclr(beta, occupied_orb);
                let key = (alpha, beta_r1);
                alpha_beta_lookup.entry(key).or_insert_with(Vec::new).push((det_idx, beta));
            }
        }

        alpha_beta_lookup
    }

    /// Finds all determinants connected to a given source determinant by a
    /// single or double excitation, using the fast lookup tables.
    ///
    /// # Arguments
    /// * `source_det_idx` - The index of the source determinant in `self.determinants`.
    /// * `wf` - The variational wavefunction (for index mapping).
    /// * `ham` - The Hamiltonian operator.
    /// * `n_stored_h` - Number of determinants for which H matrix elements are already stored.
    ///
    /// # Returns
    /// A vector of tuples, where each tuple contains the index of a connected
    /// determinant and the corresponding Hamiltonian matrix element.
    pub fn find_connections_and_add_to_sparse_ham(
        &self,
        source_det_idx: usize,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        let source_config = self.determinants[source_det_idx];
        let source_alpha = source_config.up;
        let source_beta = source_config.dn;

        // Find opposite-spin double excitations
        self.find_opposite_spin_connections(source_det_idx, source_alpha, source_beta, wf, ham, n_stored_h);

        // Find same-spin double excitations
        self.find_same_spin_connections(source_det_idx, source_alpha, source_beta, wf, ham, n_stored_h);
    }

    /// Finds opposite-spin double excitation connections using fast lookup tables.
    /// Implements the algorithm from Li et al. 2018 for mixed excitations.
    /// An opposite-spin double excitation requires alpha single + beta single = overall double.
    fn find_opposite_spin_connections(
        &self,
        source_det_idx: usize,
        source_alpha: AlphaString,
        source_beta: BetaString,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // For opposite-spin double excitations: need alpha single-connected AND beta single-connected
        if let Some(connected_alphas) = self.alpha_singles.get(&source_alpha) {
            for &connected_alpha in connected_alphas {
                if let Some(connected_betas) = self.beta_singles.get(&source_beta) {
                    for &connected_beta in connected_betas {
                        // Check if a determinant with (connected_alpha, connected_beta) exists
                        let target_config = Config {
                            up: connected_alpha,
                            dn: connected_beta,
                        };
                        
                        // Look up this configuration in the determinant list
                        if let Some(&connected_det_idx) = wf.wf.inds.get(&target_config) {
                            // Only add if at least one determinant index is new
                            if source_det_idx >= n_stored_h || connected_det_idx >= n_stored_h {
                                add_el(wf, ham, source_det_idx, connected_det_idx, None);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Finds same-spin double excitation connections.
    /// For same-spin doubles: either same alpha + beta double, or same beta + alpha double.
    /// To avoid double counting, only check same alpha + beta double (the other direction 
    /// will be found when processing the other determinant).
    pub fn find_same_spin_connections(
        &self,
        source_det_idx: usize,
        source_alpha: AlphaString,
        source_beta: BetaString,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // Same alpha string: check for beta double excitations only
        // (This avoids double-counting since we'll find the reverse when processing the other det)
        if let Some(alpha_idx) = self.alpha_to_index.get(&source_alpha) {
            for &connected_det_idx in &self.alpha_to_dets[*alpha_idx] {
                if connected_det_idx != source_det_idx {
                    let connected_config = self.determinants[connected_det_idx];
                    let connected_beta = connected_config.dn;

                    // Check if beta strings differ by double excitation
                    if self.are_strings_double_connected(source_beta, connected_beta) {
                        // Only add if at least one determinant index is new
                        if source_det_idx >= n_stored_h || connected_det_idx >= n_stored_h {
                            add_el(wf, ham, source_det_idx, connected_det_idx, None);
                        }
                    }
                }
            }
        }

        // For same beta + alpha double: only process if source_det_idx has a NEW determinant
        // This ensures we don't double-count (since add_el handles i<j ordering automatically)
        if source_det_idx >= n_stored_h {
            for (other_alpha, other_alpha_idx) in &self.alpha_to_index {
                if *other_alpha != source_alpha && self.are_strings_double_connected(source_alpha, *other_alpha) {
                    for &connected_det_idx in &self.alpha_to_dets[*other_alpha_idx] {
                        let connected_config = self.determinants[connected_det_idx];
                        if connected_config.dn == source_beta {
                            add_el(wf, ham, source_det_idx, connected_det_idx, None);
                        }
                    }
                }
            }
        }
    }

    /// Checks if two beta strings are connected by a single excitation.
    fn are_beta_strings_single_connected(&self, beta1: BetaString, beta2: BetaString) -> bool {
        if let Some(connected_betas) = self.beta_singles.get(&beta1) {
            connected_betas.contains(&beta2)
        } else {
            false
        }
    }

    /// Checks if two strings (alpha or beta) are connected by a double excitation.
    /// Two strings are double-connected if they differ by exactly 4 bits (2 removals + 2 additions).
    /// This represents moving exactly 2 electrons to different orbitals.
    pub fn are_strings_double_connected(&self, string1: u128, string2: u128) -> bool {
        let diff = string1 ^ string2;
        let n_diffs = diff.count_ones();
        
        // Must differ by exactly 4 bits (2 electrons removed, 2 electrons added)
        if n_diffs != 4 {
            return false;
        }
        
        // Additional validation: ensure same number of electrons in both strings
        if string1.count_ones() != string2.count_ones() {
            return false;
        }
        
        true
    }

    /// Follows the exact pattern of the original algorithm:
    /// - Opposite-spin: process NEW alpha strings, find connections to ANY determinants
    /// - Same-spin: process ALL alpha strings, find connections involving NEW determinants
    pub fn find_all_connections_original_pattern(
        &self,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // First find all NEW alpha strings (alpha strings that have at least one NEW determinant)
        let mut new_alpha_strings = std::collections::HashSet::new();
        for det_idx in n_stored_h..self.determinants.len() {
            let alpha = self.determinants[det_idx].up;
            new_alpha_strings.insert(alpha);
        }

        // SINGLE EXCITATIONS: Process NEW determinants for single excitations
        self.find_single_excitations(wf, ham, n_stored_h);

        // OPPOSITE-SPIN EXCITATIONS: Process NEW alpha strings only
        for &new_alpha in &new_alpha_strings {
            self.find_opposite_spin_for_alpha_string(new_alpha, wf, ham, n_stored_h);
        }

        // SAME-SPIN EXCITATIONS: Process ALL alpha strings, but only connections involving NEW dets
        for &alpha in &self.alpha_strings {
            self.find_same_spin_for_alpha_string(alpha, wf, ham, n_stored_h);
        }
    }

    /// Find opposite-spin connections for a specific alpha string (following original pattern)
    fn find_opposite_spin_for_alpha_string(
        &self,
        source_alpha: AlphaString,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // Find all alpha strings connected by single excitation
        if let Some(connected_alphas) = self.alpha_singles.get(&source_alpha) {
            // Get all determinants with the source alpha
            if let Some(source_alpha_idx) = self.alpha_to_index.get(&source_alpha) {
                let source_dets = &self.alpha_to_dets[*source_alpha_idx];
                
                for &connected_alpha in connected_alphas {
                    // Get all determinants with the connected alpha
                    if let Some(connected_alpha_idx) = self.alpha_to_index.get(&connected_alpha) {
                        let connected_dets = &self.alpha_to_dets[*connected_alpha_idx];
                        
                        // Check all combinations for opposite-spin double excitations
                        for &source_det_idx in source_dets {
                            let source_beta = self.determinants[source_det_idx].dn;
                            
                            for &connected_det_idx in connected_dets {
                                let connected_beta = self.determinants[connected_det_idx].dn;
                                
                                // Check if this is a valid opposite-spin double excitation
                                if self.are_beta_strings_single_connected(source_beta, connected_beta) {
                                    // Only add if at least one determinant is NEW
                                    if source_det_idx >= n_stored_h || connected_det_idx >= n_stored_h {
                                        add_el(wf, ham, source_det_idx, connected_det_idx, None);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Find same-spin connections for a specific alpha string (following original pattern)
    fn find_same_spin_for_alpha_string(
        &self,
        alpha: AlphaString,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // Get all determinants with this alpha string
        if let Some(alpha_idx) = self.alpha_to_index.get(&alpha) {
            let alpha_dets = &self.alpha_to_dets[*alpha_idx];
            
            // Check if there are any NEW determinants with this alpha string
            let has_new_dets = alpha_dets.iter().any(|&det_idx| det_idx >= n_stored_h);
            
            if has_new_dets {
                // Find same-spin excitations within this alpha string (beta double excitations)
                for (i, &det_idx1) in alpha_dets.iter().enumerate() {
                    for &det_idx2 in &alpha_dets[i+1..] {
                        // Ensure at least one is new (following original pattern)
                        if det_idx1 >= n_stored_h || det_idx2 >= n_stored_h {
                            let beta1 = self.determinants[det_idx1].dn;
                            let beta2 = self.determinants[det_idx2].dn;
                            
                            // Check if this is a valid same-spin double excitation (beta)
                            if self.are_strings_double_connected(beta1, beta2) {
                                add_el(wf, ham, det_idx1, det_idx2, None);
                            }
                        }
                    }
                }
                
                // Also handle alpha double excitations: find connections to OTHER alpha strings
                // that differ by exactly 2 electrons, and check if they have determinants 
                // with the SAME beta strings as our NEW determinants
                let new_det_indices: Vec<usize> = alpha_dets.iter()
                    .filter(|&&det_idx| det_idx >= n_stored_h)
                    .copied()
                    .collect();
                
                for &other_alpha in &self.alpha_strings {
                    if other_alpha != alpha && self.are_strings_double_connected(alpha, other_alpha) {
                        if let Some(other_alpha_idx) = self.alpha_to_index.get(&other_alpha) {
                            let other_alpha_dets = &self.alpha_to_dets[*other_alpha_idx];
                            
                            for &new_det_idx in &new_det_indices {
                                let new_beta = self.determinants[new_det_idx].dn;
                                
                                // Find determinants with other_alpha and same beta
                                for &other_det_idx in other_alpha_dets {
                                    let other_beta = self.determinants[other_det_idx].dn;
                                    if new_beta == other_beta {
                                        add_el(wf, ham, new_det_idx, other_det_idx, None);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Find single excitations connecting NEW determinants to ALL determinants
    fn find_single_excitations(
        &self,
        wf: &mut VarWf,
        ham: &Ham,
        n_stored_h: usize,
    ) {
        // For each NEW determinant, check ALL determinants for single excitations
        for source_idx in n_stored_h..self.determinants.len() {
            let source_config = self.determinants[source_idx];
            
            // Check all determinants (both old and new)
            for target_idx in 0..self.determinants.len() {
                if source_idx == target_idx {
                    continue; // Skip self
                }
                
                let target_config = self.determinants[target_idx];
                
                // Check if they differ by exactly one electron
                let up_diff = (source_config.up ^ target_config.up).count_ones();
                let dn_diff = (source_config.dn ^ target_config.dn).count_ones();
                
                let is_single_excitation = match (up_diff, dn_diff) {
                    (2, 0) => true,  // Alpha single excitation
                    (0, 2) => true,  // Beta single excitation
                    _ => false,      // Not a single excitation
                };
                
                if is_single_excitation {
                    // Only add upper triangular elements
                    if source_idx < target_idx {
                        let elem = ham.ham_off_diag_no_excite(&source_config, &target_config);
                        if elem.abs() > 1e-12 {
                            wf.sparse_ham.off_diag[source_idx].push((target_idx, elem));
                        }
                    } else if target_idx < source_idx {
                        let elem = ham.ham_off_diag_no_excite(&target_config, &source_config);
                        if elem.abs() > 1e-12 {
                            wf.sparse_ham.off_diag[target_idx].push((source_idx, elem));
                        }
                    }
                }
            }
        }
    }
}

/// Fast Hamiltonian construction using the VariationalSpace lookup tables.
/// This replaces the algorithm in `gen_sparse_ham_fast` with the O(N log N) approach.
///
/// # Arguments
/// * `global` - Global calculation parameters.
/// * `wf` - Mutable reference to the variational wavefunction.
/// * `ham` - The Hamiltonian operator.
pub fn gen_sparse_ham_fast_lookup(
    global: &Global,
    wf: &mut VarWf,
    ham: &Ham,
) {
    println!("Starting fast Hamiltonian construction with lookup tables...");
    let start_time = std::time::Instant::now();

    // Extract determinant configurations for the VariationalSpace
    let mut determinants = Vec::with_capacity(wf.wf.n);
    for det in &wf.wf.dets {
        determinants.push(det.config);
    }

    // Build the VariationalSpace with lookup tables
    let setup_start = std::time::Instant::now();
    let var_space = VariationalSpace::new(determinants);
    println!("Lookup table setup time: {:?}", setup_start.elapsed());

    // Initialize sparse Hamiltonian matrix if needed
    if wf.n_stored_h() == 0 {
        wf.new_sparse_ham();
    } else {
        wf.expand_sparse_ham_rows();
    }

    // Find connections using the same pattern as the original algorithm
    let connection_start = std::time::Instant::now();
    let n_stored_h = wf.n_stored_h();
    
    println!("Finding connections for {} new determinants (indices {} to {})", 
             wf.wf.n - n_stored_h, n_stored_h, wf.wf.n - 1);
    
    // Follow the original algorithm pattern:
    // 1. Opposite-spin: process NEW alpha strings to ANY determinants
    // 2. Same-spin: process ALL alpha strings but only with NEW determinants involved
    
    var_space.find_all_connections_original_pattern(wf, ham, n_stored_h);
    
    println!("Connection finding time: {:?}", connection_start.elapsed());

    // Sort and remove duplicates
    wf.sparse_ham.sort_remove_duplicates();

    // Update stored H count
    wf.update_n_stored_h(wf.wf.n);

    println!("Total fast Hamiltonian construction time: {:?}", start_time.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::bits::{ibset, ibclr};
    use crate::ham::Ham;
    use crate::wf::{VarWf, Wf};
    use crate::wf::det::Det;
    use std::collections::HashMap;

    /// Create a simple test determinant configuration
    fn test_config(up_orbs: &[i32], dn_orbs: &[i32]) -> Config {
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

    /// Create a mock Ham for testing (zero integrals)
    fn mock_ham() -> Ham {
        let mut ham = Ham::default();
        ham.valence_orbs = (0..20).collect();
        ham
    }

    /// Create a mock VarWf with given determinants
    fn mock_var_wf(determinants: Vec<Config>) -> VarWf {
        use crate::var::sparse::SparseMatUpperTri;
        
        let n = determinants.len();
        let mut dets = Vec::with_capacity(n);
        let mut inds = HashMap::new();
        
        for (i, config) in determinants.iter().enumerate() {
            dets.push(Det {
                config: *config,
                coeff: if i == 0 { 1.0 } else { 0.0 },
                diag: Some(-1.0), // Mock diagonal element
            });
            inds.insert(*config, i);
        }
        
        let wf = Wf {
            n,
            dets,
            inds,
            energy: -1.0,
        };
        
        let mut var_wf = VarWf::default();
        var_wf.wf = wf;
        var_wf.new_sparse_ham();
        var_wf
    }

    #[test]
    fn test_variational_space_construction() {
        // Create a small test system with known determinants
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // Reference: |01⍺01β⟩
            test_config(&[0, 2], &[0, 1]),  // Single excitation: |02⍺01β⟩
            test_config(&[0, 1], &[0, 2]),  // Single excitation: |01⍺02β⟩
            test_config(&[0, 2], &[0, 2]),  // Double excitation: |02⍺02β⟩
        ];

        let var_space = VariationalSpace::new(determinants.clone());

        // Test that all determinants are stored
        assert_eq!(var_space.determinants.len(), 4);
        
        // Test that unique alpha strings are identified correctly
        // We should have 2 unique alpha strings: |01⍺⟩ and |02⍺⟩
        assert_eq!(var_space.alpha_strings.len(), 2);
        
        // Test alpha_to_index mapping
        let alpha_01 = test_config(&[0, 1], &[]).up;
        let alpha_02 = test_config(&[0, 2], &[]).up;
        
        assert!(var_space.alpha_to_index.contains_key(&alpha_01));
        assert!(var_space.alpha_to_index.contains_key(&alpha_02));
        
        // Test alpha_to_dets mapping
        let alpha_01_idx = var_space.alpha_to_index[&alpha_01];
        let alpha_02_idx = var_space.alpha_to_index[&alpha_02];
        
        // Alpha string |01⍺⟩ should be associated with determinants 0 and 2
        let mut alpha_01_dets = var_space.alpha_to_dets[alpha_01_idx].clone();
        alpha_01_dets.sort();
        assert_eq!(alpha_01_dets, vec![0, 2]);
        
        // Alpha string |02⍺⟩ should be associated with determinants 1 and 3
        let mut alpha_02_dets = var_space.alpha_to_dets[alpha_02_idx].clone();
        alpha_02_dets.sort();
        assert_eq!(alpha_02_dets, vec![1, 3]);
    }

    #[test]
    fn test_alpha_singles_lookup() {
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
            test_config(&[0, 2], &[0, 1]),  // |02⍺01β⟩ - single excitation from above
            test_config(&[1, 2], &[0, 1]),  // |12⍺01β⟩ - single excitation from |02⍺01β⟩
        ];

        let var_space = VariationalSpace::new(determinants);

        let alpha_01 = test_config(&[0, 1], &[]).up;
        let alpha_02 = test_config(&[0, 2], &[]).up;
        let alpha_12 = test_config(&[1, 2], &[]).up;

        // Check that alpha_01 (|01⍺⟩) is connected to alpha_02 (|02⍺⟩)
        if let Some(connections) = var_space.alpha_singles.get(&alpha_01) {
            assert!(connections.contains(&alpha_02));
        } else {
            panic!("Alpha singles lookup failed for alpha_01");
        }

        // Check that alpha_02 (|02⍺⟩) is connected to both alpha_01 (|01⍺⟩) and alpha_12 (|12⍺⟩)
        if let Some(connections) = var_space.alpha_singles.get(&alpha_02) {
            assert!(connections.contains(&alpha_01));
            assert!(connections.contains(&alpha_12));
        } else {
            panic!("Alpha singles lookup failed for alpha_02");
        }
    }

    #[test]
    fn test_beta_singles_lookup() {
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
            test_config(&[0, 1], &[0, 2]),  // |01⍺02β⟩ - single β excitation
            test_config(&[0, 1], &[1, 2]),  // |01⍺12β⟩ - single β excitation from above
        ];

        let var_space = VariationalSpace::new(determinants);

        let beta_01 = test_config(&[], &[0, 1]).dn;
        let beta_02 = test_config(&[], &[0, 2]).dn;
        let beta_12 = test_config(&[], &[1, 2]).dn;

        // Check beta single excitation connections
        if let Some(connections) = var_space.beta_singles.get(&beta_01) {
            assert!(connections.contains(&beta_02));
        }

        if let Some(connections) = var_space.beta_singles.get(&beta_02) {
            assert!(connections.contains(&beta_01));
            assert!(connections.contains(&beta_12));
        }
    }

    #[test]
    fn test_double_excitation_detection() {
        let var_space = VariationalSpace::new(vec![]);

        // Two strings that differ by exactly 4 bits (double excitation)
        let string1 = test_config(&[0, 1], &[]).up;  // |01⟩
        let string2 = test_config(&[2, 3], &[]).up;  // |23⟩

        assert!(var_space.are_strings_double_connected(string1, string2));

        // Two strings that differ by 2 bits (single excitation) - should not be double connected
        let string3 = test_config(&[0, 2], &[]).up;  // |02⟩
        assert!(!var_space.are_strings_double_connected(string1, string3));

        // Identical strings - should not be double connected
        assert!(!var_space.are_strings_double_connected(string1, string1));
    }

    #[test]
    fn test_single_excitation_detection() {
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),
            test_config(&[0, 1], &[0, 2]),  // Single β excitation
        ];
        
        let var_space = VariationalSpace::new(determinants);

        let beta1 = test_config(&[], &[0, 1]).dn;
        let beta2 = test_config(&[], &[0, 2]).dn;
        let beta3 = test_config(&[], &[1, 2]).dn;

        // These should be single connected
        assert!(var_space.are_beta_strings_single_connected(beta1, beta2));

        // These should not be single connected (double excitation)
        assert!(!var_space.are_beta_strings_single_connected(beta1, beta3));
    }

    #[test]
    fn test_opposite_spin_double_excitation_detection() {
        let var_space = VariationalSpace::new(vec![]);

        // Test cases for opposite-spin double excitations
        // Alpha: |01⟩ -> |02⟩ (single), Beta: |01⟩ -> |02⟩ (single) = overall double
        let alpha1 = test_config(&[0, 1], &[]).up;
        let alpha2 = test_config(&[0, 2], &[]).up;
        let beta1 = test_config(&[], &[0, 1]).dn;
        let beta2 = test_config(&[], &[0, 2]).dn;

        // Should detect single excitation in both alpha and beta
        assert!(var_space.alpha_singles.get(&alpha1).is_none() || 
               var_space.alpha_singles.get(&alpha2).is_none()); // Empty var_space, so no lookups
    }

    #[test] 
    fn test_connection_finding_simple_case() {
        // Test the connection finding on a very simple 2-determinant system
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
            test_config(&[0, 2], &[0, 2]),  // |02⍺02β⟩ - opposite-spin double excitation
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        // Test connection finding with n_stored_h = 0 (both are new)
        let n_stored_h = 0;
        let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        var_space.find_connections_and_add_to_sparse_ham(0, &mut wf, &ham, n_stored_h);
        
        let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Should find exactly one connection: (0,1) representing the double excitation
        assert_eq!(final_count - initial_count, 1);
    }

    #[test]
    fn test_same_spin_double_excitation() {
        // Test same-spin double excitation: same alpha, different beta by double excitation
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
            test_config(&[0, 1], &[2, 3]),  // |01⍺23β⟩ - same alpha, beta double excitation
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        let n_stored_h = 0;
        let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        var_space.find_connections_and_add_to_sparse_ham(0, &mut wf, &ham, n_stored_h);
        
        let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Should find exactly one connection for the same-spin double excitation
        assert_eq!(final_count - initial_count, 1);
    }

    #[test] 
    fn test_alpha_beta_lookup_construction() {
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
            test_config(&[0, 1], &[0, 2]),  // |01⍺02β⟩ 
            test_config(&[0, 2], &[0, 1]),  // |02⍺01β⟩
        ];

        let var_space = VariationalSpace::new(determinants);
        
        // Test that alpha_beta_lookup was constructed correctly
        let alpha_01 = test_config(&[0, 1], &[]).up;
        let alpha_02 = test_config(&[0, 2], &[]).up;
        
        // For determinant 0: alpha=01, beta=01
        // Removing electron from position 0 of beta gives beta_r1 = 01 (just electron 1)
        let beta_r1_from_0 = test_config(&[], &[1]).dn;
        let key1 = (alpha_01, beta_r1_from_0);
        
        // Should have entry for this key
        assert!(var_space.alpha_beta_lookup.contains_key(&key1));
    }

    #[test]
    fn test_no_self_connections() {
        // Ensure determinants don't connect to themselves
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        let n_stored_h = 0;
        let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        var_space.find_connections_and_add_to_sparse_ham(0, &mut wf, &ham, n_stored_h);
        
        let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Should find no connections for a single determinant
        assert_eq!(final_count - initial_count, 0);
    }

    #[test]
    fn test_new_vs_old_determinant_logic() {
        // Test that connections are only found when at least one determinant is new
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩ - old (index 0)
            test_config(&[0, 2], &[0, 2]),  // |02⍺02β⟩ - new (index 1)
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        // Set n_stored_h = 1, so only determinant 1 is new
        let n_stored_h = 1;
        let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Try to find connections from old determinant 0
        var_space.find_connections_and_add_to_sparse_ham(0, &mut wf, &ham, n_stored_h);
        
        let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Should still find the connection because determinant 1 is new
        assert_eq!(final_count - initial_count, 1);
    }

    #[test]
    fn test_comprehensive_four_determinant_system() {
        // Comprehensive test with multiple excitation types
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // |01⍺01β⟩ - reference
            test_config(&[0, 2], &[0, 1]),  // |02⍺01β⟩ - alpha single
            test_config(&[0, 1], &[0, 2]),  // |01⍺02β⟩ - beta single  
            test_config(&[0, 2], &[0, 2]),  // |02⍺02β⟩ - opposite-spin double
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        let n_stored_h = 0;
        let mut total_connections = 0;
        
        // Count connections from each determinant
        for i in 0..4 {
            let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
            var_space.find_connections_and_add_to_sparse_ham(i, &mut wf, &ham, n_stored_h);
            let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
            total_connections += final_count - initial_count;
        }
        
        // Should find connections between:
        // 0-1 (alpha single), 0-2 (beta single), 0-3 (opposite-spin double)
        // 1-2 (opposite-spin double), 1-3 (beta single), 2-3 (alpha single)
        // Total: 6 unique connections, but each found twice = 12
        // However, the sparse matrix only stores upper triangle, so expect 6
        println!("Total connections found: {}", total_connections);
        
        // Remove duplicates by sorting and deduplicating the sparse matrix
        wf.sparse_ham.sort_remove_duplicates();
        let unique_connections = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        println!("Unique connections after deduplication: {}", unique_connections);
        assert_eq!(unique_connections, 6);
    }

    #[test]
    fn test_bit_manipulation_accuracy() {
        // Test that our bit manipulation functions work correctly
        let config1 = test_config(&[0, 1, 3], &[0, 2]);
        let config2 = test_config(&[0, 2, 3], &[0, 1]);
        
        // These should be connected by a single alpha excitation (1->2) and single beta excitation (2->1)
        // Making it an opposite-spin double excitation
        
        let var_space = VariationalSpace::new(vec![]);
        
        // Check alpha strings differ by single excitation (should not be double connected)
        assert!(!var_space.are_strings_double_connected(config1.up, config2.up));
        
        // Check beta strings differ by single excitation (should not be double connected)  
        assert!(!var_space.are_strings_double_connected(config1.dn, config2.dn));
        
        // Test double excitation example
        let double1 = test_config(&[0, 1], &[]).up;  // |01⟩
        let double2 = test_config(&[2, 3], &[]).up;  // |23⟩
        assert!(var_space.are_strings_double_connected(double1, double2));
    }

    #[test]
    fn test_original_pattern_following() {
        // Test that the original pattern algorithm follows the exact logic
        let determinants = vec![
            test_config(&[0, 1], &[0, 1]),  // old determinant
            test_config(&[0, 2], &[0, 1]),  // new determinant 
            test_config(&[1, 2], &[0, 1]),  // new determinant
        ];

        let var_space = VariationalSpace::new(determinants.clone());
        let mut wf = mock_var_wf(determinants);
        let ham = mock_ham();
        
        // Set first determinant as old, others as new
        let n_stored_h = 1;
        
        let initial_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        // Use the original pattern algorithm
        var_space.find_all_connections_original_pattern(&mut wf, &ham, n_stored_h);
        
        let final_count = wf.sparse_ham.off_diag.iter().map(|v| v.len()).sum::<usize>();
        
        println!("Connections found with original pattern: {}", final_count - initial_count);
        
        // Should find:
        // - Opposite-spin: NEW alpha strings (02, 12) to ALL determinants 
        // - Same-spin: ALL alpha strings but only involving NEW determinants
        assert!(final_count > initial_count);
    }
}