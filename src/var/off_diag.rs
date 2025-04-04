//! # Off-Diagonal Hamiltonian Element Utilities (`var::off_diag`)
//!
//! This module provides helper functions for computing and adding off-diagonal
//! Hamiltonian matrix elements (`H_ij = <D_i|H|D_j>`) to the sparse matrix
//! representation (`wf.sparse_ham`) used by the Davidson algorithm.

use crate::ham::Ham;
use crate::wf::det::Config;
use crate::wf::VarWf;

// Removed unused struct `OffDiagElemsNoHash` and its impl block (marked with cfg(test))

/// Computes (if necessary) and adds an off-diagonal Hamiltonian element `H_ij`
/// to the sparse matrix representation stored in `wf.sparse_ham`.
///
/// Handles index ordering (`i` vs `j`) to store only the upper triangle.
/// If `elem` is `None`, it computes `H_ij` using `ham.ham_off_diag_no_excite`.
/// If the computed or provided element is non-zero, it calls `add_off_diag_elem`.
/// Does nothing if `i == j`.
///
/// # Arguments
/// * `wf`: Mutable reference to the variational wavefunction, containing the sparse matrix.
/// * `ham`: The Hamiltonian operator.
/// * `i`, `j`: Indices of the two determinants in `wf.wf.dets`.
/// * `elem`: Optionally pre-computed value of `H_ij`. If `None`, it will be computed.
pub fn add_el(wf: &mut VarWf, ham: &Ham, i: usize, j: usize, elem: Option<f64>) {
    match elem {
        None => {
            if i == j {
                return;
            }
            let elem = ham.ham_off_diag_no_excite(&wf.wf.dets[i].config, &wf.wf.dets[j].config);
            if elem != 0.0 {
                add_off_diag_elem(wf, i, j, elem);
            }
        }
        Some(elem) => add_off_diag_elem(wf, i, j, elem),
    }
}

/// Adds a non-zero off-diagonal element `elem = H_ij` to the appropriate row
/// in the upper triangular sparse matrix `wf.sparse_ham.off_diag`.
/// Ensures `i < j` before pushing `(j, elem)` onto `wf.sparse_ham.off_diag[i]`.
/// Internal helper function.
fn add_off_diag_elem(wf: &mut VarWf, i: usize, j: usize, elem: f64) {
    if i < j {
        wf.sparse_ham.off_diag[i].push((j, elem));
        // if j >= wf.n_stored_h() { wf.sparse_ham.off_diag[i].push((j, elem)); }
    } else {
        wf.sparse_ham.off_diag[j].push((i, elem));
        // if i >= wf.n_stored_h() { wf.sparse_ham.off_diag[j].push((i, elem)); }
    }
}

/// Computes and adds both `H_ij` and its spin-flipped counterpart `H_i'j'` to the sparse matrix.
///
/// Assumes spin symmetry, where `H_ij = H_i'j'` if `i'` and `j'` are the spin-flips of `i` and `j`.
/// 1. Computes `elem = H_ij` using `ham.ham_off_diag_no_excite`.
/// 2. If `elem` is non-zero:
///    a. Adds `(j, elem)` or `(i, elem)` to the sparse matrix via `add_el`.
///    b. Finds the indices `i_spin_flipped` and `j_spin_flipped` corresponding to the
///       spin-flipped configurations of determinants `i` and `j` using the `wf.wf.inds` map.
///    c. Adds `(j_spin_flipped, elem)` or `(i_spin_flipped, elem)` to the sparse matrix via `add_el`.
/// Does nothing if `i == j`.
///
/// # Arguments
/// * `wf`: Mutable reference to the variational wavefunction.
/// * `ham`: The Hamiltonian operator.
/// * `i`, `j`: Indices of the two determinants.
///
/// # Panics
/// Panics if the spin-flipped configuration of determinant `i` or `j` is not found in `wf.wf.inds`.
pub fn add_el_and_spin_flipped(wf: &mut VarWf, ham: &Ham, i: usize, j: usize) {
    if i == j {
        return;
    }
    // println!("add_el_and_spin_flipped called with i,j= {}, {}", i, j);
    // println!("i det: {}, {}", wf.wf.dets[i].config.up, wf.wf.dets[i].config.dn);
    // println!("j det: {}, {}", wf.wf.dets[j].config.up, wf.wf.dets[j].config.dn);
    let elem = ham.ham_off_diag_no_excite(&wf.wf.dets[i].config, &wf.wf.dets[j].config);
    if elem != 0.0 {
        // Element
        add_el(wf, ham, i, j, Some(elem));

        // Spin-flipped element
        let i_spin_flipped = {
            let config = wf.wf.dets[i].config;
            wf.wf.inds[&Config {
                up: config.dn,
                dn: config.up,
            }]
        };
        let j_spin_flipped = {
            let config = wf.wf.dets[j].config;
            wf.wf.inds[&Config {
                up: config.dn,
                dn: config.up,
            }]
        };
        add_el(wf, ham, i_spin_flipped, j_spin_flipped, Some(elem));
    }
}
