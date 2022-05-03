//! Data structure for holding off-diagonal Hamiltonian matrix elements and putting them into the SparseMat data structure for Davidson

use crate::ham::Ham;
use crate::wf::det::Config;
use crate::wf::VarWf;

/// Off-diag elems data structure
#[cfg(test)]
#[derive(Default)]
pub struct OffDiagElemsNoHash {
    n: usize,                        // number of rows
    nonzero: Vec<Vec<(usize, f64)>>, // vector of (ind, val) pairs for each row
    nnz: Vec<usize>, // number of nonzero elements (only update when generating SparseMat, only used to determine what has been added onto the end)
}

#[cfg(test)]
impl OffDiagElemsNoHash {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            nonzero: vec![Vec::with_capacity(100); n],
            nnz: vec![0; n + 1], // 0th element is always 0; nnz[i + 1] corresponds to row i
        }
    }

    /// Just create empty rows to fill up the dimension to new size n
    pub fn expand_rows(&mut self, n: usize) {
        println!("Expanding variational H from size {} to size {}", self.n, n);
        self.nonzero
            .append(&mut vec![Vec::with_capacity(100); n - self.n]);
        self.nnz.append(&mut vec![0; n - self.n]);
        self.n = n;
    }
}

/// Add an off-diagonal element H_{ij} to off_diag_elems
/// i and j can be of any order
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

fn add_off_diag_elem(wf: &mut VarWf, i: usize, j: usize, elem: f64) {
    if i < j {
        wf.sparse_ham.off_diag[i].push((j, elem));
        // if j >= wf.n_stored_h() { wf.sparse_ham.off_diag[i].push((j, elem)); }
    } else {
        wf.sparse_ham.off_diag[j].push((i, elem));
        // if i >= wf.n_stored_h() { wf.sparse_ham.off_diag[j].push((i, elem)); }
    }
}

/// Add an off-diagonal element H_{ij}, as well as its spin-flipped counterpart, to off_diag_elems
/// i and j can be of any order
pub fn add_el_and_spin_flipped(wf: &mut VarWf, ham: &Ham, i: usize, j: usize) {
    if i == j {
        return;
    }
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
