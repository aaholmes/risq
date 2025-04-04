//! # Integral and Indexing Utilities (`utils::ints`)
//!
//! This module provides helper functions related to reading files (like FCIDUMP),
//! combining orbital indices into unique 1D indices for accessing flattened integral arrays,
//! and computing permutation sign factors for Slater-Condon rules.

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use super::bits::parity;

/// Reads a file line by line and returns an iterator over the lines.
///
/// Opens the file specified by `filename`, wraps it in a `BufReader` for efficiency,
/// and returns an iterator that yields each line as `io::Result<String>`.
///
/// # Arguments
/// * `filename`: A type that can be converted into a `Path` (e.g., `&str`).
///
/// # Returns
/// An `io::Result` containing an iterator over the lines of the file.
pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// Combines two orbital indices `p` and `q` into a single unique `usize` index.
///
/// Assumes `p` and `q` are 1-based indices (as in FCIDUMP).
/// Uses triangular numbering: `index = max(i,j)*(max(i,j)-1)/2 + min(i,j)`, where `i=|p|, j=|q|`.
/// This maps a pair `(p, q)` to a unique index suitable for accessing a flattened
/// upper (or lower) triangular matrix storing symmetric two-index quantities (like h_pq).
/// Ignores the sign of `p` and `q`.
pub fn combine_2(p: i32, q: i32) -> usize {
    let i = p.abs();
    let j = q.abs();
    if i > j {
        ((i * (i - 1)) / 2 + j) as usize
    } else {
        ((j * (j - 1)) / 2 + i) as usize
    }
}

/// Combines two `usize` indices `i` and `j` using triangular numbering.
/// Internal helper function, assumes `i, j` are already 0-based or appropriately adjusted.
fn combine_2_usize(i: usize, j: usize) -> usize {
    if i > j {
        (i * (i - 1)) / 2 + j
    } else {
        (j * (j - 1)) / 2 + i
    }
}

/// Combines four orbital indices `p, q, r, s` into a single unique `usize` index.
///
/// Assumes `p, q, r, s` are 1-based indices.
/// Uses nested triangular numbering: first combines `(p, q)` into `idx_pq` using `combine_2`,
/// then combines `(r, s)` into `idx_rs` using `combine_2`, and finally combines
/// `idx_pq` and `idx_rs` using `combine_2_usize`.
/// This maps the four indices to a unique index suitable for accessing a flattened
/// representation of the two-electron integrals (pq|rs), exploiting permutation symmetry.
pub fn combine_4(p: i32, q: i32, r: i32, s: i32) -> usize {
    combine_2_usize(combine_2(p, q), combine_2(r, s))
}

/// Calculates the permutation sign (+1 or -1) between two configurations (`det1`, `det2`)
/// that differ by exactly one spin-orbital.
///
/// This sign is needed for the Slater-Condon rules. It's determined by the parity
/// (even or odd number of electrons) between the differing orbital indices.
/// Assumes `det1` and `det2` differ by a single bit flip.
pub fn permute(det1: u128, det2: u128) -> i32 {
    if det1 > det2 {
        parity(det1 & (det1 - det2))
    } else {
        parity(det2 & (det2 - det1))
    }
}

/// Calculates the permutation sign (+1 or -1) between two configurations (`det1`, `det2`)
/// that differ by exactly two spin-orbitals.
///
/// `v` should contain the four orbital indices involved in the double excitation `(i, j -> k, l)`.
/// The sign depends on the parity of electrons between `i` and `k` and between `j` and `l`.
/// This implementation seems to calculate the parity based on the bits *between* the
/// excited orbitals, which is a common way to determine the sign factor.
/// Assumes `det1` and `det2` differ by exactly two bit flips corresponding to the indices in `v`.
pub fn permute_2(det1: u128, det2: u128, v: [i32; 4]) -> i32 {
    let diff: u128 = det1
        & det2
        & (((1 << v[0]) - 1) ^ ((1 << v[1]) - 1) ^ ((1 << v[2]) - 1) ^ ((1 << v[3]) - 1));
    parity(diff)
}
