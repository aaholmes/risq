use std::collections::HashMap;

// Determinant
struct Det {
    up: u128,
    dn: u128,
}

// Wavefunction
struct Wf {
    n: u64, // number of dets
    inds: HashMap<Det, u64>, // hashtable : det -> u64 for looking up index by det
    dets: Vec<Det>, // for looking up det by index
    coeffs: Vec<f64>,
    diags: Vec<f64>, // diagonal elements of Hamiltonian (so new diagonal elements can be computed quickly)
}

// Max-heap of double excitation triplet (r, s, |H|)
struct DoubHeap {
    rsh: (i32, i32, f64),
}

fn main() {
    println!("Hello, world!");
}
