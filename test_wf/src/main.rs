use std::collections::HashMap;

// Global variables
struct Global {
    norb: u32,
    nup: u32,
    ndn: u32,
    var_eps: f64,
}

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

// Orbital pair
struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
struct Doub {
    target: OPair,
    abs_h: f64,
}

// Hamiltonian, containing both integrals and heat-bath hashmap of double excitations
struct Ham {
    doubs: HashMap<OPair, Vec<Doub>>, // Each electron pair points to a sorted vector of double excitations
}

fn main() {
    println!("Hello, world!");
}
