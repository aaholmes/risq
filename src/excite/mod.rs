// Excitation generation module (includes usual heat-bath routines)

use std::collections::HashMap;

// Orbital pair
pub struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    target: OPair,
    abs_h: f64,
}

// Heat-bath excitation generator:
// each electron pair points to a sorted vector of double excitations
pub struct ExciteGenerator {
    same_spin_doub_generator: HashMap<i32, Vec<Doub>>,
    opp_spin_doub_generator: HashMap<OPair, Vec<Doub>>,
}

impl ExciteGenerator {
    pub fn init(&self)) -> () {
        // Initialize by sorting double excitation element for all pairs
        todo!()
    }

    pub fn double_excite(&self, det: Det) {
        // Generate all double excitations from the given determinant
        todo!()
    }
}