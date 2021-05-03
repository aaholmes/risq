// Excitation generation module:
// Includes sorted excitations for heat-bath algorithm

pub mod init;
pub mod iterator;

use std::hash::Hash;

// Orbital pair
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct OPair(pub i32, pub i32);


// Excitations
pub enum Excite {
    Double(Doub),
    Single(Sing)
}

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    pub init: OPair,
    pub target: OPair,
    pub abs_h: f64,
    pub is_alpha: Option<bool>, // if None, then opposite spin
}

// Single excitation doublet (r, max |H|)
pub struct Sing {
    pub init: i32,
    pub target: i32,
    pub max_abs_h: f64,
    pub is_alpha: bool,
}


// Simplified excitations for storing in heat-bath tensor
pub enum StoredExcite {
    Double(StoredDoub),
    Single(StoredSing)
}

// Double excitation triplet (r, s, |H|)
pub struct StoredDoub {
    pub target: OPair,
    pub abs_h: f64,
}

// Single excitation doublet (r, max |H|)
pub struct StoredSing {
    pub target: i32,
    pub max_abs_h: f64,
}