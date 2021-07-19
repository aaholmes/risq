// Excitation generation module:
// Includes sorted excitations for heat-bath algorithm

pub mod init;
pub mod iterator;

use std::hash::{Hash, Hasher};

// Orbital or orbital pair
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum Orbs {
    Double((i32, i32)),
    Single(i32)
}

// Excitations
pub struct Excite {
    pub init: Orbs,
    pub target: Orbs,
    pub abs_h: f64,
    pub is_alpha: Option<bool>, // if None, then opposite spin
}

// Simplified excitations for storing in heat-bath tensor
// Double excitation triplet (r, s, |H|)
// Single excitation doublet (r, max |H|)
// Also includes sum of remaining |H| and H^2 values for importance sampling the CDF's
#[derive(Copy, Clone)]
pub struct StoredExcite {
    pub target: Orbs,
    pub abs_h: f64,
    pub sum_remaining_abs_h: f64,
    pub sum_remaining_h_squared: f64,
}

// These impl's are only needed for testing the CDF searching sampler
impl PartialEq for StoredExcite {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}
impl Eq for StoredExcite {}

impl Hash for StoredExcite {
    // Hash using only target orb(s)
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.target.hash(state);
    }
}
