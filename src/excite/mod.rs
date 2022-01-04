// Excitation generation module:
// Includes sorted excitations for heat-bath algorithm

pub mod init;
pub mod iterator;

use std::hash::{Hash, Hasher};

/// Generalized type that can either be a single spatial orbital or a pair of them
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum Orbs {
    Double((i32, i32)),
    Single(i32)
}

/// Candidate excitation from one determinant to another.  Contains the initial orbs `init`, target
/// orb(s) `target`, absolute value of the excitation matrix element `abs_h`, and whether it is an
/// alpha-spin excitation `is_alpha` (`is_alpha = None` for opposite-spin double excitations)
pub struct Excite {
    pub init: Orbs,
    pub target: Orbs,
    pub abs_h: f64,
    pub is_alpha: Option<bool>, // if None, then opposite spin
}

/// Excitation information to be stored in `ExciteGenerator`.  Contains only `target` and `abs_h`
/// (since `init` and `is_alpha` are already known by the time `StoredExcite`s are needed).
/// Also contains `sum_remaining_abs_h` and `sum_remaining_h_squared` for importance-sampling the
/// remaining excitations
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
