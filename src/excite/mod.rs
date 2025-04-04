//! # Excitation Representation (`excite`)
//!
//! This module defines the core data structures used to represent electronic excitations
//! between Slater determinants. It distinguishes between the full description of an
//! excitation (`Excite`) and a more compact version used for storage and efficient
//! generation (`StoredExcite`).
//!
//! ## Key Components:
//! *   `Orbs`: Enum representing either a single orbital index or a pair of orbital indices,
//!     used to specify the target orbitals of an excitation.
//! *   `Excite`: Represents a complete excitation event, including initial and target orbitals,
//!     spin information, and an estimate of the Hamiltonian matrix element magnitude (`abs_h`).
//! *   `StoredExcite`: A compact representation storing only the target orbitals and `abs_h`,
//!     along with cumulative sums used for importance sampling or screening. Used within `ExciteGenerator`.
//! *   `init`: Submodule for initializing the `ExciteGenerator`.
//! *   `iterator`: Submodule providing iterators for generating excitations.

pub mod init;
pub mod iterator;

use std::hash::{Hash, Hasher};

/// Represents the spatial orbital(s) involved in an excitation target or origin.
///
/// Used to specify either a single orbital (for single excitations) or a pair
/// of orbitals (for double excitations). Indices are 0-based.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum Orbs {
    /// A pair of spatial orbitals (p, q), used for double excitations.
    Double((i32, i32)),
    /// A single spatial orbital p, used for single excitations.
    Single(i32),
}

/// Represents a complete electronic excitation event between two determinants.
///
/// Contains information about the initial orbitals (`init`), target orbitals (`target`),
/// the spin channel (`is_alpha`), and an estimate of the absolute value of the
/// Hamiltonian matrix element (`abs_h`) connecting the initial and final determinants.
#[derive(Debug, Clone)] // Added Debug, Clone
pub struct Excite {
    /// Specifies the spin channel: `Some(true)` for alpha, `Some(false)` for beta,
    /// `None` for opposite-spin double excitations.
    pub is_alpha: Option<bool>,
    /// The orbital(s) electrons are excited *from*.
    pub init: Orbs,
    /// The orbital(s) electrons are excited *to*.
    pub target: Orbs,
    /// An estimate of the absolute value of the Hamiltonian matrix element |<D_final|H|D_initial>|.
    /// Often based on approximations like diagonal elements or simplified PT estimates.
    pub abs_h: f64,
}

/// Compact representation of an excitation target, stored within `ExciteGenerator`.
///
/// When generating excitations *from* a specific initial orbital set (`init`) and spin
/// (`is_alpha`), only the target orbitals and associated Hamiltonian estimate (`abs_h`)
/// need to be stored efficiently. This struct also stores cumulative sums (`sum_remaining_*`)
/// of `abs_h` and `abs_h^2` for excitations further down the sorted list. These sums are
/// crucial for implementing screening (deterministic treatment above a threshold `eps`) and
/// importance sampling (stochastic treatment below `eps`) efficiently.
#[derive(Copy, Clone, Debug)] // Added Debug
pub struct StoredExcite {
    /// The target orbital(s) for this specific excitation pathway.
    pub target: Orbs,
    /// An estimate of the absolute value of the Hamiltonian matrix element |<D_final|H|D_initial>|.
    pub abs_h: f64,
    /// The sum of `abs_h` for all excitations *following* this one in a pre-sorted list
    /// (originating from the same `init` orbitals and `is_alpha` channel). Used for sampling.
    pub sum_remaining_abs_h: f64,
    /// The sum of `abs_h^2` for all excitations *following* this one in a pre-sorted list.
    /// Used for estimating the variance or contribution of the remaining stochastic space.
    pub sum_remaining_h_squared: f64,
}

// These impl's are only needed for testing the CDF searching sampler
impl PartialEq for StoredExcite {
    /// Equality comparison based only on the target orbitals.
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}
impl Eq for StoredExcite {}

impl Hash for StoredExcite {
    /// Hashes based only on the target orbitals. Consistent with `PartialEq`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.target.hash(state);
    }
}
