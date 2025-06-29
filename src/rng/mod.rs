//! # Random Number Generation (`rng`)
//!
//! This module provides a simple wrapper around a seeded random number generator (RNG)
//! for use in stochastic parts of the calculations. Using a seeded RNG ensures
//! reproducibility of stochastic results.

use crate::error::{RisqError, RisqResult};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Wrapper struct holding the seeded random number generator instance.
///
/// This struct encapsulates the `StdRng` from the `rand` crate, initialized with a fixed seed.
/// An instance of `Rand` should be passed to functions requiring random numbers.
#[derive(Debug)] // Added Debug derive
pub struct Rand {
    /// The seeded standard random number generator instance.
    pub rng: StdRng,
}

/// Initializes and returns a new `Rand` struct.
///
/// Creates a `StdRng` instance seeded with a fixed value (currently `1312`).
/// This ensures that sequences of random numbers generated using the returned `Rand`
/// instance are reproducible across runs.
///
/// # Errors
/// Currently never fails, but returns `RisqResult` for future extensibility.
pub fn init_rand() -> RisqResult<Rand> {
    Ok(Rand {
        rng: StdRng::seed_from_u64(1312), // Seeded for reproducibility
    })
}
