//! Contains the random number generator and its seed

use rand::rngs::StdRng;
use rand::SeedableRng;
// use rand::prelude::ThreadRng;

/// Contains the seeded random number generator.  Must be used each time a random number is generated
pub struct Rand {
    // pub rng: ThreadRng, // thread-safe random number generator
    pub rng: StdRng, // seeded rng
}

/// Initialize the random number generator + seed
pub fn init_rand() -> Rand {
    Rand {
        // rng: rand::thread_rng(),
        rng: StdRng::seed_from_u64(1312),
    }
}
