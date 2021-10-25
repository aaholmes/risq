// Contains the random number generator and its seed

use rand::rngs::StdRng;
use rand::SeedableRng;
// use rand::prelude::ThreadRng;

pub struct Rand {
    // pub rng: ThreadRng, // thread-safe random number generator
    pub rng: StdRng, // seeded rng
}

pub fn init_rand() -> Rand {
    // Initialize the random number generator + seed
    Rand {
        // rng: rand::thread_rng(),
        rng: StdRng::seed_from_u64(1312),
    }
}