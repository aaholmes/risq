//! # Alias Method Sampling (`stoch::alias`)
//!
//! Implements Vose's Alias Method for efficient O(1) sampling from a discrete
//! probability distribution.
//!
//! This method pre-computes two tables (`alias` and `alias_prob`) based on the input
//! probability distribution. Sampling then involves:
//! 1. Rolling a fair die (selecting a random index `i` uniformly).
//! 2. Flipping a biased coin (comparing a uniform random number `r` in [0,1) to `alias_prob[i]`).
//! 3. If `r < alias_prob[i]`, return `i`. Otherwise, return `alias[i]`.
//!
//! This implementation is based on the description at [https://www.keithschwarz.com/darts-dice-coins/](https://www.keithschwarz.com/darts-dice-coins/)
//! and inspired by the `vose-alias` crate, but uses vectors directly.

use rand::Rng;
use std::fmt::Debug;
use crate::rng::Rand;
use rand::distributions::Uniform;

/// Data structure for Vose's Alias Method.
///
/// Contains the pre-computed tables needed for O(1) sampling.
#[derive(Debug, Clone)]
pub struct Alias {
    /// The original normalized probability of sampling each element `i`. Stored for reference.
    pub sample_prob: Vec<f64>,
    /// The alias table. `alias[i]` stores the index of the alternative element
    /// that might be chosen when column `i` is selected.
    alias: Vec<usize>,
    /// The probability table. `alias_prob[i]` stores the probability of choosing
    /// index `i` (rather than `alias[i]`) when column `i` is selected.
    alias_prob: Vec<f64>,
    /// A distribution for sampling column indices uniformly from `0..size`.
    uniform: Uniform<usize>,
}

impl Alias {
    /// Creates a new `Alias` instance from a vector of relative probabilities.
    ///
    /// # Arguments
    /// * `rel_probs`: A vector where `rel_probs[i]` is proportional to the desired
    ///   probability of sampling index `i`. These probabilities do not need to be normalized.
    ///
    /// # Returns
    /// An `Alias` struct initialized with the pre-computed tables.
    ///
    /// # Algorithm
    /// Implements Vose's algorithm:
    /// 1. Normalizes the input probabilities (`rel_probs`).
    /// 2. Scales probabilities by `size` (number of elements).
    /// 3. Initializes `small` and `large` worklists containing indices with scaled probabilities
    ///    less than 1.0 and greater than or equal to 1.0, respectively.
    /// 4. Iteratively processes pairs from `small` and `large`:
    ///    - Sets `alias_prob[l]` for the small index `l`.
    ///    - Sets `alias[l]` to the large index `g`.
    ///    - Updates the remaining probability "mass" for the large index `g`.
    ///    - Re-categorizes `g` into `small` or `large` based on its updated probability.
    /// 5. Handles any remaining elements in `large` or `small` (due to numerical precision)
    ///    by setting their `alias_prob` to 1.0.
    pub fn new(rel_probs: Vec<f64>) -> Alias {
        let size = rel_probs.len();
        println!("New Alias with size = {}", size);

        // Normalize input probabilities
        let mut sum = 0.0;
        for p in &rel_probs {
            sum = sum + p;
        }
        let mut norm_prob: Vec<f64> = Vec::with_capacity(size);
        for p in &rel_probs {
            norm_prob.push(p / sum);
        }

        // starting the actual init
        let mut small: Vec<usize> = Vec::with_capacity(size);
        let mut large: Vec<usize> = Vec::with_capacity(size);
        let mut scaled_probability_vector: Vec<f64> = Vec::with_capacity(size);

        let mut alias: Vec<usize> = vec![0; size];
        let mut prob: Vec<f64> = vec![0.0; size];

        // multiply each prob by size
        for i in 0..size {
            let p: f64 = norm_prob[i];
            let scaled_proba = p * (size as f64);
            scaled_probability_vector.push(scaled_proba);

            if scaled_proba < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // emptying one column first
        while !(small.is_empty() || large.is_empty()) {
            // removing the element from small and large
            if let (Some(l), Some(g)) = (small.pop(), large.pop()) {
                // put g in the alias vector
                alias[l] = g;
                // getting the probability of the small element
                let p_l = scaled_probability_vector[l];
                // put it in the prob vector
                prob[l] = p_l;

                // update the probability for g
                let p_g = scaled_probability_vector[g];
                // P(l->g) = 1 - p_l, so subtract this from p_g to get remaining prob to still be accounted for
                let new_p_g = (p_g + p_l) - 1.0;
                // update scaled_probability_vector
                scaled_probability_vector[g] = new_p_g;
                if new_p_g < 1.0 {
                    small.push(g);
                } else {
                    large.push(g);
                }
            }
        }

        // finishing the init
        while !large.is_empty() {
            if let Some(g) = large.pop() {
                prob[g] = 1.0;
            }
        }

        while !small.is_empty() {
            if let Some(l) = small.pop() {
                prob[l] = 1.0;
            }
        }

        Alias {
            sample_prob: norm_prob,
            alias,
            alias_prob: prob,
            uniform: Uniform::from(0..size),
        }
    }

    /// Samples an index from the distribution in O(1) time.
    ///
    /// # Arguments
    /// * `rand`: A mutable reference to the `Rand` struct containing the RNG state.
    ///
    /// # Returns
    /// The index (`usize`) of the sampled element.
    pub fn sample(&self, rand: &mut Rand) -> usize { // Made immutable self
        let (i, r) = self.roll_die_and_flip_coin(rand);
        return self.select_element(i, r);
    }

    /// Samples an index and returns it along with its true probability from the original distribution.
    /// O(1) time complexity.
    ///
    /// # Arguments
    /// * `rand`: A mutable reference to the `Rand` struct containing the RNG state.
    ///
    /// # Returns
    /// A tuple `(index, probability)`, where `index` is the sampled index and `probability`
    /// is its corresponding value from the original normalized distribution (`self.sample_prob`).
    pub fn sample_with_prob(&self, rand: &mut Rand) -> (usize, f64) {
        let (i, r) = self.roll_die_and_flip_coin(rand);
        return self.select_element_and_prob(i, r);
    }

    /// Performs the two random steps of the Alias method: selecting a uniform column index
    /// and generating a uniform random float for comparison.
    /// Internal helper function.
    fn roll_die_and_flip_coin(&self, rand: &mut Rand) -> (usize, f64) {
        let i: usize = rand.rng.sample(self.uniform);
        let r: f64 = rand.rng.gen();
        return (i, r);
    }

    /// Selects the final element based on the die roll and coin flip.
    /// Internal helper function.
    fn select_element(&self, die: usize, coin: f64) -> usize {
        // choose randomly an element from the element vector
        return if coin < self.alias_prob[die] {
            die
        } else {
            self.alias[die]
        };
    }

    /// Selects the final element and retrieves its original probability.
    /// Internal helper function.
    fn select_element_and_prob(&self, die: usize, coin: f64) -> (usize, f64) {
        // choose randomly an element from the element vector
        return if coin < self.alias_prob[die] {
            (die, self.sample_prob[die])
        } else {
            let alias = self.alias[die];
            (alias, self.sample_prob[alias])
        };
    }

    /// Tests the sampler by drawing many samples and comparing frequencies to target probabilities.
    /// Prints the comparison to standard output.
    pub fn test(&self, n_samples: i32, rand: &mut Rand) { // Made immutable self
        // Take n_samples samples, show frequency of samples vs probability for full distribution
        let n = self.alias.len();
        let mut freq: Vec<i32> = vec![0; n];
        for _i_sample in 0..n_samples {
            let sample = self.sample(rand);
            freq[sample] += 1;
        }
        println!("Target prob, sampled prob, element");
        for i in 0..n {
            println!(
                "{:.6},   {:.6},   {}",
                self.sample_prob[i],
                (freq[i] as f64) / (n_samples as f64),
                i
            );
        }
    }
}
