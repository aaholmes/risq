//! Alias sampling
// Borrows heavily from vose-alias crate, but with some improvements:
// - Use vectors of indices to avoid all hash tables
// - Don't discretize probabilities to multiples of 1%

use rand::Rng;
use std::fmt::Debug;
// use rand::rngs::ThreadRng;
use crate::rng::Rand;
use rand::distributions::Uniform;

/// Alias sampling data structure
#[derive(Debug, Clone)]
pub struct Alias {
    // Probability of sampling each element
    pub sample_prob: Vec<f64>,
    // Internal components of sampling process
    alias: Vec<usize>,
    alias_prob: Vec<f64>,
    uniform: Uniform<usize>,
}

impl Alias {
    /// Generate new Alias struct given a vector of relative probabilities (not necessarily normalized)
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

    /// Sample an element from the Alias struct in O(1) time
    pub fn sample(&mut self, rand: &mut Rand) -> usize {
        let (i, r) = self.roll_die_and_flip_coin(rand);
        return self.select_element(i, r);
    }

    /// Sample an element and also return its sample probability
    pub fn sample_with_prob(&self, rand: &mut Rand) -> (usize, f64) {
        let (i, r) = self.roll_die_and_flip_coin(rand);
        return self.select_element_and_prob(i, r);
    }

    // This function 'rolls the unweighted die' (selects an element uniformly) and 'flips the weighted coin' (selects a uniform real between 0 and 1 for comparing to the probability of using the alias)
    fn roll_die_and_flip_coin(&self, rand: &mut Rand) -> (usize, f64) {
        let i: usize = rand.rng.sample(self.uniform);
        let r: f64 = rand.rng.gen();
        return (i, r);
    }

    // This function selects an element from the VoseAlias table given a die (a column) and a coin (the element or its alias). This function has been separated from the `sample` function to allow unit testing, but should never be called by itself.
    fn select_element(&self, die: usize, coin: f64) -> usize {
        // choose randomly an element from the element vector
        return if coin < self.alias_prob[die] {
            die
        } else {
            self.alias[die]
        };
    }

    // This function selects an element from the VoseAlias table given a die (a column) and a coin (the element or its alias). This function has been separated from the `sample` function to allow unit testing, but should never be called by itself.
    fn select_element_and_prob(&self, die: usize, coin: f64) -> (usize, f64) {
        // choose randomly an element from the element vector
        return if coin < self.alias_prob[die] {
            (die, self.sample_prob[die])
        } else {
            let alias = self.alias[die];
            (alias, self.sample_prob[alias])
        };
    }

    pub fn test(&mut self, n_samples: i32, rand: &mut Rand) {
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
