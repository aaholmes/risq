// Alias sampling module
// Borrows heavily from vose-alias crate, but with some improvements:
// - Use vectors of indices to avoid all hash tables
// - Don't discretize probabilities to multiples of 1%

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use rand::seq::SliceRandom;
use rand::Rng;
use rand::rngs::ThreadRng;
use rand::distributions::Uniform;

/////////////////////////////////////////////
// Structure Definition and Implementation //
/////////////////////////////////////////////
/// A structure containing the necessary Vose-Alias tables.
///
/// The structure contains the following attributes:
/// 1. A vector containing the elements to sample from
/// 2. The Alias table, created from the Vose-Alias initialization step
/// 3. The Probability table, created frmo the Vose-Alias initialization step
///
/// The structure is created by the function `vose_alias::new()`. See its documentation for more details.
///
/// Internally, the elements are used as indexes in `HashMap` and `Vec`. Therefore, the type `T` must implement the following traits:
/// - Copy
/// - Hash
/// - Eq
/// - Debug
#[derive(Debug, Clone)]
pub struct Alias {
    // Probability of sampling each element
    pub sample_prob: Vec<f64>,
    // Internal components of sampling process
    alias: Vec<usize>,
    alias_prob: Vec<f64>,
    rng: ThreadRng,
    uniform: Uniform<usize>,
}

impl Alias {
    /// Returns the Vose-Alias object containing the element vector as well as the alias and probability tables.
    ///
    /// The `element_vector` contains the list of elements that should be sampled from.
    /// The `probability_vector` contains the discrete probability distribution to be sampled with.
    /// `element_vector` and `probability_vector` should have the same size and `probability_vector` should describe a well-formed probability distribution.
    ///
    /// # Panics
    ///
    /// The function panics in two cases:
    /// 1. the `element_vector` and the `probability_vector` do not contain the same number of elements
    ///
    /// # Examples
    /// ```
    /// use vose_alias::VoseAlias;
    ///
    /// // Creates a Vose-Alias object from a list of Integer elements
    /// let va = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
    /// ```

    pub fn new(relative_probability_vector: Vec<f64>) -> Alias {
        let size = relative_probability_vector.len();

        // Normalize input probabilities
        let mut sum = 0.0;
        for p in &relative_probability_vector {
            sum = sum + p;
        }
        let mut norm_prob: Vec<f64> = Vec::with_capacity(size);
        for p in &relative_probability_vector {
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
            alias: alias,
            alias_prob: prob,
            rng: rand::thread_rng(),
            uniform: Uniform::from(0..size)
        }
    }

    /// Returns a sampled element from a previously created Vose-Alias object.
    ///
    /// This function uses a `VoseAlias` object previously created using the method `vose_alias::new()` to sample in linear time an element of type `T`.
    ///
    /// # Panics
    /// This function panics only if the lists created in `vose_alias::new()` are not correctly form, which would indicate a internal bug in the code.
    /// If your code panics while using this function, please fill in an issue report.
    ///
    /// # Examples
    /// ```
    /// use vose_alias::VoseAlias;
    ///
    /// // Samples an integer from a list and prints it.
    /// let va = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
    /// let element = va.sample();
    /// println!("{}", element);
    ///
    /// ```
    pub fn sample(&mut self) -> usize {
        let (i, r) = self.roll_die_and_flip_coin();
        return self.select_element(i, r);
    }

    /// Returns a sampled element from a previously created Vose-Alias object, as well as its sample probability
    ///
    /// This function uses a `VoseAlias` object previously created using the method `vose_alias::new()` to sample in linear time an element of type `T`.
    ///
    /// # Panics
    /// This function panics only if the lists created in `vose_alias::new()` are not correctly form, which would indicate a internal bug in the code.
    /// If your code panics while using this function, please fill in an issue report.
    ///
    /// # Examples
    /// ```
    /// use vose_alias::VoseAlias;
    ///
    /// // Samples an integer from a list and prints it.
    /// let va = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
    /// let (element, prob) = va.sample_with_prob();
    /// println!("{} sampled; its probability is {}", element, prob);
    ///
    /// ```
    pub fn sample_with_prob(&mut self) -> (usize, f64) {
        let (i, r) = self.roll_die_and_flip_coin();
        return self.select_element_and_prob(i, r);
    }

    /// This function 'rolls the unweighted die' (selects an element uniformly) and 'flips the weighted coin' (selects a uniform real between 0 and 1 for comparing to the probability of using the alias)
    fn roll_die_and_flip_coin(&mut self) -> (usize, f64) {
        let i: usize = self.rng.sample(self.uniform);
        let r: f64 = self.rng.gen();
        return (i, r);
    }

    /// This function selects an element from the VoseAlias table given a die (a column) and a coin (the element or its alias). This function has been separated from the `sample` function to allow unit testing, but should never be called by itself.
    fn select_element(&self, die: usize, coin: f64) -> usize {
        // choose randomly an element from the element vector
        if coin < self.alias_prob[die] {
            return die;
        } else {
            return self.alias[die];
        }
    }

    /// This function selects an element from the VoseAlias table given a die (a column) and a coin (the element or its alias). This function has been separated from the `sample` function to allow unit testing, but should never be called by itself.
    fn select_element_and_prob(&self, die: usize, coin: f64) -> (usize, f64) {
        // choose randomly an element from the element vector
        if coin < self.alias_prob[die] {
            return (die, self.sample_prob[die]);
        } else {
            let alias = self.alias[die];
            return (alias, self.sample_prob[alias]);
        }
    }

    pub fn test(&mut self, n_samples: i32) {
        // Take n_samples samples, show frequency of samples vs probability for full distribution
        let n = self.alias.len();
        let mut freq: Vec<i32> = vec![0; n];
        for _i_sample in 0..n_samples {
            let sample = self.sample();
            freq[sample] += 1;
        }
        println!("Target prob, sampled prob, element");
        for i in 0..n {
            println!("{:.6},   {:.6},   {}", self.sample_prob[i], (freq[i] as f64) / (n_samples as f64), i);
        }
    }
}

// ////////////////////////////
// // Traits Implementation  //
// ////////////////////////////
// impl<T> VoseAlias<T>
//     where
//         T: Display + Copy + Hash + Eq + Debug,
// {
//     pub fn print(&self) {
//         println!("Probability, Alias prob, Element, Alias");
//         for e in &self.elements {
//             println!("{},   {},   {},   {}",
//                      self.elem_prob.get(e).unwrap(),
//                      self.alias_prob.get(e).unwrap(),
//                      *e,
//                      match self.alias.get(e) {
//                          Some(alias ) => *alias,
//                          None => *e
//                      }
//             );
//         }
//     }
// }
//
// impl<T> PartialEq for VoseAlias<T>
//     where
//         T: Display + Copy + Hash + Eq + Debug,
// {
//     fn eq(&self, other: &Self) -> bool {
//         self.alias == other.alias
//     }
// }
//
// impl<T> Eq for VoseAlias<T> where T: Display + Copy + Hash + Eq + Debug {}
//
// ///////////
// // Tests //
// ///////////
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     ////////////////////////////////////////
//     // Tests of the Struct Implementation //
//     ////////////////////////////////////////
//     #[test]
//     fn construction_ok() {
//         VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
//     }
//
//     #[test]
//     #[should_panic]
//     fn size_not_ok() {
//         VoseAlias::new(vec![1, 2, 3], vec![0.5, 0.2, 0.2, 0.1]);
//     }
//
//     #[test]
//     // #[should_panic]
//     // fn sum_not_ok() {
//     //     VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.]);
//     // }
//     #[test]
//     #[should_panic]
//     fn new_empty_vectors() {
//         let element_vector: Vec<u16> = Vec::new();
//         let probability_vector: Vec<f64> = Vec::new();
//         VoseAlias::new(element_vector, probability_vector);
//     }
//
//     #[test]
//     fn test_roll_die_flip_coin() {
//         let element_vector = vec![1, 2, 3, 4];
//         let va = VoseAlias::new(element_vector.clone(), vec![0.5, 0.2, 0.2, 0.1]);
//         let (die, coin) = va.roll_die_and_flip_coin();
//         assert!(element_vector.contains(&die));
//         assert!(coin <= 100);
//     }
//
//     #[test]
//     fn test_select_element_ok() {
//         let va = VoseAlias::new(
//             vec![
//                 "orange",
//                 "yellow",
//                 "green",
//                 "turquoise",
//                 "grey",
//                 "blue",
//                 "pink",
//             ],
//             vec![0.125, 0.2, 0.1, 0.25, 0.1, 0.1, 0.125],
//         );
//         // column orange / alias yellow
//         let element = va.select_element("orange", 0);
//         assert!(element == "orange");
//         let element = va.select_element("orange", 0.87);
//         assert!(element == "orange");
//         let element = va.select_element("orange", 0.88);
//         assert!(element == "yellow");
//         let element = va.select_element("orange", 1);
//         assert!(element == "yellow");
//
//         // column yellow / no alias
//         let element = va.select_element("yellow", 0);
//         assert!(element == "yellow");
//         let element = va.select_element("yellow", 1);
//         assert!(element == "yellow");
//
//         // column green / alias turquoise
//         let element = va.select_element("green", 0);
//         assert!(element == "green");
//         let element = va.select_element("green", 0.7);
//         assert!(element == "green");
//         let element = va.select_element("green", 0.71);
//         assert!(element == "turquoise");
//         let element = va.select_element("green", 1);
//         assert!(element == "turquoise");
//
//         // column turquoise / alias yellow
//         let element = va.select_element("turquoise", 0);
//         assert!(element == "turquoise");
//         let element = va.select_element("turquoise", 0.72);
//         assert!(element == "turquoise");
//         let element = va.select_element("turquoise", 0.73);
//         assert!(element == "yellow");
//         let element = va.select_element("turquoise", 1);
//         assert!(element == "yellow");
//
//         // column grey / alias turquoise
//         let element = va.select_element("grey", 0);
//         assert!(element == "grey");
//         let element = va.select_element("grey", 0.7);
//         assert!(element == "grey");
//         let element = va.select_element("grey", 0.71);
//         assert!(element == "turquoise");
//         let element = va.select_element("grey", 1);
//         assert!(element == "turquoise");
//
//         // column blue / alias turquoise
//         let element = va.select_element("blue", 0);
//         assert!(element == "blue");
//         let element = va.select_element("blue", 0.7);
//         assert!(element == "blue");
//         let element = va.select_element("blue", 0.71);
//         assert!(element == "turquoise");
//         let element = va.select_element("blue", 1);
//         assert!(element == "turquoise");
//
//         // column pink / alias turquoise
//         let element = va.select_element("pink", 0);
//         assert!(element == "pink");
//         let element = va.select_element("pink", 0.87);
//         assert!(element == "pink");
//         let element = va.select_element("pink", 0.88);
//         assert!(element == "turquoise");
//         let element = va.select_element("pink", 1);
//         assert!(element == "turquoise");
//     }
//
//     #[test]
//     #[should_panic]
//     fn select_element_proba_too_high() {
//         let va = VoseAlias::new(
//             vec![
//                 "orange",
//                 "yellow",
//                 "green",
//                 "turquoise",
//                 "grey",
//                 "blue",
//                 "pink",
//             ],
//             vec![0.125, 0.2, 0.1, 0.25, 0.1, 0.1, 0.125],
//         );
//         va.select_element("yellow", 1.01);
//     }
//
//     #[test]
//     #[should_panic]
//     fn select_element_not_in_list() {
//         let va = VoseAlias::new(
//             vec![
//                 "orange",
//                 "yellow",
//                 "green",
//                 "turquoise",
//                 "grey",
//                 "blue",
//                 "pink",
//             ],
//             vec![0.125, 0.2, 0.1, 0.25, 0.1, 0.1, 0.125],
//         );
//         va.select_element("red", 1.0);
//     }
//
//     ///////////////////////////////////////
//     // Tests of the trait implementation //
//     ///////////////////////////////////////
//     #[test]
//     fn test_trait_equal() {
//         let va = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
//         let va2 = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
//         assert!(va == va2);
//     }
//
//     #[test]
//     fn test_trait_not_equali() {
//         let va = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.0, 0.3]);
//         let va2 = VoseAlias::new(vec![1, 2, 3, 4], vec![0.5, 0.2, 0.2, 0.1]);
//         assert!(va != va2);
//     }
// }
