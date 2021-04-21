// Variational epsilon iterator (to attach to wf)
// Epsilon starts at the largest value that allows at least one double excitation from the initial
// wf, then drops by a factor of 2 every iteration until it reaches the target value set in the
// input file

use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::{ExciteGenerator, Doub, OPair, Sing};
use crate::utils::bits::{bits, btest, ibset, ibclr};
use std::cmp::max;


#[derive(Clone, Copy)]
pub struct Eps {
    next: f64,
    target: f64,
}

impl Iterator for Eps {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let curr: f64 = self.next;
        self.next = if self.next / 2.0 > self.target { self.next / 2.0 } else { self.target };
        Some(curr)
    }
}

impl Default for Eps {
    fn default() -> Self {
        Eps{ next: 0.0, target: 0.0}
    }
}
