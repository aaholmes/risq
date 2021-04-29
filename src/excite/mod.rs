// Excitation generation module:
// Includes sorted excitations for heat-bath algorithm

pub mod init;
pub mod iterator;

use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::hash::Hash;

use crate::utils::bits::bit_pairs;
use crate::wf::det::{Config, Det};

use super::ham::Ham;
use super::utils::read_input::Global;

// Orbital pair
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OPair(pub i32, pub i32);

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    pub(crate) init: OPair, // For now, store the initial pair here too
    pub(crate) target: OPair,
    pub(crate) abs_h: f64,
}

// Single excitation doublet (r, max |H|)
pub struct Sing {
    pub(crate) init: i32, // Store init as in Doub
    pub(crate) target: i32,
    pub(crate) max_abs_h: f64,
}

// Excitation (unifies single and double excitations into one enum)
pub enum Excite {
    Single(Sing),
    Double(Doub),
}
