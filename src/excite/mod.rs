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


// Excitations
pub enum Excite {
    Double(Doub),
    Single(Sing)
}

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    pub init: OPair, // For now, store the initial pair here too
    pub target: OPair,
    pub abs_h: f64,
    pub is_alpha: Option<bool>, // if None, then either opposite spin double or hasn't been set yet
}

// Single excitation doublet (r, max |H|)
pub struct Sing {
    pub init: i32, // Store init as in Doub
    pub target: i32,
    pub max_abs_h: f64,
    pub is_alpha: Option<bool>, // if None, then either opposite spin double or hasn't been set yet
}


// Simplified excitations for storing in heat-bath tensor
pub enum StoredExcite {
    Double(StoredDoub),
    Single(StoredSing)
}

// Double excitation triplet (r, s, |H|)
pub struct StoredDoub {
    pub target: OPair,
    pub abs_h: f64,
}

// Single excitation doublet (r, max |H|)
pub struct StoredSing {
    pub target: i32,
    pub max_abs_h: f64,
}