//! # General Utilities (`utils`)
//!
//! This module provides various helper functions and submodules used throughout the `risq` crate.
//!
//! ## Submodules:
//! *   `bits`: Functions for bit manipulation, primarily for working with determinant configurations.
//! *   `display`: Implementations of `fmt::Display` for custom data structures.
//! *   `ints`: Utilities related to handling molecular integrals and orbital indices.
//! *   `iter`: Custom iterators, potentially for efficient looping over specific data structures or patterns.
//! *   `read_input`: Functionality for reading and parsing the `in.json` input file.

pub mod bits;
pub mod display;
pub mod ints;
pub mod iter;
pub mod read_input;
