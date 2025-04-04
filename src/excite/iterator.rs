//! # Excitation Iterators (`excite::iterator`)
//!
//! This module provides convenient iterators for generating excitations based on the
//! pre-computed data in `ExciteGenerator`. These iterators handle the logic of
//! iterating through determinants, initial orbitals, and target orbitals, applying
//! screening thresholds and validity checks.

use crate::ham::Ham;
use crate::ham::read_ints::read_ints;
use crate::utils::read_input::{read_input, Global};
use crate::excite::init::{init_excite_generator, ExciteGenerator};
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::utils::bits::valence_elecs_and_epairs;
use crate::wf::det::{Config, Det};
use crate::wf::{init_var_wf, VarWf, Wf};
use std::iter::repeat;
use std::time::Instant;
use crate::var::variational;

/// Creates an iterator yielding deterministically significant external excitations.
///
/// Iterates through each determinant (`det`) in the input wavefunction `wf`.
/// For each `det`, it iterates through all possible single and double excitations
/// originating from its occupied valence orbitals, using the pre-sorted lists in `excite_gen`.
///
/// It yields tuples `(&det, excite, excited_config)` only for excitations that satisfy:
/// 1. The estimated contribution `|H_ij * c_i|` is greater than or equal to `eps`.
///    (Note: The `take_while` implies screening based on `|H_ij| * |c_i| >= eps`, assuming `abs_h` is `|H_ij|`).
/// 2. The excitation is valid (target orbitals are unoccupied in `det`).
/// 3. The resulting `excited_config` is *not* already present in the input `wf` (i.e., it's external).
///
/// The `excite` object in the output tuple is a full `Excite` struct, reconstructed from the `StoredExcite`.
///
/// # Arguments
/// * `wf`: The source wavefunction.
/// * `excite_gen`: The pre-computed excitation generator.
/// * `eps`: The screening threshold for deterministic treatment.
pub fn dets_excites_and_excited_dets<'a>(
    wf: &'a Wf,
    excite_gen: &'a ExciteGenerator,
    eps: f64,
) -> impl Iterator<Item = (&'a Det, Excite, Config)> {
    wf.dets
        .iter() // For each det
        .flat_map(move |det| repeat(det).zip(valence_elecs_and_epairs(&det.config, excite_gen))) // For each electron and electron pair (det, (is_alpha, init_orbs))
        .flat_map(move |(det, (is_alpha, init))| {
            repeat((det, is_alpha, init))
                .zip(excite_gen.excites_from((is_alpha, &init))) // For each (det, excite)
                .take_while(move |((det, _is_alpha, _init), excite)| {
                    excite.abs_h * det.coeff.abs() >= eps
                }) // Stop searching for excites if eps threshold reached
        })
        .filter(move |((det, is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
        .map(move |((det, is_alpha, init), excite)| {
            (
                det,
                is_alpha,
                init,
                excite,
                det.config.apply_excite(is_alpha, &init, excite),
            )
        }) // Compute excited determinant configuration
        .filter(move |(_det, _is_alpha, _init, _excite, excited_det)| {
            !wf.inds.contains_key(&excited_det)
        }) // Filter excites to determinants in the exciting wavefunction
        .map(move |(det, is_alpha, init, excite, excited_det)| {
            (
                det,
                Excite {
                    is_alpha,
                    init,
                    target: excite.target,
                    abs_h: excite.abs_h,
                },
                excited_det,
            ) // Convert StoredExcite to Excite
        })
}

// Removed test function `test_dets_excites_and_excited_dets`

/// Creates an iterator yielding tuples of (det_index, det, spin, initial_orbs).
///
/// This iterates through each determinant `det` in the wavefunction `wf` and, for each `det`,
/// yields all possible single and double *initial* orbital combinations (`init`) from its
/// occupied valence orbitals, along with the corresponding spin channel (`is_alpha`).
///
/// Useful as a starting point for generating all excitations *from* the wavefunction.
///
/// # Arguments
/// * `wf`: The source wavefunction.
/// * `excite_gen`: Used only to access the `valence` orbital mask.
pub fn dets_and_excitable_orbs<'a>(
    wf: &'a Wf,
    excite_gen: &'a ExciteGenerator,
) -> impl Iterator<Item = (usize, &'a Det, Option<bool>, Orbs)> {
    wf.dets
        .iter()
        .enumerate() // For each det
        .flat_map(move |(ind, det)| {
            repeat((ind, det)).zip(valence_elecs_and_epairs(&det.config, excite_gen))
        }) // For each electron and electron pair (det, (is_alpha, init_orbs))
        .map(move |((ind, det), (is_alpha, init))| (ind, det, is_alpha, init))
}

/// Creates an iterator yielding valid, external excitations from a specific determinant and initial orbital set.
///
/// Given a source determinant `det`, a spin channel `is_alpha`, and initial orbitals `init`,
/// this iterates through all potential target excitations (`StoredExcite`) stored in `excite_gen`.
///
/// It yields tuples `(&stored_excite, excited_config)` only for excitations that are:
/// 1. Valid (target orbitals are unoccupied in `det`).
/// 2. External (the resulting `excited_config` is not already in `wf`).
///
/// Note: This iterator does *not* apply the `eps` screening threshold.
///
/// # Arguments
/// * `det`: The source determinant.
/// * `is_alpha`: The spin channel of the excitation.
/// * `init`: The initial orbitals the excitation originates from.
/// * `wf`: The wavefunction (used to check if the target is external).
/// * `excite_gen`: The pre-computed excitation generator.
pub fn excites_from_det_and_orbs<'a>(
    det: &'a Det,
    is_alpha: Option<bool>,
    init: Orbs,
    wf: &'a Wf,
    excite_gen: &'a ExciteGenerator,
) -> impl Iterator<Item = (&'a StoredExcite, Config)> {
    repeat((det, is_alpha, init))
        .zip(excite_gen.excites_from((is_alpha, &init))) // For each (det, excite)
        .filter(move |((det, is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
        .map(move |((det, is_alpha, init), excite)| {
            (excite, det.config.apply_excite(is_alpha, &init, excite))
        }) // Compute excited determinant configuration
        .filter(move |(_excite, excited_det)| !wf.inds.contains_key(&excited_det))
    // Filter excites to determinants in the exciting wavefunction
}

// Removed commented-out batch function `dets_excites_and_excited_dets_batch`

/// Creates an iterator yielding deterministically significant excitations *from* a single determinant.
///
/// Given a source determinant `det`, iterates through all possible single and double
/// excitations originating from its occupied valence orbitals.
///
/// Yields tuples `(is_alpha, init_orbs, &stored_excite)` for excitations that satisfy:
/// 1. The estimated contribution `|H_ij * c_i|` is greater than or equal to `eps`.
/// 2. The excitation is valid (target orbitals are unoccupied in `det`).
///
/// This is useful for the HCI "search" step where new determinants are added based on `eps`.
///
/// # Arguments
/// * `det`: The source determinant.
/// * `excite_gen`: The pre-computed excitation generator.
/// * `eps`: The screening threshold.
pub fn excites<'a>(
    det: &'a Det,
    excite_gen: &'a ExciteGenerator,
    eps: f64,
) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite)> {
    valence_elecs_and_epairs(&det.config, excite_gen) // For each electron and electron pair (is_alpha, init_orbs)
        .flat_map(move |(is_alpha, init)| {
            repeat((is_alpha, init))
                .zip(excite_gen.excites_from((is_alpha, &init))) // For each (det, excite)
                .take_while(move |((_is_alpha, _init), excite)| {
                    excite.abs_h * det.coeff.abs() >= eps
                })
            // Stop searching for excites if eps threshold reached
        })
        .filter(move |((is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
        .map(move |((is_alpha, init), excite)| (is_alpha, init, excite))
}

impl ExciteGenerator {
    /// Helper function to get an iterator over `StoredExcite` for a given initial orbital set and spin.
    ///
    /// Takes `init = (is_alpha, initial_orbs)` and returns an iterator over the corresponding
    /// pre-sorted `Vec<StoredExcite>` stored in the `ExciteGenerator`'s HashMaps.
    /// Panics if the `initial_orbs` key is not found in the corresponding map.
    fn excites_from(&self, init: (Option<bool>, &Orbs)) -> impl Iterator<Item = &StoredExcite> {
        match init.1 {
            Orbs::Double(_) => {
                match init.0 {
                    None => {
                        // Opposite-spin double
                        self.opp_doub_sorted_list.get(init.1).unwrap().iter()
                    }
                    Some(_) => {
                        // Same-spin double
                        self.same_doub_sorted_list.get(init.1).unwrap().iter()
                    }
                }
            }
            Orbs::Single(_) => {
                // Single
                self.sing_sorted_list.get(init.1).unwrap().iter()
            }
        }
    }
}

// Removed commented-out batch function `excites_from_batch`
// Removed commented-out test module `tests`
