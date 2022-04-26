//! Generate excitations as an iterator

use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::utils::bits::{valence_elecs, valence_elecs_and_epairs, valence_epairs};
use crate::wf::det::{Config, Det};
use crate::wf::Wf;
use std::iter::repeat;

/// Iterate over all dets and their single and double excitations that are at least as large in magnitude as eps,
/// returns (exciting det, excitation, excited config)
/// Includes single excitations that may or may not meet the threshold
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
                .take_while(move |((det, is_alpha, init), excite)| {
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
        .filter(move |(det, is_alpha, init, excite, excited_det)| {
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

// /// Same as dets_excites_and_excited_dets, but only generates excites in this batch
// pub fn dets_excites_and_excited_dets_batch<'a>(
//     wf: &'a Wf,
//     excite_gen: &'a ExciteGenerator,
//     eps: f64,
//     batch: usize,
// ) -> impl Iterator<Item = (&'a Det, Excite, Config)> {
//     wf.dets.iter() // For each det
//         .flat_map(move |det| (det, valence_elecs_and_epairs(&det.config, excite_gen))) // For each electron and electron pair
//         .flat_map(move |(det, (is_alpha, init))| {
//             excite_gen
//                 .excites_from_batch(&det.config, (is_alpha, &init), batch) // For each excite
//                 .take_while(move |excite| excite.abs_h * det.coeff.abs() >= eps) // Stop searching for excites if eps threshold reached
//                 .filter(move |excite| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
//                 .flat_map(move |excite| det.apply_excite(is_alpha, init, excite)) // Compute excited determinant configuration
//                 .filter(move |excited_det| !wf.dets.contains(&excited_det)) // Filter excites to determinants in the exciting wavefunction
//                 .map(move |excite| (det, Excite{
//                     is_alpha,
//                     init,
//                     target: excite.target,
//                     abs_h: excite.abs_h
//                 }, excited_det))
//         })
// }

/// Iterate over all single and double excitations from the given determinant (only excitations whose matrix elements
/// are larger in magnitude than eps)
pub fn excites<'a>(
    det: &'a Det,
    excite_gen: &'a ExciteGenerator,
    eps: f64,
) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite)> {
    valence_elecs_and_epairs(&det.config, excite_gen) // For each electron and electron pair (is_alpha, init_orbs)
        .flat_map(move |(is_alpha, init)| {
            repeat((is_alpha, init))
                .zip(excite_gen.excites_from((is_alpha, &init))) // For each (det, excite)
                .take_while(move |((is_alpha, init), excite)| {
                    excite.abs_h * det.coeff.abs() >= eps
                }) // Stop searching for excites if eps threshold reached
        })
        .filter(move |((is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
        .map(move |((is_alpha, init), excite)| {
            (is_alpha, init, excite)
        })
}

/// Iterate over all single and double excitations from the given determinant (only excitations whose matrix elements
/// are larger in magnitude than eps)
// pub fn excites<'a>(
//     det: &'a Det,
//     excite_gen: &'a ExciteGenerator,
//     eps: f64,
// ) -> impl Iterator<Item = Excite> {
//     valence_elecs_and_epairs(&det.config, excite_gen) // For each electron and electron pair (is_alpha, init_orbs)
//         .flat_map(move |(is_alpha, init)| {
//             repeat((is_alpha, init))
//                 .zip(excite_gen.excites_from((is_alpha, &init))) // For each (det, excite)
//                 .take_while(move |((is_alpha, init), excite)| {
//                     excite.abs_h * det.coeff.abs() >= eps
//                 }) // Stop searching for excites if eps threshold reached
//         })
//         .filter(move |((is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
//         .map(move |((is_alpha, init), excite)| {
//                 Excite {
//                     is_alpha,
//                     init,
//                     target: excite.target,
//                     abs_h: excite.abs_h,
//                 } // Convert StoredExcite to Excite
//         })
// }

/// Iterate over double excitations from the given determinant (only excitations whose matrix elements
/// are larger in magnitude than eps)
// pub fn double_excites<'a>(
//     det: &'a Det,
//     excite_gen: &'a ExciteGenerator,
//     eps: f64,
// ) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite)> {
//     valence_epairs(&det.config, excite_gen).flat_map(move |(is_alpha, orbs)| {
//         excite_gen
//             .excites_from((is_alpha, &orbs))
//             .take_while(move |excite| excite.abs_h * det.coeff.abs() >= eps)
//             .filter(move |excite| det.config.is_valid_stored(is_alpha, excite))
//             .map(move |excite| (is_alpha, orbs, excite))
//     })
// }

/// Iterate over double excitations from the given determinant (only excitations whose matrix elements
/// are larger in magnitude than eps)
/// Includes first skipped double excitation, i.e., the first one that doesn't meet the threshold
// pub fn double_excites_incl_first_skipped<'a>(
//     det: &'a Det,
//     excite_gen: &'a ExciteGenerator,
//     eps: f64,
// ) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite, bool)> {
//     valence_epairs(&det.config, excite_gen).flat_map(move |(is_alpha, orbs)| {
//         excite_gen
//             .excites_from((is_alpha, &orbs))
//             .take_while(move |excite| excite.abs_h * det.coeff.abs() >= eps)
//             .filter(move |excite| det.config.is_valid_stored(is_alpha, excite))
//             .map(move |excite| (is_alpha, orbs, excite, true))
//     })
// }

/// Iterate over single excitations from the given determinant (only excitations whose max possible matrix elements
/// are larger in magnitude than eps)
// pub fn single_excites<'a>(
//     det: &'a Det,
//     excite_gen: &'a ExciteGenerator,
//     eps: f64,
// ) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite)> {
//     valence_elecs(&det.config, excite_gen).flat_map(move |(is_alpha, orbs)| {
//         excite_gen
//             .excites_from((is_alpha, &orbs))
//             .take_while(move |excite| excite.abs_h * det.coeff.abs() >= eps)
//             .filter(move |excite| det.config.is_valid_stored(is_alpha, excite))
//             .map(move |excite| (is_alpha, orbs, excite))
//     })
// }

impl ExciteGenerator {
    /// Iterate over the excitations from these orbitals
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

    // Iterate over the excitations from these orbitals, using only excitations in this batch
    // Needs the det's config as an input because batches are determined by generated PT dets
//     fn excites_from_batch(
//         &self,
//         det: &Config,
//         init: (Option<bool>, &Orbs),
//         batch: usize,
//         n_batches: usize,
//     ) -> impl Iterator<Item = &StoredExcite> {
//         todo!()
//         // let batch_excite: usize = (det.get_batch() + batch) % n_batches;
//         // match init.1 {
//         //     Orbs::Double(_) => {
//         //         match init.0 {
//         //             None => {
//         //                 // Opposite-spin double
//         //                 self.opp_doub_sorted_list.get(init.1).unwrap().iter()
//         //             }
//         //             Some(_) => {
//         //                 // Same-spin double
//         //                 self.same_doub_sorted_list.get(init.1).unwrap().iter()
//         //             }
//         //         }
//         //     }
//         //     Orbs::Single(_) => {
//         //         // Single
//         //         self.sing_sorted_list.get(init.1).unwrap().iter()
//         //     }
//         // }
//     }
// }

// #[cfg(test)]
// mod tests {
//
//     use super::*;
//     use crate::excite::init::init_excite_generator;
//     use crate::ham::read_ints::read_ints;
//     use crate::ham::Ham;
//     use crate::utils::read_input::{read_input, Global};
//
//     #[test]
//     fn test_iter() {
//         println!("Reading input file");
//         lazy_static! {
//             static ref GLOBAL: Global = read_input("in.json").unwrap();
//         }
//
//         println!("Reading integrals");
//         lazy_static! {
//             static ref HAM: Ham = read_ints(&GLOBAL, "FCIDUMP");
//         }
//
//         println!("Initializing excitation generator");
//         lazy_static! {
//             static ref EXCITE_GEN: ExciteGenerator = init_excite_generator(&GLOBAL, &HAM);
//         }
//
//         let det = Det {
//             config: Config { up: 3, dn: 3 },
//             coeff: 1.0,
//             diag: None,
//         };
//
//         let eps = 0.1;
//         println!("About to iterate!");
//         for excite in EXCITE_GEN.truncated_excites(det, eps) {
//             println!("Got here");
//         }
//     }
// }
