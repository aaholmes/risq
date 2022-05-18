//! Generate excitations as an iterator

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

pub fn test_dets_excites_and_excited_dets(wf: &Wf, excite_gen: &ExciteGenerator) {
    // println!("Reading input file");
    // lazy_static! {
    //     static ref GLOBAL: Global = read_input("in.json").unwrap();
    // }
    //
    // println!("Reading integrals");
    // lazy_static! {
    //     static ref HAM: Ham = read_ints(&GLOBAL, "FCIDUMP");
    // }
    //
    // println!("Initializing excitation generator");
    // lazy_static! {
    //     static ref EXCITE_GEN: ExciteGenerator = init_excite_generator(&GLOBAL, &HAM);
    // }
    //
    // println!("Initializing wavefunction");
    // let mut wf: VarWf = init_var_wf(&GLOBAL, &HAM, &EXCITE_GEN);
    // wf.print();

    for (n, (det, excite, excited_det)) in dets_excites_and_excited_dets(wf, excite_gen, 1e-9).enumerate() {
        println!("{}: Det: {}", n, det);
        println!("Excite: {}", excite);
        println!("Excited det: {}\n", excited_det);
    }

}

/// Iterate over all variational dets and their corresponding excitable orbs (along with is_alpha)
pub fn dets_and_excitable_orbs<'a>(
    wf: &'a Wf,
    excite_gen: &'a ExciteGenerator, // just needed to read the valence orbitals
) -> impl Iterator<Item = (usize, &'a Det, Option<bool>, Orbs)> {
    wf.dets
        .iter()
        .enumerate() // For each det
        .flat_map(move |(ind, det)| {
            repeat((ind, det)).zip(valence_elecs_and_epairs(&det.config, excite_gen))
        }) // For each electron and electron pair (det, (is_alpha, init_orbs))
        .map(move |((ind, det), (is_alpha, init))| (ind, det, is_alpha, init))
}

/// Iterate over all excites from the given variational det and excitable orb
/// (Does not check eps threshold, but does check whether valid excite)
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
                .take_while(move |((_is_alpha, _init), excite)| {
                    excite.abs_h * det.coeff.abs() >= eps
                })
            // Stop searching for excites if eps threshold reached
        })
        .filter(move |((is_alpha, _), excite)| det.config.is_valid_stored(is_alpha, excite)) // Filter excites to already-occupied orbitals
        .map(move |((is_alpha, init), excite)| (is_alpha, init, excite))
}

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
