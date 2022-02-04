//! Generate excitations as an iterator

use crate::excite::init::ExciteGenerator;
use crate::excite::{Orbs, StoredExcite};
use crate::utils::bits::valence_epairs;
use crate::wf::det::Det;

/// Iterate over double excitations from the given determinant (only excitations whose matrix elements
/// are larger in magnitude than eps)
pub fn double_excites<'a>(
    det: &'a Det,
    excite_gen: &'a ExciteGenerator,
    eps: f64,
) -> impl Iterator<Item = (Option<bool>, Orbs, &'a StoredExcite)> {
    valence_epairs(&det.config, excite_gen).flat_map(move |(is_alpha, orbs)| {
        excite_gen
            .excites_from((is_alpha, &orbs))
            .take_while(move |excite| excite.abs_h * det.coeff.abs() >= eps)
            .filter(move |excite| det.config.is_valid_stored(is_alpha, excite))
            .map(move |excite| (is_alpha, orbs, excite))
    })
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
