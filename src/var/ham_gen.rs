// Variational Hamiltonian generation algorithms

use crate::excite::init::ExciteGenerator;
use crate::wf::Wf;
use crate::ham::Ham;
use nalgebra::DMatrix;
use crate::excite::{Excite, Orbs};
use crate::wf::det::Config;
use eigenvalues::utils::generate_random_sparse_symmetric;
use crate::utils::bits::{bit_pairs, bits};
use crate::var::sparse::SparseMat;

pub fn gen_dense_ham_connections(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> DMatrix<f64> {
    // Generate Ham as a dense matrix by using all connections to each variational determinant
    // Simplest algorithm, very slow

    let mut excite: Excite;
    let mut new_det: Option<Config>;

    // Generate Ham
    let mut ham_matrix = generate_random_sparse_symmetric(wf.n, wf.n, 0.0); //DMatrix::<f64>::zeros(wf.n, wf.n);
    for (i_det, det) in wf.dets.iter().enumerate() {

        // Diagonal element
        ham_matrix[(i_det, i_det)] = det.diag;

        // Double excitations
        // Opposite spin
        for i in bits(det.config.up) {
            for j in bits(det.config.dn) {
                for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                    excite = Excite {
                        init: Orbs::Double((i, j)),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: None
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        // Same spin
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for (i, j) in bit_pairs(*config) {
                for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                    excite = Excite {
                        init: Orbs::Double((i, j)),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: Some(*is_alpha)
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_doub(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        // Single excitations
        for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
            for i in bits(*config) {
                for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
                    excite = Excite {
                        init: Orbs::Single(i),
                        target: stored_excite.target,
                        abs_h: stored_excite.abs_h,
                        is_alpha: Some(*is_alpha)
                    };
                    new_det = det.config.safe_excite_det(&excite);
                    match new_det {
                        Some(d) => {
                            // Valid excite: add to H*psi
                            match wf.inds.get(&d) {
                                // Compute matrix element and add to H*psi
                                // TODO: Do this in a cache efficient way
                                Some(ind) => {
                                    if *ind > i_det as usize {
                                        ham_matrix[(i_det as usize, *ind)] = ham.ham_sing(&det.config, &d);
                                        ham_matrix[(*ind, i_det as usize)] = ham_matrix[(i_det as usize, *ind)];
                                    }
                                },
                                _ => {}
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }
    ham_matrix
}


// pub fn gen_sparse_ham_partial(wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator) -> SparseMat {
//     // Generate Ham as a sparse matrix by using partial connections (either as in "Fast SHCI" or in my faster notes)
//
//     let mut out: SparseMat = SparseMat::new(wf.n);
//     out
// }
