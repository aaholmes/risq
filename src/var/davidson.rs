// Module for just performing matrix-free Davidson
// For now, just use simple diagonal preconditioning

use crate::wf::Wf;
use itertools::enumerate;
use std::cmp::min;
// use std::intrinsics::sqrtf64;
use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::wf::det::{Config, Det};
use crate::excite::{Excite, Orbs};
use crate::utils::bits::{bits, bit_pairs};

// pub fn optimize(&mut wf: Wf, hb_eps: f64, dav_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) {
//     // Optimize coefficients of wf, and update its energy
//     let mut new_coeffs: Vec<f64> = vec![];
//     (wf.energy, new_coeffs) = davidson(wf, hb_eps, dav_eps, ham, excite_gen);
//     for (i, new_coeff) in enumerate(new_coeffs.iter()) {
//         wf.dets[i].coeff = *new_coeff;
//     }
// }

// fn davidson(&wf: Wf, hb_eps: f64, dav_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) -> (f64, Vec<f64>) {
//     // Davidson algorithm; returns (eigenvalue, eigenvector) pair
//
//     // Current energy (at the end of a given iteration)
//     let energy: Option<f64> = None;
//
//     // Current solution vector
//     let mut w = vec![0.0f64; wf.n];
//     let mut hw = vec![0.0f64; wf.n];
//
//     // Max number of iterations can't be more than the number of states
//     let max_iters: i32 = min(20, wf.n_states);
//
//     // Krylov space vectors
//     let mut v = vec![vec![0.0f64; wf.n], max_iters];
//     let mut hv = vec![vec![0.0f64; wf.n], max_iters];
//
//     // Krylov Hamiltonian
//     let mut h_krylov  = vec![vec![0.0f64, max_iters], max_iters];
//
//     // Initialize: Start with v[0] = wf.coeffs
//     for (i, det) in enumerate(wf.dets.iter()) {
//         v[0][i] = det.coeff;
//     }
//
//     // Loop over iterations:
//     // Multiply v[iter] by H to get hv[iter] (TODO)
//     // Compute Krylov H, diagonalize it
//     // If energy has not changed, converged (exit)
//     // Else, update current solution vector w (and hw)
//     // Compute new krylov vector (diagonally- or heat-bath preconditioned), orthonormalize
//
//     for iter in 0..max_iters {
//
//         // Normalize v[iter]
//         let mut norm: f64 = 1.0; // dot_product(wf.dets.coeff, wf.dets.coeff);
//         norm = norm.sqrt();
//         for i in 0..wf.n {
//             v[iter][i] = v[iter][i] / norm;
//         }
//
//         // Multiply v[iter] by H to get hv[iter]
//         hv[iter] = wf.approx_matmul_variational(ham, excite_gen, hb_eps);
//
//         // Compute Krylov H, diagonalize it
//         // TODO: examine whether, for heat-bath approximated Hv's, it is more accurate to average over all of the v*Hv components
//
//         h_krylov[iter][iter] = dot_product(v[iter], hv[iter]);
//
//         for i in 0..iter {
//             h_krylov[i][iter] = dot_product(v[iter], hv[i]);
//             h_krylov[iter][i] = h_krylov[i][iter];
//         }
//
//         // Diagonalize it
//
//         // If energy has not changed, converged (exit)
//         // Else, update current solution vector w (and hw)
//         // Compute new krylov vector (diagonally  or heat-bath preconditioned), orthogonalize with Gram-Schmidt
//     }
//
//
//     todo!()
// }

// pub fn approx_matmul_variational(n: usize, &dets: Vec<Det>, &input_coeffs: Vec<f64>, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> Vec<f64> {
//     // Approximate matrix-vector multiplication within variational space only
//     // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
//     // they meet the cutoff is as expensive as actually calculating the matrix element)
//     let mut local_eps: f64;
//     let mut excite: Excite;
//     let mut new_det: Option<Config>;
//
//     // Diagonal component
//     let mut out: Vec<f64> = vec![0.0f64; n];
//     for (i, det) in enumerate(dets.iter()) {
//         out[i] = det.diag * det.coeff;
//     }
//
//     // Off-diagonal component
//
//     // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
//     // only add if it is already in variational wf
//     for det in dets {
//         local_eps = eps / det.coeff.abs();
//         // Double excitations
//         // Opposite spin
//         if excite_gen.max_opp_doub >= local_eps {
//             for i in bits(det.config.up) {
//                 for j in bits(det.config.dn) {
//                     for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
//                         if stored_excite.abs_h < local_eps { break; }
//                         excite = Excite {
//                             init: Orbs::Double((i, j)),
//                             target: stored_excite.target,
//                             abs_h: stored_excite.abs_h,
//                             is_alpha: None
//                         };
//                         new_det = det.config.safe_excite_det(&excite);
//                         match new_det {
//                             Some(d) => {
//                                 // Valid excite: add to H*psi
//                                 match self.inds.get(&d) {
//                                     // Compute matrix element and add to H*psi
//                                     // TODO: Do this in a cache efficient way
//                                     Some(ind) => {
//                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                             None => {}
//                         }
//                     }
//                 }
//             }
//         }
//
//         // Same spin
//         if excite_gen.max_same_doub >= local_eps {
//             for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
//                 for (i, j) in bit_pairs(*config) {
//                     for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
//                         if stored_excite.abs_h < local_eps { break; }
//                         excite = Excite {
//                             init: Orbs::Double((i, j)),
//                             target: stored_excite.target,
//                             abs_h: stored_excite.abs_h,
//                             is_alpha: Some(*is_alpha)
//                         };
//                         new_det = det.config.safe_excite_det(&excite);
//                         match new_det {
//                             Some(d) => {
//                                 // Valid excite: add to H*psi
//                                 // Valid excite: add to H*psi
//                                 match self.inds.get(&d) {
//                                     // Compute matrix element and add to H*psi
//                                     // TODO: Do this in a cache efficient way
//                                     Some(ind) => {
//                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                             None => {}
//                         }
//                     }
//                 }
//             }
//         }
//
//         // Single excitations
//         if excite_gen.max_sing >= local_eps {
//             for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
//                 for i in bits(*config) {
//                     for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
//                         if stored_excite.abs_h < local_eps { break; }
//                         excite = Excite {
//                             init: Orbs::Single(i),
//                             target: stored_excite.target,
//                             abs_h: stored_excite.abs_h,
//                             is_alpha: Some(*is_alpha)
//                         };
//                         new_det = det.config.safe_excite_det(&excite);
//                         match new_det {
//                             Some(d) => {
//                                 // Valid excite: add to H*psi
//                                 match self.inds.get(&d) {
//                                     // Compute matrix element and add to H*psi
//                                     // TODO: Do this in a cache efficient way
//                                     Some(ind) => {
//                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                             None => {}
//                         }
//                     }
//                 }
//             }
//         }
//     } // for det in self.dets
//
//     out
// }
