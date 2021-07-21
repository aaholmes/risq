// // Module for just performing matrix-free Davidson
// // For now, just use simple diagonal preconditioning
//
use crate::wf::Wf;
// use itertools::enumerate;
// use std::cmp::min;
// // use std::intrinsics::sqrtf64;
use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};
// use nalgebra::DMatrix;
// use crate::excite::{Orbs, Excite};
// use crate::wf::det::Config;
// use eigenvalues::algorithms::davidson::DavidsonError;
use crate::var::ham_gen::gen_dense_ham_connections;
// use crate::var::ham_gen::{gen_dense_ham_connections, gen_sparse_ham_partial};
use std::time::Instant;
// use std::intrinsics::offset;
// //use crate::wf::det::{Config, Det};
// //use crate::excite::{Excite, Orbs};
// //use crate::utils::bits::{bits, bit_pairs};
// use nalgebra::linalg::SymmetricEigen;
// //use sprs::io::SymmetryMode::Symmetric;
// use nalgebra::DMatrix;

pub fn dense_optimize(wf: &mut Wf, coeff_eps: f64, energy_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) {
    // Generate Ham as a dense matrix
    // Optimize using davidson
    // Just to get something working - need to replace with efficient algorithm soon!

    let start: Instant = Instant::now();
    let ham_matrix = gen_dense_ham_connections(wf, ham, excite_gen);
    println!("Time for Ham gen with dim = {}: {:?}", wf.n, start.elapsed());

    if wf.n <= 8 {
        println!("H to diagonalize: {}", ham_matrix);
    }

    // Davidson
    let dav = Davidson::new (ham_matrix, 1, DavidsonCorrection::DPR, SpectrumTarget::Lowest, coeff_eps, energy_eps );
    match dav {
        Ok(eig) => {
            wf.energy = eig.eigenvalues[0];
            for i in 0..wf.n {
                wf.dets[i].coeff = eig.eigenvectors[(i, 0)];
            }
        }
        Err(err) => {
            println!("Error! {}", err);
        }
    }
}

// pub fn sparse_optimize(wf: &mut Wf, coeff_eps: f64, energy_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) {
//     // Generate Ham as a sparse matrix
//     // Optimize using davidson
//
//     let ham_matrix = gen_sparse_ham_partial(wf, ham, excite_gen);
//
//     // Davidson
//     let dav = Davidson::new (ham_matrix, 1, DavidsonCorrection::DPR, SpectrumTarget::Lowest, coeff_eps, energy_eps );
//     match dav {
//         Ok(eig) => {
//             wf.energy = eig.eigenvalues[0];
//             for i in 0..wf.n {
//                 wf.dets[i].coeff = eig.eigenvectors[(i, 0)];
//             }
//         }
//         Err(err) => {
//             println!("Error! {}", err);
//         }
//     }
// }


// pub fn optimize(&mut wf: Wf, hb_eps: f64, dav_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) {
//     // Optimize coefficients of wf, and update its energy
//     let mut new_coeffs: Vec<f64> = vec![];
//     (wf.energy, new_coeffs) = davidson(wf, hb_eps, dav_eps, ham, excite_gen);
//     for (i, new_coeff) in enumerate(new_coeffs.iter()) {
//         wf.dets[i].coeff = *new_coeff;
//     }
// }

// fn davidson(wf: &Wf, hb_eps: f64, dav_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) -> (f64, Vec<f64>) {
//     // Lanczos/Davidson algorithm; returns (eigenvalue, eigenvector) pair
//
//     // Current energy (at the end of a given iteration)
//     let mut energy: f64 = 0.0;
//
//     // Previous energy (for checking convergence)
//     let mut last_energy: f64 = 0.0;
//
//     // Current solution vector
//     let mut w = vec![0.0f64; wf.n];
//     let mut hw = vec![0.0f64; wf.n];
//
//     // Max number of iterations can't be more than the number of states
//     let max_iters: i32 = min(20, wf.n_states);
//
//     // Krylov space vectors
//     let mut v = vec![vec![0.0f64; wf.n]; max_iters as usize];
//     let mut hv = vec![vec![0.0f64; wf.n]; max_iters as usize];
//
//     // Krylov Hamiltonian
//     let mut h_krylov: DMatrix::<f64> = DMatrix::zeros(max_iters as usize, max_iters as usize);
//
//     // Initialize: Start with v[0] = wf.coeffs
//     for (i, det) in enumerate(wf.dets.iter()) {
//         v[0][i] = det.coeff;
//     }
//
//     // Loop over iterations:
//     // Multiply v[iter] by H to get hv[iter]
//     // Compute Krylov H, diagonalize it
//     // If energy has not changed, converged (exit)
//     // Else, update current solution vector w (and hw)
//     // Compute new krylov vector (diagonally- or heat-bath preconditioned), orthonormalize
//
//     for iter in 0..max_iters {
//
//         // Normalize v[iter]
//         let mut norm: f64 = v[iter].dot(v[iter]);
//         norm = norm.sqrt();
//         for i in 0..wf.n {
//             v[iter][i] = v[iter][i] / norm;
//         }
//
//         // Multiply v[iter] by H to get hv[iter]
//         hv[iter] = wf.approx_matmul_variational(v[iter], ham, excite_gen, hb_eps);
//
//         // Compute Krylov H, diagonalize it
//         // TODO: examine whether, for heat-bath approximated Hv's, it is more accurate to average over all of the v*Hv components
//
//         h_krylov[(iter, iter)] = v[iter].dot(hv[iter]);
//
//         for i in 0..iter {
//             h_krylov[(i, iter)] = v[iter as usize].dot(hv[i]);
//             h_krylov[(iter, i)] = h_krylov[(i, iter)];
//         }
//
//         // Diagonalize it, update w (and hw) as the solution
//         let mut eigenvec;
//         {
//             const SIZE: usize = iter;
//             let mut eigen: SymmetricEigen<f64, SIZE> = SymmetricEigen::new(h_krylov[(..iter + 1, ..iter + 1)]);
//             // let mut eigen: SymmetricEigen<f64, iter> = SymmetricEigen::new(h_krylov[.. iter + 1][.. iter + 1]);
//             energy = eigen.eigenvalues[0];
//             eigenvec = eigen.eigenvectors[0];
//         }
//         println!("Eigenvalue: {}", energy);
//
//         // If energy has not changed, converged (exit)
//         if iter > 0 {
//             if (energy - last_energy).abs() < 1e-9f64 { break; }
//         }
//
//         // Else, update current solution vector w (and hw)
//         w = eigenvec.dot(&v[.. (iter + 1) as usize]);
//         hw = eigenvec.dot(&hv[.. (iter + 1) as usize]);
//
//         last_energy = energy.unwrap();
//         // Compute new krylov vector (diagonally  or heat-bath preconditioned), orthogonalize with Gram-Schmidt
//          let krylov_type = KrylovType::Davidson;
//          v[(iter + 1) as usize] = next_krylov(wf, w, hw, energy, krylov_type, ham, excite_gen);
//
//         // Orthogonalize
//         v[(iter + 1) as usize] = orthogonalize(&v, iter);
//     }
//
//
//     todo!()
// }
//
// pub enum KrylovType {
//     Lanczos,
//     Davidson,
//     Heatbath(f64) // Large eps value to use here in approxiating (E0 - H)^-1
// }
//
// pub fn next_krylov(wf: Wf, w: Vec<f64>, hw: Vec<f64>, e0: f64, krylov_type: KrylovType, ham: &Ham, excite_gen: &ExciteGenerator) -> Vec<f64> {
//     // Generate the next Krylov vector using one of three algorithms
//     match krylov_type {
//         KrylovType::Lanczos => {
//             // v[iter + 1] = Hv[iter]
//             hw
//         }
//         KrylovType::Davidson => {
//             // v[iter + 1] = (Hv[iter] - E0 v[iter]) / (H_diag - E0)
//             let mut out = vec![0.0f64; wf.n];
//             for i in 0..wf.n {
//                 out[i] = (hw[i] - e0 * w[i]) / (e0 - wf.dets[i].diag);
//             }
//             out
//         }
//         KrylovType::Heatbath(hb_eps) => {
//             // Use 1 / (1 - eps) ~ 1 + eps:
//             // 1 / (E_0 - H_diag - H_off) = 1 / (E_0 - H_diag) * 1 / (1 - H_off / (E_0 - H_diag))
//             // ~ 1 / (E_0 - H_diag) * 1 + H_off / (E_0 - H_diag)
//             // v[iter + 1] = (E0 v[iter] - Hv[iter]) * (1 / (E_0 - H_diag) + H_off / (E_0 - H_diag) ^ 2),
//             // where H_off is heatbath-approximated so it can be performed efficiently
//             let mut out = wf.approx_matmul_variational(&w, ham, excite_gen, hb_eps);
//             for i in 0..wf.n {
//                 out[i] = (hw[i] - e0 * w[i]) / (e0 - wf.dets[i].diag)
//                     * (1.0f64 + (out[i] - wf.dets[i].diag * w[i]) / (e0 - wf.dets[i].diag)); // subtract off the diagonal in the numerator
//             }
//             out
//         }
//     }
// }
//
// pub fn orthogonalize(v: &Vec<Vec<f64>>, iter: i32) -> Vec<f64> {
//     // Orthogonalize v[iter] with respect to all previous v
//     todo!()
// }
//
// // DELETE FOLLOWING (already in wf/mod.rs)
//
// // pub fn approx_matmul_variational(n: usize, &dets: Vec<Det>, &input_coeffs: Vec<f64>, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> Vec<f64> {
// //     // Approximate matrix-vector multiplication within variational space only
// //     // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
// //     // they meet the cutoff is as expensive as actually calculating the matrix element)
// //     let mut local_eps: f64;
// //     let mut excite: Excite;
// //     let mut new_det: Option<Config>;
// //
// //     // Diagonal component
// //     let mut out: Vec<f64> = vec![0.0f64; n];
// //     for (i, det) in enumerate(dets.iter()) {
// //         out[i] = det.diag * det.coeff;
// //     }
// //
// //     // Off-diagonal component
// //
// //     // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
// //     // only add if it is already in variational wf
// //     for det in dets {
// //         local_eps = eps / det.coeff.abs();
// //         // Double excitations
// //         // Opposite spin
// //         if excite_gen.max_opp_doub >= local_eps {
// //             for i in bits(det.config.up) {
// //                 for j in bits(det.config.dn) {
// //                     for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
// //                         if stored_excite.abs_h < local_eps { break; }
// //                         excite = Excite {
// //                             init: Orbs::Double((i, j)),
// //                             target: stored_excite.target,
// //                             abs_h: stored_excite.abs_h,
// //                             is_alpha: None
// //                         };
// //                         new_det = det.config.safe_excite_det(&excite);
// //                         match new_det {
// //                             Some(d) => {
// //                                 // Valid excite: add to H*psi
// //                                 match self.inds.get(&d) {
// //                                     // Compute matrix element and add to H*psi
// //                                     // TODO: Do this in a cache efficient way
// //                                     Some(ind) => {
// //                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
// //                                     },
// //                                     _ => {}
// //                                 }
// //                             }
// //                             None => {}
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //
// //         // Same spin
// //         if excite_gen.max_same_doub >= local_eps {
// //             for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
// //                 for (i, j) in bit_pairs(*config) {
// //                     for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
// //                         if stored_excite.abs_h < local_eps { break; }
// //                         excite = Excite {
// //                             init: Orbs::Double((i, j)),
// //                             target: stored_excite.target,
// //                             abs_h: stored_excite.abs_h,
// //                             is_alpha: Some(*is_alpha)
// //                         };
// //                         new_det = det.config.safe_excite_det(&excite);
// //                         match new_det {
// //                             Some(d) => {
// //                                 // Valid excite: add to H*psi
// //                                 // Valid excite: add to H*psi
// //                                 match self.inds.get(&d) {
// //                                     // Compute matrix element and add to H*psi
// //                                     // TODO: Do this in a cache efficient way
// //                                     Some(ind) => {
// //                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
// //                                     },
// //                                     _ => {}
// //                                 }
// //                             }
// //                             None => {}
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //
// //         // Single excitations
// //         if excite_gen.max_sing >= local_eps {
// //             for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
// //                 for i in bits(*config) {
// //                     for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
// //                         if stored_excite.abs_h < local_eps { break; }
// //                         excite = Excite {
// //                             init: Orbs::Single(i),
// //                             target: stored_excite.target,
// //                             abs_h: stored_excite.abs_h,
// //                             is_alpha: Some(*is_alpha)
// //                         };
// //                         new_det = det.config.safe_excite_det(&excite);
// //                         match new_det {
// //                             Some(d) => {
// //                                 // Valid excite: add to H*psi
// //                                 match self.inds.get(&d) {
// //                                     // Compute matrix element and add to H*psi
// //                                     // TODO: Do this in a cache efficient way
// //                                     Some(ind) => {
// //                                         out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
// //                                     },
// //                                     _ => {}
// //                                 }
// //                             }
// //                             None => {}
// //                         }
// //                     }
// //                 }
// //             }
// //         }
// //     } // for det in self.dets
// //
// //     out
// // }
