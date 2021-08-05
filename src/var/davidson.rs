// Davidson module
// For now, just use simple diagonal preconditioning

use crate::wf::Wf;
use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget};
use crate::var::ham_gen::{gen_sparse_ham_doubles, gen_sparse_ham_fast};
use std::time::Instant;
use crate::utils::read_input::Global;
use eigenvalues::algorithms::davidson::DavidsonError;
use nalgebra::DMatrix;
use crate::excite::Excite;


// pub fn dense_optimize(wf: &mut Wf, coeff_eps: f64, energy_eps: f64, ham: &Ham, excite_gen: &ExciteGenerator) {
//     // Generate Ham as a dense matrix
//     // Optimize using davidson
//     // Just to get something working - need to replace with efficient algorithm soon!
//
//     let start: Instant = Instant::now();
//     let ham_matrix = gen_dense_ham_connections(wf, ham, excite_gen);
//     println!("Time for Ham gen with dim = {}: {:?}", wf.n, start.elapsed());
//
//     if wf.n <= 8 {
//         println!("H to diagonalize: {}", ham_matrix);
//     }
//
//     // Davidson
//     let dav = Davidson::new (ham_matrix, 1, None, DavidsonCorrection::DPR, SpectrumTarget::Lowest, coeff_eps, energy_eps );
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

pub fn sparse_optimize(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf, coeff_eps: f64, energy_eps: f64, init_last_iter: bool) {
    // Generate Ham as a sparse matrix
    // Optimize using davidson

    let start_gen_sparse_ham: Instant = Instant::now();
    // let sparse_ham = gen_sparse_ham_doubles(wf, ham, excite_gen);
    let sparse_ham = gen_sparse_ham_fast(global, wf, ham, false);
    println!("Time to generate sparse H: {:?}", start_gen_sparse_ham.elapsed());

    // Davidson
    let start_dav: Instant = Instant::now();
    let mut dav: Result<Davidson, DavidsonError>;
    let init: Option<DMatrix<f64>> = {
        if init_last_iter { // Use inital guess
            let mut init = DMatrix::from_vec(wf.n, 1, vec![0.0; wf.n]);
            for (i, det) in wf.dets.iter().enumerate() {
                init[(i, 0)] = det.coeff;
            };
            Some(init)
        } else { // No initial guess
            None
        }
    };
    dav = Davidson::new(
        sparse_ham, 1, init, DavidsonCorrection::DPR,
        SpectrumTarget::Lowest, coeff_eps,
        energy_eps
    );
    println!("Time to perform Davidson diagonalization: {:?}", start_dav.elapsed());

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
