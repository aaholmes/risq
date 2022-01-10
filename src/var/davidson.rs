//! Davidson module
//! For now, just use simple diagonal preconditioning

use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::utils::read_input::Global;
use crate::var::eigenvalues::algorithms::davidson::{Davidson, DavidsonError};
use crate::var::eigenvalues::algorithms::{DavidsonCorrection, SpectrumTarget};
use crate::var::ham_gen::gen_sparse_ham_fast;
use crate::wf::Wf;
use nalgebra::DMatrix;
use std::time::Instant;

/// Generate Ham as a sparse matrix, and optimize using Davidson
pub fn sparse_optimize(
    global: &Global,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    wf: &mut Wf,
    coeff_eps: f64,
    energy_eps: f64,
    init_last_iter: bool,
) {

    let start_gen_sparse_ham: Instant = Instant::now();
    gen_sparse_ham_fast(global, wf, ham, excite_gen);
    println!(
        "Time to generate sparse H: {:?}",
        start_gen_sparse_ham.elapsed()
    );

    // Davidson
    let start_dav: Instant = Instant::now();
    let dav: Result<Davidson, DavidsonError>;
    let init: Option<DMatrix<f64>> = {
        if init_last_iter {
            // Use inital guess
            let mut init = DMatrix::from_vec(wf.n, 1, vec![0.0; wf.n]);
            for (i, det) in wf.dets.iter().enumerate() {
                init[(i, 0)] = det.coeff;
            }
            Some(init)
        } else {
            // No initial guess
            None
        }
    };
    // let eps_hpr = wf.eps;
    dav = Davidson::new(
        &wf.sparse_ham,
        1,
        init,
        DavidsonCorrection::DPR, //HPR(eps_hpr),
        SpectrumTarget::Lowest,
        coeff_eps,
        energy_eps,
        ham,
        excite_gen,
        wf,
    );
    println!(
        "Time to perform Davidson diagonalization: {:?}",
        start_dav.elapsed()
    );

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
