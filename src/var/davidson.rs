// Module for just performing matrix-free Davidson
// For now, just use simple diagonal preconditioning

// use crate::wf::Wf;
// use itertools::enumerate;

// impl Wf {
//     pub fn optimize(&mut self, hb_eps: f64, dav_eps: f64) {
//         // Optimize coefficients of wf, and update its energy
//         let mut new_coeffs: Vec<f64> = vec![];
//         (self.energy, new_coeffs) = self.davidson(hb_eps, dav_eps);
//         for (i, new_coeff) in enumerate(new_coeffs.iter()) {
//             self.dets[i].coeff = *new_coeff;
//         }
//     }
//
//     fn davidson(&self, hb_eps: f64, dav_eps: f64) -> (f64, Vec<f64>) {
//         // Davidson algorithm; returns (eigenvalue, eigenvector) pair
//
//     }
// }
