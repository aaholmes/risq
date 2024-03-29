use crate::excite::init::ExciteGenerator;
use crate::ham::Ham;
use crate::rng::Rand;
use crate::stoch::{matmul_sample_remaining, ImpSampleDist, ScreenedSampler};
use rolling_stats::Stats;

/// Sample the 'diagonal' contribution to the PT energy (i.e., the term coming from the diagonal of
/// the matrix V \[1 / (E_0 - E_a)\] V), and update the Welford statistics for the energy
pub fn sample_diag_update_welford(
    screened_sampler: &ScreenedSampler,
    excite_gen: &ExciteGenerator,
    ham: &Ham,
    rand: &mut Rand,
    e0: f64,
    enpt2_diag: &mut Stats<f64>,
) {
    // Sample with probability proportional to (Hc)^2
    let (sampled_det_info, sampled_prob) = matmul_sample_remaining(
        screened_sampler,
        ImpSampleDist::HcSquared,
        excite_gen,
        ham,
        rand,
    );

    match sampled_det_info {
        None => {
            // Sampled excitation not valid
        }
        Some((exciting_det, excite, _target_det)) => {
            // println!(
            //     "Sampled excitation: Sampled det = {}, Sample prob = {}, (H_ai c_i)^2 / p = {}",
            //     target_det,
            //     sampled_prob,
            //     excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff
            //         / sampled_prob
            // );
            let e_a: f64 = exciting_det.new_diag(ham, &excite);
            let energy: f64 = excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff
                / (e0 - e_a)
                / sampled_prob;
            enpt2_diag.update(energy);
        }
    }
}
