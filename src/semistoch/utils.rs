use crate::excite::init::ExciteGenerator;
use crate::excite::iterator::double_excites;
use crate::excite::{Orbs, StoredExcite};
use crate::ham::Ham;
use crate::rng::Rand;
use crate::stoch::{matmul_sample_remaining, ImpSampleDist, ScreenedSampler};
use crate::wf::det::{Config, Det};
use crate::wf::Wf;
use rand::distributions::{Distribution, Uniform};
use rolling_stats::Stats;
use std::collections::HashMap;

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
        rand
    );

    match sampled_det_info {
        None => {
            println!(
                "Sampled excitation not valid! Sample prob = {}",
                sampled_prob
            );
        }
        Some((exciting_det, excite, target_det)) => {
            println!(
                "Sampled excitation: Sampled det = {}, Sample prob = {}, (H_ai c_i)^2 / p = {}",
                target_det,
                sampled_prob,
                excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff
                    / sampled_prob
            );
            let e_a: f64 = exciting_det.new_diag(ham, &excite);
            let energy: f64 = excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff
                / (e0 - e_a)
                / sampled_prob;
            enpt2_diag.update(energy);
        }
    }
}

/// Holds the off-diagonal samples: (E_a,
/// sum_i H_{ai} c_i w_i / p_i, sum_i (H_{ai} c_i w_i / p_i) ^ 2)
/// for each PT det
#[derive(Default)]
struct OffDiagSamples {
    pub n: i32,
    pub pt_energies_and_sums: HashMap<Config, (f64, f64, f64)>,
}

impl OffDiagSamples {
    /// Clear data structure to start collecting a new batch of samples
    // pub fn clear(&mut self) {
    //     self.n = 0;
    //     self.pt_energies_and_sums = Default::default();
    // }

    /// Add a (variational det, PT det) pair
    /// Note that all the relevant information is contained in pt_det, so var_det is not
    /// even needed except for efficiently computing new pt_det energies
    fn add_var_and_pt(
        &mut self,
        var_det: &Det,
        is_alpha: Option<bool>,
        init_orbs: Orbs,
        excite: &StoredExcite,
        pt_config: Config,
        w: i32,
        prob: f64,
        ham: &Ham,
    ) {
        // x_{ai} = H_{ai} c_i w_i / p_i
        let x_ai: f64 =
            ham.ham_off_diag_no_excite(&var_det.config, &pt_config) * var_det.coeff * (w as f64)
                / prob;
        match self.pt_energies_and_sums.get_mut(&pt_config) {
            None => {
                // compute diagonal element e_a
                let e_a: f64 = var_det.new_diag_stored(ham, is_alpha, init_orbs, excite);
                self.pt_energies_and_sums
                    .insert(pt_config, (e_a, x_ai, x_ai * x_ai));
            }
            Some((_, s, s_sq)) => {
                // update sum and sum_sq, but don't re-compute diagonal element
                *s += x_ai;
                *s_sq += x_ai * x_ai;
            }
        }
    }

    /// Add the PT samples corresponding to a new variational det
    pub fn add_new_var_det(
        &mut self,
        wf: &Wf,
        var_det: &Det,
        w: i32,
        prob: f64,
        excite_gen: &ExciteGenerator,
        ham: &Ham,
    ) {
        self.n += w;
        for (is_alpha, init_orbs, stored_excite) in double_excites(var_det, excite_gen, 1e-9) {
            let pt_config = var_det
                .config
                .apply_excite(is_alpha, init_orbs, stored_excite);
            if !wf.inds.contains_key(&pt_config) {
                self.add_var_and_pt(
                    var_det,
                    is_alpha,
                    init_orbs,
                    stored_excite,
                    pt_config,
                    w,
                    prob,
                    ham,
                );
            }
        }
    }

    /// Compute the off-diagonal contribution to the PT energy estimate
    /// using the stored samples
    pub fn pt_energy(&self, e0: f64) -> f64 {
        let mut e_pt: f64 = 0.0;
        for (e_a, s, s_sq) in self.pt_energies_and_sums.values() {
            e_pt += (*s * *s - *s_sq) / (e0 - e_a);
        }
        e_pt / (self.n as f64) / ((self.n - 1) as f64)
    }
}

/// Sample the 'off-diagonal' contribution to the PT energy (i.e., the term coming from the off-diagonal of
/// the matrix V \[1 / (E_0 - E_a)\] V), and update the Welford statistics for the energy
pub fn sample_off_diag_update_welford(
    wf: &Wf,
    excite_gen: &ExciteGenerator,
    ham: &Ham,
    n_samples_per_batch: i32,
    rand: &mut Rand,
    enpt2_off_diag: &mut Stats<f64>,
) {
    // Sample i with probability proportional to |c_i| max_a |H_{ai}|
    // Interestingly, this is approximately uniform!
    // We just use uniform sampling for now

    let mut counts: HashMap<usize, i32> = HashMap::default();
    let uniform_dist: Uniform<usize> = Uniform::from(0..wf.n);
    for _i_sample in 0..n_samples_per_batch {
        // Sample a variational det i, and add it to the samples data structure
        let i = uniform_dist.sample(&mut rand.rng);
        match counts.get_mut(&i) {
            None => {
                counts.insert(i, 1);
            }
            Some(w) => {
                *w += 1;
            }
        }
    }

    // Obtain var_dets and their counts {w}
    let mut off_diag: OffDiagSamples = OffDiagSamples::default();
    let prob: f64 = 1.0 / (wf.n as f64);
    for (i, w) in counts {
        off_diag.add_new_var_det(wf, &wf.dets[i], w, prob, excite_gen, ham);
    }
    assert_eq!(n_samples_per_batch, off_diag.n);

    let off_diag_estimate: f64 = off_diag.pt_energy(wf.energy);
    println!(
        "Sampled off-diagonal energy this batch: {:.4}",
        off_diag_estimate
    );
    enpt2_off_diag.update(off_diag_estimate);
}
