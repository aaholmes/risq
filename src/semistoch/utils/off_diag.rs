use crate::excite::init::ExciteGenerator;
use crate::excite::{Orbs, StoredExcite};
use crate::ham::Ham;
use crate::rng::Rand;
use crate::stoch::{matmul_sample_remaining, ImpSampleDist, ScreenedSampler};
use crate::wf::det::{Config, Det};
use crate::wf::Wf;
use rand::distributions::{Distribution, Uniform};
use rolling_stats::Stats;
use std::collections::HashMap;
use crate::excite::iterator::excites;

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
    /// For single excitations, first compares matrix element to eps (since
    /// only an upper bound has been computed at this point)
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
        eps: f64,
    ) {
        let h_ai: f64 = ham.ham_off_diag_no_excite(&var_det.config, &pt_config);

        // If single excite, we only had max |h|, so may still need to reject here
        match excite.target {
            Orbs::Single(_) => {
                if (h_ai * var_det.coeff).abs() < eps {
                    return;
                }
            },
            _ => {},
        }
        assert!((h_ai * var_det.coeff).abs() >= eps, "Got an excitation smaller than eps!");

        // x_{ai} = H_{ai} c_i w_i / p_i
        let x_ai: f64 =
             h_ai * var_det.coeff * (w as f64)
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

    /// Add the PT samples corresponding to a new variational det (only for the off-diagonal term)
    pub fn add_new_var_det(
        &mut self,
        wf: &Wf,
        var_det: &Det,
        w: i32,
        prob: f64,
        excite_gen: &ExciteGenerator,
        ham: &Ham,
        eps: Option<f64>
    ) {
        self.n += w;
        let mut eps_local: f64 = 1e-9; // This epsilon should be effectively zero, not the usual eps_var or eps_pt_dtm
        if let Some(e) = eps {
            eps_local = eps.unwrap();
        }
        for (is_alpha, init_orbs, stored_excite) in excites(var_det, excite_gen, eps_local) {
            let pt_config = var_det
                .config
                .apply_excite(is_alpha, &init_orbs, stored_excite);
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
                    eps_local
                );
            }
        }
    }

    /// Compute the off-diagonal contribution to the PT energy estimate
    /// using the stored samples
    pub fn off_diag_pt_energy(&self, e0: f64) -> f64 {
        let mut e_pt: f64 = 0.0;
        for (e_a, s, s_sq) in self.pt_energies_and_sums.values() {
            e_pt += (*s * *s - *s_sq) / (e0 - e_a);
        }
        e_pt / (self.n as f64) / ((self.n - 1) as f64)
    }
}

/// Sample the 'off-diagonal' contribution to the PT energy (i.e., the term coming from the off-diagonal of
/// the matrix V \[1 / (E_0 - E_a)\] V), and update the Welford statistics for the energy
/// If eps is not None, then evaluate at eps=0 and subtract eps=eps.unwrap()
pub fn sample_off_diag_update_welford(
    wf: &Wf,
    excite_gen: &ExciteGenerator,
    ham: &Ham,
    n_samples_per_batch: i32,
    rand: &mut Rand,
    enpt2_off_diag: &mut Stats<f64>,
    eps: Option<f64>
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
    let mut off_diag_screened: OffDiagSamples = OffDiagSamples::default();
    let prob: f64 = 1.0 / (wf.n as f64);
    for (i, w) in counts {
        off_diag.add_new_var_det(wf, &wf.dets[i], w, prob, excite_gen, ham, None);
        if let Some(e) = eps {off_diag_screened.add_new_var_det(wf, &wf.dets[i], w, prob, excite_gen, ham, eps);}
    }
    assert_eq!(n_samples_per_batch, off_diag.n);

    let mut off_diag_estimate: f64 = off_diag.off_diag_pt_energy(wf.energy);
    if let Some(e) = eps {off_diag_estimate -= off_diag_screened.off_diag_pt_energy(wf.energy);}
    println!(
        "Sampled off-diagonal energy this batch: {:.4}",
        off_diag_estimate
    );
    enpt2_off_diag.update(off_diag_estimate);
}
