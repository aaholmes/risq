//! # Semistochastic Methods (`semistoch`)
//!
//! This module implements semistochastic approaches, primarily focusing on
//! Epstein-Nesbet perturbation theory (ENPT2) combined with stochastic sampling.
//! The goal is to capture the bulk of the PT2 correction deterministically while
//! efficiently estimating the remaining contributions via sampling.

mod utils;

use crate::excite::init::ExciteGenerator;
use crate::excite::iterator::{dets_and_excitable_orbs, excites_from_det_and_orbs};
use crate::excite::{Excite, Orbs};
use crate::ham::Ham;
use crate::pt::{pt, pt_off_diag, PtSamples};
use crate::rng::Rand;
use crate::semistoch::utils::diag::sample_diag_update_welford;
use crate::semistoch::utils::off_diag::sample_off_diag_update_welford;
use crate::stoch::alias::Alias;
use crate::stoch::{generate_screened_sampler, DetOrbSample, ScreenedSampler};
use crate::stoch::{matmul_sample_remaining, ImpSampleDist};
use crate::utils::read_input::Global;
use crate::wf::det::Det;
use crate::wf::Wf;
use rolling_stats::Stats;
use std::time::Instant;

/// Implements an importance-sampled semistochastic ENPT2 calculation.
///
/// This variant appears to:
/// 1. Compute a deterministic PT2 contribution (`dtm_enpt2`) using a threshold `global.eps_pt_dtm`.
/// 2. Stochastically sample the remaining contributions (those below `eps_pt_dtm` but above `eps_var`)
///    using importance sampling, weighted by `(H_ai * c_i)^2`.
/// 3. Combine the deterministic and stochastic estimates to get the final PT2 energy and uncertainty.
///
/// # Arguments
/// * `input_wf`: The converged variational wavefunction.
/// * `global`: Global calculation parameters.
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator.
/// * `rand`: Mutable random number generator state.
///
/// # Returns
/// A tuple `(pt2_energy, pt2_std_dev)` containing the estimated PT2 energy and its standard deviation.
pub fn importance_sampled_semistoch_enpt2(
    input_wf: &Wf,
    global: &Global,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    rand: &mut Rand,
) -> (f64, f64) {
    // Basically just the original SHCI except the sum on "a" is importance sampled
    // i.e., instead of computing the sum deterministically, sample according to some
    // importance sampled probability distribution, specifically, P(a|i)P(i) ~ (H_{ai} c_i)^2,
    // since the leading term has this depenence

    // Old algorithm: Sample i (p_i ~ |c_i|), sum on a (sum on all a minus sum on screened a)
    // New algorithm:
    // Idea 1 (do this one): Sample i (smarter), then sample a (from all), sample a (screened), take difference
    // Idea 2: Sample ai (from all), then sample ai (screened), take difference

    // Probability of sampling each det: sum_a (H_{ai} c_i)^2 (over terms larger than eps?)

    println!("Importance sampled semistoch ENPT2");

    let start_dtm_enpt2: Instant = Instant::now();

    // Create a sampler that importance samples all excitations smaller than eps_var (since excitations that exceed eps_var would have already been used to construct wf_var)
    let (_, screened_sampler_lt_var_eps) =
        input_wf.approx_matmul_external_semistoch_singles(ham, excite_gen, global.eps_var);

    // Compute deterministic approximation to H psi using large PT eps (i.e., the eps that determines whether a deterministic or stochastic contribution)
    let (dtm_result, screened_sampler_lt_pt_eps) =
        input_wf.approx_matmul_external_semistoch_singles(ham, excite_gen, global.eps_pt_dtm);

    println!("Time for dtm ENPT2: {:?}", start_dtm_enpt2.elapsed());

    // Compute approximate delta E using approx H psi
    let mut dtm_enpt2: f64 = 0.0;
    for det in dtm_result.dets {
        dtm_enpt2 += det.coeff * det.coeff / (input_wf.energy - det.diag.unwrap());
    }
    println!(
        "Deterministic approximation to Delta E using eps = {} ({} dets): {:.6}",
        global.eps_pt_dtm, dtm_result.n, dtm_enpt2
    );

    // Sample dets from wf with probability |c|
    // Update estimate of difference between exact and approximate delta E using deterministic application of H
    // let eps_pt: f64 = 1e-9; // essentially zero in the SHCI paper
    let mut stoch_enpt2: Stats<f64> = Stats::new();
    let mut samples_gt_eps: PtSamples = Default::default(); // data structure to contain sampled contributions to PT that exceed eps_dtm_pt
    let mut samples_all: PtSamples = Default::default(); // data structure to contain sampled contributions to PT that are less than eps_dtm_pt

    let n_batches_max = 43;

    let start_quadratic: Instant = Instant::now();
    for i_batch in 0..n_batches_max {
        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);

        samples_all.clear(); // For estimating the full PT correction
        samples_gt_eps.clear(); // For estimating the deterministic component of the PT correction

        for _i_sample in 0..global.n_samples_per_batch {
            // Collect samples that exceed eps
            // TODO: Make this only sample greater than eps component, but this works for now

            // Sample with probability proportional to (Hc)^2
            let (sampled_det_info, sampled_prob) = matmul_sample_remaining(
                &screened_sampler_lt_var_eps,
                ImpSampleDist::HcSquared,
                excite_gen,
                ham,
                rand,
            );

            match sampled_det_info {
                None => {
                    // println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
                }
                Some((exciting_det, excite, target_det)) => {
                    // println!("Sampled excitation: Sampled det = {}, Sample prob = {}, (H_ai c_i)^2 / p = {}", target_det, sampled_prob, excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff / sampled_prob);
                    // Collect this sample for the quadratic contribution
                    // This is analogous to the SHCI contribution, but only applies to the (<eps)
                    // terms. The sampling probability (Hc)^2 was chosen because the largest terms
                    // in this component are (Hc)^2 / (E_0 - E_a).
                    if i_batch == 42 {
                        println!("Adding sample to 'all' samples: exciting det: {}, target det: {}, sampled prob: {}", exciting_det, target_det, sampled_prob);
                    }
                    samples_all.add_sample_compute_diag(
                        exciting_det,
                        &excite,
                        target_det,
                        sampled_prob,
                        ham,
                    );
                    if excite.abs_h * exciting_det.coeff.abs() > global.eps_pt_dtm {
                        if i_batch == 42 {
                            println!("Adding sample to '> eps_pt' samples: exciting det: {}, target det: {}, sampled prob: {}", exciting_det, target_det, sampled_prob);
                        }
                        samples_gt_eps.add_sample_compute_diag(
                            exciting_det,
                            &excite,
                            target_det,
                            sampled_prob,
                            ham,
                        );
                    }
                }
            }

            // Collect samples less than eps

            // Sample with probability proportional to (Hc)^2
            let (sampled_det_info, sampled_prob) = matmul_sample_remaining(
                &screened_sampler_lt_pt_eps,
                ImpSampleDist::HcSquared,
                excite_gen,
                ham,
                rand,
            );

            match sampled_det_info {
                None => {
                    // println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
                }
                Some((exciting_det, excite, target_det)) => {
                    // println!("Sampled excitation: Sampled det = {}, Sample prob = {}, (H_ai c_i)^2 / p = {}", target_det, sampled_prob, excite.abs_h * excite.abs_h * exciting_det.coeff * exciting_det.coeff / sampled_prob);
                    // Collect this sample for the quadratic contribution
                    // This is analogous to the SHCI contribution, but only applies to the (<eps)
                    // terms. The sampling probability (Hc)^2 was chosen because the largest terms
                    // in this component are (Hc)^2 / (E_0 - E_a).
                    if i_batch == 42 {
                        println!("Adding sample to 'all' samples: exciting det: {}, target det: {}, sampled prob: {}", exciting_det, target_det, sampled_prob);
                    }
                    samples_all.add_sample_compute_diag(
                        exciting_det,
                        &excite,
                        target_det,
                        sampled_prob,
                        ham,
                    );
                }
            }
        }

        let e_stoch_all: f64 =
            samples_all.pt_estimator(input_wf.energy, global.n_samples_per_batch);
        let e_stoch_screened: f64 =
            samples_gt_eps.pt_estimator(input_wf.energy, global.n_samples_per_batch);
        println!("Energy components: {} {}", e_stoch_all, e_stoch_screened);

        let sampled_e: f64 = e_stoch_all - e_stoch_screened;
        println!("Sampled energy this batch = {}", sampled_e);

        stoch_enpt2.update(sampled_e);
        println!(
            "Current estimate of stochastic component: {:.4} +- {:.4}",
            stoch_enpt2.mean, stoch_enpt2.std_dev
        );

        if i_batch > 9 && stoch_enpt2.std_dev <= global.target_uncertainty {
            println!("Target uncertainty reached!");
            break;
        }
    }
    println!("Time for sampling: {:?}", start_quadratic.elapsed());

    println!(
        "Stochastic component: {:.4} +- {:.4}",
        stoch_enpt2.mean, stoch_enpt2.std_dev
    );

    (dtm_enpt2 + stoch_enpt2.mean, stoch_enpt2.std_dev)
}

fn std_err(stats: &Stats<f64>) -> f64 {
    stats.std_dev / (stats.count as f64).sqrt()
}

pub fn new_semistoch_enpt2_dtm_diag_singles(
    input_wf: &Wf,
    global: &Global,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    rand: &mut Rand,
) -> (f64, f64) {
    // Loop over all single excites: Sum their diagonal contributions, and for the ones for which
    // |H_ai c_i| > eps_dtm_pt, generate wf of their perturbative contributions

    // Loop over double excites |H_ai c_i| > eps_dtm_pt: add to diagonal contribution, and also
    // use for wf of perturbative contributions.

    // Compute dtm PT expression from these quantities

    // Sample either one double excitation for updating the diagonal term, or a batch of variational
    // dets for updating the off-diagonal term

    // Repeat until convergence

    let start_dtm: Instant = Instant::now();

    // For making screened sampler
    let mut det_orbs: Vec<DetOrbSample> = vec![];

    // For making det_sampler
    let mut rel_probs: Vec<f64> = vec![0.0; input_wf.n];

    let mut excite: Excite;
    let mut h_ai_c_i: f64;
    let mut diag: f64;
    let mut h_psi: Wf = Wf::default();
    // let mut h_psi_sq: Wf = Wf::default();
    // let mut e: f64;
    let mut e_pt_diag_doubles: f64 = 0.0;
    let mut e_pt_diag_singles: f64 = 0.0;

    // Loop over all doubles that exceed eps, and all singles
    // Separate out the iteration over var_dets and orbs from the iteration over excites leaving those orbs
    // so that we can set up the sampler object for the remaining excites
    // (i.e., don't use the iterator dets_excites_and_excited_dets here)
    //for (var_det, excite, pt_config) in dets_excites_and_excited_dets(input_wf, excite_gen, global.eps_pt_dtm) {
    for (i_det, var_det, is_alpha, init) in dets_and_excitable_orbs(input_wf, excite_gen) {
        for (stored_excite, pt_config) in
            excites_from_det_and_orbs(var_det, is_alpha, init, input_wf, excite_gen)
        {
            // Filters out excitations into variational wf
            // For doubles, check whether exceeds eps
            // Edit: do this for singles too! This introduces a bias in its current implementation,
            // but dramatically accelerates dtm component
            // if matches!(init, Orbs::Double(_))
            //     && stored_excite.abs_h * var_det.coeff.abs() < global.eps_pt_dtm
            // {
            if stored_excite.abs_h * var_det.coeff.abs() < global.eps_pt_dtm {
                // Threshold reached: set up sampler and go to next det/excitable orb
                det_orbs.push(DetOrbSample {
                    det: var_det,
                    init,
                    is_alpha,
                    sum_abs_h: stored_excite.sum_remaining_abs_h,
                    sum_h_squared: stored_excite.sum_remaining_h_squared,
                    sum_abs_hc: var_det.coeff.abs() * stored_excite.sum_remaining_abs_h,
                    sum_hc_squared: var_det.coeff
                        * var_det.coeff
                        * stored_excite.sum_remaining_h_squared,
                });
                break;
            }

            // Create excite to make calculation of off_diag and new diag elements easier
            excite = Excite {
                is_alpha,
                init,
                target: stored_excite.target,
                abs_h: stored_excite.abs_h,
            };

            // Compute off-diagonal element times var_det.coeff
            h_ai_c_i = ham.ham_off_diag(&var_det.config, &pt_config, &excite) * var_det.coeff;

            // Update relative probabilities is this exceeds the previous |c_i| max_a |H_{ai}|
            if h_ai_c_i.abs() > rel_probs[i_det] {
                rel_probs[i_det] = h_ai_c_i.abs();
            }

            // Off-diagonal term
            // Compute diagonal element in O(N) time only if necessary, store diag energy for next step
            // For single excitations: Check whether this excite actually exceeds eps
            if matches!(init, Orbs::Single(_)) && h_ai_c_i.abs() < global.eps_pt_dtm {
                // Compute or lookup diag term for later, but don't add to h_psi
                if let Some(ind) = h_psi.inds.get_mut(&pt_config) {
                    diag = h_psi.dets[*ind].diag.unwrap();
                } else {
                    diag = var_det.new_diag(ham, &excite);
                }
            } else {
                // Compute or lookup diag term and add h_ai_c_i to h_psi
                if let Some(ind) = h_psi.inds.get_mut(&pt_config) {
                    diag = h_psi.dets[*ind].diag.unwrap();
                    h_psi.dets[*ind].coeff += h_ai_c_i;
                    // h_psi_sq.dets[*ind].coeff += h_ai_c_i * h_ai_c_i;
                } else {
                    diag = var_det.new_diag(ham, &excite);
                    h_psi.push(Det {
                        config: pt_config,
                        coeff: h_ai_c_i,
                        diag: Some(diag),
                    });
                    // h_psi_sq.push(Det {
                    //     config: pt_config,
                    //     coeff: h_ai_c_i * h_ai_c_i,
                    //     diag: Some(diag),
                    // });
                }
            }

            // Diagonal term
            // e = h_ai_c_i * h_ai_c_i / (input_wf.energy - diag);
            // if let Orbs::Double(_) = excite.init {
            //     // Double excite, exceeds eps_pt_dtm
            //     e_pt_diag_doubles += e;
            // } else {
            //     // Single excite
            //     e_pt_diag_singles += e;
            // }
        }
    }

    // let e_pt_off_diag = pt_off_diag(&h_psi, &h_psi_sq, input_wf.energy);
    // let e_pt_dtm: f64 = e_pt_diag_doubles + e_pt_diag_singles + e_pt_off_diag;
    let e_pt_dtm: f64 = pt(&h_psi, input_wf.energy);

    // println!("\nDeterministic component of PT energy:");
    // println!(
    //     "  Diagonal Doubles: {:.6} ({:.1}%)",
    //     e_pt_diag_doubles,
    //     100.0 * e_pt_diag_doubles / e_pt_dtm
    // );
    // println!(
    //     "  Diagonal Singles: {:.6} ({:.1}%)",
    //     e_pt_diag_singles,
    //     100.0 * e_pt_diag_singles / e_pt_dtm
    // );
    // println!(
    //     "  Off-diagonal:     {:.6} ({:.1}%)",
    //     e_pt_off_diag,
    //     100.0 * e_pt_off_diag / e_pt_dtm
    // );
    // println!("  Total:            {:.6}\n", e_pt_dtm);

    println!(
        "Time for deterministic component: {:?}",
        start_dtm.elapsed()
    );

    // Prepare stochastic component

    // Diagonal part: for sampling (D_i, D_a) that are a double excite apart
    // (this sets up both importance sampling distributions:
    // |Hc| and (Hc)^2, but if this is time consuming we can only set up one)
    println!("Setting up screened sampler");
    let start_setup_screened_sampler: Instant = Instant::now();
    let screened_sampler: ScreenedSampler = generate_screened_sampler(det_orbs);
    println!(
        "Time for sampling setup: {:?}",
        start_setup_screened_sampler.elapsed()
    );

    // Off-diagonal part: for sampling D_i with probability proportional to |c_i| max_a |H_{ai}|
    println!("\ni, |c_i|, |c_i| max_a |H_ai| =");
    for i in 0..10 {
        println!("{}  {}  {}", i, input_wf.dets[i].coeff.abs(), rel_probs[i]);
    }
    println!("...");
    for i in input_wf.n - 5..input_wf.n {
        println!("{}  {}  {}", i, input_wf.dets[i].coeff.abs(), rel_probs[i]);
    }
    let det_sampler: Alias = Alias::new(rel_probs);

    // Stochastic component

    // Initially, take samples of several batches of each of the diagonal and off-diagonal components
    // Keep track of the uncertainties in the two components
    // While total uncertainty is too high, take a batch of the component that has the higher uncertainty
    // of the two

    // Need functions: sample_diag, sample_off_diag
    // sample_diag is trivial: just sample one (a, i), compute E(a, i), and add it to the Welford struct
    // sample_off_diag is more complicated:
    // Loop over samples in the batch:
    // For each, add to the hash table {a: E_a, {i: w_i, H_{ai} c_i / q_i}}
    // Then, computing the energy is straightforward:
    // Loop over a:
    // For each, compute the two sums over that a's i values, and use that to update the energy

    let start_pt: Instant = Instant::now();

    let mut enpt2_diag: Stats<f64> = Stats::new();
    let mut enpt2_off_diag: Stats<f64> = Stats::new();

    let n_diag_init: i32 = 100;
    let n_off_diag_init: i32 = 10;

    println!(
        "\nCollecting {} initial samples of the diagonal contribution to E_PT",
        n_diag_init
    );
    for _i_batch in 0..n_diag_init {
        // Sample diag, update Welford
        sample_diag_update_welford(
            &screened_sampler,
            excite_gen,
            ham,
            rand,
            input_wf.energy,
            &mut enpt2_diag,
        );
    }
    println!(
        "Initial estimate of the diagonal contribution: {:.4} +- {:.4}",
        enpt2_diag.mean,
        std_err(&enpt2_diag)
    );

    println!(
        "\nCollecting {} initial samples of the off-diagonal contribution to E_PT",
        n_off_diag_init
    );
    for _i_batch in 0..n_off_diag_init {
        // Sample off_diag, update Welford
        sample_off_diag_update_welford(
            input_wf,
            excite_gen,
            ham,
            &det_sampler,
            global.n_samples_per_batch,
            rand,
            &mut enpt2_off_diag,
            Some(global.eps_pt_dtm),
        );
    }
    println!(
        "Initial estimate of the off-diagonal contribution: {:.4} +- {:.4}",
        enpt2_off_diag.mean,
        std_err(&enpt2_off_diag)
    );

    let mut total_std_err: f64 = (std_err(&enpt2_diag) * std_err(&enpt2_diag)
        + std_err(&enpt2_off_diag) * std_err(&enpt2_off_diag))
    .sqrt();
    println!(
        "After init, diag and off-diag components: {} +- {}, {} +- {}, total: {} +- {}",
        enpt2_diag.mean,
        std_err(&enpt2_diag),
        enpt2_off_diag.mean,
        std_err(&enpt2_off_diag),
        enpt2_diag.mean + enpt2_off_diag.mean,
        total_std_err
    );

    while total_std_err > global.target_uncertainty {
        // Collect another batch of the less certain component
        if std_err(&enpt2_diag) >= 0.5 * std_err(&enpt2_off_diag) {
            // Sample diag, update Welford
            for _i_diag in 0..global.n_samples_per_batch {
                sample_diag_update_welford(
                    &screened_sampler,
                    excite_gen,
                    ham,
                    rand,
                    input_wf.energy,
                    &mut enpt2_diag,
                );
            }
        } else {
            // Sample off_diag, update Welford
            sample_off_diag_update_welford(
                input_wf,
                excite_gen,
                ham,
                &det_sampler,
                global.n_samples_per_batch,
                rand,
                &mut enpt2_off_diag,
                Some(global.eps_pt_dtm),
            );
        }
        total_std_err = (std_err(&enpt2_diag) * std_err(&enpt2_diag)
            + std_err(&enpt2_off_diag) * std_err(&enpt2_off_diag))
        .sqrt();
        println!(
            "diag ({} samples) and off-diag ({} batches) components: {} +- {}, {} +- {}, total: {} +- {}",
            enpt2_diag.count, enpt2_off_diag.count,
            enpt2_diag.mean,
            std_err(&enpt2_diag),
            enpt2_off_diag.mean,
            std_err(&enpt2_off_diag),
            enpt2_diag.mean + enpt2_off_diag.mean,
            total_std_err
        );
    }

    println!(
        "Time for new stochastic PT algorithm: {:?}",
        start_pt.elapsed()
    );

    (
        e_pt_dtm + enpt2_diag.mean + enpt2_off_diag.mean,
        total_std_err,
    )
}

/// Implements a purely stochastic ENPT2 calculation
///
/// This function seems to skip any deterministic component and directly samples contributions
/// using the `PtSamples` structure and the `pt_estimator` function, which implements the
/// unbiased estimator from the SHCI papers. It samples excitations originating from variational
/// determinants (`i`) selected with probability proportional to `|c_i|`.
///
/// # Arguments
/// * `input_wf`: The converged variational wavefunction.
/// * `global`: Global calculation parameters.
/// * `ham`: The Hamiltonian operator.
/// * `excite_gen`: Pre-computed excitation generator.
/// * `rand`: Mutable random number generator state.
///
/// # Returns
/// A tuple `(pt2_energy, pt2_std_dev)` containing the estimated PT2 energy and its standard deviation.
pub fn fast_stoch_enpt2(
    input_wf: &Wf,
    global: &Global,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    rand: &mut Rand,
) -> (f64, f64) {
    // Stochastic Epstein-Nesbet PT2
    // In earlier SHCI paper, used the following strategy:
    // 1. Compute ENPT2 deterministically using large eps
    // 2. Sample some variational determinants, compute difference between large-eps and small-eps
    //    ENPT2 expressions for them
    // Here, we take a different (smarter?) approach that introduces importance sampling:
    // 1. Importance-sample the large set of H*psi terms that had been screened out in the variational
    //    stage
    // Main advantage: uses importance sampling, can take larger batch sizes since we don't have to
    // compute all connections.

    // Compute deterministic component (even though not used), and create sampler object for sampling remaining component
    let start_dtm_enpt2: Instant = Instant::now();
    let (_, screened_sampler) =
        input_wf.approx_matmul_external_skip_singles(ham, excite_gen, global.eps_var);
    println!("Time for sampling setup: {:?}", start_dtm_enpt2.elapsed());

    // Stochastic component
    let mut stoch_enpt2_quadratic: Stats<f64> = Stats::new(); // unbiased contribution from samples with themselves
    let mut samples: PtSamples = Default::default(); // data structure to contain sampled contributions to PT

    let start_quadratic: Instant = Instant::now();
    let n_batches_max = 10;
    for i_batch in 0..n_batches_max {
        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);
        samples.clear();
        for _i_sample in 0..global.n_samples_per_batch {
            // Sample with probability proportional to (Hc)^2
            let (sampled_det_info, sampled_prob) = matmul_sample_remaining(
                &screened_sampler,
                ImpSampleDist::HcSquared,
                excite_gen,
                ham,
                rand,
            );

            match sampled_det_info {
                None => {}
                Some((exciting_det, excite, target_det)) => {
                    // Collect this sample for the quadratic contribution
                    // This is analogous to the SHCI contribution, but only applies to the (<eps)
                    // terms. The sampling probability (Hc)^2 was chosen because the largest terms
                    // in this component are (Hc)^2 / (E_0 - E_a).
                    samples.add_sample_compute_diag(
                        exciting_det,
                        &excite,
                        target_det,
                        sampled_prob,
                        ham,
                    );
                }
            }
        }

        // Collect the samples, evaluate their contributions a la the original SHCI paper
        let sampled_e: f64 = samples.pt_estimator(input_wf.energy, samples.n);
        println!("Sampled energy this batch = {}", sampled_e);
        stoch_enpt2_quadratic.update(sampled_e);
        println!(
            "Current estimate of stochastic component: {:.4} +- {:.4}",
            stoch_enpt2_quadratic.mean, stoch_enpt2_quadratic.std_dev
        );

        if i_batch > 9 && stoch_enpt2_quadratic.std_dev <= global.target_uncertainty / 2f64.sqrt() {
            println!("Target uncertainty reached!");
            break;
        }
    }
    println!("Time for sampling: {:?}", start_quadratic.elapsed());

    (stoch_enpt2_quadratic.mean, stoch_enpt2_quadratic.std_dev)
}

/// Old algorithm (2017) for semistochastic ENPT2
pub fn old_semistoch_enpt2(
    input_wf: &Wf,
    global: &Global,
    ham: &Ham,
    excite_gen: &ExciteGenerator,
    use_optimal_probs: bool,
    rand: &mut Rand,
) -> (f64, f64) {
    // If use_optimal_probs, then sample with probability proportional to sum of remaining (Hc)^2;
    // else, use probability proportional to |c|

    // Compute deterministic approximation to H psi using large eps
    let start_dtm_enpt2: Instant = Instant::now();
    let (dtm_result, optimal_probs) =
        input_wf.approx_matmul_external_dtm_only(ham, excite_gen, global.eps_pt_dtm);
    println!("Time for dtm ENPT2: {:?}", start_dtm_enpt2.elapsed());

    // Compute approximate delta E using approx H psi
    let mut dtm_enpt2: f64 = 0.0;
    for det in dtm_result.dets {
        dtm_enpt2 += det.coeff * det.coeff / (input_wf.energy - det.diag.unwrap());
    }
    println!(
        "Deterministic approximation to Delta E using eps = {} ({} dets): {:.6}",
        global.eps_pt_dtm, dtm_result.n, dtm_enpt2
    );

    // Sample dets from wf with probability |c|
    // Update estimate of difference between exact and approximate delta E using deterministic application of H
    let eps_pt: f64 = 1e-9; // essentially zero in the SHCI paper
    let mut stoch_enpt2: Stats<f64> = Stats::new();
    let mut samples_all: PtSamples = Default::default(); // data structure to contain sampled contributions to PT
    let mut samples_large_eps: PtSamples = Default::default(); // data structure to contain sampled contributions to PT that exceed eps

    // Setup Alias sampling of var wf
    let mut var_probs: Vec<f64> = vec![0.0; input_wf.n];
    for (i_det, det) in input_wf.dets.iter().enumerate() {
        if use_optimal_probs {
            // P ~ sum_remaining (Hc)^2
            var_probs[i_det] = optimal_probs[i_det];
        } else {
            // P ~ |c|
            var_probs[i_det] = det.coeff.abs() as f64;
        }
    }
    let prob_norm: f64 = var_probs.iter().sum::<f64>();
    for var_prob in var_probs.iter_mut() {
        *var_prob /= prob_norm;
    }
    println!("Setting up wf sampler");
    println!("Normalization: {}", var_probs.iter().sum::<f64>());
    let wf_sampler: Alias = Alias::new(var_probs);
    println!("Done setting up wf sampler");
    // println!("Alias sampler:");
    // wf_sampler.print();
    //
    // println!("Testing alias sampler...");
    // wf_sampler.test(1000000);

    let n_batches_max = 1000;

    let start_quadratic: Instant = Instant::now();
    for i_batch in 0..n_batches_max {
        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);

        samples_all.clear();
        samples_large_eps.clear();

        for _i_sample in 0..global.n_samples_per_batch {
            // Sample with prob var_probs(:)
            let (sampled_var_det_ind, sampled_prob) = wf_sampler.sample_with_prob(rand);
            let sampled_var_det = input_wf.dets[sampled_var_det_ind];
            // let sampled_var_det = wf_sampler.sample();
            // let sampled_prob = sampled_var_det.coeff.abs() / prob_norm;

            // Generate (wastefully) all PT dets connected to the sampled variational det

            // Collect samples using small eps (eps_pt)
            let mut v_times_sampled_var_det = sampled_var_det
                .approx_matmul_external_dtm_only_compute_diags(input_wf, ham, excite_gen, eps_pt);
            for target_det in v_times_sampled_var_det.dets {
                samples_all.add_sample_diag_already_stored(
                    sampled_var_det,
                    target_det,
                    sampled_prob,
                );
            }

            // Collect samples using large eps (eps) corresponding to deterministic part
            v_times_sampled_var_det = sampled_var_det
                .approx_matmul_external_dtm_only_compute_diags(
                    input_wf,
                    ham,
                    excite_gen,
                    global.eps_pt_dtm,
                );
            for target_det in v_times_sampled_var_det.dets {
                samples_large_eps.add_sample_diag_already_stored(
                    sampled_var_det,
                    target_det,
                    sampled_prob,
                );
            }
        }

        let sampled_e: f64 = samples_all.pt_estimator(input_wf.energy, global.n_samples_per_batch)
            - samples_large_eps.pt_estimator(input_wf.energy, global.n_samples_per_batch);

        println!("Sampled energy this batch = {}", sampled_e);
        stoch_enpt2.update(sampled_e);
        println!(
            "Current estimate of stochastic component: {:.4} +- {:.4}",
            stoch_enpt2.mean,
            std_err(&stoch_enpt2)
        );

        if i_batch > 9 && std_err(&stoch_enpt2) <= global.target_uncertainty {
            println!("Target uncertainty reached!");
            break;
        }
    }
    println!("Time for sampling: {:?}", start_quadratic.elapsed());

    println!(
        "Stochastic component: {:.4} +- {:.4}",
        stoch_enpt2.mean,
        std_err(&stoch_enpt2)
    );

    (dtm_enpt2 + stoch_enpt2.mean, std_err(&stoch_enpt2))
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::excite::init::init_excite_generator;
//     use crate::ham::read_ints::read_ints;
//     use crate::ham::Ham;
//     use crate::utils::read_input::{read_input, Global};
//     use crate::wf::init_var_wf;
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
//         println!("Initializing wavefunction");
//         let mut wf: Wf = init_var_wf(&GLOBAL, &HAM, &EXCITE_GEN);
//         println!("Done initializing wf");
//         wf.print();
//
//         let eps = 0.1;
//         let n_samples = 10;
//         println!("Calling semistoch_matmul!");
//         semistoch_matmul(wf, &HAM, &EXCITE_GEN, eps, n_samples)
//     }
// }
