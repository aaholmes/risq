// Semistochastic methods mod; for now, just includes semistochastic ENPT2

use crate::wf::Wf;
use crate::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::stoch::{matmul_sample_remaining, ImpSampleDist};
use rolling_stats::Stats;
use crate::pt::PtSamples;
use vose_alias::VoseAlias;
use crate::wf::det::Det;
use serde::de::Expected;
// use rand::seq::index::sample;

pub fn semistoch_enpt2(input_wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, n_batches: i32, n_samples_per_batch: i32) -> (f64, f64) {
    // Semistochastic Epstein-Nesbet PT2
    // In earlier SHCI paper, used the following strategy:
    // 1. Compute ENPT2 deterministically using large eps
    // 2. Sample some variational determinants, compute difference between large-eps and small-eps
    //    ENPT2 expressions for them
    // Here, we take a different (smarter?) approach that introduces importance sampling:
    // 1. Compute ENPT2 deterministically using large eps, store the approximate H*psi used
    // 2. Importance-sample a large set of H*psi terms that had been screened out in step 1.
    // 3. Use these along with the stored approximate H*psi from step 1 to compute the contribution
    //    linear in the samples
    // 4. Use the samples and the original SHCI paper's unbiased expression to compute the term
    //    quadratic in the samples.
    // Main advantage: uses importance sampling, can take larger batch sizes since we don't have to
    // compute all connections, and makes use of stored approximate H*psi directly (rather than sampling it).

    // Compute deterministic component, and create sampler object for sampling remaining component
    let (dtm_result, screened_sampler) = input_wf.approx_matmul_external_no_singles(ham, excite_gen, eps);
    // println!("Result of screened_matmul:");
    // dtm_result.print();

    // Compute dtm approx to observable using dtm_result
    // Simple ENPT2
    let mut dtm_enpt2: f64 = 0.0;
    let mut energy_sample: f64 = 0.0;
    for det in &dtm_result.dets {
        dtm_enpt2 += (det.coeff * det.coeff) / (input_wf.energy - det.diag);
    }
    println!("Deterministic approximation to Delta E using eps = {}: {:.4}", eps, dtm_enpt2);

    // Stochastic component
    let mut stoch_enpt2: Stats<f64> = Stats::new(); // samples overlap with deterministic part
    let mut stoch_enpt2_2: Stats<f64> = Stats::new(); // unbiased contribution from samples with themselves
    let mut samples: PtSamples = Default::default(); // data structure to contain sampled contributions to PT

    for i_batch in 0..n_batches {
        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);
        samples.clear();
        for i_sample in 0..n_samples_per_batch {

            // Sample with probability proportional to (Hc)^2
            let (sampled_det_info, sampled_prob) = matmul_sample_remaining(
                &screened_sampler, ImpSampleDist::HcSquared, excite_gen, ham
            );

            match sampled_det_info {
                None => {
                    //println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
                }
                Some((exciting_det, excite, target_det)) => {
                    // Update expectation values using determinant d, sample probability sampled_prob
                    // This contribution tends to be small, as it is the (>eps)/(<eps) cross term,
                    // so for now we don't worry about whether sampling is optimal distribution,
                    // but could easily change that
                    //println!("Sampled config: {} with probability = {}", target_det.config, sampled_prob);
                    //println!("Coeff/sampled_prob = {}", target_det.coeff / sampled_prob);
                    // If sampled_det is in dtm_result, then update the stoch_enpt2 energy
                    match dtm_result.inds.get(&target_det.config) {
                        None => {}
                        Some(ind) => {
                            energy_sample = dtm_result.dets[*ind].coeff * target_det.coeff /
                                sampled_prob / (input_wf.energy - dtm_result.dets[*ind].diag);
                            //println!("Sampled energy: {}", energy_sample);
                            stoch_enpt2.update(energy_sample);
                        }
                    }
                    // Regardless of whether it was used to update the energy above, collect this sample for the next contribution
                    // This is analogous to the SHCI contribution, but only applies to the (<eps)
                    // terms. The sampling probability (Hc)^2 was chosen because the largest terms
                    // in this component are (Hc)^2 / (E_0 - E_a).
                    samples.add_sample_compute_diag(exciting_det, &excite, target_det, sampled_prob, ham);
                },
            }
        }

        println!("Stochastic component projected against deterministic component: {}", stoch_enpt2);

        // Collect the samples, evaluate their contributions a la the original SHCI paper
        println!("Collected samples:");
        samples.print();
        let sampled_e: f64 = samples.pt_estimator(input_wf.energy, samples.n);
        println!("Sampled energy this batch = {}", sampled_e);
        stoch_enpt2_2.update(sampled_e);

    }

    println!("Stochastic components: {:.4} +- {:.4},   {:.4} +- {:.4}", 2f64 * stoch_enpt2.mean,
             2f64 * stoch_enpt2.std_dev, stoch_enpt2_2.mean, stoch_enpt2_2.std_dev);

    (dtm_enpt2 + 2f64 * stoch_enpt2.mean + stoch_enpt2_2.mean,
     2f64 * stoch_enpt2.std_dev + stoch_enpt2_2.std_dev)

}


pub fn old_semistoch_enpt2(input_wf: &Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, n_batches: i32, n_samples_per_batch: i32) -> (f64, f64) {
    // Old algorithm (2017) for semistochastic ENPT2

    // Compute deterministic approximation to H psi using large eps
    let dtm_result = input_wf.approx_matmul_external_dtm_only(ham, excite_gen, eps);

    // Compute approximate delta E using approx H psi
    let mut dtm_enpt2: f64 = 0.0;
    for det in dtm_result.dets {
        dtm_enpt2 += det.coeff * det.coeff / (input_wf.energy - det.diag);
    }
    println!("Deterministic approximation to Delta E using eps = {}: {:.4}", eps, dtm_enpt2);

    // Sample dets from wf with probability |c|
    // Update estimate of difference between exact and approximate delta E using deterministic application of H
    let eps_pt: f64 = 1e-9; // essentially zero in the SHCI paper
    let mut stoch_enpt2: Stats<f64> = Stats::new();
    let mut samples_all: PtSamples = Default::default(); // data structure to contain sampled contributions to PT
    let mut samples_large_eps: PtSamples = Default::default(); // data structure to contain sampled contributions to PT that exceed eps

    // Setup Alias sampling of wf with p_i = |c_i|
    let mut probs_abs_c : Vec<f64> = vec![0.0; input_wf.n];
    // let mut prob_norm: f64 = 0.0;
    for (i_det, det) in input_wf.dets.iter().enumerate() {
        // prob_norm += det.coeff.abs() as f64;
        probs_abs_c[i_det] = det.coeff.abs() as f64;
    }
    let prob_norm:f64 = probs_abs_c.iter().sum::<f64>();
    for i_prob in 0..input_wf.n {
        probs_abs_c[i_prob] = probs_abs_c[i_prob] / prob_norm;
        // probs_abs_c[i_prob] = (probs_abs_c[i_prob] as f64 / prob_norm) as f64;
    }
    println!("Setting up wf sampler");
    println!("Normalization: {}", probs_abs_c.iter().sum::<f64>());
    let wf_sampler: VoseAlias<Det> = VoseAlias::new(input_wf.dets.clone(), probs_abs_c);
    println!("Done setting up wf sampler");
    // println!("Alias sampler:");
    // wf_sampler.print();
    //
    // println!("Testing alias sampler...");
    // wf_sampler.test(1000000);

    for i_batch in 0..n_batches {

        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);

        samples_all.clear();
        samples_large_eps.clear();

        for i_sample in 0..n_samples_per_batch {
            // Sample with prob |c|
            let sampled_var_det = wf_sampler.sample();
            let sampled_prob = sampled_var_det.coeff.abs() / prob_norm;

            // Generate (wastefully) all PT dets connected to the sampled variational det

            // Collect samples using small eps (eps_pt)
            let mut v_times_sampled_var_det = sampled_var_det.approx_matmul_external_dtm_only_compute_diags(input_wf, ham, excite_gen, eps_pt);
            for target_det in v_times_sampled_var_det.dets {
                samples_all.add_sample_diag_already_stored(sampled_var_det, target_det, sampled_prob, ham);
            }

            // Collect samples using large eps (eps) corresponding to deterministic part
            v_times_sampled_var_det = sampled_var_det.approx_matmul_external_dtm_only_compute_diags(input_wf, ham, excite_gen, eps);
            for target_det in v_times_sampled_var_det.dets {
                samples_large_eps.add_sample_diag_already_stored(sampled_var_det, target_det, sampled_prob, ham);
            }
        }

        let mut sampled_e: f64 = samples_all.pt_estimator(input_wf.energy, n_samples_per_batch)
            - samples_large_eps.pt_estimator(input_wf.energy, n_samples_per_batch);

        println!("Sampled energy this batch = {}", sampled_e);
        stoch_enpt2.update(sampled_e);

    }

    println!("Stochastic component: {:.4} +- {:.4}", stoch_enpt2.mean, stoch_enpt2.std_dev);

    (dtm_enpt2 + stoch_enpt2.mean,
     stoch_enpt2.std_dev)

}


// pub fn semistoch_matmul(input_wf: Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, n_samples: i32) {
//     // Sketch of how semistoch_matmul would work in practice
//
//     // Compute deterministic component, and create sampler object for sampling remaining component
//     let (dtm_result, screened_sampler) = input_wf.approx_matmul(ham, excite_gen, eps);
//
//     // Compute dtm approx to observable using dtm_result
//     println!("Deterministic component:");
//     dtm_result.print();
//
//     for i in 0..n_samples {
//         println!("\nCollecting sample {}...", i);
//         let (sampled_det, sampled_prob) = matmul_sample_remaining(&screened_sampler, excite_gen, ham);
//         match sampled_det {
//             Some(d) => {
//                 // Update expectation values using determinant d, sample probability sampled_prob
//                 println!("Sampled config: {} with probability = {}", d, sampled_prob);
//                 println!("Coeff/sampled_prob = {}", d.coeff / sampled_prob);
//             },
//             None => {
//                 println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
//             }
//         }
//     }
// }


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