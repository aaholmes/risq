use crate::wf::Wf;
use crate::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::stoch::matmul_sample_remaining;
use rolling_stats::Stats;
use crate::pt::PtSamples;

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
    println!("Result of screened_matmul:");
    dtm_result.print();

    // Compute dtm approx to observable using dtm_result
    // Simple ENPT2
    let mut dtm_enpt2: f64 = 0.0;
    let mut energy_sample: f64 = 0.0;
    for det in &dtm_result.dets {
        dtm_enpt2 += (det.coeff * det.coeff) / (input_wf.energy - det.diag);
    }
    let mut stoch_enpt2: Stats<f64> = Stats::new(); // samples overlap with deterministic part
    let mut stoch_enpt2_2: Stats<f64> = Stats::new(); // unbiased contribution from samples with themselves
    let mut samples: PtSamples = Default::default(); // data structure to contained sampled contributions to PT

    for i_batch in 0..n_batches {
        // Sample a batch of samples, updating the stoch component of the energy for each sample
        println!("\n Starting batch {}", i_batch);
        for i_sample in 0..n_samples_per_batch {
            //println!("\nCollecting sample {}...", i_sample);
            let (sampled_det_info, sampled_prob) = matmul_sample_remaining(&screened_sampler, excite_gen, ham);
            match sampled_det_info {
                None => {
                    //println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
                }
                Some((exciting_det, excite, target_det)) => {
                    // Update expectation values using determinant d, sample probability sampled_prob
                    //println!("Sampled config: {} with probability = {}", target_det.config, sampled_prob);
                    //println!("Coeff/sampled_prob = {}", target_det.coeff / sampled_prob);
                    // If sampled_det is in dtm_result, then update the stoch_enpt2 energy
                    match dtm_result.inds.get(&target_det.config) {
                        None => {}
                        Some(ind) => {
                            energy_sample = dtm_result.dets[*ind].coeff * target_det.coeff / sampled_prob / (input_wf.energy - dtm_result.dets[*ind].diag);
                            //println!("Sampled energy: {}", energy_sample);
                            stoch_enpt2.update(energy_sample);
                        }
                    }
                    // Regardless of whether it was used to update the energy above, collect this sample for the next contribution
                    samples.add_sample(exciting_det, &excite, target_det, sampled_prob, ham);
                },
            }
        }

        println!("Stochastic component projected against deterministic component: {}", stoch_enpt2);

        // Collect the samples, evaluate their contributions a la the original SHCI paper
        stoch_enpt2_2.update(samples.unbiased_pt_estimator(input_wf.energy));

    }

    // Multiply the first component by 2
    // println!("Stochastic component: {} +- {}", 2f64 * stoch_enpt2.mean, 2f64 * stoch_enpt2.std_dev);
    println!("Stochastic components: {:.4} +- {:.4},   {:.4} +- {:.4}", 2f64 * stoch_enpt2.mean, 2f64 * stoch_enpt2.std_dev, stoch_enpt2_2.mean, stoch_enpt2_2.std_dev);
    (dtm_enpt2 + 2f64 * stoch_enpt2.mean + stoch_enpt2_2.mean, stoch_enpt2_2.std_dev)
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