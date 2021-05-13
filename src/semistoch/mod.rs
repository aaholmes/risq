use crate::wf::Wf;
use crate::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::stoch::matmul_sample_remaining;

pub fn semistoch_matmul(input_wf: Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, n_samples: i32) {
    // Sketch of how semistoch_matmul would work in practice

    // Compute deterministic component, and  create sampler object for sampling remaining component
    let (dtm_result, screened_sampler) = input_wf.approx_matmul(ham, excite_gen, eps);

    // Compute dtm approx to observable using dtm_result
    println!("Deterministic component:");
    dtm_result.print();

    for i in 0..n_samples {
        println!("\nCollecting sample {}...", i);
        let (sampled_det, sampled_prob) = matmul_sample_remaining(&screened_sampler, excite_gen, ham);
        match sampled_det {
            Some(d) => {
                // Update expectation values using determinant d, sample probability sampled_prob
                println!("Sampled config: {} with probability = {}", d, sampled_prob);
                println!("Coeff/sampled_prob = {}", d.coeff / sampled_prob);
            },
            None => {
                println!("Sampled excitation not valid! Sample prob = {}", sampled_prob);
            }
        }
    }
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