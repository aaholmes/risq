use crate::wf::Wf;
use crate::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::stoch::matmul_sample_remaining;

pub fn semistoch_matmul(input_wf: Wf, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, n_samples: i32) {
    // Sketch of how semistoch_matmul would work in practice

    let (dtm_result, screened_sampler) = input_wf.approx_matmul(ham, excite_gen, eps);

    // Compute dtm approx to observable using dtm_result

    for i in 0..n_samples {
        (sampled_det, sampled_prob) = matmul_sample_remaining(screened_sampler, excite_gen, ham);
        match sampled_det {
            Some(d) => {
                // Update expectation values using determinant d, sample probability sampled_prob
            },
            None => {}
        }
    }
}
