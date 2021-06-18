#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate itertools;

extern crate alloc;

mod ham;
use ham::Ham;
use ham::read_ints::read_ints;

mod excite;
use excite::init::{ExciteGenerator, init_excite_generator};

pub mod wf;
use wf::{Wf, init_var_wf};

mod var;
// use var::variational;

mod utils;
mod stoch;
mod semistoch;
mod pt;

use utils::read_input::{Global, read_input};
use crate::semistoch::semistoch_enpt2;
use crate::var::variational;

fn main() {

    println!("ESP - Electronic Structure Package\nAdam A Holmes, 2021\n");

    println!("Reading input file");
    lazy_static! {
        static ref GLOBAL: Global = read_input("in.json").unwrap();
    }

    println!("Reading integrals");
    lazy_static! {
        static ref HAM: Ham = read_ints(&GLOBAL, "FCIDUMP");
    }

    println!("Initializing excitation generator");
    lazy_static! {
        static ref EXCITE_GEN: ExciteGenerator = init_excite_generator(&GLOBAL, &HAM);
    }

    println!("Initializing wavefunction");
    let mut wf: Wf = init_var_wf(&GLOBAL, &HAM, &EXCITE_GEN);
    wf.print();

    println!("Variational stage");
    variational(&HAM, &EXCITE_GEN, &mut wf);

    // let eps = 1.0;
    // let n_samples = 1000;
    // // println!("Calling semistoch_matmul!");
    // // semistoch_matmul(wf, &HAM, &EXCITE_GEN, eps, n_samples);
    //
    // println!("Calling semistoch ENPT2!");
    // let (e_pt2, std_dev) = semistoch_enpt2(&wf, &HAM, &EXCITE_GEN, eps, 100, 1000);
    // println!("Variational energy: {:.6}", wf.energy);
    // println!("PT energy: {:.6} +- {:.6}", e_pt2, std_dev);
    // println!("Total energy: {:.6} +- {:.6}", wf.energy + e_pt2, std_dev);
    //
    // // println!("Computing variational wavefunction and energy");
    // // variational(&HAM, &EXCITE_GEN, &mut wf);

}
