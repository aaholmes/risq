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
use var::variational;

mod utils;
mod stoch;
mod semistoch;

use utils::read_input::{Global, read_input};
use crate::semistoch::semistoch_matmul;

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

    let eps = 0.1;
    let n_samples = 10;
    println!("Calling semistoch_matmul!");
    semistoch_matmul(wf, &HAM, &EXCITE_GEN, eps, n_samples);

    // println!("Computing variational wavefunction and energy");
    // variational(&HAM, &EXCITE_GEN, &mut wf);

}
