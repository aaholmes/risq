#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate itertools;

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
use utils::read_input::{Global, read_input};

fn main() {

    println!("ESP - Electronic Structure Package\nAdam A Holmes, 2021\n");

    println!("Reading input file");
    lazy_static! {
        static ref global: Global = read_input("in.json").unwrap();
    }

    println!("Reading integrals");
    lazy_static! {
        static ref ham: Ham = read_ints(&global, "FCIDUMP");
    }

    println!("Initializing excitation generator");
    lazy_static! {
        static ref excite_gen: ExciteGenerator = init_excite_generator(&global, &ham);
    }

    println!("Initializing wavefunction");
    let mut wf: Wf = init_var_wf(&global, &ham, &excite_gen);
    wf.print();

    println!("Computing variational wavefunction and energy");
    variational(&ham, &excite_gen, &mut wf);

}
