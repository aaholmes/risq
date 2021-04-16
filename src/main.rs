#[macro_use]
extern crate lazy_static;

mod ham;
use ham::{Ham, read_ints};

mod excite;
use excite::{ExciteGenerator, init_excite_generator};

mod wf;
use wf::{Wf, init_wf};

mod var;
use var::variational;

mod utils;
use utils::read_input::{Global, read_input};

fn main() {

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
    let mut wf: Wf = init_wf(&global, &ham, &excite_gen);
    wf.print();

    for i in 0..20 {
        println!("Eps = {}", wf.eps_iter.next().unwrap());
    }

    // println!("Computing variational wavefunction and energy");
    // variational(&global, &ham, &excite_gen, &mut wf);

}
