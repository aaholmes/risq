#![crate_name = "risq"]
#![crate_type = "bin"]
#![doc(html_root_url = "https://aaholmes.github.io/risq/")]
#![doc(
    html_logo_url = "https://wherethewindsblow.com/wp-content/uploads/2020/11/crab_dice_red_white.jpg"
)]

//! Rust Implementation of Semistochastic Quantum chemistry (`risq`) implements an efficient selected
//! configuration interaction algorithm called Semistochastic Heat-bath Configuration Interaction
//! (SHCI).

#[macro_use]
extern crate lazy_static;
extern crate alloc;
use std::time::Instant;
mod ham;
use ham::read_ints::read_ints;
use ham::Ham;
mod excite;
use excite::init::{init_excite_generator, ExciteGenerator};
pub mod wf;
use wf::{init_var_wf, VarWf};
mod pt;
mod rng;
mod semistoch;
mod stoch;
mod utils;
mod var;
// mod projector;

use crate::pt::perturbative;
use crate::var::variational;
use utils::read_input::{read_input, Global};

fn main() {
    let start: Instant = Instant::now();

    println!(" //==================================================================\\\\");
    println!("//   Rust Implementation of Semistochastic Quantum chemistry (RISQ)   \\\\");
    println!("\\\\                        Adam A Holmes, 2021                         //");
    println!(" \\\\==================================================================//");

    println!("\n\n=====\nSetup\n=====\n");
    let start_setup: Instant = Instant::now();

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
    let mut wf: VarWf = init_var_wf(&GLOBAL, &HAM, &EXCITE_GEN);
    wf.print();
    println!("Time for setup: {:?}", start_setup.elapsed());

    println!("\n\n=================\nVariational stage\n=================\n");
    let start_var: Instant = Instant::now();
    variational(&GLOBAL, &HAM, &EXCITE_GEN, &mut wf);
    println!("Time for variational stage: {:?}", start_var.elapsed());

    println!("\n\n==================\nPerturbative stage\n==================\n");
    let start_enpt2: Instant = Instant::now();
    perturbative(&GLOBAL, &HAM, &EXCITE_GEN, &wf.wf);
    println!("Time for perturbative stage: {:?}", start_enpt2.elapsed());

    println!("Total Time: {:?}", start.elapsed());
}
