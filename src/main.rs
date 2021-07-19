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

mod utils;
mod stoch;
mod semistoch;
mod pt;

use utils::read_input::{Global, read_input};
use crate::var::variational;
use crate::semistoch::{semistoch_enpt2, old_semistoch_enpt2, fast_semistoch_enpt2, faster_semistoch_enpt2};

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

    let eps_pt = GLOBAL.eps_pt;
    let n_batches = 10;
    // let n_samples_per_batch_old = 10;
    let n_samples_per_batch_new = 20;

    // println!("\nCalling semistoch ENPT2 the old way with p ~ |c| using eps_pt = {}", eps_pt);
    // let (e_pt2, std_dev) = old_semistoch_enpt2(&wf, &HAM, &EXCITE_GEN, eps_pt, n_batches, n_samples_per_batch_old, false);
    // println!("Variational energy: {:.6}", wf.energy);
    // println!("PT energy: {:.6} +- {:.6}", e_pt2, std_dev);
    // println!("Total energy (old): {:.6} +- {:.6}", wf.energy + e_pt2, std_dev);

    println!("Calling semistoch ENPT2 the new way!");
    let (e_pt2, std_dev) = faster_semistoch_enpt2(&wf, &HAM, &EXCITE_GEN, eps_pt, n_batches, n_samples_per_batch_new);
    println!("Variational energy: {:.6}", wf.energy);
    println!("PT energy: {:.6} +- {:.6}", e_pt2, std_dev);
    println!("Total energy (new): {:.6} +- {:.6}", wf.energy + e_pt2, std_dev);

}
