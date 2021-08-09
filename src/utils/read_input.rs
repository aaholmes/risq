// Read input file into Global variable data structure

extern crate serde;
extern crate serde_json;

use serde::Deserialize;
use serde_json::{from_reader, Result};

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Deserialize, Debug)]
pub struct Global {
    pub norb: i32,
    pub norb_core: i32, // freezes the norb_core orbs with lowest diagonal fock elements
    pub nup: i32,
    pub ndn: i32,
    pub n_states: i32, // n_states > 1 not yet implemented
    pub eps_var: f64,
    pub eps_pt_dtm: f64, // division between deterministic and stochastic components of semistochastic PT
    pub opp_algo: i32, // 1-3 for which of the opposite-spin algorithms to use (currently use 2)
    pub same_algo: i32, // 1-2 for which of the same-spin algorithms to use (currently use 1)
}

pub fn read_input<P: AsRef<Path>>(path: P) -> Result<Global> {

    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let global: Global = from_reader(reader)?;

    println!("\nInput:");
    println!("  norb: {}", global.norb);
    println!("  norb_core: {}", global.norb_core);
    println!("  nup: {}", global.nup);
    println!("  ndn: {}", global.ndn);
    println!("  n_states: {}", global.n_states);
    println!("  eps_var: {}", global.eps_var);
    println!("  eps_pt_dtm: {}", global.eps_pt_dtm);
    println!("  opp_algo: {}", global.opp_algo);
    println!("  same_algo: {}\n", global.same_algo);

    // if n_states != 1 {
    //     Err("n_states > 1 not yet implemented!")
    // }
    // if !(opp_algo >=1 && opp_algo <= 3) {
    //     Err("opp_algo must be 1, 2, or 3!")
    // }
    // if !(same_algo >=1 && same_algo <= 2) {
    //     return Err("same_algo must be 1 or 2!");
    // }

    Ok(global)
}