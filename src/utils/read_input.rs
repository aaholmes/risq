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
    pub norb_core: i32,
    pub nup: i32,
    pub ndn: i32,
    pub n_states: i32,
    pub eps_var: f64,
    pub eps_pt: f64,
}

pub fn read_input<P: AsRef<Path>>(path: P) -> Result<Global> {

    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let global: Global = from_reader(reader)?;

    Ok(global)
}