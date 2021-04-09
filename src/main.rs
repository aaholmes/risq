#[macro_use]
extern crate lazy_static;

mod ham;
use ham::Ham;

mod excite;

mod wf;
use wf::{init_wf, Det};

//mod var;

mod utils;
use utils::bits::bits;
use utils::read_input::{Global, read_input};

fn main() {

    println!("Reading input file");
    lazy_static! {
        static ref global: Global = read_input("in.json").unwrap();
    }

    println!("Reading integrals");
    let mut ham: Ham = Ham::default();
    ham.read_ints(&global, "FCIDUMP");

    println!("Initializing wavfefuntion");
    let mut wf = init_wf(&global, &ham);
    wf.print();

}
