#[macro_use]
extern crate lazy_static;

mod ham;
use ham::Ham;

mod excite;

mod wf;
use wf::{init_wf, Det};

mod utils;
use utils::read_input::{Global, read_input};
use utils::bits::bits;

fn main() {
    println!("Reading input file");
    lazy_static! {
        static ref global: Global = read_input("in.json").unwrap();
    }

    println!("Testing init wf and add a det");
    let mut wf = init_wf(&global);
    wf.print();
    wf.add_det(Det { up: 23, dn: 27 });
    wf.print();

    println!("Reading integrals");
    let mut ham: Ham = Ham::default();
    ham.read_ints(&global, "FCIDUMP");
    //println!("Nuc term: {}", ham.ints.nuc);
    //println!("{} {} {}", ham.get_int(0, 0, 0, 0), ham.get_int(1, -1, 0, 0), ham.get_int(-4, 3, 2, 1));
    for i in bits(27) {
        println!("{}", i);
    }
    let d = Det { up: 63, dn: 63 };
    let hf = ham.ham_diag(&d);
    println!("Energy of det {} {} is {}", d.up, d.dn, hf);
}
