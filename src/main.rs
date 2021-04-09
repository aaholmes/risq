mod global;

mod ham;
use ham::Ham;

mod excite;

mod wf;
use wf::{init_wf, Det};

mod utils;
use utils::bits::bits;

fn main() {
    println!("Testing init wf and add a det");
    let mut wf = init_wf();
    wf.print();
    wf.add_det(Det { up: 23, dn: 27 });
    wf.print();

    println!("Reading input file");
    let mut ham: Ham = Ham::default();
    ham.read_ints("FCIDUMP");
    //println!("Nuc term: {}", ham.ints.nuc);
    //println!("{} {} {}", ham.get_int(0, 0, 0, 0), ham.get_int(1, -1, 0, 0), ham.get_int(-4, 3, 2, 1));
    for i in bits(27) {
        println!("{}", i);
    }
    let d = Det { up: 63, dn: 63 };
    let hf = ham.ham_diag(&d);
    println!("Energy of det {} {} is {}", d.up, d.dn, hf);
}
