use std::collections::HashMap;

// Global variables
static NORB: u32 = 28;
static NUP: u32 = 4;
static NDN: u32 = 3;
static EPS: f64 = 1e-6;

// Determinant
struct Det {
    up: u128,
    dn: u128,
}

// Wavefunction
#[derive(Default)]
struct Wf {
    n: u64,                  // number of dets
    inds: HashMap<Det, u64>, // hashtable : det -> u64 for looking up index by det
    dets: Vec<Det>,          // for looking up det by index
    coeffs: Vec<f64>,        // coefficients
    diags: Vec<f64>, // diagonal elements of Hamiltonian (so new diagonal elements can be computed quickly)
    energy: f64,     // variational energy
}

// Orbital pair
struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
struct Doub {
    target: OPair,
    abs_h: f64,
}

// Hamiltonian, containing both integrals and heat-bath hashmap of double excitations
struct Ham {
    doubs: HashMap<OPair, Vec<Doub>>, // Each electron pair points to a sorted vector of double excitations
}

impl Det {
    fn print(&self) {
        println!("{} {}", format!("{:b}", self.up), format!("{:b}", self.dn));
    }
}

impl Wf {
    fn print(&self) {
        println!(
            "Wavefunction has {} dets with energy {}",
            self.n, self.energy
        );
        for (d, c) in self.dets.iter().zip(self.coeffs.iter()) {
            println!("{} {} {}", format!("{:b}", d.up), format!("{:b}", d.dn), c);
        }
    }
}

// Init wf to the HF det (only needs to be called once)
fn init_wf() -> Wf {
    let mut wf: Wf = Wf::default();
    wf.n = 1;
    let one: u128 = 1;
    let hf = Det {
        up: ((one << NUP) - 1),
        dn: ((one << NDN) - 1),
    };
    //TODO: Implement Eq, Hash so that we can use this hashmap
    //wf.inds.insert(hf, 0);
    wf.dets.push(hf);
    wf.coeffs.push(1.0);
    //TODO: compute diag elem (only time it ever needs to be calculated directly)
    wf.diags.push(1.0);
    wf.energy = wf.diags[0];
    wf
}

fn main() {
    let wf = init_wf();
    println!("n = {}", wf.n);
    wf.dets[0].print();
    wf.print();
}
