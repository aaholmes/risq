use std::collections::HashMap;

// Global variables
static NORB: u32 = 8;
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

fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

impl Wf {
    fn print(&self) {
        println!(
            "Wavefunction has {} dets with energy {}",
            self.n, self.energy
        );
        for (d, c) in self.dets.iter().zip(self.coeffs.iter()) {
            println!("{} {}   {}", fmt_det(d.up), fmt_det(d.dn), c);
        }
    }

    fn add_det(&mut self, d: Det) {
        //TODO: implement hashmap
        //if (d in self.inds) {
        self.n += 1;
        //TODO: implement hashmap
        //wf.inds.insert(d, self.n);
        self.dets.push(d);
        self.coeffs.push(0.0);
        //TODO: implement diag elems
        self.diags.push(1.0);
        //}
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
    let mut wf = init_wf();
    wf.print();
    wf.add_det(Det { up: 23, dn: 25 });
    wf.print();
}
