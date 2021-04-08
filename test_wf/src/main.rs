extern crate lexical;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// Global variables
static NORB: i32 = 28;
static NUP: i32 = 6;
static NDN: i32 = 6;
static EPS: f64 = 1e-6;

pub struct SpinDet{
    det: u128
}

impl SpinDet {
    fn new(d: u128) -> SpinDet {
        SpinDet {det: d}
    }
}

impl IntoIterator for SpinDet {
    type Item = i32;
    type IntoIter = SpinDetIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        SpinDetIntoIterator {
            bits_left: self.det,
        }
    }
}

pub struct SpinDetIntoIterator {
    bits_left: u128,
}

impl Iterator for SpinDetIntoIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        // If bits_left = 0, return None
        if self.bits_left == 0 {
            return None;
        }
        let res: i32 = self.bits_left.trailing_zeros() as i32; // Because orbs start with 1
        self.bits_left &= !(1 << res);
        Some(res)
    }
}

// Spin determinant
// Syntax: for i in bits(det): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
   SpinDet::new(det).into_iter()
}

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

#[derive(Default)]
struct Ints {
    nuc: f64, // Nuclear-nuclear integral
    one_body: Vec<f64>, // One-body integrals
    two_body: Vec<f64>, // Two-body integrals
}

// Hamiltonian, containing both integrals and heat-bath hashmap of double excitations
#[derive(Default)]
struct Ham {
    // Heat-bath double excitation generator:
    // each electron pair points to a sorted vector of double excitations
    doub_generator: HashMap<OPair, Vec<Doub>>,
    // Integrals are a one-index vector; to get any integral, use Ham.get_int(p, q, r, s)
    ints: Ints,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn combine_2(p: i32, q: i32) -> usize {
    // Combine 2 indices in a unique way
    let i = p.abs();
    let j = q.abs();
    if i < j {
        ((i * (i - 1)) / 2 + j) as usize
    } else {
        ((j * (j - 1)) / 2 + i) as usize
    }
}

fn combine_2_usize(i: usize, j: usize) -> usize {
    // Combine 2 indices in a unique way
    if i < j {
        (i * (i - 1)) / 2 + j
    } else {
        (j * (j - 1)) / 2 + i
    }
}

fn combine_4(p: i32, q: i32, r: i32, s: i32) -> usize {
    // Combine 4 indices in a unique way
    combine_2_usize(combine_2(p, q), combine_2(r, s))   
}

impl Ham {
    pub fn read_ints(&mut self, filename: &str) {
        // Read integrals, put them into self.ints
        self.ints.one_body = vec![0.0; combine_2(NORB, NORB) + 1];
        self.ints.two_body = vec![0.0; combine_4(NORB, NORB, NORB, NORB) + 1];
        if let Ok(lines) = read_lines(filename) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(read_str) = line {
                    let mut str_split = read_str.split_whitespace();
                    let i: f64 = lexical::parse(str_split.next().unwrap()).unwrap();
                    let p: i32 = lexical::parse(str_split.next().unwrap()).unwrap();
                    let q: i32 = lexical::parse(str_split.next().unwrap()).unwrap();
                    let r: i32 = lexical::parse(str_split.next().unwrap()).unwrap();
                    let s: i32 = lexical::parse(str_split.next().unwrap()).unwrap();
                    if p == 0 && q == 0 && r == 0 && s == 0 {
                        self.ints.nuc = i;
                    } else if r == 0 && s == 0 {
                        self.ints.one_body[combine_2(p, q)] = i;
                    } else {
                        self.ints.two_body[combine_4(p, q, r, s)] = i;
                    }
                }
            }
        }
    }

    fn get_int(&self, p: i32, q: i32, r: i32, s: i32) -> f64 {
        // Get the integral corresponding to pqrs
        // incorporates symmetries p-q, r-s, pq-rs
        // Insensitive to whether indices are positive or negative (up or dn spin)
        if p == 0 && q == 0 && r == 0 && s == 0 {
            self.ints.nuc
        } else if r == 0 && s == 0 {
            self.ints.one_body[combine_2(p, q)]
        } else {
            self.ints.two_body[combine_4(p, q, r, s)]
        }
    }

    pub fn ham_diag(det: &Det) -> f64 {
        // Get the diagonal element corresponding to this determinant
        // Should only be called once
        todo!()
    }

    pub fn ham_sing(det1: &Det, det2: &Det) -> f64 {
        // Get the single excitation matrix element corresponding to
        // the excitation from det1 to det2
        todo!()
    }

    pub fn ham_doub(det1: &Det, det2: &Det) -> f64 {
        // Get the double excitation matrix element corresponding to
        // the excitation from det1 to det2
        if det1.dn == det2.dn {
            // Same spin, up
            todo!()
        } else if det1.up == det2.up {
            // Same spin, dn
            todo!()
        } else {
            // Opposite spin
            todo!()
        }
    }
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
    println!("Testing init wf and add a det");
    let mut wf = init_wf();
    wf.print();
    wf.add_det(Det { up: 23, dn: 27 });
    wf.print();

    println!("Reading input file");
    let mut ham: Ham = Ham::default();
    ham.read_ints("FCIDUMP");
    println!("Nuc term: {}", ham.ints.nuc);
    println!("1 1 term: {}", ham.ints.one_body[combine_2(1, 1)]);
    println!("1 3 term: {}", ham.ints.one_body[combine_2(1, 3)]);
    println!("1 2 3 4 term: {}", ham.ints.two_body[combine_4(1, 2, 3, 4)]);
    println!("{} {} {}", ham.get_int(0, 0, 0, 0), ham.get_int(1, -1, 0, 0), ham.get_int(-4, 3, 2, 1));
    for i in bits(27) {
        println!("{}", i);
    }
}
