extern crate lexical;

use std::collections::HashMap;

use super::global::{EPS, NDN, NORB, NUP};
use super::utils::ints::{combine_2, combine_4, read_lines};
use super::wf::Det;

// Orbital pair
pub struct OPair(i32, i32);

// Double excitation triplet (r, s, |H|)
pub struct Doub {
    target: OPair,
    abs_h: f64,
}

#[derive(Default)]
pub struct Ints {
    nuc: f64,           // Nuclear-nuclear integral
    one_body: Vec<f64>, // One-body integrals
    two_body: Vec<f64>, // Two-body integrals
}

// Hamiltonian, containing both integrals and heat-bath hashmap of double excitations
#[derive(Default)]
pub struct Ham {
    // Heat-bath double excitation generator:
    // each electron pair points to a sorted vector of double excitations
    doub_generator: HashMap<OPair, Vec<Doub>>,
    // Integrals are a one-index vector; to get any integral, use Ham.get_int(p, q, r, s)
    ints: Ints,
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
