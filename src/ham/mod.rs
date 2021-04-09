extern crate lexical;
use lexical::parse;

use std::collections::HashMap;

use super::global::{EPS, NDN, NORB, NUP};
use super::utils::bits::{bits, det_bits};
use super::utils::ints::{combine_2, combine_4, permute, permute_2, read_lines};
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
        // Ints are stored starting with index 1 (following the FCIDUMP file they're read from)
        self.ints.one_body = vec![0.0; combine_2(NORB + 1, NORB + 1)];
        self.ints.two_body = vec![0.0; combine_4(NORB + 1, NORB + 1, NORB + 1, NORB + 1)];
        if let Ok(lines) = read_lines(filename) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(read_str) = line {
                    let mut str_split = read_str.split_whitespace();
                    let mut i: f64 = 0.0;
                    match parse(str_split.next().unwrap()) {
                        Ok(v) => i = v,
                        Err(_) => continue, // Skip header lines that don't begin with a float
                    }
                    let p: i32 = parse(str_split.next().unwrap()).unwrap();
                    let q: i32 = parse(str_split.next().unwrap()).unwrap();
                    let r: i32 = parse(str_split.next().unwrap()).unwrap();
                    let s: i32 = parse(str_split.next().unwrap()).unwrap();
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

    pub fn ham_diag(self, det: &Det) -> f64 {
        // Get the diagonal element corresponding to this determinant
        // Should only be called once
        println!("Warning: Computing diagonal element (should only happen once!)");

        // nuclear-nuclear component
        let mut diag: f64 = self.ints.nuc;
        //println!("nuclear energy= {}", diag);

        // one-body component
        for i in det_bits(det) {
            //println!("Updating one-body part for orbital {}", i);
            diag += self.get_int(i + 1, i + 1, 0, 0);
        }

        // two-body component
        for i in bits(det.up) {
            for j in bits(det.up) {
                diag += self.get_int(i + 1, i + 1, j + 1, j + 1) - self.get_int(i + 1, j + 1, j + 1, i + 1);
            }
            for j in bits(det.dn) {
                diag += self.get_int(i + 1, i + 1, j + 1, j + 1);
            }
        }
        for i in bits(det.dn) {
            for j in bits(det.dn) {
                diag += self.get_int(i + 1, i + 1, j + 1, j + 1) - self.get_int(i + 1, j + 1, j + 1, i + 1);
            }
            for j in bits(det.up) {
                diag += self.get_int(i + 1, i + 1, j + 1, j + 1);
            }
        }
        diag
    }

    pub fn ham_sing(self, det1: &Det, det2: &Det) -> f64 {
        // Get the single excitation matrix element corresponding to
        // the excitation from det1 to det2
        let mut out: f64;
        if det1.dn == det2.dn {
            let i: i32 = (det1.up & !det2.up).trailing_zeros() as i32;
            let j: i32 = (det2.up & !det1.up).trailing_zeros() as i32;
            // One-body term
            out = (permute(det1.up, [i, j]) as f64) * self.get_int(i + 1, j + 1, 0, 0);
            // Two-body term
            for k in bits(det1.up) {
                out += self.get_int(i + 1, j + 1, k + 1, k + 1) - self.get_int(i + 1, k + 1, k + 1, j + 1);
            }
            out *= permute(det1.up, [i, j]) as f64;
        } else {
            let i: i32 = (det1.dn & !det2.dn).trailing_zeros() as i32;
            let j: i32 = (det2.dn & !det1.dn).trailing_zeros() as i32;
            // One-body term
            out = (permute(det1.dn, [i, j]) as f64) * self.get_int(i + 1, j + 1, 0, 0);
            // Two-body term
            for k in bits(det1.dn) {
                out += self.get_int(i + 1, j + 1, k + 1, k + 1) - self.get_int(i + 1, k + 1, k + 1, j + 1);
            }
            out *= permute(det1.dn, [i, j]) as f64;
        }
        out
    }

    pub fn ham_doub(&self, det1: &Det, det2: &Det) -> f64 {
        // Get the double excitation matrix element corresponding to
        // the excitation from det1 to det2
        if det1.dn == det2.dn {
            // Same spin, up
            let mut ind: [i32; 4] = [0; 4];
            let mut n = 0;
            for i in bits(det1.up & !det2.up) {
                ind[n] = i;
                n += 1;
            }
            for i in bits(det2.up & !det1.up) {
                ind[n] = i;
                n += 1;
            }
            (permute_2(det1.up, ind) as f64) * (self.get_int(ind[0], ind[1], ind[2], ind[3]) - self.get_int(ind[0], ind[3], ind[2], ind[1]))
        } else if det1.up == det2.up {
            // Same spin, dn
            let mut ind: [i32; 4] = [0; 4];
            let mut n = 0;
            for i in bits(det1.dn & !det2.dn) {
                ind[n] = i;
                n += 1;
            }
            for i in bits(det2.dn & !det1.dn) {
                ind[n] = i;
                n += 1;
            }
            (permute_2(det1.dn, ind) as f64) * (self.get_int(ind[0], ind[1], ind[2], ind[3]) - self.get_int(ind[0], ind[3], ind[2], ind[1]))
        } else {
            // Opposite spin
            let mut ind1: [i32; 2] = [0; 2];
            let mut n = 0;
            for i in bits(det1.up & !det2.up) {
                ind1[n] = i;
                n += 1;
            }
            for i in bits(det2.up & !det1.up) {
                ind1[n] = i;
                n += 1;
            }
            let mut ind2: [i32; 2] = [0; 2];
            let mut n = 0;
            for i in bits(det1.dn & !det2.dn) {
                ind2[n] = i;
                n += 1;
            }
            for i in bits(det2.dn & !det1.dn) {
                ind2[n] = i;
                n += 1;
            }
            ((permute(det1.up, ind1) * permute(det1.dn, ind2)) as f64) * self.get_int(ind1[0], ind2[0], ind1[1], ind2[1])
        }
    }
}
