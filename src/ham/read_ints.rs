// Read integrals from an FCIDUMP file into the Ham data structure

extern crate lexical;
use lexical::parse;

use crate::utils::read_input::Global;
use crate::ham::Ham;
use crate::utils::ints::{read_lines, combine_2, combine_4};

#[derive(Default)]
pub struct Ints {
    pub(crate) nuc: f64,           // Nuclear-nuclear integral
    pub(crate) one_body: Vec<f64>, // One-body integrals
    pub(crate) two_body: Vec<f64>, // Two-body integrals
}

pub fn read_ints(global: &Global, filename: &str) -> Ham {
    // Read integrals, put them into self.ints
    // Ints are stored starting with index 1 (following the FCIDUMP file they're read from)
    let mut ham: Ham = Ham::default();
    ham.ints.one_body = vec![0.0; combine_2(global.norb + 1, global.norb + 1)];
    ham.ints.two_body = vec![0.0; combine_4(global.norb + 1, global.norb + 1, global.norb + 1, global.norb + 1)];
    if let Ok(lines) = read_lines(filename) {
        // Consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(read_str) = line {
                let mut str_split = read_str.split_whitespace();
                let i: f64;
                match parse(str_split.next().unwrap()) {
                    Ok(v) => i = v,
                    Err(_) => continue, // Skip header lines that don't begin with a float
                }
                let p: i32 = parse(str_split.next().unwrap()).unwrap();
                let q: i32 = parse(str_split.next().unwrap()).unwrap();
                let r: i32 = parse(str_split.next().unwrap()).unwrap();
                let s: i32 = parse(str_split.next().unwrap()).unwrap();
                if p == 0 && q == 0 && r == 0 && s == 0 {
                    ham.ints.nuc = i;
                } else if r == 0 && s == 0 {
                    ham.ints.one_body[combine_2(p, q)] = i;
                } else {
                    ham.ints.two_body[combine_4(p, q, r, s)] = i;
                }
            }
        }
    }
    ham
}
