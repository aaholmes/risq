//! Read integrals from an FCIDUMP file into the Ham data structure

extern crate lexical;
use lexical::parse;

use crate::ham::Ham;
use crate::utils::ints::{combine_2, combine_4, read_lines};
use crate::utils::read_input::Global;
use std::cmp::Ordering::Equal;

#[derive(Default)]
pub struct Ints {
    pub(crate) nuc: f64,           // Nuclear-nuclear integral
    pub(crate) one_body: Vec<f64>, // One-body integrals
    pub(crate) two_body: Vec<f64>, // Two-body integrals
}

/// Read integrals, put them into self.ints
/// Ints are stored starting with index 1 (following the FCIDUMP file they're read from)
/// Also, create core_orbs and valence_orbs lists using the diagonal Fock elements to determine
/// which norb_core orbitals to freeze
pub fn read_ints(global: &Global, filename: &str) -> Ham {
    let mut ham: Ham = Ham::default();
    // ham.diag_computed = false;
    ham.ints.one_body = vec![0.0; combine_2(global.norb + 1, global.norb + 1)];
    ham.ints.two_body = vec![
        0.0;
        combine_4(
            global.norb + 1,
            global.norb + 1,
            global.norb + 1,
            global.norb + 1
        )
    ];
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

        // Determine core and valence orbs using the diagonal Fock elements
        ham.core_orbs = Vec::with_capacity(global.norb as usize);
        ham.valence_orbs = Vec::with_capacity(global.norb as usize);

        // Sort diagonal elements in increasing order
        let mut fock_diag: Vec<f64> = Vec::with_capacity(global.norb as usize);
        let mut inds: Vec<i32> = Vec::with_capacity(global.norb as usize);
        for i in 0..global.norb {
            fock_diag.push(ham.one_body(i, i));
            inds.push(i);
        }
        fock_diag
            .iter()
            .zip(&inds)
            .collect::<Vec<_>>()
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));

        for (i, (_, ind)) in fock_diag
            .into_iter()
            .zip(inds)
            .collect::<Vec<_>>()
            .iter()
            .enumerate()
        {
            if i < global.norb_core as usize {
                ham.core_orbs.push(*ind);
            } else {
                ham.valence_orbs.push(*ind);
            }
        }
        println!("Core orbs: {:?}", ham.core_orbs);
        println!("Valence orbs: {:?}", ham.valence_orbs);
        //
        // // Finally, compute the screens of orbs that provide nonzero two-body contributions
        // // to single excitations
        // ham.screen_single_nonzero_direct = vec![vec![0; global.norb as usize]; global.norb as usize];
        // ham.screen_single_nonzero_direct_plus_exchange = vec![vec![0; global.norb as usize]; global.norb as usize];
        // for p in global.norb_core + 1 .. global.norb {
        //     for r in global.norb_core + 1 .. global.norb {
        //         if p == r { continue; }
        //         for q in global.norb_core + 1 .. global.norb {
        //             if ham.ints.tw            for k in bits(det1.up) {
        //                 out += self.direct_plus_exchange(i, k, j, k);
        //             }
        //             for k in bits(det1.dn) {
        //                 out += self.direct(i, k, j, k);
        //             }o
        //         }
        //     }
        // }
    }
    ham
}
