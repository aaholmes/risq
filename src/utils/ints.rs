use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use super::bits::parity;

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn combine_2(p: i32, q: i32) -> usize {
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

pub fn combine_4(p: i32, q: i32, r: i32, s: i32) -> usize {
    // Combine 4 indices in a unique way
    combine_2_usize(combine_2(p, q), combine_2(r, s))
}

pub fn permute(det: u128, v: [i32; 2]) -> i32 {
    // Expect Vec to be a 2-index vector
    let diff: u128 = det & (((1 << v[0]) - 1) ^ ((1 << v[1]) - 1));
    parity(diff)
}

pub fn permute_2(det: u128, v: [i32; 4]) -> i32 {
    // Expect Vec to be a 4-index vector
    let diff: u128 = det & (((1 << v[0]) - 1) ^ ((1 << v[1]) - 1) ^ ((1 << v[2]) - 1) ^ ((1 << v[3]) - 1));
    parity(diff)
}
