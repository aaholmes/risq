use super::super::wf::Det;

// Spin determinant
// Syntax: for i in bits(det: u128): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
    Bits::new(det).into_iter()
}

pub fn det_bits(det: &Det) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
}

struct Bits {
    det: u128,
}

impl Bits {
    fn new(d: u128) -> Bits {
        Bits { det: d }
    }
}

impl IntoIterator for Bits {
    type Item = i32;
    type IntoIter = BitsIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        BitsIntoIterator {
            bits_left: self.det,
        }
    }
}

struct BitsIntoIterator {
    bits_left: u128,
}

impl Iterator for BitsIntoIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.bits_left == 0 {
            return None;
        };
        let res: i32 = self.bits_left.trailing_zeros() as i32;
        self.bits_left &= !(1 << res);
        Some(res)
    }
}

pub fn parity(mut n: u128) -> i32 {
    // Returns 1 if even number of bits, -1 if odd number
    n ^= n >> 64;
    n ^= n >> 32;
    n ^= n >> 16;
    n ^= n >> 8;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    1 - 2 * ((n & 1) as i32)
}

fn parity_brute_force(n: u128) -> i32 {
    let mut out: i32 = 0;
    for _ in bits(n) {
        out ^= 1;
    }
    1 - 2 * out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity() {
        for i in vec![14, 15, 27, 1919, 4958202, 15<<64, 1<<127, (1<<126) + (1<<65)] {
            println!("Parity({}) = {} = {}", i, parity(i), parity_brute_force(i));
            assert_eq!(parity(i), parity_brute_force(i));
        }
    }
}