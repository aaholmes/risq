use super::super::wf::Det;

// Spin determinant
// Syntax: for i in bits(det: u128): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
    Bits::new(det).into_iter()
}

pub fn det_bits(det: &Det) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
}




pub fn ibset(n: u128, b: i32) -> u128 {
    n | (1 << b)
}

pub fn ibclr(n: u128, b: i32) -> u128 {
    n & !(1 << b)
}

pub fn btest(n: u128, b: i32) -> bool {
    !(n & (1 << b) == 0)
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
        for i in vec![
            14,
            15,
            27,
            1919,
            4958202,
            15 << 64,
            1 << 127,
            (1 << 126) + (1 << 65),
        ] {
            println!("Parity({}) = {} = {}", i, parity(i), parity_brute_force(i));
            assert_eq!(parity(i), parity_brute_force(i));
        }
    }
}
