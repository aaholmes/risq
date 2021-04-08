// Spin determinant
// Syntax: for i in bits(det): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
    Bits::new(det).into_iter()
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
