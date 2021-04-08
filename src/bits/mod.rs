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