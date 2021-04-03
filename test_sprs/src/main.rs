fn main() {
    use sprs::{CsMat, CsVec};
    let a = CsMat::new_csc((5, 5),
                       vec![0, 2, 4, 5],
                       vec![0, 1, 0, 2, 2],
                       vec![1., 2., 3., 4., 5.]);
    let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    let y = &a * &x;
}
