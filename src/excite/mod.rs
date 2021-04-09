// Excitation generation module (includes usual heat-bath routines)

//use std::collections::HashMap;
//
//// Orbital pair
//pub struct OPair(i32, i32);
//
//// Double excitation triplet (r, s, |H|)
//pub struct Doub {
//    target: OPair,
//    abs_h: f64,
//}
//
//// Heat-bath excitation generator:
//// each electron pair points to a sorted vector of double excitations
//pub struct ExciteGenerator {
//    same_spin_doub_generator: HashMap<i32, Vec<Doub>>,
//    opp_spin_doub_generator: HashMap<OPair, Vec<Doub>>,
//}
//
//impl ExciteGenerator {
//    pub fn init(&self, ham: Ham) -> () {
//        // Initialize by sorting double excitation element for all pairs
//        // Same spin
//        let mut v: Vec<Doub>;
//        for p in 0..NORB {
//            for q in p+1..NORB {
//                v = vec![];
//                for r in 0..NORB {
//                    if p == r || q == r { continue; };
//                    for s in r+1..NORB {
//                        if p == s || q == s { continue; };
//                        // Compute H elem
//                        v.push(r, s, (ham.get_int(p + 1, q + 1, r + 1, s + 1) - ham.get_int(p + 1, q + 1, s + 1, r + 1)).abs());
//                    }
//                }
//                // Sort v
//
//                self.same_spin_doub_generator.insert(combine_2(p, q), v);
//            }
//        }
//    }
//
//    pub fn double_excite(&self, det: Det, eps: f64) {
//        // Generate all double excitations from the given determinant
//        // larger in magnitude than eps
//        // Same spin doubles
//        for i in bits(det.up) {
//            for j in bits(det.up) {
//                for excite in self.same_spin_doub_generator[combine_2(i, j)] {
//                    if (excite.abs_h < eps) {
//                        break;
//                    }
//                    // Add this excitation here
//                }
//            }
//        }
//    }
//}