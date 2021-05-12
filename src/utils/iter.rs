// // Iterator tools
//
// // Empty iterator
// pub fn empty<T>() -> impl Iterator<Item=T> {
//     Empty::new().into_iter()
// }
//
// // Backend for bits()
//
// struct Empty<T> {
//     none: Option<T>
// }
//
// impl<T> Empty<T> {
//     fn new() -> Empty<T> {
//         Empty { none: None }
//     }
// }
//
// impl<T> IntoIterator for Empty<T> {
//     type Item = T;
//     type IntoIter = EmptyIntoIterator<T>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         EmptyIntoIterator { none: None }
//     }
// }
//
// struct EmptyIntoIterator<T> {
//     none: Option<T>
// }
//
// impl<T> Iterator for EmptyIntoIterator<T> {
//     type Item = T;
//
//     fn next(&mut self) -> Option<T> {
//         None
//     }
// }
