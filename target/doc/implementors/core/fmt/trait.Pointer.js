(function() {var implementors = {};
implementors["crossbeam_epoch"] = [{"text":"impl&lt;T:&nbsp;?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a> + <a class=\"trait\" href=\"crossbeam_epoch/trait.Pointable.html\" title=\"trait crossbeam_epoch::Pointable\">Pointable</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a> for <a class=\"struct\" href=\"crossbeam_epoch/struct.Atomic.html\" title=\"struct crossbeam_epoch::Atomic\">Atomic</a>&lt;T&gt;","synthetic":false,"types":["crossbeam_epoch::atomic::Atomic"]},{"text":"impl&lt;T:&nbsp;?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a> + <a class=\"trait\" href=\"crossbeam_epoch/trait.Pointable.html\" title=\"trait crossbeam_epoch::Pointable\">Pointable</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a> for <a class=\"struct\" href=\"crossbeam_epoch/struct.Shared.html\" title=\"struct crossbeam_epoch::Shared\">Shared</a>&lt;'_, T&gt;","synthetic":false,"types":["crossbeam_epoch::atomic::Shared"]}];
implementors["itertools"] = [{"text":"impl&lt;'a, I&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a> for <a class=\"struct\" href=\"itertools/structs/struct.Format.html\" title=\"struct itertools::structs::Format\">Format</a>&lt;'a, I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/iter/traits/iterator/trait.Iterator.html\" title=\"trait core::iter::traits::iterator::Iterator\">Iterator</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;I::<a class=\"type\" href=\"https://doc.rust-lang.org/1.54.0/core/iter/traits/iterator/trait.Iterator.html#associatedtype.Item\" title=\"type core::iter::traits::iterator::Iterator::Item\">Item</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a>,&nbsp;</span>","synthetic":false,"types":["itertools::format::Format"]}];
implementors["nalgebra"] = [{"text":"impl&lt;T, R:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, C:&nbsp;<a class=\"trait\" href=\"nalgebra/base/dimension/trait.Dim.html\" title=\"trait nalgebra::base::dimension::Dim\">Dim</a>, S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a> for <a class=\"struct\" href=\"nalgebra/base/struct.Matrix.html\" title=\"struct nalgebra::base::Matrix\">Matrix</a>&lt;T, R, C, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"nalgebra/base/trait.Scalar.html\" title=\"trait nalgebra::base::Scalar\">Scalar</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/fmt/trait.Pointer.html\" title=\"trait core::fmt::Pointer\">Pointer</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"nalgebra/base/storage/trait.Storage.html\" title=\"trait nalgebra::base::storage::Storage\">Storage</a>&lt;T, R, C&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;<a class=\"struct\" href=\"nalgebra/base/default_allocator/struct.DefaultAllocator.html\" title=\"struct nalgebra::base::default_allocator::DefaultAllocator\">DefaultAllocator</a>: <a class=\"trait\" href=\"nalgebra/base/allocator/trait.Allocator.html\" title=\"trait nalgebra::base::allocator::Allocator\">Allocator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.54.0/std/primitive.usize.html\">usize</a>, R, C&gt;,&nbsp;</span>","synthetic":false,"types":["nalgebra::base::matrix::Matrix"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()