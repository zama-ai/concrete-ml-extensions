type Scalar = u64;

mod compression;
mod computations;
mod encryption;
mod ml;
mod fhext_classes;

#[cfg(not(feature = "swift_bindings"))]
mod lib_python;

#[cfg(feature = "swift_bindings")]
mod lib_swift;

#[cfg(not(feature = "swift_bindings"))]
pub use self::lib_python::*; // Re-export items from old lib

#[cfg(feature = "swift_bindings")]
pub use self::lib_swift::*; // Re-export items from new lib
