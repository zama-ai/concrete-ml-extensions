mod compression;
mod computations;
mod encryption;
mod ml;

#[cfg(not(feature = "use_lib2"))]
mod lib;

#[cfg(feature = "use_lib2")]
mod lib2;

#[cfg(not(feature = "use_lib2"))]
pub use self::lib::*; // Re-export items from old lib

#[cfg(feature = "use_lib2")]
pub use self::lib2::*; // Re-export items from new lib
