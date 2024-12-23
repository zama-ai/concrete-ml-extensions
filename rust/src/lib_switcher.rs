// Conditionally include `lib2.rs` if the `use_lib2` feature is enabled.
#[cfg(feature = "use_lib2")]
mod lib2;

// Include the default `lib.rs` if the `use_lib2` feature is not enabled.
#[cfg(not(feature = "use_lib2"))]
mod lib1;

#[cfg(not(feature = "use_lib2"))]
pub use self::lib1::*; // Re-export items from the default library.

#[cfg(feature = "use_lib2")]
pub use self::lib2::*; // Re-export items from `lib2`.
