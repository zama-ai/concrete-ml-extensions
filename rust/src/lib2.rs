uniffi::setup_scaffolding!();

#[uniffi::export]
pub fn say_hello() {
    println!("Hello from lib2!");
}

#[uniffi::export]
pub fn add(a: u64, b: u64) -> u64 {
    a + b
}
