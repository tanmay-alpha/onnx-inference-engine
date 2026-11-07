//! Link the CLI against the Crucible shared library (`libcrucible.so`,
//! `libcrucible.dylib`, or `crucible.lib`/`crucible.dll`), built with
//! `cmake -S engine -B <dir> -DCRUCIBLE_BUILD_SHARED=ON && cmake --build <dir> --target crucible`.
//!
//! `CRUCIBLE_LIB_DIR` must point at the directory containing that library.
//! At run time the dynamic loader must also find it (e.g. `LD_LIBRARY_PATH`).

fn main() {
    println!("cargo:rerun-if-env-changed=CRUCIBLE_LIB_DIR");
    let dir = std::env::var("CRUCIBLE_LIB_DIR").unwrap_or_else(|_| {
        panic!(
            "CRUCIBLE_LIB_DIR is not set. Build the engine's shared library first:\n  \
             cmake -S engine -B build/shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRUCIBLE_BUILD_SHARED=ON\n  \
             cmake --build build/shared --target crucible\n\
             then build the CLI with CRUCIBLE_LIB_DIR=<absolute path to build/shared>."
        )
    });
    if !std::path::Path::new(&dir).is_dir() {
        panic!("CRUCIBLE_LIB_DIR='{dir}' is not a directory");
    }
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=crucible");
}
