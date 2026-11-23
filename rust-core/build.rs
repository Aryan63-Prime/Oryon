use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let shader = PathBuf::from("src/oryon_matvec.spv");
    println!("cargo:rerun-if-changed={}", shader.display());

    let bytes = fs::read(&shader).unwrap_or_default();
    let mut words = Vec::with_capacity((bytes.len() + 3) / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    fs::write(
        out_dir.join("oryon_matvec_spv.rs"),
        format!("pub const MATVEC_SPIRV: &[u32] = &{:?};\n", words),
    )
    .unwrap();
}
