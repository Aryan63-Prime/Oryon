//! Qualcomm Hexagon NPU (QNN HTP) runtime capability probe.
//! Checks whether FastRPC DSP transport and Qualcomm QNN HTP libraries are reachable
//! on this Snapdragon device; falls back cleanly to ARM64 NEON Oryon V2 CPU otherwise.

use std::path::Path;

#[derive(Debug, Clone)]
pub struct HardwareStatus {
    pub has_fastrpc_device: bool,
    pub has_qnn_htp_lib: bool,
    pub active_backend: &'static str,
}

pub fn probe_hardware(native_lib_dir: &str) -> HardwareStatus {
    let has_fastrpc_device =
        Path::new("/dev/adsprpc-smd").exists() || Path::new("/dev/fastrpc-cdsp").exists();

    let candidate_paths = [
        format!("{}/libQnnHtp.so", native_lib_dir),
        "/vendor/lib64/libQnnHtp.so".to_string(),
        "/system/vendor/lib64/libQnnHtp.so".to_string(),
    ];
    let has_qnn_htp_lib = candidate_paths.iter().any(|p| Path::new(p).exists());

    // Active execution uses ARM64 NEON SIMD until a compiled QNN HTP context binary (.qnn)
    // and Qualcomm QNN SDK libraries are bundled into app/src/main/jniLibs/arm64-v8a/.
    let active_backend = if has_fastrpc_device && has_qnn_htp_lib {
        "Hexagon NPU Ready (QNN HTP detected) + ARM64 NEON"
    } else if has_fastrpc_device {
        "Snapdragon 8 Elite ARM64 NEON (Hexagon FastRPC present; bundle QNN SDK for NPU)"
    } else {
        "ARM64 NEON CPU"
    };

    HardwareStatus {
        has_fastrpc_device,
        has_qnn_htp_lib,
        active_backend,
    }
}
