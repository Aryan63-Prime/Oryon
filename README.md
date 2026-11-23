# Oryon AI — Native Android App (Rust JNI + GGUF + Hexagon NPU Probe)

A standalone Android app (`arm64-v8a`, Jetpack Compose + Rust JNI) for your **iQOO 13 (Snapdragon 8 Elite, 12 GB RAM)**—no Termux required.

## What's Inside
- **`rust-core/`**: Native Rust shared library (`liboryon_lm.so`) with:
  - Fixed BOS handling, zero dead-code warnings, and tuned thread-spawn threshold (`>= 256` rows)
  - **Native GGUF v2/v3 loader** (`Q4_0`, `Q8_0`, `F16`, `F32`) with embedded tokenizer support for Llama-3 / Qwen2.5 / SmolLM2
  - **ARM64 NEON SIMD** quantized dot products + 6-thread scoped parallelism
  - **Qualcomm Hexagon NPU (QNN HTP) hardware probe** (`/dev/adsprpc-smd` + `libQnnHtp.so` detection, with automatic ARM64 NEON fallback)
- **`app/`**: Material 3 dark-mode Jetpack Compose UI with streaming token output, live `tok/s` telemetry, thread/temperature sliders, and an Android file picker (`Load GGUF`) so you can load any `Q4_0` or `Q8_0` `.gguf` file from your phone's Downloads folder.
- **Bundled starter checkpoint**: `app/src/main/assets/oryon_micro_q8.bin` + `tokenizer.bin` so the app launches and verifies immediately offline.

## Get an Installable APK (Zero PC / Zero Termux)
1. Create a free GitHub repository and upload the contents of `oryon-android/`.
2. Open the **Actions** tab → **Build Oryon AI Android APK (ARM64)** → download `oryon-ai-iqoo13-apk` (`app-debug.apk`).
3. Tap `app-debug.apk` on your iQOO 13 to install.

*(Or build locally in Android Studio after running `cargo ndk -t arm64-v8a -o ../app/src/main/jniLibs build --release` inside `rust-core/`.)*

## Loading a Real 3B–7B Model (No Termux)
1. Download a `Q4_0` or `Q8_0` `.gguf` file directly in your phone's browser (for example, `qwen2.5-3b-instruct-q4_0.gguf` from [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)).
2. Open **Oryon AI**, tap **Load GGUF**, and select the downloaded file.

## Enabling Qualcomm Hexagon NPU (QNN HTP)
Qualcomm's Hexagon NPU libraries are proprietary and cannot be redistributed in this archive. To enable NPU detection/execution, download the [Qualcomm AI Engine Direct (QNN) SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) and place `libQnnHtp.so`, `libQnnSystem.so`, and the Hexagon v79 skel library into `app/src/main/jniLibs/arm64-v8a/` before building.
