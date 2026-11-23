use crate::gguf::load_gguf;
use crate::model::{RunState, Transformer};
use crate::npu_qnn::probe_hardware;
use crate::npu_qnn::qnn_status;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::BpeTokenizer;
use crate::vulkan_gpu::VulkanGpu;
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jfloat, jint, jlong, jstring};
use jni::JNIEnv;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicBool, Ordering};

struct EngineHandle {
    model: Transformer,
    tokenizer: BpeTokenizer,
    state: RunState,
    backend_info: String,
    backend_mode: i32,
    active_backend: String,
    gpu: VulkanGpu,
    cancelled: AtomicBool,
    native_lib_dir: String,
}

fn now_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeProbeHardware(
    mut env: JNIEnv,
    _class: JClass,
    native_lib_dir: JString,
) -> jstring {
    let dir: String = env
        .get_string(&native_lib_dir)
        .map(|s| s.into())
        .unwrap_or_default();
    let status = probe_hardware(&dir);
    env.new_string(status.active_backend)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeLoadModel(
    mut env: JNIEnv,
    _class: JClass,
    model_path: JString,
    tokenizer_path: JString,
    native_lib_dir: JString,
    max_ctx: jint,
) -> jlong {
    let m_path: String = match env.get_string(&model_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    let t_path: String = env
        .get_string(&tokenizer_path)
        .map(|s| s.into())
        .unwrap_or_default();
    let lib_dir: String = env
        .get_string(&native_lib_dir)
        .map(|s| s.into())
        .unwrap_or_default();

    let loaded = if m_path.to_lowercase().ends_with(".gguf") {
        load_gguf(&m_path, (max_ctx as usize).max(256))
    } else {
        Transformer::load_oryn(&m_path).and_then(|m| {
            let tok = BpeTokenizer::load(&t_path)?;
            Ok((m, tok))
        })
    };

    match loaded {
        Ok((model, tokenizer)) => {
            let _hw = probe_hardware(&lib_dir);
            let gpu = VulkanGpu::probe();
            let active_backend = if gpu.available {
                format!("GPU: {} (Vulkan)", gpu.name)
            } else {
                "CPU: ARM64 NEON (6 threads)".to_string()
            };
            let info = format!(
                "{:.2}M | {} | {:.1} MB RAM | {}",
                model.config.param_count() as f64 / 1e6,
                model.config.quant_type.name(),
                model.estimated_ram_mb(),
                active_backend
            );
            let state = RunState::new(&model.config);
            let handle = Box::new(EngineHandle {
                model,
                tokenizer,
                state,
                backend_info: info,
                backend_mode: 0,
                active_backend,
                gpu,
                cancelled: AtomicBool::new(false),
                native_lib_dir: lib_dir,
            });
            Box::into_raw(handle) as jlong
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeModelInfo(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    if handle == 0 {
        return std::ptr::null_mut();
    }
    let eng = unsafe { &*(handle as *const EngineHandle) };
    env.new_string(&eng.backend_info)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeSetBackend(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    mode: jint,
) -> jstring {
    if handle == 0 {
        return env.new_string("CPU: ARM64 NEON (6 threads)")
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut());
    }
    let eng = unsafe { &mut *(handle as *mut EngineHandle) };
    let status = match mode {
        1 => "CPU: ARM64 NEON (6 threads)".to_string(),
        2 => {
            if eng.gpu.available {
                format!("GPU: {} (Vulkan)", eng.gpu.gpu_name())
            } else {
                "CPU: ARM64 NEON (Vulkan unavailable)".to_string()
            }
        }
        3 => {
            let (ok, qnn) = qnn_status(&eng.native_lib_dir);
            if ok {
                qnn.to_string()
            } else {
                format!("CPU: ARM64 NEON ({}; fallback)", qnn)
            }
        }
        _ => {
            if eng.gpu.available {
                format!("GPU: {} (Vulkan)", eng.gpu.gpu_name())
            } else {
                "CPU: ARM64 NEON (6 threads)".to_string()
            }
        }
    };
    eng.backend_mode = mode;
    eng.active_backend = status.clone();
    eng.backend_info = eng.backend_info
        .rsplit_once(" | ")
        .map(|(prefix, _)| format!("{} | {}", prefix, status))
        .unwrap_or_else(|| status.clone());
    env.new_string(status)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeCancelGeneration(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        let eng = unsafe { &*(handle as *const EngineHandle) };
        eng.cancelled.store(true, Ordering::Release);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeGenerate(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    prompt: JString,
    max_tokens: jint,
    threads: jint,
    temperature: jfloat,
    min_p: jfloat,
    callback: JObject,
) -> jstring {
    if handle == 0 {
        return std::ptr::null_mut();
    }
    let eng = unsafe { &mut *(handle as *mut EngineHandle) };
    let prompt_str: String = match env.get_string(&prompt) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };

    let n_threads = (threads as usize).clamp(1, 8);
    let max_gen = (max_tokens as usize).clamp(1, 1024);
    let prompt_tokens = eng.tokenizer.encode(&prompt_str, eng.model.config.add_bos);
    let max_pos = eng
        .model
        .config
        .max_seq_len
        .min(prompt_tokens.len() + max_gen);

    let mut sampler = Sampler::new(
        SamplerConfig {
            temperature,
            min_p,
            ..Default::default()
        },
        now_seed(),
    );

    let mut history = Vec::with_capacity(max_pos);
    let mut pos = 0usize;
    let mut gpu_executed = false;

    for &pt in &prompt_tokens {
        if pos >= max_pos {
            break;
        }
        if eng.backend_mode == 2 {
            eng.model.forward(pt as usize, pos, &mut eng.state, n_threads,
                Some(&mut eng.gpu), &mut gpu_executed);
        } else {
            eng.model.forward(pt as usize, pos, &mut eng.state, n_threads, None, &mut gpu_executed);
        }
        history.push(pt);
        pos += 1;
    }

    let t0 = Instant::now();
    let mut generated = 0usize;
    let mut full_out = String::new();
    eng.cancelled.store(false, Ordering::Release);

    while generated < max_gen && pos < max_pos {
        if eng.cancelled.load(Ordering::Acquire) {
            break;
        }
        let next_tok = sampler.sample(&mut eng.state.logits, &history);
        if next_tok == eng.tokenizer.eos_id {
            break;
        }
        let piece = eng.tokenizer.decode_token(next_tok);
        full_out.push_str(&piece);

        // Stream token piece directly to Kotlin UI callback
        if let Ok(jpiece) = env.new_string(&piece) {
            let _ = env.call_method(
                &callback,
                "onToken",
                "(Ljava/lang/String;)V",
                &[JValue::Object(&jpiece)],
            );
        }

        // Stop cleanly on Q&A turn boundary for micro checkpoints
        if full_out.ends_with("\n\nQ:") || full_out.ends_with("<|im_end|>") {
            break;
        }

        history.push(next_tok);
        if eng.backend_mode == 2 {
            eng.model.forward(next_tok as usize, pos, &mut eng.state, n_threads,
                Some(&mut eng.gpu), &mut gpu_executed);
        } else {
            eng.model.forward(next_tok as usize, pos, &mut eng.state, n_threads, None, &mut gpu_executed);
        }
        pos += 1;
        generated += 1;
    }

    let secs = t0.elapsed().as_secs_f64().max(1e-6);
    let cancelled = eng.cancelled.load(Ordering::Acquire);
    let displayed_backend = if eng.backend_mode == 2 && gpu_executed {
        format!("GPU: {} (Vulkan)", eng.gpu.gpu_name())
    } else if eng.backend_mode == 2 {
        "CPU: ARM64 NEON (Vulkan unavailable)".to_string()
    } else {
        eng.active_backend.clone()
    };
    let stats = format!(
        "{} tokens | {:.1} tok/s | {}{}",
        generated,
        generated as f64 / secs,
        displayed_backend,
        if cancelled { " | cancelled" } else { "" }
    );
    env.new_string(stats)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_com_oryon_ai_OryonEngine_nativeRelease(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut EngineHandle);
        }
    }
}
