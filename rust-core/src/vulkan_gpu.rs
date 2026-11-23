use ash::{vk, Entry};
use crate::quant::WeightMatrix;

include!(concat!(env!("OUT_DIR"), "/oryon_matvec_spv.rs"));

pub struct VulkanGpu {
    pub available: bool,
    pub name: String,
    initialized: bool,
    pipeline_ready: bool,
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    physical: Option<vk::PhysicalDevice>,
    queue: Option<vk::Queue>,
    queue_family: Option<u32>,
    command_pool: Option<vk::CommandPool>,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set: Option<vk::DescriptorSet>,
    buffers: Option<PersistentBuffers>,
    weight_cache: std::collections::HashMap<usize, Vec<f32>>,
}

struct PersistentBuffers {
    x: vk::Buffer,
    w: vk::Buffer,
    s: vk::Buffer,
    y: vk::Buffer,
    memory_x: vk::DeviceMemory,
    memory_w: vk::DeviceMemory,
    memory_y: vk::DeviceMemory,
    capacity: usize,
}

impl VulkanGpu {
    pub fn probe() -> Self {
        let entry = match unsafe { Entry::load() } {
            Ok(entry) => entry,
            Err(_) => return unavailable("Vulkan unavailable"),
        };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"oryon-lm")
            .api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(instance) => instance,
            Err(_) => return unavailable("Vulkan instance unavailable"),
        };
        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(_) => {
                unsafe { instance.destroy_instance(None) };
                return unavailable("No Vulkan physical device");
            }
        };

        let mut best: Option<(usize, vk::PhysicalDeviceProperties)> = None;
        for (index, device) in devices.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(*device) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                if best.is_none() || index < best.as_ref().unwrap().0 {
                    best = Some((index, props));
                }
            }
        }
        match best {
            Some((device_index, props)) => {
                let name = String::from_utf8_lossy(
                    &props.device_name
                        .iter()
                        .map(|&b| b as u8)
                        .take_while(|&b| b != 0)
                        .collect::<Vec<_>>(),
                )
                .trim()
                .to_string();
                if name.is_empty() {
                    unavailable("Vulkan GPU")
                } else {
                    init_compute(instance, devices[device_index], name)
                }
            }
            None => unavailable("No Vulkan GPU"),
        }
    }

    pub fn has_vulkan_gpu(&self) -> bool {
        self.available
    }

    pub fn gpu_name(&self) -> &str {
        &self.name
    }

    pub fn dispatch_matvec(
        &mut self,
        matrix: &crate::quant::WeightMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> bool {
        if !self.available || !self.initialized || !self.pipeline_ready || matrix.rows() < 256
            || input.len() != matrix.cols()
            || output.len() != matrix.rows()
        {
            return false;
        }
        if !matches!(matrix,
            crate::quant::WeightMatrix::F32 { .. }
            | crate::quant::WeightMatrix::F32Mmap { .. }
            | crate::quant::WeightMatrix::Q4_K { .. }
            | crate::quant::WeightMatrix::Q6_K { .. }
            | crate::quant::WeightMatrix::Q4KMmap { .. }
            | crate::quant::WeightMatrix::Q6KMmap { .. }
        ) {
            return false;
        }
        self.dispatch_f32(matrix, input, output)
    }

    fn dispatch_f32(
        &mut self,
        matrix: &crate::quant::WeightMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> bool {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let payload_key = matrix as *const WeightMatrix as usize;
        if !self.weight_cache.contains_key(&payload_key) {
            let payload = match matrix {
            WeightMatrix::F32 { data, .. } => data.clone(),
            WeightMatrix::F32Mmap { mmap, offset, cols: mmap_cols, .. } => {
                let rows = matrix.rows();
                let mmap_cols = *mmap_cols;
                let mut data = vec![0.0f32; rows * mmap_cols];
                for r in 0..rows {
                    let start = offset + r * mmap_cols * 4;
                    for i in 0..mmap_cols {
                        data[r * mmap_cols + i] = f32::from_le_bytes(mmap[start + i * 4..start + i * 4 + 4].try_into().unwrap());
                    }
                }
                data
            }
            WeightMatrix::Q4KMmap { .. } | WeightMatrix::Q6KMmap { .. } => {
                let mut row = vec![0.0f32; cols];
                let mut data = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    matrix.dequantize_row(r, &mut row);
                    data[r * cols..(r + 1) * cols].copy_from_slice(&row);
                }
                data
            }
            WeightMatrix::Q4_K { .. } | WeightMatrix::Q6_K { .. } => {
                let mut data = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    let mut row = vec![0.0f32; cols];
                    matrix.dequantize_row(r, &mut row);
                    data[r * cols..(r + 1) * cols].copy_from_slice(&row);
                }
                data
            }
            _ => return false,
        };
            self.weight_cache.insert(payload_key, payload);
        }
        let data = match self.weight_cache.get(&payload_key) {
            Some(payload) => payload.as_slice(),
            None => return false,
        };
        let device = match self.device.as_ref() { Some(d) => d, None => return false };
        let physical = match self.physical { Some(p) => p, None => return false };
        let pool = match self.command_pool { Some(p) => p, None => return false };
        let set = match self.descriptor_set { Some(s) => s, None => return false };
        let pipeline = match self.pipeline { Some(p) => p, None => return false };
        let layout = match self.pipeline_layout { Some(l) => l, None => return false };

        let required = (input.len() * 4 + data.len() * 4 + output.len() * 4)
            .next_multiple_of(64);
        if self.buffers.is_none() {
            let instance = match self.instance.as_ref() { Some(i) => i, None => return false };
            if let Ok(b) = allocate_persistent(instance, device, physical, required) {
                self.buffers = Some(b);
            } else {
                return false;
            }
        } else if self.buffers.as_ref().map(|b| b.capacity).unwrap_or(0) < required {
            let instance = match self.instance.as_ref() { Some(i) => i, None => return false };
            if let Ok(b) = allocate_persistent(instance, device, physical, required) {
                self.buffers = Some(b);
            } else {
                return false;
            }
        }
        let buffers = match self.buffers.as_ref() { Some(b) => b, None => return false };
        unsafe {
            let x_ptr = device.map_memory(buffers.memory_x, 0, (input.len() * 4) as u64,
                vk::MemoryMapFlags::empty());
            if x_ptr.is_err() { return false; }
            std::ptr::copy_nonoverlapping(input.as_ptr(), x_ptr.unwrap().cast(), input.len());
            device.unmap_memory(buffers.memory_x);
            let w_ptr = device.map_memory(buffers.memory_w, 0, (data.len() * 4) as u64,
                vk::MemoryMapFlags::empty());
            if w_ptr.is_err() { return false; }
            std::ptr::copy_nonoverlapping(data.as_ptr(), w_ptr.unwrap().cast(), data.len());
            device.unmap_memory(buffers.memory_w);

            let x_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.x).offset(0).range((input.len() * 4) as u64);
            let w_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.w).offset(0).range((data.len() * 4) as u64);
            let s_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.s).offset(0).range(4);
            let y_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.y).offset(0).range((output.len() * 4) as u64);
            let desc_writes = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&x_info)),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&w_info)),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&s_info)),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&y_info)),
            ];
            device.update_descriptor_sets(&desc_writes, &[]);

            let command_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let command = match device.allocate_command_buffers(&command_info) { Ok(c) => c[0], Err(_) => return false };
            let begin = vk::CommandBufferBeginInfo::default();
            if device.begin_command_buffer(command, &begin).is_err() { return false; }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE,
                layout, 0, std::slice::from_ref(&set), &[]);
            let push = [matrix.rows() as u32, matrix.cols() as u32, 0u32];
            let push_bytes = std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), push.len() * 4);
            device.cmd_push_constants(command, layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            device.cmd_dispatch(command, ((matrix.rows() as u32) + 63) / 64, 1, 1);
            if device.end_command_buffer(command).is_err() { return false; }
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
            let queue = match self.queue { Some(q) => q, None => return false };
            if device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null()).is_err() { return false; }
            if device.queue_wait_idle(queue).is_err() { return false; }

            let y_ptr = device.map_memory(buffers.memory_y,
                0,
                (output.len() * 4) as u64, vk::MemoryMapFlags::empty());
            if y_ptr.is_err() { return false; }
            std::ptr::copy_nonoverlapping(y_ptr.unwrap().cast::<f32>(), output.as_mut_ptr(), output.len());
            device.unmap_memory(buffers.memory_y);
            let _ = device.free_command_buffers(pool, std::slice::from_ref(&command));
        }
        true
    }
}

fn unavailable(reason: &str) -> VulkanGpu {
    VulkanGpu {
        available: false,
        name: reason.to_string(),
        initialized: false,
        pipeline_ready: false,
        instance: None,
        device: None,
        physical: None,
        queue: None,
        queue_family: None,
        command_pool: None,
        pipeline: None,
        pipeline_layout: None,
        descriptor_set_layout: None,
        descriptor_pool: None,
        descriptor_set: None,
        buffers: None,
        weight_cache: std::collections::HashMap::new(),
    }
}

fn init_compute(instance: ash::Instance, physical: vk::PhysicalDevice, name: String) -> VulkanGpu {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let queue_family = match families.iter().position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
        Some(index) => index as u32,
        None => return unavailable("No Vulkan compute queue"),
    };
    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities);
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    let device = match unsafe { instance.create_device(physical, &device_info, None) } {
        Ok(device) => device,
        Err(_) => return unavailable("Vulkan compute device unavailable"),
    };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let mut gpu = VulkanGpu {
        available: true,
        name,
        initialized: true,
        pipeline_ready: false,
        instance: Some(instance),
        device: Some(device),
        physical: Some(physical),
        queue: Some(queue),
        queue_family: Some(queue_family),
        command_pool: None,
        pipeline: None,
        pipeline_layout: None,
        descriptor_set_layout: None,
        descriptor_pool: None,
        descriptor_set: None,
        buffers: None,
        weight_cache: std::collections::HashMap::new(),
    };
    gpu.pipeline_ready = init_pipeline(&mut gpu).is_ok();
    gpu
}

fn init_pipeline(gpu: &mut VulkanGpu) -> Result<(), vk::Result> {
    let device = gpu.device.as_ref().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let module_info = vk::ShaderModuleCreateInfo::default().code(MATVEC_SPIRV);
    let module = unsafe { device.create_shader_module(&module_info, None)? };
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe { device.create_descriptor_set_layout(&set_info, None)? };
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(12);
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let compute = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&compute), None)
    }.map_err(|(_, result)| result)?[0];
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes).max_sets(1);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&set_layout));
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };
    let queue_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(gpu.queue_family.unwrap_or(0))
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&queue_info, None)? };
    unsafe { device.destroy_shader_module(module, None) };
    gpu.pipeline = Some(pipeline);
    gpu.pipeline_layout = Some(pipeline_layout);
    gpu.descriptor_set_layout = Some(set_layout);
    gpu.descriptor_pool = Some(descriptor_pool);
    gpu.descriptor_set = Some(descriptor_set);
    gpu.command_pool = Some(command_pool);
    Ok(())
}

fn allocate_persistent(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    total_size: usize,
) -> Result<PersistentBuffers, vk::Result> {
    let total = total_size as u64;
    let create = vk::BufferCreateInfo::default()
        .size(total)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let x = unsafe { device.create_buffer(&create, None)? };
    let w = unsafe { device.create_buffer(&create, None)? };
    let s = unsafe { device.create_buffer(&create, None)? };
    let y = unsafe { device.create_buffer(&create, None)? };
    let req = unsafe { device.get_buffer_memory_requirements(x) };
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let memory_type = props.memory_types[..props.memory_type_count as usize]
        .iter().enumerate()
        .find(|(index, t)| req.memory_type_bits & (1 << index) != 0
            && t.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            && t.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
        .map(|(index, _)| index as u32)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let alloc_x = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(memory_type);
    let alloc_w = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(memory_type);
    let alloc_y = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(memory_type);
    let memory_x = unsafe { device.allocate_memory(&alloc_x, None)? };
    let memory_w = unsafe { device.allocate_memory(&alloc_w, None)? };
    let memory_y = unsafe { device.allocate_memory(&alloc_y, None)? };
    unsafe {
        device.bind_buffer_memory(x, memory_x, 0)?;
        device.bind_buffer_memory(w, memory_w, 0)?;
        device.bind_buffer_memory(y, memory_y, 0)?;
    }
    Ok(PersistentBuffers {
        x, w, s, y,
        memory_x, memory_w, memory_y,
        capacity: total_size,
    })
}
