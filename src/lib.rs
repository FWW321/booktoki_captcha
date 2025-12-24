pub mod model;
pub mod data; 

use std::sync::{LazyLock, Mutex};

use burn_ndarray::NdArray; 
use burn::{
    module::Module,
    record::{BinBytesRecorder, HalfPrecisionSettings, Recorder},
    tensor::Tensor,
};
use image::ImageReader;
use std::io::Cursor;
use model::{Model, IMG_HEIGHT, IMG_WIDTH};

// Use NdArray backend for CPU inference
type Backend = NdArray;

/// A thread-safe captcha solver using a pre-trained ResNet model.
struct CaptchaSolver {
    model: Model<Backend>,
    device: <Backend as burn::tensor::backend::Backend>::Device,
}

// Embed the trained model binary (FP16 format)
static MODEL_BYTES: &[u8] = include_bytes!("../model/captcha_model.bin");


// 全岛唯一的 Solver 实例 (Wrapped in Mutex for thread safety)
static SOLVER: LazyLock<Mutex<CaptchaSolver>> = LazyLock::new(|| Mutex::new(CaptchaSolver::new()));

impl CaptchaSolver {
    /// 初始化一个新实例。
    /// 注意：由于使用了单例模式，通常不直接调用此方法，而是使用全局入口。
    fn new() -> Self {
        let device = Default::default();
        
        // 初始化模型结构
        let model = Model::new(&device);

        // 从嵌入的二进制文件中加载权重
        let record = BinBytesRecorder::<HalfPrecisionSettings>::default()
            .load(MODEL_BYTES.to_vec(), &device)
            .expect("Failed to load embedded model");

        let model = model.load_record(record);

        Self { model, device }
    }

    /// 静态预处理函数：解码 -> 调整大小 -> 归一化
    /// 这是一个纯 CPU 计算函数，不涉及模型，应在锁外执行。
    fn preprocess(image_bytes: &[u8]) -> Result<Vec<f32>, String> {
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .decode()
            .map_err(|e| e.to_string())?;

        let gray = img.resize_exact(
            IMG_WIDTH as u32, 
            IMG_HEIGHT as u32, 
            image::imageops::FilterType::Triangle
        ).to_luma8();

        let mut pixel_data = Vec::with_capacity(IMG_WIDTH * IMG_HEIGHT);
        for pixel in gray.pixels() {
            let val = pixel.0[0] as f32 / 255.0;
            pixel_data.push((val - 0.5) / 0.5);
        }
        Ok(pixel_data)
    }

    /// 核心推理函数：Tensor 转换 -> Forward -> Argmax
    /// 此函数非常快 (~2ms)，需要持有锁。
    fn inference(&self, pixel_data: Vec<f32>) -> Result<String, String> {
        let input_tensor = Tensor::<Backend, 1>::from_floats(pixel_data.as_slice(), &self.device)
            .reshape([1, 1, IMG_HEIGHT, IMG_WIDTH]);

        let output = self.model.forward(input_tensor); // [1, 4, 10]
        
        let predicted = output.argmax(2).squeeze::<1>(); 
        
        let indices: Vec<i64> = predicted
            .into_data()
            .to_vec::<i64>()
            .expect("Failed to read tensor data");

        Ok(indices.iter().map(|i| i.to_string()).collect())
    }
}

/// 解决验证码的便捷函数。
/// 优化：在获取锁之前进行图像预处理，最大化并发性能。
pub fn solve_captcha(image_bytes: &[u8]) -> Result<String, String> {
    // 1. 无锁预处理 (耗时大头)
    let pixels = CaptchaSolver::preprocess(image_bytes)?;

    // 2. 获取锁并快速推理
    SOLVER.lock()
        .map_err(|e| format!("Failed to acquire solver lock: {}", e))?
        .inference(pixels)
}

// todo: 扩展库提供一个 solve_batch(images: Vec<&[u8]>) 方法，利用CPU 的 SIMD 指令一次算多张