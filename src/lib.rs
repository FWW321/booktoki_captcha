pub mod model;
pub mod data; 

// 引入 NdArray (注意大写 A)
use burn_ndarray::NdArray; 
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::Tensor,
};
use image::ImageReader;
use std::io::Cursor;
use model::{Model, IMG_HEIGHT, IMG_WIDTH};

type Backend = NdArray;

pub struct CaptchaSolver {
    model: Model<Backend>,
    device: <Backend as burn::tensor::backend::Backend>::Device,
}

// =========================================================
// ⚠️ 第一次编译训练时请保持注释！
// 训练生成 model/captcha_model.bin 后，再取消下面这行的注释
static MODEL_BYTES: &[u8] = include_bytes!("../model/captcha_model.bin");
// =========================================================

impl CaptchaSolver {
    // 临时 new 方法 (用于通过编译进行训练)
    // pub fn new() -> Self {
    //    panic!("请先运行训练脚本生成模型文件，然后修改 src/lib.rs 取消 MODEL_BYTES 的注释！");
    // }

    // 🟢 训练完成后，取消 static MODEL_BYTES 的注释，并启用这个 new 方法
    pub fn new() -> Self {
        let device = Default::default();
        let model = Model::new(&device);

        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(MODEL_BYTES.to_vec(), &device)
            .expect("Failed to load embedded model");

        let model = model.load_record(record);

        Self { model, device }
    }
    

    pub fn solve(&self, image_bytes: &[u8]) -> Result<String, String> {
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .decode()
            .map_err(|e| e.to_string())?;

        // 优化1：保持与训练时一致的插值算法 (Triangle)
        let gray = img.resize_exact(IMG_WIDTH as u32, IMG_HEIGHT as u32, image::imageops::FilterType::Triangle)
            .to_luma8();

        let mut pixel_data = Vec::with_capacity(IMG_WIDTH * IMG_HEIGHT);
        for pixel in gray.pixels() {
            // 优化2：保持与训练时一致的归一化 [-1.0, 1.0]
            let val = pixel.0[0] as f32 / 255.0;
            pixel_data.push((val - 0.5) / 0.5);
        }

        let input_tensor = Tensor::<Backend, 1>::from_floats(pixel_data.as_slice(), &self.device)
            .reshape([1, 1, IMG_HEIGHT, IMG_WIDTH]);

        // 推理
        let output = self.model.forward(input_tensor); // [1, 4, 10]
        
        // 【修复点 1】：squeeze::<1>() 不再需要参数 (0)
        // argmax(2) 得到 [1, 4, 1]，squeeze::<1>() 自动压缩为 [4]
        let predicted = output.argmax(2).squeeze::<1>(); 
        
        // 【修复点 2】：TensorData 没有 .value 字段了，使用 .to_vec::<i64>()
        let indices: Vec<i64> = predicted
            .into_data()
            .to_vec::<i64>()
            .expect("Failed to read tensor data");

        Ok(indices.iter().map(|i| i.to_string()).collect())
    }
}

pub fn solve_captcha(image_bytes: &[u8]) -> Result<String, String> {
    let solver = CaptchaSolver::new();
    solver.solve(image_bytes)
}