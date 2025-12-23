pub mod model;
pub mod data; 

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
pub struct CaptchaSolver {
    model: Model<Backend>,
    device: <Backend as burn::tensor::backend::Backend>::Device,
}

// Embed the trained model binary (FP16 format)
static MODEL_BYTES: &[u8] = include_bytes!("../model/captcha_model.bin");

impl CaptchaSolver {
    /// Initializes a new solver instance.
    /// Loads the embedded model into memory.
    pub fn new() -> Self {
        let device = Default::default();
        
        // Initialize model structure
        let model = Model::new(&device);

        // Load weights from embedded binary using HalfPrecisionSettings
        let record = BinBytesRecorder::<HalfPrecisionSettings>::default()
            .load(MODEL_BYTES.to_vec(), &device)
            .expect("Failed to load embedded model");

        let model = model.load_record(record);

        Self { model, device }
    }
    
    /// Solves a captcha from image bytes (JPEG/PNG).
    /// Returns the 4-digit string.
    pub fn solve(&self, image_bytes: &[u8]) -> Result<String, String> {
        // Decode image
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .decode()
            .map_err(|e| e.to_string())?;

        // Preprocessing: Resize using Triangle filter (matches training)
        let gray = img.resize_exact(
            IMG_WIDTH as u32, 
            IMG_HEIGHT as u32, 
            image::imageops::FilterType::Triangle
        ).to_luma8();

        // Normalize pixels to [-1.0, 1.0]
        let mut pixel_data = Vec::with_capacity(IMG_WIDTH * IMG_HEIGHT);
        for pixel in gray.pixels() {
            let val = pixel.0[0] as f32 / 255.0;
            pixel_data.push((val - 0.5) / 0.5);
        }

        // Create tensor [1, 1, H, W]
        let input_tensor = Tensor::<Backend, 1>::from_floats(pixel_data.as_slice(), &self.device)
            .reshape([1, 1, IMG_HEIGHT, IMG_WIDTH]);

        // Inference
        let output = self.model.forward(input_tensor); // [1, 4, 10]
        
        // Post-processing
        let predicted = output.argmax(2).squeeze::<1>(); 
        
        let indices: Vec<i64> = predicted
            .into_data()
            .to_vec::<i64>()
            .expect("Failed to read tensor data");

        Ok(indices.iter().map(|i| i.to_string()).collect())
    }
}

/// Convenience function to solve a captcha in one shot.
/// Note: Re-initializing the solver every time is inefficient for batch processing.
/// Use `CaptchaSolver` struct for repeated use.
pub fn solve_captcha(image_bytes: &[u8]) -> Result<String, String> {
    let solver = CaptchaSolver::new();
    solver.solve(image_bytes)
}