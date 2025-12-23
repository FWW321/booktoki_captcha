use crate::model::{IMG_HEIGHT, IMG_WIDTH};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use image::ImageReader;
use rand::seq::SliceRandom;
use rand::rng;

#[derive(Clone, Debug)]
pub struct CaptchaItem {
    // 预处理好的像素数据 (60x160 = 9600 floats)
    pub pixels: Vec<f32>,
    pub label: [i64; 4],
}

#[derive(Clone)]
pub struct CaptchaDataset {
    items: Vec<CaptchaItem>,
}

impl CaptchaDataset {
    pub fn new(root: &str) -> Self {
        let mut items = Vec::new();
        if let Ok(entries) = std::fs::read_dir(root) {
            let entries: Vec<_> = entries.flatten().collect();
            println!("Preprocessing {} images...", entries.len());
            
            for entry in entries {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "png" || e == "jpg") {
                    let stem = path.file_stem().unwrap().to_str().unwrap();
                    if stem.len() == 4 && stem.chars().all(char::is_numeric) {
                        let mut label = [0; 4];
                        for (i, c) in stem.chars().enumerate() {
                            label[i] = c.to_digit(10).unwrap() as i64;
                        }

                        // 核心优化：只在加载数据集时解码一次
                        let img = ImageReader::open(&path).unwrap().decode().unwrap();
                        // 优化1：使用 Triangle (双线性) 插值，比 Nearest 更平滑，保留更多特征
                        let gray = img.resize_exact(IMG_WIDTH as u32, IMG_HEIGHT as u32, image::imageops::FilterType::Triangle).to_luma8();
                        
                        let mut pixels = Vec::with_capacity(IMG_WIDTH * IMG_HEIGHT);
                        for pixel in gray.pixels() {
                            // 优化2：归一化到 [-1.0, 1.0] 范围
                            // (x / 255.0 - 0.5) / 0.5
                            let val = pixel.0[0] as f32 / 255.0;
                            pixels.push((val - 0.5) / 0.5);
                        }

                        items.push(CaptchaItem { pixels, label });
                    }
                }
            }
        }
        println!("Successfully loaded {} images into RAM", items.len());
        Self { items }
    }

    /// Splits the dataset into two datasets (train, valid) based on the ratio.
    /// Ratio is for the first dataset (e.g., 0.8 for 80% train).
    pub fn split(mut self, ratio: f32) -> (Self, Self) {
        let mut rng = rng();
        self.items.shuffle(&mut rng);
        
        let split_idx = (self.items.len() as f32 * ratio) as usize;
        let valid_items = self.items.split_off(split_idx);
        
        println!("Split dataset: {} training, {} validation", self.items.len(), valid_items.len());
        
        (self, Self { items: valid_items })
    }

    /// 离线数据增强：将数据集扩充 N 倍
    /// 包含：随机噪声、Cutout
    pub fn augment(&mut self, factor: usize) {
        println!("🔨 Augmenting dataset {}x ...", factor);
        let original_items = self.items.clone();
        let mut rng = rng();
        use rand::Rng;

        for _ in 0..factor {
            for item in &original_items {
                let mut new_pixels = item.pixels.clone();
                
                // 1. Random Noise
                for p in new_pixels.iter_mut() {
                    if rng.random_bool(0.15) { // 稍微增加概率
                         *p += rng.random_range(-0.15..0.15); // 稍微增加强度
                         *p = p.clamp(-1.0, 1.0);
                    }
                }

                // 2. Random Cutout (Occlusion)
                if rng.random_bool(0.4) { // 增加遮挡概率
                    let cut_h = rng.random_range(8..20); // 更大的遮挡
                    let cut_w = rng.random_range(8..20);
                    let start_y = rng.random_range(0..(IMG_HEIGHT - cut_h));
                    let start_x = rng.random_range(0..(IMG_WIDTH - cut_w));
                    
                    for y in start_y..(start_y + cut_h) {
                        for x in start_x..(start_x + cut_w) {
                            new_pixels[y * IMG_WIDTH + x] = -1.0; // Black
                        }
                    }
                }

                // 3. Random Translation (Shift) - 更激进的平移
                if rng.random_bool(0.6) {
                    let shift_x = rng.random_range(-15..15); 
                    let shift_y = rng.random_range(-8..8);   
                    let mut shifted_pixels = vec![-1.0; new_pixels.len()]; 
                    
                    for y in 0..IMG_HEIGHT {
                        for x in 0..IMG_WIDTH {
                            let old_y = y as i32 - shift_y;
                            let old_x = x as i32 - shift_x;
                            
                            if old_y >= 0 && old_y < IMG_HEIGHT as i32 && old_x >= 0 && old_x < IMG_WIDTH as i32 {
                                shifted_pixels[y * IMG_WIDTH + x] = new_pixels[old_y as usize * IMG_WIDTH + old_x as usize];
                            }
                        }
                    }
                    new_pixels = shifted_pixels;
                }

                // 4. Random Shear (Slant) - 模拟斜体/倾斜
                // x' = x + alpha * y
                if rng.random_bool(0.5) {
                    let shear_factor = rng.random_range(-0.2..0.2); // 倾斜程度
                    let mut sheared_pixels = vec![-1.0; new_pixels.len()];

                    for y in 0..IMG_HEIGHT {
                        let shift = (y as f32 * shear_factor) as i32;
                        for x in 0..IMG_WIDTH {
                            let old_x = x as i32 - shift;
                            if old_x >= 0 && old_x < IMG_WIDTH as i32 {
                                sheared_pixels[y * IMG_WIDTH + x] = new_pixels[y * IMG_WIDTH + old_x as usize];
                            }
                        }
                    }
                    new_pixels = sheared_pixels;
                }

                self.items.push(CaptchaItem {
                    pixels: new_pixels,
                    label: item.label,
                });
            }
        }
        
        // Shuffle again
        self.items.shuffle(&mut rng);
        println!("✨ Dataset expanded to {} images", self.items.len());
    }

    /// 将所有数据一次性上传到 GPU，返回 (Images, Targets)
    pub fn to_gpu_tensors<B: Backend>(self, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 2, Int>) {
        let batch_size = self.items.len();
        println!("🚀 Uploading {} images to GPU...", batch_size);
        
        let mut all_pixels = Vec::with_capacity(batch_size * IMG_HEIGHT * IMG_WIDTH);
        let mut all_labels = Vec::with_capacity(batch_size * 4);

        for item in self.items {
            all_pixels.extend_from_slice(&item.pixels);
            all_labels.extend_from_slice(&item.label);
        }

        let images = Tensor::<B, 1>::from_floats(all_pixels.as_slice(), device)
            .reshape([batch_size, 1, IMG_HEIGHT, IMG_WIDTH]);

        let targets = Tensor::<B, 1, Int>::from_ints(all_labels.as_slice(), device)
            .reshape([batch_size, 4]);
            
        println!("✅ Upload complete!");

        (images, targets)
    }
}

impl Dataset<CaptchaItem> for CaptchaDataset {
    fn get(&self, index: usize) -> Option<CaptchaItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct CaptchaBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> CaptchaBatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

#[derive(Clone, Debug)]
pub struct CaptchaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, CaptchaItem, CaptchaBatch<B>> for CaptchaBatcher<B> {
    fn batch(&self, items: Vec<CaptchaItem>, device: &B::Device) -> CaptchaBatch<B> {
        let batch_size = items.len();
        let mut all_pixels = Vec::with_capacity(batch_size * IMG_HEIGHT * IMG_WIDTH);
        let mut all_labels = Vec::with_capacity(batch_size * 4);

        for item in items {
            // 这里现在只是简单的内存拷贝，极快
            all_pixels.extend_from_slice(&item.pixels);
            all_labels.extend_from_slice(&item.label);
        }

        let images = Tensor::<B, 1>::from_floats(all_pixels.as_slice(), device)
            .reshape([batch_size, 1, IMG_HEIGHT, IMG_WIDTH]);

        let targets = Tensor::<B, 1, Int>::from_ints(all_labels.as_slice(), device)
            .reshape([batch_size, 4]);

        CaptchaBatch { images, targets }
    }
}