use booktoki_captcha::{model::Model, data::CaptchaDataset};
use burn::{
    prelude::*,
    optim::{AdamWConfig, Optimizer},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend, 
    backend::Autodiff,
    module::{Module, AutodiffModule},
};
use burn_tch::{LibTorch, LibTorchDevice};
use std::f64::consts::PI;

fn main() {
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("TORCH_NUM_THREADS", "1");
    }
    env_logger::init();
    
    let device = LibTorchDevice::Cuda(0);
    println!("🚀 Starting Training V3 (Deep & Narrow + Cosine LR) on: {:?}", device);

    type MyBackend = LibTorch<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let batch_size = 128; 
    let base_lr = 3e-4; // 稍微提高初始 LR，因为我们会快速下降
    let num_epochs = 500;

    let dataset = CaptchaDataset::new("./data");
    let (mut dataset_train, dataset_remaining) = dataset.split(0.8);
    let (dataset_test, _) = dataset_remaining.split(1.0);

    dataset_train.augment(9); // 10x data

    let (train_images, train_targets) = dataset_train.to_gpu_tensors::<MyAutodiffBackend>(&device);
    let (test_images, test_targets)   = dataset_test.to_gpu_tensors::<MyBackend>(&device);

    let train_len = train_images.dims()[0];
    let test_len = test_images.dims()[0];

    println!("Dataset sizes: Train={}, Test={}", train_len, test_len);

    let mut model = Model::<MyAutodiffBackend>::new(&device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(1e-4)
        .init();

    let mut best_acc = 0.0;

    for epoch in 1..=num_epochs {
        // Cosine Annealing LR Schedule
        // LR = 0.5 * base_lr * (1 + cos(pi * epoch / num_epochs))
        let lr = 0.5 * base_lr * (1.0 + (PI * epoch as f64 / num_epochs as f64).cos());
        
        if epoch % 50 == 0 {
             println!("📉 LR -> {:.2e}", lr);
        }

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for i in (0..train_len).step_by(batch_size) {
            let end = (i + batch_size).min(train_len);
            let images_batch = train_images.clone().slice([i..end]);
            let targets_batch = train_targets.clone().slice([i..end]);

            let item = model.forward_classification(images_batch, targets_batch);
            let grads = item.loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            total_loss += item.loss.into_scalar();
            batch_count += 1;
        }

        let avg_loss = total_loss as f64 / batch_count as f64;

        if epoch % 5 == 0 || epoch == 1 {
            let model_valid = model.valid();
            let mut full_match = 0;
            let mut char_correct = 0;
            let mut total_samples = 0;
            
            for i in (0..test_len).step_by(batch_size) {
                 let end = (i + batch_size).min(test_len);
                 let b_size = end - i;
                 let images_batch = test_images.clone().slice([i..end]);
                 let targets_batch = test_targets.clone().slice([i..end]); 

                 let output = model_valid.forward(images_batch); 
                 let predicted = output.argmax(2).reshape([b_size, 4]); 
                 
                 let eq_elements = predicted.clone().equal(targets_batch.clone()).int(); 
                 char_correct += eq_elements.clone().sum().into_scalar() as usize;
                 
                 let row_sums = eq_elements.sum_dim(1).reshape([b_size]); 
                 let row_sums_vec = row_sums.into_data().to_vec::<i64>().unwrap();
                 for sum in row_sums_vec {
                     if sum == 4 { full_match += 1; }
                 }
                 
                 if i == 0 && epoch % 10 == 0 {
                     let p = predicted.slice([0..1]).into_data();
                     let t = targets_batch.slice([0..1]).into_data();
                     println!("   [Debug] Sample 0 -> Pred: {:?}, Tgt: {:?}", p, t);
                 }

                 total_samples += b_size;
            }
            
            let acc_full = full_match as f64 / total_samples as f64 * 100.0;
            let acc_char = char_correct as f64 / (total_samples * 4) as f64 * 100.0;
            println!("Epoch {:3}/{} - Loss: {:.4} - Full Acc: {:6.2}% - Char Acc: {:6.2}%", 
                     epoch, num_epochs, avg_loss, acc_full, acc_char);
            
            // Save Best Model
            if acc_full > best_acc {
                best_acc = acc_full;
                println!("🔥 New Best Model! Acc: {:.2}% - Saving...", best_acc);
                
                let record = model.clone().into_record();
                let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
                    .record(record, ())
                    .expect("Failed to serialize model");
                    
                if !std::path::Path::new("model").exists() {
                    std::fs::create_dir("model").unwrap();
                }
                std::fs::write("./model/captcha_model.bin", bytes).expect("Failed to write file");
            }

        } else {
             println!("Epoch {:3}/{} - Loss: {:.4}", epoch, num_epochs, avg_loss);
        }
    }
}
