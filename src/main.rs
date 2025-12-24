use booktoki_captcha::solve_captcha;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // 1. Get arguments or pick random
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 {
        // Predict specific file
        let path = &args[1];
        predict_one(path);
    } else {
        // Predict a few random samples from data folder
        println!("No file specified. Testing random samples from ./data ...");
        
        if let Ok(entries) = fs::read_dir("./data") {
            let mut count = 0;
            for entry in entries {
                if count >= 5 { break; } // Test 5 images
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "jpg" || e == "png") {
                        predict_one(path.to_str().unwrap_or_default());
                        count += 1;
                    }
                }
            }
        } else {
             println!("Warning: ./data directory not found.");
        }
    }
}

fn predict_one(path: &str) {
    if path.is_empty() { return; }
    
    match fs::read(path) {
        Ok(image_bytes) => {
            match solve_captcha(&image_bytes) {
                Ok(result) => {
                    let filename = Path::new(path).file_name().unwrap_or_default().to_string_lossy();
                    println!("File: {:<20} -> Prediction: {}", filename, result);
                }
                Err(e) => {
                    eprintln!("Error predicting {}: {}", path, e);
                }
            }
        },
        Err(e) => eprintln!("Failed to read file {}: {}", path, e),
    }
}