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
        let entries = fs::read_dir("./data").expect("Failed to read ./data directory");
        let mut count = 0;
        for entry in entries {
            if count >= 5 { break; } // Test 5 images
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "jpg" || e == "png") {
                predict_one(path.to_str().expect("Path is not valid UTF-8"));
                count += 1;
            }
        }
    }
}

fn predict_one(path: &str) {
    let image_bytes = fs::read(path).expect("Failed to read image file");
    
    match solve_captcha(&image_bytes) {
        Ok(result) => {
            let filename = Path::new(path).file_name().unwrap().to_str().unwrap();
            println!("File: {:<20} -> Prediction: {}", filename, result);
        }
        Err(e) => {
            eprintln!("Error predicting {}: {}", path, e);
        }
    }
}
