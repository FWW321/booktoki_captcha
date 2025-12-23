use std::fs;
use std::path::Path;
use image::{ImageReader, ImageFormat};

fn main() {
    let data_dir = "./data"; // 你的图片目录
    println!("🧹 开始清洗数据集: {}", data_dir);

    if !Path::new(data_dir).exists() {
        eprintln!("❌ 目录不存在: {}", data_dir);
        return;
    }

    let mut count_fixed = 0;
    let mut count_deleted = 0;
    let mut count_ok = 0;

    let entries = fs::read_dir(data_dir).expect("无法读取目录");

    for entry in entries.flatten() {
        let path = entry.path();
        
        // 只处理文件
        if path.is_file() {
            // 1. 尝试猜测文件的真实格式
            let reader_result = ImageReader::open(&path)
                .and_then(|r| r.with_guessed_format());

            match reader_result {
                Ok(reader) => {
                    // 获取检测到的格式
                    if let Some(format) = reader.format() {
                        // 获取正确的后缀名
                        let correct_ext = match format {
                            ImageFormat::Png => "png",
                            ImageFormat::Jpeg => "jpg",
                            ImageFormat::Gif => "gif",
                            ImageFormat::WebP => "webp",
                            ImageFormat::Bmp => "bmp",
                            _ => "bin", // 其他生僻格式
                        };

                        // 获取当前的后缀名 (转小写)
                        let current_ext = path.extension()
                            .and_then(|e| e.to_str())
                            .map(|s| s.to_lowercase())
                            .unwrap_or_default();

                        // 2. 如果后缀不匹配，进行重命名
                        if current_ext != correct_ext {
                            let new_path = path.with_extension(correct_ext);
                            println!("🔧 修复后缀: {:?} -> .{}", path.file_name().unwrap(), correct_ext);
                            
                            if let Err(e) = fs::rename(&path, &new_path) {
                                eprintln!("   重命名失败: {}", e);
                            } else {
                                count_fixed += 1;
                            }
                        } else {
                            count_ok += 1;
                        }
                    } else {
                        // 3. 虽然 ImageReader 打开了，但识别不出格式（可能是坏损的图片头）
                        println!("🗑️ 删除未知格式文件: {:?}", path.file_name().unwrap());
                        fs::remove_file(path).unwrap_or_default();
                        count_deleted += 1;
                    }
                },
                Err(_) => {
                    // 4. 根本打不开（例如：其实是 HTML 文本、PHP 源码、空文件）
                    println!("🗑️ 删除无效文件(非图片): {:?}", path.file_name().unwrap());
                    fs::remove_file(path).unwrap_or_default();
                    count_deleted += 1;
                }
            }
        }
    }

    println!("--------------------------------");
    println!("🎉 清洗完成！");
    println!("✅ 正常图片: {}", count_ok);
    println!("🔧 修复后缀: {}", count_fixed);
    println!("🗑️ 删除垃圾: {}", count_deleted);
}