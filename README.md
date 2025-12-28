# Booktoki Captcha Solver

基于 Rust 和 [Burn](https://github.com/burn-rs/burn) 的验证码识别库。它专门针对 4 位数字验证码（booktoki 网站）进行了训练和优化

## 安装

在你的 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
booktoki_captcha = { git = "https://github.com/FWW321/booktoki_captcha.git" }
```

> **注意**：本项目目前针对 Nightly 编译器进行了优化（开启了 SIMD）。如果你使用的是 Stable 版本，请在 `Cargo.toml` 中通过 `default-features = false, features = ["std"]` 禁用 SIMD 特性。

## 使用方法

直接传入图片字节即可。库会自动处理模型加载、线程安全和预处理逻辑。

```rust
use booktoki_captcha::solve_captcha;
use std::fs;

fn main() {
    // 1. 读取图片字节 (支持 JPG, PNG, WEBP 等)
    let image_bytes = fs::read("data/1234.jpg").expect("Failed to read file");

    // 2. 识别验证码
    // 第一次调用会自动加载嵌入的模型（约 10-30ms），后续推理仅需 ~1-2ms。
    match solve_captcha(&image_bytes) {
        Ok(code) => println!("识别结果: {}", code),
        Err(e) => eprintln!("识别失败: {}", e),
    }
}
```

## 开发与训练

如果你需要重新训练模型或修改网络结构，请遵循以下步骤。

### 前置要求
*   Rust 工具链
*   CUDA 环境（仅训练需要）
*   LibTorch (由 `burn-tch` 自动处理，可能需要设置环境变量)

### 1. 准备数据
将标注好的验证码图片放入 `data/` 目录，文件名格式为 `{label}.jpg` (例如 `1234.jpg`)。

### 2. 训练模型
训练脚本会自动进行数据增强（噪声、切割、形变）并使用 GPU 训练。

```powershell
$env:LIBTORCH_BYPASS_VERSION_CHECK = 1

# 运行训练
cargo run --release --bin train --features training
```

训练完成后，最优模型会自动保存为 `model/captcha_model.bin`。

### 3. 准备推理
**注意**：在发布或构建库之前，请确保 `src/lib.rs` 中的模型加载路径正确，并且对应的二进制文件存在。

### 4. 运行测试
使用 `booktoki_captcha` 工具测试模型效果：

```powershell
# 预测单张图片
cargo run --release --bin booktoki_captcha -- data/test.jpg

# 随机预测 data 目录下的图片
cargo run --release --bin booktoki_captcha
```