# Booktoki Captcha Solver

`booktoki_captcha` 是一个基于 Rust 和 [Burn](https://github.com/burn-rs/burn) 深度学习框架的高性能验证码识别库。它专门针对 4 位数字验证码（如 booktoki 网站）进行了训练和优化。

## 特性

*   **高性能 CPU 推理**: 使用 `burn-ndarray` 后端，无需显卡即可快速推理。
*   **零依赖部署**: 训练好的模型权重直接嵌入到二进制文件中，无需分发额外的模型文件。
*   **极速响应**: 采用 `std::sync::LazyLock` 实现全局单例，模型仅在第一次调用时加载。
*   **并发优化**: 核心推理锁（Mutex）仅包裹最快的矩阵运算部分，图像解码与预处理在锁外并发执行，极致压榨多核 CPU 性能。
*   **硬件加速**: 默认开启 `simd` 特性（针对 Nightly 编译器），利用 CPU 矢量指令集实现数倍的推理提速。
*   **极简 API**: 仅暴露单个 `solve_captcha` 函数，无需手动管理实例或生命周期。

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

### 架构优势

1.  **单例模式 (Lazy Singleton)**: 全局共享一个模型实例，避免了每个请求都重新加载模型导致的内存抖动和 CPU 尖峰。
2.  **细粒度锁**: 只有真正的 `model.forward()`（推理）阶段需要持有 `Mutex`。耗时的图像解码（CPU 密集型）在锁外并发进行。这意味着即使在高并发下，整体延迟也非常稳定。
3.  **零文件依赖**: 模型权重以二进制形式嵌入在 `.dll` / `.exe` 中，部署时只需要一个可执行文件。

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
# 跳过 burn 的 libtorch 版本检测（如果遇到版本不匹配错误）
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

## 架构细节
*   **输入**: 160x60 灰度图像
*   **骨干网络**: Custom SE-ResNet (3 Stages)
*   **参数量**: < 1M params (轻量级)
*   **后端**: 
    *   训练: `burn-tch` (LibTorch + CUDA)
    *   推理: `burn-ndarray` (Pure Rust CPU)
