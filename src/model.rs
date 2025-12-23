use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig,
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu, Dropout, DropoutConfig,
        Sigmoid,
    },
    prelude::*,
};

pub const IMG_HEIGHT: usize = 60;
pub const IMG_WIDTH: usize = 160;
const NUM_CLASSES: usize = 10;
const CAPTCHA_LENGTH: usize = 4;
const POOL_WIDTH: usize = 8; 

// ========================================================================
// SE Block (保持不变，用于去噪)
// ========================================================================
#[derive(Module, Debug)]
pub struct SeBlock<B: Backend> {
    pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
    sigmoid: Sigmoid,
}

impl<B: Backend> SeBlock<B> {
    pub fn new(channels: usize, reduction: usize, device: &B::Device) -> Self {
        let reduced = (channels / reduction).max(4); 
        Self {
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc1: LinearConfig::new(channels, reduced).init(device),
            fc2: LinearConfig::new(reduced, channels).init(device),
            activation: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, _, _] = x.dims();
        let y = self.pool.forward(x.clone()); 
        let y = y.reshape([b, c]);
        let y = self.fc1.forward(y);
        let y = self.activation.forward(y);
        let y = self.fc2.forward(y);
        let y = self.sigmoid.forward(y);
        let y = y.reshape([b, c, 1, 1]);
        x.mul(y)
    }
}

// ========================================================================
// SE-BasicBlock (回归 ResNet 结构，但加入 SE 增强)
// ========================================================================
#[derive(Module, Debug)]
pub struct SeBasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    se: SeBlock<B>,       // 新增：注意力模块
    activation: Relu,
    downsample: Option<Conv2d<B>>,
}

impl<B: Backend> SeBasicBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        // 标准 3x3 卷积，不使用分组卷积，保证特征提取能力
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        // SE 模块放在残差相加之前
        let se = SeBlock::new(out_channels, 8, device); // Reduction=8, 轻量化

        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1])
                    .with_stride([stride, stride])
                    .init(device))
        } else {
            None
        };

        Self { conv1, bn1, conv2, bn2, se, activation: Relu::new(), downsample }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.downsample {
            Some(conv) => conv.forward(input.clone()),
            None => input.clone(),
        };

        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        
        // 关键：在残差相加前通过 SE 模块过滤特征
        let x = self.se.forward(x);

        let x = x.add(residual);
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    
    // 3 Stages, Max 64 Channels (体积控制)
    layer1: SeBasicBlock<B>,
    layer1_2: SeBasicBlock<B>, 

    layer2: SeBasicBlock<B>,
    layer2_2: SeBasicBlock<B>,

    layer3: SeBasicBlock<B>,
    layer3_2: SeBasicBlock<B>, 
    layer3_3: SeBasicBlock<B>, // 多加一层深度，处理复杂扭曲

    // 移除了 layer4 (128 channels)，因为 64 通道配合 SE 通常足够，且能大幅减小体积
    
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc: Linear<B>, 
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        // Initial Conv
        let conv1 = Conv2dConfig::new([1, 32], [5, 5]) 
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let bn1 = BatchNormConfig::new(32).init(device);

        // Stage 1: 32 channels
        let layer1 = SeBasicBlock::new(32, 32, 1, device);
        let layer1_2 = SeBasicBlock::new(32, 32, 1, device);

        // Stage 2: 64 channels
        let layer2 = SeBasicBlock::new(32, 64, 2, device);
        let layer2_2 = SeBasicBlock::new(64, 64, 1, device);

        // Stage 3: 64 channels (保持 64，不再升到 128)
        // 我们用更多的层数代替更多的通道，这样参数更少，非线性能力更强
        let layer3 = SeBasicBlock::new(64, 64, 2, device); 
        let layer3_2 = SeBasicBlock::new(64, 64, 1, device);
        let layer3_3 = SeBasicBlock::new(64, 64, 1, device);

        // Pooling: [B, 64, 1, 8]
        // 注意：这里通道是 64，比原来的 128 少了一半
        let pool = AdaptiveAvgPool2dConfig::new([1, POOL_WIDTH]).init();
        
        // Dropout 回归 0.5，防止过拟合
        let dropout = DropoutConfig::new(0.5).init(); 
        
        // Linear: 64 * 8 = 512 -> 40
        // 全连接层参数也减少了一半
        let fc = LinearConfig::new(64 * POOL_WIDTH, CAPTCHA_LENGTH * NUM_CLASSES).init(device);
        
        Self {
            conv1, bn1, relu: Relu::new(),
            layer1, layer1_2,
            layer2, layer2_2,
            layer3, layer3_2, layer3_3,
            pool, dropout, fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let batch_size = input.dims()[0];

        let x = self.conv1.forward(input);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        let x = self.layer1.forward(x);
        let x = self.layer1_2.forward(x);

        let x = self.layer2.forward(x);
        let x = self.layer2_2.forward(x);

        let x = self.layer3.forward(x);
        let x = self.layer3_2.forward(x);
        let x = self.layer3_3.forward(x);

        let x = self.pool.forward(x); // -> [B, 64, 1, 8]
        let x = x.reshape([batch_size, 64 * POOL_WIDTH]);

        let x = self.dropout.forward(x);
        let x = self.fc.forward(x); // -> [B, 40]

        x.reshape([batch_size, CAPTCHA_LENGTH, NUM_CLASSES])
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> burn::train::ClassificationOutput<B> {
        let batch_size = images.dims()[0];
        let output = self.forward(images);
        
        let output_flat = output.reshape([batch_size * CAPTCHA_LENGTH, NUM_CLASSES]);
        let targets_flat = targets.reshape([batch_size * CAPTCHA_LENGTH]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&output_flat.device())
            .forward(output_flat.clone(), targets_flat.clone());

        burn::train::ClassificationOutput::new(loss, output_flat, targets_flat)
    }
}