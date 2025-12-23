use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig,
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu, Dropout, DropoutConfig,
    },
    prelude::*,
};

pub const IMG_HEIGHT: usize = 60;
pub const IMG_WIDTH: usize = 160;
const NUM_CLASSES: usize = 10;
const CAPTCHA_LENGTH: usize = 4;
const POOL_WIDTH: usize = 8; 

#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    activation: Relu,
    downsample: Option<Conv2d<B>>,
}

impl<B: Backend> BasicBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
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

        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1])
                    .with_stride([stride, stride])
                    .init(device))
        } else {
            None
        };

        Self { conv1, bn1, conv2, bn2, activation: Relu::new(), downsample }
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
        let x = x.add(residual);
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    
    // 我们将层数加倍，通道减半
    layer1: BasicBlock<B>,
    layer1_2: BasicBlock<B>, // Extra depth
    
    layer2: BasicBlock<B>,
    layer2_2: BasicBlock<B>, // Extra depth
    
    layer3: BasicBlock<B>,
    layer3_2: BasicBlock<B>, // Extra depth
    layer3_3: BasicBlock<B>, // Extra depth

    layer4: BasicBlock<B>,
    layer4_2: BasicBlock<B>, // Extra depth
    
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc: Linear<B>, 
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        // Initial Conv
        let conv1 = Conv2dConfig::new([1, 32], [5, 5]) // 5x5 for better initial receptive field
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let bn1 = BatchNormConfig::new(32).init(device);

        // Deep & Narrow Architecture
        // Stage 1: 32 channels
        let layer1 = BasicBlock::new(32, 32, 1, device);
        let layer1_2 = BasicBlock::new(32, 32, 1, device);

        // Stage 2: 64 channels
        let layer2 = BasicBlock::new(32, 64, 2, device);
        let layer2_2 = BasicBlock::new(64, 64, 1, device);

        // Stage 3: 128 channels (STOP HERE, NO 256)
        // More layers here to process complex distortions
        let layer3 = BasicBlock::new(64, 128, 2, device);
        let layer3_2 = BasicBlock::new(128, 128, 1, device);
        let layer3_3 = BasicBlock::new(128, 128, 1, device);

        // Stage 4: Still 128 channels, just downsample spatial dims
        let layer4 = BasicBlock::new(128, 128, 2, device); // Stride 2
        let layer4_2 = BasicBlock::new(128, 128, 1, device);

        // Pooling: [B, 128, 1, 8]
        let pool = AdaptiveAvgPool2dConfig::new([1, POOL_WIDTH]).init();
        let dropout = DropoutConfig::new(0.5).init(); 
        
        // Linear: 128 * 8 = 1024 -> 40
        // Previous was 2048 -> 40. This is 2x smaller too.
        let fc = LinearConfig::new(128 * POOL_WIDTH, CAPTCHA_LENGTH * NUM_CLASSES).init(device);
        
        Self {
            conv1, bn1, relu: Relu::new(),
            layer1, layer1_2,
            layer2, layer2_2,
            layer3, layer3_2, layer3_3,
            layer4, layer4_2,
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

        let x = self.layer4.forward(x);
        let x = self.layer4_2.forward(x);

        let x = self.pool.forward(x); // -> [B, 128, 1, 8]
        let x = x.reshape([batch_size, 128 * POOL_WIDTH]);

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
        let output = self.forward(images); // [B, 4, 10]
        
        let output_flat = output.reshape([batch_size * CAPTCHA_LENGTH, NUM_CLASSES]);
        let targets_flat = targets.reshape([batch_size * CAPTCHA_LENGTH]);

        // Standard CrossEntropy is fine. Label smoothing is usually done by modifying targets,
        // but Burn's built-in CrossEntropyLoss doesn't support soft labels easily yet without OneHot.
        // For simplicity and speed, we stick to standard CE but rely on Dropout and Data Augmentation for regularization.
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&output_flat.device())
            .forward(output_flat.clone(), targets_flat.clone());

        burn::train::ClassificationOutput::new(loss, output_flat, targets_flat)
    }
}