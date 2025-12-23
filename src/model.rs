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

/// Input image height
pub const IMG_HEIGHT: usize = 60;
/// Input image width
pub const IMG_WIDTH: usize = 160;

const NUM_CLASSES: usize = 10;
const CAPTCHA_LENGTH: usize = 4;
const POOL_WIDTH: usize = 8; 

/// Squeeze-and-Excitation Block for channel-wise attention.
/// Helps the model focus on informative channels and suppress noise.
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

/// ResNet Basic Block with SE Attention.
/// Standard 3x3 convolutions are used to effectively handle character overlap.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    se: SeBlock<B>,
    activation: Relu,
    downsample: Option<Conv2d<B>>,
}

impl<B: Backend> ResBlock<B> {
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

        // Reduction=8 for lightweight attention
        let se = SeBlock::new(out_channels, 8, device);

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
        
        let x = self.se.forward(x);

        let x = x.add(residual);
        self.activation.forward(x)
    }
}

/// Captcha Solver Model (SE-ResNet Lite).
/// Optimized for accuracy on sticky characters while maintaining low parameter count.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    
    // Stage 1
    layer1: ResBlock<B>,
    layer1_2: ResBlock<B>, 

    // Stage 2
    layer2: ResBlock<B>,
    layer2_2: ResBlock<B>,

    // Stage 3 (Deep but narrow to save parameters)
    layer3: ResBlock<B>,
    layer3_2: ResBlock<B>, 
    layer3_3: ResBlock<B>, 

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
        let layer1 = ResBlock::new(32, 32, 1, device);
        let layer1_2 = ResBlock::new(32, 32, 1, device);

        // Stage 2: 64 channels
        let layer2 = ResBlock::new(32, 64, 2, device);
        let layer2_2 = ResBlock::new(64, 64, 1, device);

        // Stage 3: 64 channels (Max 64 to keep model size ~1MB)
        let layer3 = ResBlock::new(64, 64, 2, device); 
        let layer3_2 = ResBlock::new(64, 64, 1, device);
        let layer3_3 = ResBlock::new(64, 64, 1, device);

        // Pooling -> [B, 64, 1, 8]
        let pool = AdaptiveAvgPool2dConfig::new([1, POOL_WIDTH]).init();
        
        // High dropout to prevent overfitting on small dataset
        let dropout = DropoutConfig::new(0.5).init(); 
        
        // FC: 512 -> 40
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

        let x = self.pool.forward(x); 
        let x = x.reshape([batch_size, 64 * POOL_WIDTH]);

        let x = self.dropout.forward(x);
        let x = self.fc.forward(x);

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