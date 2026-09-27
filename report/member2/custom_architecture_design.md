# Custom CNN Architecture Design

## 1. Introduction

Two custom convolutional neural network architectures were developed for the
EuroSAT image-classification dataset. All input images have three RGB channels
and a spatial resolution of 64 × 64 pixels.

Model A is a standard CNN that uses conventional convolutional layers. Model B
is a lightweight CNN that uses depthwise-separable convolution. Model B was
designed to contain fewer than 100,000 trainable parameters so that it is more
appropriate for deployment on a resource-constrained device.

---

## 2. Model A: Standard CNN

### 2.1 Architecture

Model A uses three standard convolutional layers. Each convolution is followed
by the ReLU activation function and 2 × 2 max pooling. Adaptive average pooling
converts the final feature maps into one value per channel. Two fully connected
layers perform classification.

| Stage | Layer | Configuration | Output shape |
|---|---|---|---|
| Input | Image | RGB, 64 × 64 | 3 × 64 × 64 |
| 1 | Convolution + ReLU | 3 × 3, 3 → 32, padding 1 | 32 × 64 × 64 |
| 2 | Max pooling | 2 × 2, stride 2 | 32 × 32 × 32 |
| 3 | Convolution + ReLU | 3 × 3, 32 → 64, padding 1 | 64 × 32 × 32 |
| 4 | Max pooling | 2 × 2, stride 2 | 64 × 16 × 16 |
| 5 | Convolution + ReLU | 3 × 3, 64 → 128, padding 1 | 128 × 16 × 16 |
| 6 | Max pooling | 2 × 2, stride 2 | 128 × 8 × 8 |
| 7 | Adaptive average pooling | Output size 1 × 1 | 128 × 1 × 1 |
| 8 | Flatten | 128 features | 128 |
| 9 | Fully connected + ReLU | 128 → 64 | 64 |
| 10 | Dropout | Regularisation | 64 |
| 11 | Fully connected | 64 → 10 | 10 logits |

### 2.2 Parameter calculation

For a standard convolutional layer with a bias term:

Parameter count = Cout × (Kh × Kw × Cin + 1)

where Cin is the number of input channels, Cout is the number of output
channels and Kh × Kw is the kernel size.

First convolution:

32 × (3 × 3 × 3 + 1) = 896

Second convolution:

64 × (3 × 3 × 32 + 1) = 18,496

Third convolution:

128 × (3 × 3 × 64 + 1) = 73,856

For a fully connected layer:

Parameter count = Nout × (Nin + 1)

First fully connected layer:

64 × (128 + 1) = 8,256

Output layer:

10 × (64 + 1) = 650

Therefore, the total trainable parameter count is:

896 + 18,496 + 73,856 + 8,256 + 650 = 102,154 parameters

Pooling, ReLU, adaptive average pooling, flattening and dropout do not contain
trainable parameters.

---

## 3. Model B: Lightweight CNN

### 3.1 Depthwise-separable convolution

A standard convolution processes the spatial dimensions and combines input
channels in a single operation. A depthwise-separable convolution divides this
operation into two stages:

1. A 3 × 3 depthwise convolution independently processes each input channel.
2. A 1 × 1 pointwise convolution combines information from different channels.

This separation substantially reduces the number of parameters and arithmetic
operations.

### 3.2 Architecture

Model B contains four depthwise-separable convolution blocks. Each block uses a
3 × 3 depthwise convolution, a 1 × 1 pointwise convolution and ReLU activation.
A 2 × 2 max-pooling layer reduces the spatial dimensions after each block.

| Stage | Layer | Configuration | Output shape |
|---|---|---|---|
| Input | Image | RGB, 64 × 64 | 3 × 64 × 64 |
| 1 | Depthwise-separable block | 3 → 16 | 16 × 64 × 64 |
| 2 | Max pooling | 2 × 2, stride 2 | 16 × 32 × 32 |
| 3 | Depthwise-separable block | 16 → 32 | 32 × 32 × 32 |
| 4 | Max pooling | 2 × 2, stride 2 | 32 × 16 × 16 |
| 5 | Depthwise-separable block | 32 → 64 | 64 × 16 × 16 |
| 6 | Max pooling | 2 × 2, stride 2 | 64 × 8 × 8 |
| 7 | Depthwise-separable block | 64 → 96 | 96 × 8 × 8 |
| 8 | Max pooling | 2 × 2, stride 2 | 96 × 4 × 4 |
| 9 | Adaptive average pooling | Output size 1 × 1 | 96 × 1 × 1 |
| 10 | Flatten | 96 features | 96 |
| 11 | Fully connected | 96 → 10 | 10 logits |

### 3.3 Parameter calculation

The trainable parameter count of a depthwise-separable convolution block is:

Depthwise parameters = Cin × (Kh × Kw + 1)

Pointwise parameters = Cout × (Cin + 1)

Total block parameters = Depthwise parameters + Pointwise parameters

Block 1, 3 → 16:

Depthwise = 3 × (3 × 3 + 1) = 30

Pointwise = 16 × (3 + 1) = 64

Block 1 total = 30 + 64 = 94

Block 2, 16 → 32:

Depthwise = 16 × (3 × 3 + 1) = 160

Pointwise = 32 × (16 + 1) = 544

Block 2 total = 160 + 544 = 704

Block 3, 32 → 64:

Depthwise = 32 × (3 × 3 + 1) = 320

Pointwise = 64 × (32 + 1) = 2,112

Block 3 total = 320 + 2,112 = 2,432

Block 4, 64 → 96:

Depthwise = 64 × (3 × 3 + 1) = 640

Pointwise = 96 × (64 + 1) = 6,240

Block 4 total = 640 + 6,240 = 6,880

Classification layer:

10 × (96 + 1) = 970

Therefore, the total trainable parameter count is:

94 + 704 + 2,432 + 6,880 + 970 = 11,080 parameters

The parameter limit specified for Model B is 100,000 parameters.

11,080 ≤ 100,000

Therefore, Model B satisfies the lightweight-model requirement.

---

## 4. Activation-function selection

ReLU was selected as the activation function for both custom models. ReLU is
defined as:

ReLU(x) = max(0, x)

ReLU is computationally inexpensive because it only compares each input value
with zero. It introduces non-linearity without adding trainable parameters and
helps reduce the vanishing-gradient problem compared with saturating activation
functions such as sigmoid. These properties make ReLU suitable for models
intended for resource-constrained devices.

---

## 5. Parameter and size comparison

The theoretical FP32 model size was calculated using:

Model size in bytes = number of parameters × 4 bytes

| Measurement | Model A | Model B |
|---|---:|---:|
| Trainable parameters | 102,154 | 11,080 |
| Theoretical FP32 size| 399.04 KB | 43.28 KB |
| Actual serialized size  | 402.82 KB | 49.67 KB |

Model B contains 91,074 fewer parameters than Model A. This represents an
approximate parameter reduction of 89.15%.

The actual serialized sizes may be slightly larger than the theoretical sizes
because a saved PyTorch file can contain metadata in addition to parameter
values.

---

## 6. Trade-off of depthwise-separable convolution

The principal advantage of Model B is its substantially smaller parameter count
and memory requirement. It should also require fewer multiplication and
addition operations than the standard CNN. Therefore, it is more suitable for a
resource-constrained edge device.

However, the reduced parameter count also limits the model's representational
capacity. Because depthwise convolution initially processes channels
independently, Model B may learn less complex feature relationships than Model
A. This could produce lower classification accuracy.

The final selection should therefore consider test accuracy, precision, recall,
model size and computational cost rather than choosing a model based only on
accuracy or size.