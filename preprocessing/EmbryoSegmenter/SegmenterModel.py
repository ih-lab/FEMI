import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# SegmenterModel.py
#
# Defines the PyTorch model used to segment the embryo from the background
# in a still frame taken from a time-lapse microscope video. The model is a
# small convolutional encoder/decoder (an autoencoder-style "U-Net without
# skip connections") that takes in a single-channel (grayscale) 224x224
# image and outputs a 224x224 probability mask indicating which pixels
# belong to the embryo.
#
# This model is only used during PREPROCESSING (see preprocessing_images.py)
# to crop each raw image tightly around the embryo before it is fed into the
# main FEMI foundation model. It is a separate, much smaller network from
# FEMI itself and is trained independently (its pretrained weights are
# expected to live in `EmbryoSegmenter/weights/`).
# --------------------------------------------------------------------------


class BasicConv2d(nn.Module):
    """
    A small helper block: Conv2d -> BatchNorm -> ReLU.

    This mirrors the "basic conv block" pattern used in Inception-style
    networks: every convolution is immediately followed by batch
    normalization (for training stability) and a ReLU activation (to
    introduce non-linearity). Bias is disabled on the conv layer because the
    BatchNorm that follows already learns its own shift term, so a
    separate conv bias would be redundant.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        # padding=1 keeps the spatial size unchanged for a 3x3 kernel (the
        # kernel_size itself is passed in via **kwargs by the caller).
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConv2dSigmoid(nn.Module):
    """
    Same as BasicConv2d, but ends with a Sigmoid instead of ReLU.

    This is used only for the final output layer of the decoder, since we
    want the network to output values in the [0, 1] range that can be
    interpreted as "probability this pixel belongs to the embryo" (i.e. a
    segmentation mask), rather than an unbounded activation.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.sigmoid(x)

# A copy of Inception V3 allowing a dynamic number of input channels
class SegmenterModel(nn.Module):
    """
    Encoder/decoder CNN that maps a 224x224x1 grayscale frame to a 224x224x1
    segmentation mask.

    Architecture summary:
      - Encoder: 6 conv blocks, each followed by a 2x2 max-pool (except the
        very last one), progressively halving the spatial resolution while
        doubling the channel count (1 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512).
        This produces a compact 7x7x512 bottleneck representation.
      - Decoder: 6 conv blocks that mirror the encoder in reverse, using
        `F.interpolate` (simple upsampling, not learned transposed
        convolutions) to double the spatial resolution back up to 224x224,
        while halving the channel count at each step, ending in a single
        224x224x1 sigmoid output (the predicted mask).

    Note: there are no skip connections between encoder and decoder layers
    (unlike a classic U-Net), so all spatial detail must be reconstructed
    from the 7x7x512 bottleneck alone. This keeps the model small/fast but
    may limit fine boundary accuracy compared to a full U-Net.

    Because every layer's spatial size and channel count is fixed ahead of
    time (the input is always resized to 224x224 before being passed in,
    see `load_image` in preprocessing_images.py), the computational cost of
    a single forward pass through this network is CONSTANT — it does not
    depend on the size of the original source image, only on this fixed
    224x224 input resolution.
    """

    def __init__(self):
        super(SegmenterModel, self).__init__()

        #encoder
        # Comment shows the output tensor shape (HxWxC) after each block.
        # Spatial size halves after every max-pool call in forward().
        self.conv1 = BasicConv2d(1, 16, kernel_size=3) # 224x224x16
        self.conv2 = BasicConv2d(16, 32, kernel_size=3) # 112x112x32
        self.conv3 = BasicConv2d(32, 64, kernel_size=3) #56x56x64
        self.conv4 = BasicConv2d(64, 128, kernel_size=3) # 28x28x128
        self.conv5 = BasicConv2d(128, 256, kernel_size=3) # 14x14x256
        self.conv6 = BasicConv2d(256, 512, kernel_size=3) #7x7x512

        #decoder
        # Mirrors the encoder: each block halves the channel count while
        # `F.interpolate` in forward() doubles the spatial resolution.
        self.deconv5 = BasicConv2d(512, 256, kernel_size=3) # 7x7x256
        self.deconv4 = BasicConv2d(256, 128, kernel_size=3) # 14x14x128
        self.deconv3 = BasicConv2d(128, 64, kernel_size=3) # 28x28x64
        self.deconv2 = BasicConv2d(64, 32, kernel_size=3) # 56x56x32
        self.deconv1 = BasicConv2d(32, 16, kernel_size=3) # 112x112x16
        self.decoded = BasicConv2dSigmoid(16, 1, kernel_size=3) # 224x224x1


    def forward(self, x):
        # ----------------------- Encoder (downsampling) -----------------------
        # Input x: (batch, 1, 224, 224) - single-channel grayscale frame.
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 112x112x16
        x = self.conv2(x) # 112x112x32
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 56x56x32
        x = self.conv3(x) #56x56x64
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 28x28x64
        x = self.conv4(x) # 28x28x128
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 14x14x128
        x = self.conv5(x) # 14x14x256
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 7x7x256
        x = self.conv6(x) #7x7x512
        # x is now the 7x7x512 bottleneck - the most compressed representation.

        #decoder
        # ----------------------- Decoder (upsampling) -----------------------
        # `F.interpolate(x, scale_factor=2)` performs simple (non-learned)
        # nearest-neighbor upsampling by default, doubling H and W each time.
        x = self.deconv5(x) # 7x7x256
        x = F.interpolate(x, scale_factor=2) # 14x14x256
        x = self.deconv4(x) # 14x14x128
        x = F.interpolate(x, scale_factor=2) # 28x28x128
        x = self.deconv3(x) # 28x28x64
        x = F.interpolate(x, scale_factor=2) # 56x56x64
        x = self.deconv2(x) # 56x56x32
        x = F.interpolate(x, scale_factor=2) # 112x112x32
        x = self.deconv1(x) # 112x112x16
        x = F.interpolate(x, scale_factor=2) # 224x224x16
        x = self.decoded(x) # 224x224x1
        # Final output: (batch, 1, 224, 224) mask of values in [0, 1], where
        # values closer to 1 indicate higher confidence the pixel belongs to
        # the embryo. Thresholding happens downstream in segment_image().

        return x


class SegmenterModelSkip(nn.Module):
    """
    U-Net variant of `SegmenterModel`: same encoder and the same
    `BasicConv2d`/`BasicConv2dSigmoid` building blocks, but each decoder
    stage concatenates the matching encoder stage's pre-pool feature map
    (a "skip connection") before its conv, so boundary detail doesn't have
    to be reconstructed from the 7x7x512 bottleneck alone.

    Only 5 of ~72 weight tensors differ in shape from `SegmenterModel`: the
    `conv.weight` of `deconv4`/`deconv3`/`deconv2`/`deconv1`/`decoded` (their
    input channel count doubles from the concatenation). Everything else -
    the whole encoder, `deconv5` (which consumes the bottleneck directly,
    nothing to concatenate there), and even the BatchNorm parameters of
    those 5 changed layers (their output channel count is unchanged) -
    matches `SegmenterModel` exactly by name and shape.

    Trained via `train_segmenter.py --skip-connections`; checkpoints live in
    `EmbryoSegmenter/weights_skip/`. `preprocessing_images.py` uses this
    model (loaded from `weights_skip/latest.pt`) instead of `SegmenterModel`.
    `SegmenterModel` above is kept as-is for benchmarking (`eval_segmenter.py`
    configs A/B) and is not removed.
    """

    def __init__(self):
        super(SegmenterModelSkip, self).__init__()

        #encoder - identical to SegmenterModel's
        self.conv1 = BasicConv2d(1, 16, kernel_size=3)
        self.conv2 = BasicConv2d(16, 32, kernel_size=3)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3)
        self.conv4 = BasicConv2d(64, 128, kernel_size=3)
        self.conv5 = BasicConv2d(128, 256, kernel_size=3)
        self.conv6 = BasicConv2d(256, 512, kernel_size=3)

        #decoder - each stage's input channel count doubles to make room
        # for the concatenated skip connection from the matching encoder stage.
        self.deconv5 = BasicConv2d(512, 256, kernel_size=3)
        self.deconv4 = BasicConv2d(256 + 256, 128, kernel_size=3)
        self.deconv3 = BasicConv2d(128 + 128, 64, kernel_size=3)
        self.deconv2 = BasicConv2d(64 + 64, 32, kernel_size=3)
        self.deconv1 = BasicConv2d(32 + 32, 16, kernel_size=3)
        self.decoded = BasicConv2dSigmoid(16 + 16, 1, kernel_size=3)

    def forward(self, x):
        # ----------------------- Encoder (downsampling) -----------------------
        # Keep each pre-pool feature map (c1..c5) around for the skip
        # connections below - this is the only structural difference from
        # SegmenterModel.forward(), which discards them.
        c1 = self.conv1(x)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.conv2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.conv3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        c4 = self.conv4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)
        c5 = self.conv5(p4)
        p5 = F.max_pool2d(c5, kernel_size=2, stride=2)
        c6 = self.conv6(p5)  # 7x7x512 bottleneck

        # ----------------------- Decoder (upsampling + skip connections) -----
        x = self.deconv5(c6)  # no skip here - c6 IS the direct input already
        x = F.interpolate(x, scale_factor=2)
        x = self.deconv4(torch.cat([x, c5], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.deconv3(torch.cat([x, c4], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.deconv2(torch.cat([x, c3], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.deconv1(torch.cat([x, c2], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.decoded(torch.cat([x, c1], dim=1))
        return x


