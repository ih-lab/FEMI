import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConv2dSigmoid(nn.Module):

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

    def __init__(self):
        super(SegmenterModel, self).__init__()

        #encoder
        self.conv1 = BasicConv2d(1, 16, kernel_size=3) # 224x224x16
        self.conv2 = BasicConv2d(16, 32, kernel_size=3) # 112x112x32
        self.conv3 = BasicConv2d(32, 64, kernel_size=3) #56x56x64
        self.conv4 = BasicConv2d(64, 128, kernel_size=3) # 28x28x128
        self.conv5 = BasicConv2d(128, 256, kernel_size=3) # 14x14x256
        self.conv6 = BasicConv2d(256, 512, kernel_size=3) #7x7x512

        #decoder
        self.deconv5 = BasicConv2d(512, 256, kernel_size=3) # 7x7x256
        self.deconv4 = BasicConv2d(256, 128, kernel_size=3) # 14x14x128
        self.deconv3 = BasicConv2d(128, 64, kernel_size=3) # 28x28x64
        self.deconv2 = BasicConv2d(64, 32, kernel_size=3) # 56x56x32
        self.deconv1 = BasicConv2d(32, 16, kernel_size=3) # 112x112x16
        self.decoded = BasicConv2dSigmoid(16, 1, kernel_size=3) # 224x224x1


    def forward(self, x):
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

        #decoder
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

        return x



