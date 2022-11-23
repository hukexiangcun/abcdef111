import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.decoder1 import build_decoder1
from .s2cnn.spherical import dis_convolution
from .networks import FPA
from .basic_layers import ResidualBlock
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=6,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder1= build_decoder1(num_classes, backbone, BatchNorm)
        
        # self.fpn = FPA(256)
        # self.resblock = ResidualBlock(256,256)
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        
        x = input
        
        x, low_level_feat = self.backbone(x)
        
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = self.aspp(x)
        
        x = self.decoder(x, low_level_feat)
        x = self.decoder1(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    
    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    
class Conv_P(nn.Module):
    def __init__(self, batch_size):
        super(Conv_P, self).__init__()
        
        self.dis_convolution = dis_convolution(batch_size)
    def forward(self, input):
        x = self.dis_convolution(input)
        return x
    def get_1x_lr_params_p(self):
        modules = [self.dis_convolution]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
class Conv_N(nn.Module):
    def __init__(self):
        super(Conv_N, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu0 = nn.ReLU(inplace=False)
    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.relu0(x) 
        return x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def get_1x_lr_params_n(self):
        modules = [self.conv0]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


