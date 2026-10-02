import torch
import torch.nn as nn
import torchvision
from torch.ao.quantization import QuantStub, DeQuantStub


class QuantizableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)
        return out

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(
            self,
            [["conv1", "bn1", "relu"], ["conv2", "bn2"]],
            inplace=True
        )
        if self.downsample is not None:
            # downsample is usually Sequential(conv, bn)
            torch.ao.quantization.fuse_modules(
                self.downsample,
                [["0", "1"]],
                inplace=True
            )


class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        if pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            base_model = torchvision.models.resnet18(weights=weights)
        else:
            base_model = torchvision.models.resnet18(weights=None)

        # Adapt first conv for grayscale
        original_conv = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Replace fc
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)

        # Replace residual blocks with quantizable blocks
        base_model.layer1 = nn.Sequential(*[QuantizableBasicBlock(b) for b in base_model.layer1])
        base_model.layer2 = nn.Sequential(*[QuantizableBasicBlock(b) for b in base_model.layer2])
        base_model.layer3 = nn.Sequential(*[QuantizableBasicBlock(b) for b in base_model.layer3])
        base_model.layer4 = nn.Sequential(*[QuantizableBasicBlock(b) for b in base_model.layer4])

        self.quant = QuantStub()
        self.model = base_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # fuse top layers
        torch.ao.quantization.fuse_modules(
            self.model,
            [["conv1", "bn1", "relu"]],
            inplace=True
        )

        # fuse residual blocks
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.model, layer_name)
            for block in layer:
                block.fuse_model()


def load_baseline_weights_into_qat_model(qat_model, baseline_state_dict):
    """
    Load matching weights from baseline model into QAT-ready model.
    The baseline keys are under 'model.*' while qat model keys are under 'model.*' too,
    but residual blocks changed class, not parameter names for conv/bn/fc.
    """
    missing, unexpected = qat_model.load_state_dict(baseline_state_dict, strict=False)
    return missing, unexpected