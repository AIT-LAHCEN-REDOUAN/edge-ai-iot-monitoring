import torch
import torch.nn as nn
from torchvision import models


class ResNet18BinaryClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18(weights=None)

        # Change first conv layer to accept 1-channel grayscale input instead of 3-channel RGB
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def main():
    print("🧪 TESTING RESNET18 MODEL...\n")

    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False)

    dummy_input = torch.randn(8, 1, 128, 128)
    output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expected: (8, 2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n🎯 ResNet18 test completed successfully.")


if __name__ == "__main__":
    main()