# ============================================================
# SpaceNet 6 - Model Definitions & Checkpoint Loading
# ============================================================

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34, ResNet34_Weights


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)


# ============================================================
# U-NET BUILDING BLOCKS
# Exact architecture used in the notebook
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels // 2 + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        # Handle spatial mismatch
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat(
            [skip, x],
            dim=1
        )

        return self.conv(x)


# ============================================================
# E1 - SAR U-NET BASELINE
# ============================================================

class SARUNet(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=1
    ):

        super().__init__()

        self.inc = DoubleConv(
            in_channels,
            64
        )

        self.down1 = DownBlock(
            64,
            128
        )

        self.down2 = DownBlock(
            128,
            256
        )

        self.down3 = DownBlock(
            256,
            512
        )

        self.down4 = DownBlock(
            512,
            1024
        )

        self.up1 = UpBlock(
            1024,
            512,
            512
        )

        self.up2 = UpBlock(
            512,
            256,
            256
        )

        self.up3 = UpBlock(
            256,
            128,
            128
        )

        self.up4 = UpBlock(
            128,
            64,
            64
        )

        self.outc = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(
            x5,
            x4
        )

        x = self.up2(
            x,
            x3
        )

        x = self.up3(
            x,
            x2
        )

        x = self.up4(
            x,
            x1
        )

        return self.outc(x)


# ============================================================
# RESNET34 ENCODER
# Exact architecture used in E2/E3/E4
# ============================================================

class ResNet34Encoder(nn.Module):

    def __init__(self, encoder):

        super().__init__()

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    def forward(self, x):

        x0 = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x1 = self.layer1(
            self.maxpool(x0)
        )

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

        return (
            x0,
            x1,
            x2,
            x3,
            x4
        )


# ============================================================
# E2 / E3 / E4 - RESNET34 U-NET
# ============================================================

class ResNet34UNet(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        pretrained=False
    ):

        super().__init__()

        # ----------------------------------------------------
        # ResNet34 backbone
        #
        # pretrained=False by default: inference models are always
        # built from a trained checkpoint, so downloading ImageNet
        # weights would be wasted time and disk. Set pretrained=True
        # only to reproduce the training-time initialization.
        # ----------------------------------------------------

        if pretrained:

            backbone = resnet34(
                weights=ResNet34_Weights.DEFAULT
            )

        else:

            backbone = resnet34(
                weights=None
            )

        # ----------------------------------------------------
        # Adapt RGB -> 4 channels
        # ----------------------------------------------------

        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        if pretrained:

            with torch.no_grad():

                # Original RGB weights
                new_conv.weight[:, :3] = (
                    old_conv.weight
                )

                # Fourth channel = mean RGB weights
                new_conv.weight[:, 3:4] = (
                    old_conv.weight.mean(
                        dim=1,
                        keepdim=True
                    )
                )

        elif in_channels > 3:

            with torch.no_grad():

                # No ImageNet weights to carry over. Replicate the
                # random stem weights across the extra channels so the
                # 4-channel input has a sane scale before the
                # checkpoint is loaded.
                stem_init = old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                ).repeat(
                    1, in_channels, 1, 1
                )

                new_conv.weight.copy_(
                    stem_init * (
                        3.0 / in_channels
                    )
                )

        backbone.conv1 = new_conv

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = ResNet34Encoder(
            backbone
        )

        # ResNet feature channels:
        #
        # x0 = 64
        # x1 = 64
        # x2 = 128
        # x3 = 256
        # x4 = 512

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up1 = UpBlock(
            512,
            256,
            256
        )

        self.up2 = UpBlock(
            256,
            128,
            128
        )

        self.up3 = UpBlock(
            128,
            64,
            64
        )

        self.up4 = UpBlock(
            64,
            64,
            64
        )

        self.final_up = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.outc = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        x0, x1, x2, x3, x4 = (
            self.encoder(x)
        )

        x = self.up1(
            x4,
            x3
        )

        x = self.up2(
            x,
            x2
        )

        x = self.up3(
            x,
            x1
        )

        x = self.up4(
            x,
            x0
        )

        x = self.final_up(x)

        logits = self.outc(x)

        if return_features:

            return logits, x4

        return logits


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    state_dict = {
        key.replace(
            "module.",
            "",
            1
        ): value

        for key, value in state_dict.items()
    }

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model


# ============================================================
# E1 MODEL LOADER
# ============================================================

def load_e1_model():

    model = SARUNet(
        in_channels=4,
        out_channels=1
    )

    checkpoint = (
        CHECKPOINT_DIR
        / "E1_SAR_UNet_best.pth"
    )

    load_checkpoint(
        model,
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# E2 MODEL LOADER
# ============================================================

def load_e2_model():

    model = ResNet34UNet(
        in_channels=4,
        out_channels=1
    )

    checkpoint = (
        CHECKPOINT_DIR
        / "E2_ResNet34_UNet_best.pth"
    )

    load_checkpoint(
        model,
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# E3 MODEL LOADER
# ============================================================

def load_e3_model():

    model = ResNet34UNet(
        in_channels=4,
        out_channels=1
    )

    checkpoint = (
        CHECKPOINT_DIR
        / "E3_Optical_ResNet34_UNet_best.pth"
    )

    load_checkpoint(
        model,
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# E4 MODEL LOADER
# ============================================================

def load_e4_model():

    model = ResNet34UNet(
        in_channels=4,
        out_channels=1
    )

    checkpoint = (
        CHECKPOINT_DIR
        / "E4_SAR_KD_best.pth"
    )

    load_checkpoint(
        model,
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# TEST ALL CHECKPOINTS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpaceNet 6 - Loading All Model Checkpoints")
    print("=" * 60)

    print(f"\nDevice available: {DEVICE}")

    print(
        f"Checkpoint directory:\n"
        f"{CHECKPOINT_DIR}\n"
    )

    # --------------------------------------------------------
    # E1
    # --------------------------------------------------------

    try:

        model = load_e1_model()

        print(
            "E1: checkpoint loaded successfully"
        )

        del model

    except Exception as e:

        print(
            f"E1: FAILED\n{e}"
        )

    # --------------------------------------------------------
    # E2
    # --------------------------------------------------------

    try:

        model = load_e2_model()

        print(
            "E2: checkpoint loaded successfully"
        )

        del model

    except Exception as e:

        print(
            f"E2: FAILED\n{e}"
        )

    # --------------------------------------------------------
    # E3
    # --------------------------------------------------------

    try:

        model = load_e3_model()

        print(
            "E3: checkpoint loaded successfully"
        )

        del model

    except Exception as e:

        print(
            f"E3: FAILED\n{e}"
        )

    # --------------------------------------------------------
    # E4
    # --------------------------------------------------------

    try:

        model = load_e4_model()

        print(
            "E4: checkpoint loaded successfully"
        )

        del model

    except Exception as e:

        print(
            f"E4: FAILED\n{e}"
        )

    print(
        "\n✓ Model checkpoint test complete"
    )