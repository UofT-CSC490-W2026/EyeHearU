"""
ASL Sign Classifier — CNN backbone + Transformer encoder + classification head.

Architecture overview:
  1. CNN backbone (ResNet / MobileNet) extracts spatial features from the input image
  2. Features are projected into a sequence of patch embeddings
  3. A lightweight Transformer encoder captures relationships between patches
  4. A classification head produces logits for each sign class

This design allows:
  - Transfer learning from ImageNet-pretrained backbones
  - The Transformer to learn spatial relationships in hand shapes
  - Future extension to video (temporal sequences) by changing the input
"""

import torch
import torch.nn as nn
import torchvision.models as models
import math


class PatchProjection(nn.Module):
    """
    Takes CNN feature maps of shape (B, C, H, W) and projects them
    into a sequence of patch embeddings of shape (B, num_patches, d_model).
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, d_model, H, W)
        x = self.projection(x)
        B, D, H, W = x.shape
        # Flatten spatial dims: (B, d_model, H*W) → (B, H*W, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the patch sequence."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ASLClassifier(nn.Module):
    """
    Full ASL sign classifier.

    Args:
        num_classes: Number of sign classes to predict
        backbone: Name of the CNN backbone ("resnet18", "resnet50", "mobilenet_v3_small")
        pretrained: Whether to use ImageNet-pretrained weights
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of Transformer encoder layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes: int = 62,
        backbone: str = "resnet18",
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- CNN Backbone ---
        self.backbone, backbone_out_channels = self._build_backbone(backbone, pretrained)

        # --- Patch projection ---
        self.patch_proj = PatchProjection(backbone_out_channels, d_model)

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        # CLS token — learnable, prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def _build_backbone(self, name: str, pretrained: bool):
        """Build and return the CNN backbone, stripping the final FC/pool layers."""
        weights = "IMAGENET1K_V1" if pretrained else None

        if name == "resnet18":
            model = models.resnet18(weights=weights)
            out_channels = 512
            # Remove avgpool and fc
            backbone = nn.Sequential(*list(model.children())[:-2])
        elif name == "resnet50":
            model = models.resnet50(weights=weights)
            out_channels = 2048
            backbone = nn.Sequential(*list(model.children())[:-2])
        elif name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            out_channels = 576
            backbone = model.features
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return backbone, out_channels

    def freeze_backbone(self):
        """Freeze CNN backbone parameters (for initial training epochs)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze CNN backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        B = x.size(0)

        # 1. Extract CNN features: (B, 3, 224, 224) → (B, C, 7, 7) for resnet18
        features = self.backbone(x)

        # 2. Project to patch embeddings: (B, C, 7, 7) → (B, 49, d_model)
        patches = self.patch_proj(features)

        # 3. Prepend CLS token: (B, 49, d_model) → (B, 50, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)

        # 4. Add positional encoding
        patches = self.pos_enc(patches)

        # 5. Transformer encoder
        encoded = self.transformer_encoder(patches)

        # 6. Take the CLS token output for classification
        cls_output = encoded[:, 0, :]

        # 7. Classify
        logits = self.classifier(cls_output)

        return logits
