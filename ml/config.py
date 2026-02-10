"""
Central configuration for ML training and evaluation.

Adjust these values when experimenting with different setups.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    # Paths
    raw_data_dir: Path = Path("../data/raw")
    processed_data_dir: Path = Path("../data/processed")

    # Target vocabulary — start with the most common/useful signs
    # Professor feedback: be specific about which words/scenarios
    # We focus on: greetings, basic needs, restaurant ordering, medical emergencies
    target_vocab: list[str] = field(default_factory=lambda: [
        # Greetings & basics
        "hello", "goodbye", "please", "thank you", "sorry", "yes", "no",
        "help", "stop", "wait",
        # Self-identification
        "name", "my", "your", "me", "you",
        # Common needs
        "water", "food", "bathroom", "medicine", "pain", "sick",
        # Restaurant scenario
        "eat", "drink", "hot", "cold", "more", "enough", "check",
        # Medical scenario
        "hurt", "emergency", "doctor", "allergic",
        # Numbers 1-10 for basic communication
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        # Letters A-Z (fingerspelling fallback)
        *[chr(c) for c in range(ord("A"), ord("Z") + 1)],
    ])

    # Image preprocessing
    image_size: int = 224
    num_workers: int = 4

    # Dataset splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # CNN Backbone
    backbone: str = "resnet18"  # resnet18, resnet50, mobilenet_v3_small
    pretrained: bool = True
    backbone_freeze_epochs: int = 2  # freeze backbone for first N epochs

    # Transformer encoder head
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dim: int = 256
    transformer_dropout: float = 0.1

    # Classification
    num_classes: int = 62  # 26 letters + 36 words (adjust based on target_vocab)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 5

    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 2

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5

    # Device
    device: str = "mps"  # "cpu", "cuda", or "mps" (Apple Silicon)

    # Logging
    use_wandb: bool = False
    wandb_project: str = "eye-hear-u"


@dataclass
class Config:
    """Top-level config combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
