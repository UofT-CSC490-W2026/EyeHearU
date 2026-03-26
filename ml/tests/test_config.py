"""Unit tests for ml/config.py."""

from pathlib import Path

from config import Config, DataConfig, ModelConfig, TrainConfig


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.num_classes == 2000
        assert cfg.num_frames == 16
        assert cfg.frame_height == 224
        assert cfg.frame_width == 224
        assert cfg.num_workers == 4
        assert isinstance(cfg.processed_data_dir, Path)

    def test_custom_values(self):
        cfg = DataConfig(num_classes=100, num_frames=32)
        assert cfg.num_classes == 100
        assert cfg.num_frames == 32


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.backbone == "r3d_18"
        assert cfg.pretrained is True
        assert cfg.backbone_freeze_epochs == 3
        assert cfg.head_dropout == 0.5

    def test_custom_values(self):
        cfg = ModelConfig(backbone="mc3_18", pretrained=False, head_dropout=0.3)
        assert cfg.backbone == "mc3_18"
        assert cfg.pretrained is False
        assert cfg.head_dropout == 0.3


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 1e-4
        assert cfg.epochs == 30
        assert cfg.early_stopping_patience == 5
        assert cfg.scheduler == "cosine"
        assert cfg.warmup_epochs == 2
        assert isinstance(cfg.checkpoint_dir, Path)
        assert cfg.save_every_n_epochs == 5
        assert cfg.device == "mps"
        assert cfg.use_wandb is False
        assert cfg.wandb_project == "eye-hear-u"

    def test_custom_values(self):
        cfg = TrainConfig(batch_size=16, epochs=10, device="cpu")
        assert cfg.batch_size == 16
        assert cfg.epochs == 10
        assert cfg.device == "cpu"


class TestConfig:
    def test_default_composition(self):
        cfg = Config()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.train, TrainConfig)

    def test_nested_defaults(self):
        cfg = Config()
        assert cfg.data.num_classes == 2000
        assert cfg.model.backbone == "r3d_18"
        assert cfg.train.batch_size == 8

    def test_custom_nested(self):
        cfg = Config(
            data=DataConfig(num_classes=500),
            model=ModelConfig(backbone="mc3_18"),
            train=TrainConfig(epochs=5),
        )
        assert cfg.data.num_classes == 500
        assert cfg.model.backbone == "mc3_18"
        assert cfg.train.epochs == 5

    def test_independent_instances(self):
        """Each Config() creates independent sub-configs."""
        cfg1 = Config()
        cfg2 = Config()
        cfg1.data.num_classes = 999
        assert cfg2.data.num_classes == 2000
