from pathlib import Path
from types import SimpleNamespace

import pytest

from openhgnn.config import Config
from openhgnn.trainerflow.dhcan_trainer import DHCANTrainer


def test_hcan_default_config_is_utf8_and_loadable():
    config_path = Path(__file__).parents[1] / "openhgnn" / "config.ini"

    config = Config(
        file_path=str(config_path),
        model="HCAN",
        dataset="HGBn-ACM",
        task="node_classification",
        gpu=-1,
    )

    assert config.hidden_dim == 64
    assert config.max_hop == 2
    assert config.early_stop_metric == "Micro_f1"


def test_dhcan_uses_decoupled_training_defaults():
    config_path = Path(__file__).parents[1] / "openhgnn" / "config.ini"

    config = Config(
        file_path=str(config_path),
        model="DHCAN",
        dataset="ogbn-mag",
        task="node_classification",
        gpu=0,
    )

    assert config.cache_device == "cpu"
    assert config.cache_target_on_gpu is True
    assert config.batch_size == 65536


def test_dhcan_trainer_rejects_unsupported_tasks():
    with pytest.raises(ValueError, match="node classification only"):
        DHCANTrainer(SimpleNamespace(task="link_prediction"))
