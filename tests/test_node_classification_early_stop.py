from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("dgl")

from openhgnn.trainerflow.node_classification import NodeClassification


def _flow(metric):
    flow = object.__new__(NodeClassification)
    flow.args = SimpleNamespace(early_stop_metric=metric)
    flow.model = Mock()
    return flow


def test_early_stop_uses_loss_by_default():
    flow = _flow("loss")
    stopper = Mock()
    flow._early_stop_step(
        stopper, 1.25, {"Micro_f1": 0.8, "Macro_f1": 0.7}
    )

    stopper.loss_step.assert_called_once_with(1.25, flow.model)


def test_early_stop_uses_named_metric_case_insensitively():
    flow = _flow("micro_F1")
    stopper = Mock()
    flow._early_stop_step(
        stopper, 1.25, {"Micro_f1": 0.8, "Macro_f1": 0.7}
    )

    stopper.step_score.assert_called_once_with(0.8, flow.model)


def test_early_stop_rejects_unknown_metric():
    with pytest.raises(ValueError, match="not found"):
        _flow("accuracy")._early_stop_step(Mock(), 1.25, {"Micro_f1": 0.8})
