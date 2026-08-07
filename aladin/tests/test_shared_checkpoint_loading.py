"""
Regression test for the `shared_from` option on
nnUNetPredictor.initialize_from_trained_model_folder (see
nnUNet/nnunetv2/inference/predict_from_raw_data.py).

UNetSegmenter builds two predictors (sliding-window and full-context) from the exact same
trained checkpoint files for each modelpath (see aladin/src/aladin/backend/segmenter.py). Before
this change, both predictors independently read the same (potentially multi-GB) fold checkpoints
from disk, roughly doubling RAM usage. `shared_from` lets the second predictor reuse the first
predictor's already-loaded weights instead of reading them from disk again.

This test mocks out the disk/plans/network-building dependencies so it can run without real
trained weights, and only checks the mechanism itself: that passing `shared_from` avoids
re-reading checkpoint files and results in the exact same (not copied) parameter objects.
"""
import torch

from nnunetv2.inference import predict_from_raw_data as pfrd


class _FakePlansManager:
    def __init__(self, plans):
        self.plans = plans

    def get_configuration(self, configuration_name):
        return _FakeConfigurationManager(configuration_name)

    def get_label_manager(self, dataset_json):
        return _FakeLabelManager()


class _FakeLabelManager:
    num_segmentation_heads = 1


class _FakeConfigurationManager:
    def __init__(self, configuration_name):
        self.configuration_name = configuration_name
        self.network_arch_class_name = "fake_arch"
        self.network_arch_init_kwargs = {}
        self.network_arch_init_kwargs_req_import = []


class _FakeTrainerClass:
    @staticmethod
    def build_network_architecture(*args, **kwargs):
        return torch.nn.Identity()


def _make_fake_checkpoint(fold):
    return {
        "trainer_name": "FakeTrainer",
        "init_args": {"configuration": "fake_config"},
        "inference_allowed_mirroring_axes": None,
        "network_weights": {"weight": torch.full((4,), float(fold))},
    }


def test_shared_from_avoids_reloading_checkpoints_from_disk(monkeypatch):
    load_calls = []

    def fake_torch_load(path, map_location=None, weights_only=None):
        load_calls.append(path)
        fold = int(str(path).split("fold_")[1][0])
        return _make_fake_checkpoint(fold)

    monkeypatch.setattr(pfrd, "load_json", lambda path: {})
    monkeypatch.setattr(pfrd, "PlansManager", _FakePlansManager)
    monkeypatch.setattr(pfrd, "determine_num_input_channels", lambda *a, **k: 1)
    monkeypatch.setattr(pfrd, "recursive_find_python_class", lambda *a, **k: _FakeTrainerClass)
    monkeypatch.setattr(pfrd.torch, "load", fake_torch_load)

    first = pfrd.nnUNetPredictor(device=torch.device("cpu"))
    first.initialize_from_trained_model_folder(
        "fake_model_dir", use_folds=(0, 1), checkpoint_name="checkpoint_best.pth",
    )
    assert len(load_calls) == 2  # one torch.load per fold

    second = pfrd.nnUNetPredictor(device=torch.device("cpu"))
    second.initialize_from_trained_model_folder(
        "fake_model_dir", use_folds=(0, 1), checkpoint_name="checkpoint_best.pth",
        shared_from=first,
    )

    # No additional checkpoint files were read from disk for the second predictor.
    assert len(load_calls) == 2

    # The second predictor reuses the exact same parameter/metadata objects, not copies.
    assert second.list_of_parameters is first.list_of_parameters
    assert second.trainer_name == first.trainer_name
    assert second.configuration_name == first.configuration_name
    assert second.allowed_mirroring_axes == first.allowed_mirroring_axes


def test_shared_from_lstm_predictor_gets_independent_patch_size(monkeypatch):
    """The LSTM predictor mutates configuration_manager.patch_size after init; this must not
    leak back into the source predictor's configuration_manager (they must not share the same
    ConfigurationManager instance, only the same underlying weight tensors)."""
    def fake_torch_load(path, map_location=None, weights_only=None):
        fold = int(str(path).split("fold_")[1][0])
        return _make_fake_checkpoint(fold)

    monkeypatch.setattr(pfrd, "load_json", lambda path: {})
    monkeypatch.setattr(pfrd, "PlansManager", _FakePlansManager)
    monkeypatch.setattr(pfrd, "determine_num_input_channels", lambda *a, **k: 1)
    monkeypatch.setattr(pfrd, "recursive_find_python_class", lambda *a, **k: _FakeTrainerClass)
    monkeypatch.setattr(pfrd.torch, "load", fake_torch_load)

    sliding = pfrd.nnUNetWithClassificationPredictor(device=torch.device("cpu"))
    sliding.initialize_from_trained_model_folder(
        "fake_model_dir", use_folds=(0,), checkpoint_name="checkpoint_best.pth",
    )

    fullcontext = pfrd.nnUNetLSTMWithClassificationPredictor(device=torch.device("cpu"))
    fullcontext.initialize_from_trained_model_folder(
        "fake_model_dir", use_folds=(0,), checkpoint_name="checkpoint_best.pth",
        shared_from=sliding,
    )

    assert fullcontext.configuration_manager.patch_size == [6144]
    assert sliding.configuration_manager is not fullcontext.configuration_manager
    assert fullcontext.list_of_parameters is sliding.list_of_parameters
