"""
Tests that the ALADIN backbone can be built and run end-to-end with randomly
initialized weights, without downloading anything from Hugging Face.

This relies on UNetSegmenter(random_weights=True), which builds the nnU-Net
network architecture from a local plans file (see aladin/src/aladin/backend/segmenter.py,
UNetSegmenter._initialize_random_weights) instead of loading a trained checkpoint.
"""
import os

import pytest
import wfdb

from aladin import ALADIN
from aladin.core import Record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLANS_FOLDER = os.path.join(REPO_ROOT, "data", "nnUNet_plans")
DEMO_DIR = os.path.join(REPO_ROOT, "data", "demo")
MODELPATH = "ClassificationTrainer__nnUNetWithClassificationPlans__1d_tiny_encoding_decoding"


@pytest.fixture
def demo_record():
    rec = wfdb.rdrecord(os.path.join(DEMO_DIR, "A01986"))
    ecg = {"II": rec.p_signal[:, 0]}
    return Record(ecg, rec.fs, "DEMO", "A01986")


def make_random_weights_aladin(**overrides):
    kwargs = dict(
        modelpaths=[MODELPATH],
        usefullcontext=False,
        random_weights=True,
        plans_folder=PLANS_FOLDER,
        num_input_channels=1,
    )
    kwargs.update(overrides)
    return ALADIN(**kwargs)


def test_plans_folder_available():
    # Sanity check for the fixture data the rest of this module depends on.
    assert os.path.isdir(PLANS_FOLDER)
    assert os.path.isfile(os.path.join(DEMO_DIR, "A01986.hea"))


def test_random_weights_does_not_download_from_huggingface(monkeypatch, demo_record):
    import aladin.configuration as configuration

    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_model_folder() (the Hugging Face download path) should not be called")

    monkeypatch.setattr(configuration, "get_model_folder", fail_if_called)

    aladin = make_random_weights_aladin()
    aladin.segment(demo_record)


def test_aladin_runs_with_random_weights(demo_record):
    aladin = make_random_weights_aladin()

    aladin.segment(demo_record)

    assert demo_record.delineations.p.binary.shape[0] > 0
    assert demo_record.delineations.qrs.binary.shape == demo_record.delineations.p.binary.shape
