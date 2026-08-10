"""
Integration test that runs the full ALADIN pipeline (segmentation + reflection + diagnosis)
end-to-end against real trained weights, using only fold 0 instead of the full 5-fold ensemble
(see UNetSegmenter's use_folds, and aladin.configuration.get_model_folder's allow_patterns
support, which together mean only fold 0's checkpoint is downloaded from Hugging Face).

This requires network access to download the weights from the public fastlib/ALADIN Hugging Face
repo (no account/token needed -- see aladin.configuration.get_model_folder). It is skipped
automatically when there's no network access, since it cannot run in the regular offline test job.
"""
import os
import socket

import pytest
import wfdb

from aladin import ALADIN
from aladin.core import Record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(REPO_ROOT, "data", "demo")
MODELPATH = "1_lead_model"


def _has_network_access():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_network_access(),
    reason="Requires network access to download weights from the public fastlib/ALADIN Hugging Face repo.",
)


@pytest.fixture
def stanford1_record():
    rec = wfdb.rdrecord(os.path.join(DEMO_DIR, "STANFORD1"))
    ecg = {"II": rec.p_signal[:, 0]}
    return Record(ecg, rec.fs, "DEMO", "STANFORD1")


@pytest.fixture
def wrong_lead_record():
    rec = wfdb.rdrecord(os.path.join(DEMO_DIR, "STANFORD1"))
    ecg = {"I": rec.p_signal[:, 0]}
    return Record(ecg, rec.fs, "DEMO", "STANFORD1")


def test_aladin_analysis_with_single_fold(stanford1_record):
    aladin = ALADIN(
        modelpaths=[MODELPATH],
        use_folds=(0,),
        debug={},
    )

    aladin.analyse(stanford1_record)

    assert stanford1_record.delineations.p.binary.shape[0] > 0
    assert stanford1_record.delineations.qrs.binary.shape == stanford1_record.delineations.p.binary.shape
    assert isinstance(stanford1_record.diagnosis, list)

def test_aladin_analysis_with_empty_record():

    rec = wfdb.rdrecord(os.path.join(DEMO_DIR, "STANFORD1"))
    ecg = {}

    with pytest.raises(ValueError, match="empty"):
        rec = Record(ecg, rec.fs, "DEMO", "STANFORD1")

def test_aladin_analysis_with_wrong_lead_record(wrong_lead_record):
    aladin = ALADIN(
        modelpaths=[MODELPATH],
        use_folds=(0,),
        debug={},
    )

    with pytest.raises(ValueError, match="valid"):
        aladin.analyse(wrong_lead_record)