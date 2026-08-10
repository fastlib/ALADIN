"""
Integration test for ALADIN(modelpaths="auto"), which picks between the pretrained 1-lead and
3-lead models per record based on which leads are available (see
aladin.configuration.select_model_for_leads and UNetSegmenter._select_and_load). Uses only fold 0
of each model instead of the full 5-fold ensemble, same as test_single_fold_analysis.py.

This requires network access to download the weights from the public fastlib/ALADIN Hugging Face
repo (no account/token needed). It is skipped automatically when there's no network access.
"""
import os
import socket

import pytest
import wfdb

from aladin import ALADIN
from aladin.core import Record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(REPO_ROOT, "data", "demo")


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
def stanford1_signals():
    rec = wfdb.rdrecord(os.path.join(DEMO_DIR, "STANFORD1"))
    return {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}, rec.fs


def test_auto_mode_selects_1_lead_model_for_single_lead_record(stanford1_signals):
    signals, fs = stanford1_signals
    record = Record({"II": signals["II"]}, fs, "DEMO", "STANFORD1_II")

    aladin = ALADIN(modelpaths="auto", use_folds=(0,), usefullcontext=False, debug={})

    aladin.segment(record)

    assert aladin.segmenter._active_modelname == "1_lead_model"
    assert record.delineations.p.binary.shape[0] > 0


def test_auto_mode_selects_3_lead_model_when_ii_v1_v6_present(stanford1_signals):
    signals, fs = stanford1_signals
    record = Record(
        {"II": signals["II"], "V1": signals["V1"], "V6": signals["V6"]},
        fs, "DEMO", "STANFORD1_3LEAD",
    )

    aladin = ALADIN(modelpaths="auto", use_folds=(0,), usefullcontext=False, debug={})

    with pytest.warns(UserWarning, match="lead II only"):
        aladin.segment(record)

    assert aladin.segmenter._active_modelname == "3_lead_model"
    assert record.delineations.p.binary.shape[0] > 0


def test_auto_mode_falls_back_to_1_lead_model_without_v1_and_v6(stanford1_signals):
    signals, fs = stanford1_signals
    record = Record({"II": signals["II"], "V1": signals["V1"]}, fs, "DEMO", "STANFORD1_II_V1")

    aladin = ALADIN(modelpaths="auto", use_folds=(0,), usefullcontext=False, debug={})

    with pytest.warns(UserWarning, match="1-lead model is still being used"):
        aladin.segment(record)

    assert aladin.segmenter._active_modelname == "1_lead_model"


def test_auto_mode_errors_without_lead_ii(stanford1_signals):
    signals, fs = stanford1_signals
    record = Record({"V1": signals["V1"], "V6": signals["V6"]}, fs, "DEMO", "STANFORD1_NO_II")

    aladin = ALADIN(modelpaths="auto", use_folds=(0,), usefullcontext=False, debug={})

    with pytest.raises(ValueError, match="[Ll]ead II"):
        aladin.segment(record)


def test_auto_mode_switches_and_unloads_previous_model(stanford1_signals):
    signals, fs = stanford1_signals
    one_lead = Record({"II": signals["II"]}, fs, "DEMO", "STANFORD1_II")
    three_lead = Record(
        {"II": signals["II"], "V1": signals["V1"], "V6": signals["V6"]},
        fs, "DEMO", "STANFORD1_3LEAD",
    )

    aladin = ALADIN(modelpaths="auto", use_folds=(0,), usefullcontext=False, debug={})

    aladin.segment(one_lead)
    assert aladin.segmenter._active_modelname == "1_lead_model"
    assert aladin.segmenter.n_leads == 1

    with pytest.warns(UserWarning):
        aladin.segment(three_lead)
    assert aladin.segmenter._active_modelname == "3_lead_model"
    assert aladin.segmenter.n_leads == 3
