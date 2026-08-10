"""
Unit tests for aladin.configuration.select_model_for_leads, which decides between the pretrained
1-lead and 3-lead models based on which ECG leads are available (see
aladin/src/aladin/configuration.py). Pure logic, no model loading / network access involved.
"""
import warnings

import pytest

from aladin.configuration import select_model_for_leads


def test_lead_ii_only_selects_1_lead_model():
    assert select_model_for_leads(["II"]) == "1_lead_model"


def test_ii_v1_v6_selects_3_lead_model():
    assert select_model_for_leads(["II", "V1", "V6"]) == "3_lead_model"


def test_full_12_lead_selects_3_lead_model():
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    with pytest.warns(UserWarning, match="lead II only"):
        assert select_model_for_leads(leads) == "3_lead_model"


def test_missing_lead_ii_raises():
    with pytest.raises(ValueError, match="[Ll]ead II"):
        select_model_for_leads(["I", "V1", "V6"])


def test_no_leads_raises():
    with pytest.raises(ValueError, match="[Ll]ead II"):
        select_model_for_leads([])


def test_multiple_leads_without_v1_v6_falls_back_to_1_lead_model():
    with pytest.warns(UserWarning) as record:
        result = select_model_for_leads(["I", "II", "III"])

    assert result == "1_lead_model"
    messages = [str(w.message) for w in record]
    assert any("lead II only" in m for m in messages)
    assert any("1-lead model is still being used" in m for m in messages)


def test_more_than_one_lead_always_warns_about_rhythm_lead():
    with pytest.warns(UserWarning, match="lead II only"):
        select_model_for_leads(["II", "V1"])
