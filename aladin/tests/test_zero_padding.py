"""
UNetSegmenter.prepare_ecg feeds nnU-Net a signal whose length is a multiple of 256 samples (the
patch-size granularity the sliding-window/full-context predictors expect). Real recordings can be
shorter than that after resampling to the model's target sampling rate (204.8 Hz), so short ECGs
(e.g. under 10s) get zero-padded at the end rather than rejected. record.before_padding tracks the
pre-pad length so the padding can be cropped back off after inference (see
UNetSegmenter.undo_resampling).

These tests exercise only preprocess()/prepare_ecg() directly -- no model needs to be loaded, so
they run with no trained weights and no network access.
"""
import numpy as np
import pytest

from aladin.core import Record
from aladin.backend.segmenter import UNetSegmenter


def _make_short_record(duration_s, fs=250, lead="II"):
    ecg = {lead: np.random.rand(round(fs * duration_s))}
    return Record(ecg, fs, "DEMO", f"SHORT_{duration_s}s")


@pytest.fixture
def segmenter():
    seg = UNetSegmenter(modelpaths="auto")
    seg.n_leads = 1
    return seg


def test_short_ecg_is_zero_padded_to_multiple_of_256(segmenter):
    record = _make_short_record(duration_s=3)  # well under 10s

    segmenter.preprocess(record)
    sig, _ = segmenter.prepare_ecg(record)

    assert sig.shape[-1] % 256 == 0
    # Padding only makes the signal longer, and only ever whole zeros appended at the end.
    assert sig.shape[-1] >= record.before_padding
    assert record.before_padding < sig.shape[-1]
    assert np.all(sig[..., record.before_padding:] == 0)
    assert not np.all(sig[..., :record.before_padding] == 0)


def test_signal_already_a_multiple_of_256_after_resampling_is_not_padded(segmenter):
    # 625 samples at fs=250 resample to exactly 512 samples at the model's 204.8 Hz target rate
    # (512 = 2 * 256), so no zero-padding should be appended even though ~2.5s is short.
    record = _make_short_record(duration_s=625 / 250, fs=250)

    segmenter.preprocess(record)
    sig, _ = segmenter.prepare_ecg(record)

    assert record.before_padding == 512
    assert sig.shape[-1] == 512  # no padding was appended


def test_original_length_is_preserved_for_undo_resampling(segmenter):
    record = _make_short_record(duration_s=3, fs=250)

    segmenter.preprocess(record)
    sig, _ = segmenter.prepare_ecg(record)

    assert record.original_length == 750
    # before_padding is the resampled (204.8 Hz), pre-zero-pad length -- shorter than the padded
    # signal handed to the model, but reconstructable back to original_length via undo_resampling.
    assert record.before_padding < record.original_length
