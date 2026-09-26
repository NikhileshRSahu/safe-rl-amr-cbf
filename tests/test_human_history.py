import numpy as np
import pytest

from benchmark.human_history import HumanTrackHistory


def test_history_rejects_future_or_non_monotonic_samples():
    h = HumanTrackHistory(history_len=4)
    h.update("human-0", 1.0, [0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        h.update("human-0", 1.0, [0.1, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        h.sequence("human-0", now=0.9)


def test_history_sequence_is_fixed_length_padded_and_causal():
    h = HumanTrackHistory(history_len=4)
    h.update("human-0", 1.0, [1.0, 2.0], [0.5, -0.25])
    h.update("human-0", 1.2, [1.1, 1.95], [0.5, -0.25])
    seq = h.sequence("human-0", now=1.5)
    assert seq.features.shape == (4, 5)
    assert seq.mask.shape == (4,)
    assert seq.mask.tolist() == [False, False, True, True]
    assert np.allclose(seq.features[-1, :4], [1.1, 1.95, 0.5, -0.25])
    assert seq.features[-1, 4] == pytest.approx(1.2)
    assert seq.observation_age == pytest.approx(0.3)
    assert np.isfinite(seq.features).all()


def test_empty_history_is_valid_and_duplicate_refresh_is_suppressed():
    h = HumanTrackHistory(history_len=3)
    empty = h.sequence("missing", now=2.0)
    assert empty.features.shape == (3, 5)
    assert not empty.mask.any()
    assert empty.observation_age == pytest.approx(0.0)
    assert h.update_if_new("human-1", 2.0, [0.0, 0.0], [0.0, 0.0])
    assert not h.update_if_new("human-1", 2.0, [0.0, 0.0], [0.0, 0.0])
    assert h.sequence("human-1", now=2.1).mask.sum() == 1
