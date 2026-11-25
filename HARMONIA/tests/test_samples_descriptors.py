"""Timbre coordinates of the sampled instruments and the analog codec that uses them (#46)."""

import json

import numpy as np
import pytest

from src import charter_v3
from src.samples import descriptors
from src.synth import v3_params as P
from src.v3 import codecs

IDS = charter_v3.load().raw["sample_instruments"]["ids"]


def sample_preset(choice: int, level: float = 0.8, **values) -> np.ndarray:
    return P.physical_from_dict({"osc_1_waveform": 2, "filter_cutoff": 1500, "sample_instrument": choice,
                                 "sample_level": level, **values})


# The table in git describes every instrument of the charter, inside the codec's range (run
# scripts/samples/instrument_descriptors.py when the charter gets new instruments).
def test_table_describes_every_charter_instrument():
    raw = json.loads(descriptors.PATH.read_text(encoding="utf-8"))
    assert set(raw["instruments"]) == set(IDS)
    table = descriptors.load()
    assert table.table.shape == (len(IDS) + 1, raw["dims"]) and codecs.ANALOG_DIM == P.N_PARAMS + raw["dims"]
    assert not np.isnan(table.table).any() and np.abs(table.table).max() <= 1.0


# Coordinates keep who sounds like whom: two families of sounds stay apart, the farthest one sits at SPAN.
def test_fit_keeps_neighbours():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((2, 64))
    emb = np.concatenate([c + 0.1 * rng.standard_normal((5, 64)) for c in centres])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    fitted = descriptors.fit(emb, [f"i{k}" for k in range(10)], dims=3)
    coords = np.array(list(fitted["instruments"].values()))
    assert np.abs(coords).max() == pytest.approx(descriptors.SPAN, abs=1e-4)
    gaps = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    assert all((gaps[k].argmin() < 5) == (k < 5) for k in range(10))
    assert descriptors.neighbours_kept(emb, coords) > 0.0


# An instrument the table does not describe is never chosen, and has no coordinates.
def test_undescribed_instruments_are_never_chosen():
    table = descriptors.Descriptors.from_raw({"dims": 2, "instruments": {IDS[0]: [0.5, 0.5], IDS[4]: [-0.5, 0.0]}})
    assert table.nearest([0.0, 0.0]) == 5 and table.nearest([0.4, 0.6]) == 1
    assert np.array_equal(table.coordinates(2), [0.0, 0.0]) and np.array_equal(table.coordinates(0), [0.0, 0.0])


# The codec writes the instrument as a switch and coordinates, and reads it back as the same instrument.
def test_codec_round_trips_an_instrument():
    table = descriptors.load()
    twins = {c for c in range(1, len(IDS) + 1) if table.nearest(table.coordinates(c)) != c}
    assert len(twins) <= 2  # the same marimba recordings are in VSCO 2 and VCSL
    for choice in (1, 20, len(IDS)):
        physical = sample_preset(choice, sample_tune=-5, sample_start=120)
        vector = codecs.analog_encode(physical)
        assert vector.shape == (codecs.ANALOG_DIM,) and vector.min() >= -1 and vector.max() <= 1
        assert vector[codecs.INSTRUMENT] == 1.0
        np.testing.assert_allclose(vector[P.N_PARAMS:], table.coordinates(choice))
        np.testing.assert_allclose(codecs.analog_decode(vector), physical, rtol=1e-6, atol=1e-6)


# A sample source that plays nothing is always written the same way, whatever its unused settings.
def test_silent_sample_source_is_canonical():
    reference = codecs.analog_encode(sample_preset(0, level=0.0))
    for physical in (sample_preset(0, level=0.7, sample_tune=3), sample_preset(12, level=0.0, sample_start=300)):
        vector = codecs.analog_encode(physical)
        np.testing.assert_array_equal(vector, reference)
        back = codecs.analog_decode(vector)
        assert back[codecs.INSTRUMENT] == 0 and back[codecs.LEVEL] == 0 and back[codecs.TUNE] == 0
    assert reference[codecs.INSTRUMENT] == -1.0 and not reference[P.N_PARAMS:].any()


# A generator's coordinates choose the nearest instrument; with the switch off, nothing plays.
def test_decode_takes_the_nearest_instrument():
    table = descriptors.load()
    vector = codecs.analog_encode(sample_preset(7))
    vector[P.N_PARAMS:] += 0.01
    assert codecs.analog_decode(vector)[codecs.INSTRUMENT] == table.nearest(table.coordinates(7) + 0.01)
    vector[codecs.INSTRUMENT] = -0.2
    assert codecs.analog_decode(vector)[codecs.INSTRUMENT] == 0


# Charter 3.0 vectors (the v3 / v4 generators) are unchanged; other sizes are refused.
def test_codec_sizes():
    physical = sample_preset(3)
    old = codecs.analog_encode(physical, dim=codecs.ANALOG_DIM_V3)
    np.testing.assert_allclose(old, codecs.analog_encode(physical)[:46])
    assert codecs.analog_decode(old)[codecs.LEVEL] == 0.0
    with pytest.raises(ValueError):
        codecs.analog_encode(physical, dim=P.N_PARAMS)
    with pytest.raises(ValueError):
        codecs.analog_decode(np.zeros(P.N_PARAMS))
    assert codecs.analog_continuous(codecs.ANALOG_DIM)[[codecs.LEVEL, codecs.START]].all()
    assert not codecs.analog_continuous(codecs.ANALOG_DIM)[[codecs.INSTRUMENT, codecs.TUNE, P.N_PARAMS]].any()
