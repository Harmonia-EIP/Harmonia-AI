"""The sample source of charter 3.1 (src/samples/player.py, src/synth/engine_v3.py), on a pack made of test tones."""

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from src import charter_v3  # noqa: E402
from src.samples import pitch  # noqa: E402
from src.samples.player import SampleBank, instrument_ids  # noqa: E402
from src.synth import v3_params as P  # noqa: E402
from src.synth.engine_v3 import render, render_batch  # noqa: E402
from src.v3 import codecs  # noqa: E402

RATE = 32000
PIANO = 1  # charter choice of the first catalogue instrument


def tone(note: float, seconds: float = 3.0, level: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    f = pitch.frequency(note)
    return (level * sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 7)) * np.exp(-t)).astype(np.float32)


@pytest.fixture()
def bank(tmp_path):
    """A pack whose first instrument has notes 48 and 72, two velocity layers and two round robins at 72."""
    lib = tmp_path / "lib"
    lib.mkdir()
    regions = []
    for key, lo, hi in ((48, 0, 60), (72, 61, 127)):
        for lovel, hivel, level in ((1, 63, 0.2), (64, 127, 0.6)):
            for rr in ((1, 2) if key == 72 else (1,)):
                name = f"lib/n{key}_v{lovel}_rr{rr}.flac"
                sf.write(tmp_path / name, tone(key, level=level), RATE)
                regions.append({"file": name, "key": key, "lokey": lo, "hikey": hi, "lovel": lovel, "hivel": hivel,
                                "round_robin": rr, "trigger": "attack", "sample_rate": RATE, "channels": 1,
                                "frames": 3 * RATE})
    catalog = {"version": 1, "instruments": [{"id": instrument_ids()[PIANO], "name": "Test", "library": "lib",
                                               "family": "keys", "octave_shift": 0, "gain": 1.5, "regions": regions}]}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return SampleBank(tmp_path)


def sample_preset(level: float = 1.0, choice: int = PIANO, **values) -> np.ndarray:
    p = P.physical_from_dict({"filter_cutoff": 20000, "amp_sustain": 1.0, "amp_release": 300, **values})
    p[P.INDEX["sample_instrument"]] = choice
    p[P.INDEX["sample_level"]] = level
    return p


def test_region_choice(bank):
    assert bank.has(PIANO) and not bank.has(2) and not bank.has(0)
    assert bank.region(PIANO, 50, 100)["file"] == "lib/n48_v64_rr1.flac"
    assert bank.region(PIANO, 50, 30)["file"] == "lib/n48_v1_rr1.flac"
    takes = {bank.region(PIANO, 70, 100, take)["file"] for take in range(4)}
    assert takes == {"lib/n72_v64_rr1.flac", "lib/n72_v64_rr2.flac"}  # round robins alternate
    assert bank.region(2, 60, 100) is None  # an instrument this pack does not hold
    audio, rate = bank.audio(bank.region(PIANO, 48, 100))
    assert rate == RATE and audio.shape == (1, 3 * RATE)


# The recording is transposed to the played note, and sample_tune moves it by semitones.
@pytest.mark.parametrize("note", [43, 55, 67, 79])
def test_sample_plays_the_note(bank, note):
    audio = render(sample_preset(), note=note, bank=bank)
    assert np.abs(audio).max() > 0.05
    assert pitch.octave_shift(audio.T, 48000, note) == 0
    tuned = render(sample_preset(sample_tune=12), note=note, bank=bank)
    assert pitch.octave_shift(tuned.T, 48000, note + 12) == 0


# Level 0, no instrument, or a pack that does not hold it: exactly the charter 3.0 sound.
def test_no_sample_changes_nothing(bank):
    core = P.physical_from_dict({"osc_1_waveform": 2, "filter_cutoff": 2000, "unison_voices": 3})[:P.CORE_PARAMS]
    reference = render(core, note=57)
    assert np.array_equal(render(P.complete(core), note=57, bank=bank), reference)
    for p in (sample_preset(level=0.0, **{"osc_1_waveform": 2, "filter_cutoff": 2000, "unison_voices": 3}),
              sample_preset(choice=0, **{"osc_1_waveform": 2, "filter_cutoff": 2000, "unison_voices": 3}),
              sample_preset(choice=2, **{"osc_1_waveform": 2, "filter_cutoff": 2000, "unison_voices": 3})):
        p[P.INDEX["amp_sustain"]], p[P.INDEX["amp_release"]] = P.DEFAULTS[P.INDEX["amp_sustain"]], P.DEFAULTS[P.INDEX["amp_release"]]
        assert np.array_equal(render(p, note=57, bank=bank), reference)


# Start skips into the recording; the batch renderer plays what the single one does.
def test_start_and_batch(bank):
    full = render(sample_preset(), note=60, bank=bank)
    late = render(sample_preset(sample_start=800), note=60, bank=bank)
    assert np.abs(late[:, :2400]).max() < np.abs(full[:, :2400]).max()  # the recording decays: later is softer
    batch = render_batch(np.stack([sample_preset(), sample_preset(level=0.5)]), np.array([45, 69]), 100 / 127, 1.5,
                         4.0, 48000, 7, bank=bank)
    single = render(sample_preset(), note=45, seed=7, bank=bank).mean(axis=0)
    np.testing.assert_allclose(batch[0, 0], single, atol=1e-6)
    assert batch.shape == (2, 2, 4 * 48000) and np.abs(batch[1]).max() > 0


# Charter 3.0 vectors (what the v3 / v4 generators make) encode and decode without the sample source.
def test_codecs_old_and_new_sizes():
    physical = sample_preset(sample_tune=-5)
    full = codecs.analog_encode(physical)
    assert full.shape == (P.N_PARAMS,)
    old = codecs.analog_encode(physical, dim=codecs.ANALOG_DIM_V3)
    assert old.shape == (46,)
    back = codecs.analog_decode(old)
    assert back.shape == (P.N_PARAMS,) and back[P.INDEX["sample_level"]] == 0.0
    np.testing.assert_allclose(codecs.analog_decode(full), physical, atol=1e-6)


def test_charter_lists_every_instrument_once():
    charter = charter_v3.load()
    ids = charter.raw["sample_instruments"]["ids"]
    spec = charter.by_id["sample_instrument"]
    assert len(ids) == len(set(ids)) == spec.max and spec.choices[0] == "Off"
    assert charter.by_id["sample_level"].default == 0.0  # existing presets keep their sound


def test_charter_instrument_list_is_up_to_date():
    from pathlib import Path

    from scripts.samples import charter_instruments as CI

    charter = json.loads(Path(CI.CHARTER).read_text(encoding="utf-8"))
    catalogue = json.loads(Path(CI.INSTRUMENTS).read_text(encoding="utf-8"))
    assert CI.updated(json.loads(json.dumps(charter)), catalogue) == charter, "run scripts/samples/charter_instruments.py"
