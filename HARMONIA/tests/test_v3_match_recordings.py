"""Choice of the recordings to imitate (scripts/v3/match_recordings.py): licences, voice classes, order."""

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from scripts.v3 import match_recordings as M  # noqa: E402

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY = "http://creativecommons.org/licenses/by/3.0/"
BY_NC = "http://creativecommons.org/licenses/by-nc/3.0/"
SAMPLING = "http://creativecommons.org/licenses/sampling+/1.0/"


def fake_fsd50k(root, clips):
    """clips: (fname, labels, licence, seconds)."""
    audio = root / "FSD50K.dev_audio"
    audio.mkdir(parents=True)
    (root / "FSD50K.metadata").mkdir()
    info = {}
    for fname, _, licence, seconds in clips:
        sf.write(audio / f"{fname}.wav", np.zeros(int(seconds * 8000), dtype=np.float32), 8000)
        info[fname] = {"title": f"clip {fname}", "license": licence, "uploader": f"author {fname}"}
    (root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").write_text(json.dumps(info), encoding="utf-8")
    emb = np.random.default_rng(0).standard_normal((len(clips), 512)).astype(np.float16)
    np.savez(root / "emb.npz", fname=np.array([c[0] for c in clips]), labels=np.array([c[1] for c in clips]),
             title=np.array([f"clip {c[0]}" for c in clips]), emb=emb)
    return root / "emb.npz"


@pytest.fixture()
def fsd(tmp_path):
    clips = [
        ("1", "Slam,Door", CC0, 1.0),
        ("2", "Slam,Door", BY, 1.0),
        ("3", "Slam,Door", BY_NC, 1.0),          # non-commercial licence: left out
        ("4", "Shatter,Glass", SAMPLING, 1.0),   # sampling+ licence: left out
        ("5", "Shatter,Glass", CC0, 1.0),
        ("6", "Speech,Human_voice", CC0, 1.0),   # voice: a preset cannot stand for it
        ("7", "Laughter,Human_voice", BY, 1.0),
        ("8", "Thunder", CC0, 0.1),              # too short
        ("9", "Thunder", CC0, 30.0),             # too long
        ("10", "Thunder", BY, 2.0),
        ("11", "Slam,Door", CC0, 1.0),
    ]
    return tmp_path, fake_fsd50k(tmp_path, clips)


# Only CC0 and CC BY recordings of non-voice classes, of a usable length, are imitated.
def test_only_free_licences_and_non_voice_classes(fsd):
    root, emb = fsd
    picked = M.pick_clips(root, emb, per_class=8, seed=0)
    assert {c["fname"] for c in picked} == {"1", "2", "5", "10", "11"}
    assert all(any(k in c["license"] for k in M.LICENSES) for c in picked)
    assert all(c["uploader"].startswith("author") for c in picked)  # the author travels with the preset


# per_class bounds every class, and classes take turns so a stopped run stays balanced.
def test_per_class_and_round_robin(fsd):
    root, emb = fsd
    picked = M.pick_clips(root, emb, per_class=2, seed=0)
    leaves = [c["labels"].split(",")[0] for c in picked]
    assert leaves.count("Slam") == 2
    first_round = leaves[:3]
    assert sorted(first_round) == ["Shatter", "Slam", "Thunder"]


# The choice is cached: a resumed run gets the same recordings without reading the audio files again.
def test_choice_is_cached(fsd, tmp_path):
    root, emb = fsd
    cache = tmp_path / "out" / "picked.json"
    first = M.pick_clips(root, emb, per_class=2, seed=0, cache=cache)
    assert cache.exists()
    for wav in (root / "FSD50K.dev_audio").glob("*.wav"):
        wav.unlink()
    again = M.pick_clips(root, emb, per_class=2, seed=0, cache=cache)
    assert [c["fname"] for c in again] == [c["fname"] for c in first]
    np.testing.assert_array_equal(again[0]["emb"], first[0]["emb"])


# --classes keeps only the asked classes (the instruments for the sample source), and is part of the cache key.
def test_classes(fsd, tmp_path):
    root, emb = fsd
    cache = tmp_path / "out" / "picked.json"
    picked = M.pick_clips(root, emb, per_class=8, seed=0, cache=cache, classes=("Thunder",))
    assert [c["fname"] for c in picked] == ["10"]
    assert {c["fname"] for c in M.pick_clips(root, emb, per_class=8, seed=0, cache=cache)} == {"1", "2", "5", "10", "11"}


def normalized_presets(clips: int, pop: int) -> np.ndarray:
    from src.synth import v3_params as P

    return np.repeat(P.normalize(P.DEFAULTS)[None, None, :], clips, axis=0).repeat(pop, axis=1)


# With --samples, half of each recording's first presets play its candidate instruments; the other half are
# the generator's draws as they are.
def test_add_instruments():
    from src.synth import v3_params as P

    candidates = np.array([[3, 7], [10, 11]])
    out = M.add_instruments(normalized_presets(2, 6), candidates, np.random.default_rng(0))
    physical = np.array([[M.written(p) for p in clip] for clip in out])
    instrument, level = physical[..., P.INDEX["sample_instrument"]], physical[..., P.INDEX["sample_level"]]
    assert (instrument[:, :3] == 0).all() and (level[:, :3] == 0).all()
    assert set(instrument[0, 3:]) == {3, 7} and set(instrument[1, 3:]) == {10, 11}
    assert ((level[:, 3:] >= 0.5) & (level[:, 3:] <= 1.0)).all()


# The sample source's own moves only ever choose a candidate instrument, and keep octaves in range.
def test_mutate_instruments_stays_among_candidates():
    from src.synth import v3_params as P

    rng = np.random.default_rng(1)
    candidates = np.array([[3, 7, 9]])
    kids = M.add_instruments(normalized_presets(1, 400), candidates, rng)
    for _ in range(5):
        kids = M.mutate_instruments(kids, candidates, rng)
    physical = np.array([M.written(p) for p in kids[0]])
    instrument = physical[:, P.INDEX["sample_instrument"]]
    assert set(instrument) <= {0, 3, 7, 9} and len(set(instrument)) == 4
    tune = physical[:, P.INDEX["sample_tune"]]
    assert set(tune) <= {-24, -12, 0, 12, 24} and len(set(tune)) > 1


# A found preset is written with a silent sample source switched off (instrument 0, defaults).
def test_written_switches_off_what_cannot_be_heard():
    from src.synth import v3_params as P

    preset = normalized_presets(1, 1)[0, 0]
    M._set_instrument(preset, np.array(True), np.array(5), np.array(0.01))
    preset[P.INDEX["sample_tune"]] = 0.75
    physical = M.written(preset)
    assert physical[P.INDEX["sample_instrument"]] == 0 and physical[P.INDEX["sample_tune"]] == 0
    M._set_instrument(preset, np.array(True), np.array(5), np.array(0.6))
    physical = M.written(preset)
    assert physical[P.INDEX["sample_instrument"]] == 5 and physical[P.INDEX["sample_tune"]] == 12
    assert len(M.written(preset[:P.CORE_PARAMS])) == P.N_PARAMS


# The instruments tried for a recording are those that sound most like it.
def test_candidate_instruments(tmp_path):
    from src.samples.player import instrument_ids

    ids = instrument_ids()
    emb = np.eye(4, 512, dtype=np.float32)
    np.savez(tmp_path / "clap.npz", ids=np.array([ids[2], ids[5], ids[8], ids[9]]), emb=emb)
    targets = np.array([0.1, 0.9, 0.3, 0.0] + [0.0] * 508, dtype=np.float32)[None, :]
    assert M.candidate_instruments(targets, tmp_path / "clap.npz", count=2).tolist() == [[5, 8]]
