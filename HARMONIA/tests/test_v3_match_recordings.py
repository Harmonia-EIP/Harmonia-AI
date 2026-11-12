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
