"""Sampled instruments (src/samples): file names, octave check, catalogue of a small library made of test tones."""

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from src.samples import catalog, names, pitch  # noqa: E402

SR = 22050


def tone(note: float, seconds: float = 1.0) -> np.ndarray:
    """A saw-like test tone (not training data: it only checks the octave detection)."""
    t = np.arange(int(seconds * SR)) / SR
    f = pitch.frequency(note)
    return (sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 9)) * np.exp(-2 * t) * 0.3).astype(np.float32)


@pytest.mark.parametrize("name, note, level, rr, release", [
    ("VlnEns_susVib_A2_v1.wav", 45, 1, 1, False),
    ("GrandPno_Main_Sus_C7_v4_rr1.wav", 96, 4, 1, False),
    ("KSHarp_D4_mf.wav", 62, 4, 1, False),
    ("LLVln_Pizz_A3_f_RR2.wav", 57, 5, 2, False),
    ("Tuba3_sus_A#0_v1_rr1_Mid.wav", 22, 1, 1, False),
    ("GPiano_rel_A#-1_v2_rr1_Player.wav", 10, 2, 1, True),
    ("Releases/Upright1_Rel_C0_vl1_rr1.wav", 12, 1, 1, True),
    ("Trombone_Short_D#3_v2_4.wav", 51, 2, 4, False),
])
def test_file_names(name, note, level, rr, release):
    assert names.read(name) == names.Named(note=note, level=level, round_robin=rr, release=release)


def test_unpitched_names_have_no_note():
    assert names.read("snare1_click.wav").note is None


# The octave check finds a shift of a whole octave in either direction, and nothing in noise.
@pytest.mark.parametrize("note", [40, 57, 69, 81])
def test_octave_shift(note):
    audio = tone(note)
    assert pitch.octave_shift(audio, SR, note) == 0
    assert pitch.octave_shift(audio, SR, note - 12) == 12
    assert pitch.octave_shift(audio, SR, note + 12) == -12
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    assert pitch.octave_shift(noise, SR, note) is None


def test_agreement_is_required():
    assert pitch.agreed_shift([12, 12, 12, None, 12]) == 12
    assert pitch.agreed_shift([0, 12, 0, 12]) is None       # split vote: left out rather than played wrong
    assert pitch.agreed_shift([12, None, None]) is None      # too few clear votes


def test_key_and_velocity_ranges():
    assert catalog.key_ranges([48, 52, 60]) == {48: (0, 50), 52: (51, 56), 60: (57, 127)}
    assert catalog.velocity_ranges(1) == [(1, 127)]
    assert catalog.velocity_ranges(3) == [(1, 42), (43, 85), (86, 127)]


# A folder named one octave low ("A2" for what sounds A3) is shifted up and mapped across the keyboard.
def test_folder_instrument(tmp_path):
    root = tmp_path / "lib"
    folder = root / "Strings" / "Violin" / "Sustains"
    folder.mkdir(parents=True)
    for name in ("C3", "E3", "G3", "C4"):
        note = names.read(f"x_{name}_v1.wav").note
        for layer in (1, 2):
            sf.write(folder / f"Vln_sus_{name}_v{layer}_rr1.wav", tone(note + 12) * layer / 2, SR)
    instrument, why = catalog.folder_instrument("lib", root, folder, sorted(folder.iterdir()))
    assert why == ""
    assert instrument["id"] == "lib/strings/violin" and instrument["name"] == "Strings · Violin"
    assert instrument["family"] == "strings" and instrument["octave_shift"] == 12
    regions = instrument["regions"]
    assert {r["key"] for r in regions} == {60, 64, 67, 72}
    low = [r for r in regions if r["key"] == 60]
    assert [(r["lokey"], r["hikey"], r["lovel"], r["hivel"]) for r in low] == [(0, 62, 1, 63), (0, 62, 64, 127)]
    assert regions[0]["file"] == "Strings/Violin/Sustains/Vln_sus_C3_v1_rr1.wav" and regions[0]["sample_rate"] == SR
    summary = catalog.summary(instrument)
    assert summary["range"] == [60, 72] and summary["notes"] == 4 and summary["layers"] == 2


def test_unpitched_or_mixed_folders_are_left_out(tmp_path):
    root = tmp_path / "lib"
    folder = root / "Percussion"
    folder.mkdir(parents=True)
    for i in range(6):
        sf.write(folder / f"hit_{i}.wav", np.random.default_rng(i).standard_normal(SR // 4).astype(np.float32), SR)
    instrument, why = catalog.folder_instrument("lib", root, folder, sorted(folder.iterdir()))
    assert instrument is None and "named notes" in why
