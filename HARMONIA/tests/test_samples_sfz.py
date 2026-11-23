"""SFZ reading (src/samples/sfz.py): headers, inheritance, keys, defines and includes."""

import pytest

from src.samples.sfz import note_number, parse


@pytest.mark.parametrize("text, number", [("60", 60), ("c4", 60), ("C4", 60), ("a0", 21), ("f#3", 54), ("eb2", 39),
                                          ("c-1", 0), ("g9", 127)])
def test_note_names(text, number):
    assert note_number(text) == number


def test_regions_inherit_their_headers(tmp_path):
    sfz = tmp_path / "violin.sfz"
    sfz.write_text("""
// a comment
<control> default_path=samples/
<global> volume=-3 loop_mode=one_shot
<group> lovel=1 hivel=64 seq_length=2
<region> sample=Violin Sus C4 p.wav lokey=b3 hikey=c#4 pitch_keycenter=c4 seq_position=1
<region> sample=Violin Sus C4 p rr2.wav lokey=b3 hikey=c#4 pitch_keycenter=c4 seq_position=2 volume=-1
<group> lovel=65 hivel=127
<region> sample=Violin Sus C4 f.wav key=60 /* inline */ tune=-12
<region> sample=Violin rel.wav trigger=release key=60
""", encoding="utf-8")
    regions = parse(sfz)
    assert [r.sample for r in regions] == ["samples/Violin Sus C4 p.wav", "samples/Violin Sus C4 p rr2.wav",
                                           "samples/Violin Sus C4 f.wav", "samples/Violin rel.wav"]
    first, second, loud, release = regions
    assert (first.lokey, first.hikey, first.pitch_keycenter, first.lovel, first.hivel) == (59, 61, 60, 1, 64)
    assert first.volume == -3 and second.volume == -1          # the region overrides the global
    assert (first.seq_length, second.seq_position) == (2, 2)
    assert (loud.lovel, loud.hivel, loud.tune, loud.seq_length) == (65, 127, -12, 1)  # a new group resets the old one
    assert loud.loop_mode == "one_shot" and release.trigger == "release"


def test_defines_and_includes(tmp_path):
    (tmp_path / "map.sfz").write_text("<region> sample=$DIR/A3.flac key=57\n", encoding="utf-8")
    sfz = tmp_path / "piano.sfz"
    sfz.write_text('#define $DIR samples\n<group> amp_veltrack=90\n#include "map.sfz"\n', encoding="utf-8")
    (region,) = parse(sfz)
    assert region.sample == "samples/A3.flac" and region.pitch_keycenter == 57
    assert region.extra == {"amp_veltrack": "90"}


def test_regions_without_a_sample_are_skipped(tmp_path):
    sfz = tmp_path / "x.sfz"
    sfz.write_text("<region> lokey=0 hikey=127\n<region> sample=a.wav pitch_keycenter=sample\n", encoding="utf-8")
    (region,) = parse(sfz)
    assert region.pitch_keycenter == 60 and region.extra["pitch_keycenter"] == "sample"
