"""Sample packs (scripts/samples/build_packs.py): the listening copy and the archives."""

import tarfile

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("soxr")

from scripts.samples import build_packs as B  # noqa: E402


def region(key, lovel, hivel, rr=1, trigger="attack"):
    return {"file": f"x_{key}_{lovel}_{rr}.wav", "key": key, "lokey": 0, "hikey": 127, "lovel": lovel, "hivel": hivel,
            "round_robin": rr, "trigger": trigger, "sample_rate": 44100, "channels": 2, "frames": 44100}


# The server's copy keeps one take per recorded note: the middle layer, first round robin, any velocity.
def test_listen_regions():
    instrument = {"regions": [region(60, 1, 42), region(60, 43, 85), region(60, 43, 85, rr=2), region(60, 86, 127),
                              region(64, 1, 63), region(64, 64, 127), region(60, 1, 127, trigger="release")]}
    chosen = B.listen_regions(instrument)
    assert [(r["key"], r["file"]) for r in chosen] == [(60, "x_60_43_1.wav"), (64, "x_64_64_1.wav")]
    assert all((r["lovel"], r["hivel"], r["round_robin"]) == (1, 127, 1) for r in chosen)


# Full copies keep rate, channels and frames; listening copies are mono, shorter, at LISTEN_RATE.
def test_conversions(tmp_path):
    source = tmp_path / "a.wav"
    stereo = np.random.default_rng(0).uniform(-0.5, 0.5, (44100 * 6, 2)).astype(np.float32)
    sf.write(source, stereo, 44100, subtype="PCM_24")
    full, listen = tmp_path / "full" / "a.flac", tmp_path / "listen" / "a.flac"
    B._convert((str(source), str(full), False))
    B._convert((str(source), str(listen), True))
    a, rate = sf.read(full, always_2d=True)
    assert (rate, a.shape, sf.info(str(full)).subtype) == (44100, (44100 * 6, 2), "PCM_24")
    np.testing.assert_allclose(a, sf.read(source, always_2d=True)[0], atol=1e-6)  # lossless
    b, rate = sf.read(listen, always_2d=True)
    assert rate == B.LISTEN_RATE and b.shape == (int(B.LISTEN_SECONDS * B.LISTEN_RATE), 1)
    assert abs(b[-1, 0]) < 1e-3  # faded out


def test_archives_are_split(tmp_path):
    folder = tmp_path / "packs"
    (folder / "lib").mkdir(parents=True)
    for i in range(5):
        (folder / "lib" / f"{i}.flac").write_bytes(bytes(1000))
    meta = tmp_path / "catalog.json"
    meta.write_text("{}", encoding="utf-8")
    old = B.MAX_ARCHIVE
    B.MAX_ARCHIVE = 2500
    try:
        written = B.archives("full-lib", "samples-0", folder, [f"lib/{i}.flac" for i in range(5)], tmp_path,
                             {"catalog.json": meta})
    finally:
        B.MAX_ARCHIVE = old
    assert [w["files"] for w in written] == [2, 2, 1]
    assert written[0]["asset"] == "harmonia-samples-full-lib-samples-0-part1.tar"
    with tarfile.open(tmp_path / written[2]["asset"]) as tar:
        assert tar.getnames() == ["catalog.json", "lib/4.flac"]
