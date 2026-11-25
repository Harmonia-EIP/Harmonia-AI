"""v4 text corpus (scripts/v4/build_corpus_v4.py): which recordings and which descriptions are kept."""

import json

from scripts.v4 import build_corpus_v4 as C

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY = "http://creativecommons.org/licenses/by/3.0/"
BY_NC = "http://creativecommons.org/licenses/by-nc/3.0/"
SAMPLING = "http://creativecommons.org/licenses/sampling+/1.0/"


def fake_fsd50k(root, clips):
    """clips: (fname, description, licence)."""
    (root / "FSD50K.metadata").mkdir(parents=True)
    (root / "FSD50K.ground_truth").mkdir()
    info = {f: {"title": f"clip {f}", "description": d, "tags": [], "license": lic, "uploader": f"author {f}"}
            for f, d, lic in clips}
    (root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").write_text(json.dumps(info), encoding="utf-8")
    rows = "fname,labels,mids,split\n" + "".join(f"{f},Slam,/m/x,train\n" for f, _, _ in clips)
    (root / "FSD50K.ground_truth" / "dev.csv").write_text(rows, encoding="utf-8")
    return root


# Only CC0 and CC BY recordings, and only descriptions that say something about the sound.
def test_free_licences_and_useful_descriptions(tmp_path):
    root = fake_fsd50k(tmp_path, [
        ("1", "A heavy wooden door slams shut. Recorded with a zoom h4n", CC0),
        ("2", "Small glass bottle shattering on concrete", BY),
        ("3", "A heavy wooden door slams shut", BY_NC),        # non-commercial: left out
        ("4", "Metal gate creaking slowly", SAMPLING),         # sampling+: left out
        ("5", "Recorded at home", CC0),                        # nothing about the sound
    ])
    rows, free_clips, _ = C.descriptions(root)
    assert free_clips == 3
    assert [(r["fname"], r["en"]) for r in rows] == [("1", "A heavy wooden door slams shut"),
                                                     ("2", "Small glass bottle shattering on concrete")]
    assert rows[0]["uploader"] == "author 1" and rows[0]["license"] == CC0 and rows[0]["labels"] == "Slam"


def test_free():
    assert C.free(CC0) and C.free(BY) and C.free("https://creativecommons.org/licenses/by/4.0/")
    assert not C.free(BY_NC) and not C.free(SAMPLING)
