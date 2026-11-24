"""Credits written next to the exported weights (scripts/v3/export_model.py)."""

import json

from scripts.v3 import export_model as E

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY = "http://creativecommons.org/licenses/by/3.0/"


# The authors of the CC BY descriptions the text side learnt from are credited; CC0 needs no credit.
def test_descriptions_are_credited(tmp_path):
    descriptions = tmp_path / "descriptions.jsonl"
    rows = [{"fname": "1", "en": "door slams", "uploader": "ann", "license": CC0},
            {"fname": "2", "en": "glass breaks", "uploader": "bob", "license": BY}]
    descriptions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "model"
    out.mkdir()
    files = E.export_notices(out, "harmonia_v4.0", None, 0.45, 30.0, descriptions)
    credits = json.loads((out / "ATTRIBUTION_DESCRIPTIONS.json").read_text(encoding="utf-8"))
    assert credits == [{"freesound_id": "2", "author": "bob", "license": BY}]
    notice = (out / "NOTICE.md").read_text(encoding="utf-8")
    assert "harmonia_v4.0" in notice and "2 FSD50K recordings" in notice
    assert set(files) == {"LICENSE", "NOTICE.md", "ATTRIBUTION_DESCRIPTIONS.json"}


def test_without_descriptions_nothing_changes(tmp_path):
    out = tmp_path / "model"
    out.mkdir()
    files = E.export_notices(out, "harmonia_v3.3", None, 0.45, 30.0)
    assert set(files) == {"LICENSE", "NOTICE.md"}
    assert "descriptions" not in (out / "NOTICE.md").read_text(encoding="utf-8")


# Imitations of several runs: a recording imitated twice is credited once, from the last run; a model whose
# generator plays recorded instruments says where they come from.
def test_imitations_of_several_runs_and_samples(tmp_path):
    def run(name, rows):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"title": f"t{r[0]}", "uploader": "ann", "license": CC0, "fname": r[0],
                                            "cos": r[1], "distance": 10.0}) + "\n" for r in rows), encoding="utf-8")
        return path

    first, second = run("a", [("1", 0.5), ("2", 0.6)]), run("b", [("2", 0.3), ("3", 0.8)])
    from scripts.v3.match_recordings import read_imitations

    assert {r["fname"]: r["cos"] for r in read_imitations([first, second])} == {"1": 0.5, "2": 0.3, "3": 0.8}
    out = tmp_path / "model"
    out.mkdir()
    E.export_notices(out, "harmonia_v5", [first, second], 0.45, 30.0, samples=True)
    credits = json.loads((out / "ATTRIBUTION.json").read_text(encoding="utf-8"))
    assert [c["freesound_id"] for c in credits] == ["1", "3"]  # recording 2's last imitation is not close enough
    notice = (out / "NOTICE.md").read_text(encoding="utf-8")
    assert "Salamander Grand Piano" in notice and "CREDITS.md" in notice
