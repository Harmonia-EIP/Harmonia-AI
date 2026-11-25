"""Names of the recorded instruments for the prior (scripts/samples/instrument_texts.py, #47)."""

from scripts.samples import instrument_texts as T


def region(key, lovel, hivel, rr=1, trigger="attack", file=None):
    return {"file": file or f"Strings/Solo Violin/arco-vib/v_{key}_{hivel}_{rr}.wav", "key": key, "lovel": lovel,
            "hivel": hivel, "round_robin": rr, "trigger": trigger}


# The names are the catalogue's and the authors' folders, without the folders that say nothing.
def test_names_come_from_the_libraries():
    original = {"regions": [region(60, 1, 127)]}
    assert T.names({"name": "Solo Violin · Arco Vib"}, original) == ["Solo Violin, Arco Vib", "Strings, Solo Violin, arco-vib"]
    piano = {"regions": [region(21, 1, 127, file="samples/A0v1.flac")]}
    assert T.names({"name": "Salamander Grand Piano"}, piano) == ["Salamander Grand Piano"]


# A few notes over the range, each at its middle and loudest layer, first round robin only.
def test_chosen_regions_spread_over_the_range():
    regions = [region(k, lo, hi, rr) for k in (40, 50, 60, 70, 80, 90)
               for lo, hi in ((1, 42), (43, 85), (86, 127)) for rr in (1, 2)]
    regions.append(region(60, 1, 127, trigger="release"))
    chosen = T.chosen_regions({"regions": regions}, notes=3)
    assert sorted({r["key"] for r in chosen}) == [40, 60, 90]
    assert all(r["round_robin"] == 1 and r["trigger"] == "attack" for r in chosen)
    assert sorted(r["hivel"] for r in chosen if r["key"] == 60) == [85, 127]
