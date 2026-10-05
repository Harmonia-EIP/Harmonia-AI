#!/usr/bin/env python3
"""Build the side-by-side listening page (HTML + WAV files) from the judged benchmark.

For a selection of prompts it renders, at A3 on the engine each system ships with:
- v1 (synthetic_v1) with the French prompt on the current app (what users get today);
- v1 with the English prompt on the current app (v1's best case);
- v2 (default mode) with the French prompt on the fixed app.

    python scripts/v2/listening_page.py --scores data/v2/bench/scores.json --out data/v2/listening
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import PARAM_NAMES  # noqa: E402
from src.synth.engine import ENGINE_APP_1_0, ENGINE_APP_1_1, render_preset  # noqa: E402
from src.synth.params import to_physical  # noqa: E402

SR = 44100
PROMPTS = [
    "a door slamming in a cathedral", "door slam", "glass breaking", "thunder", "footsteps in an empty hall",
    "soft piano", "church organ", "deep sub bass", "warm analog pad", "bright supersaw lead", "kick drum",
    "eerie horror drone", "wind howling through a cave", "cinematic impact",
]
COLUMNS = [
    ("v1_synthetic", ENGINE_APP_1_0, "fr", "Aujourd'hui", "IA v1 · prompt français · app actuelle"),
    ("v1_synthetic", ENGINE_APP_1_0, "en", "v1, en anglais", "IA v1 · prompt anglais · app actuelle"),
    ("v2_retrieval", ENGINE_APP_1_1, "fr", "Harmonia v2", "IA v2 · prompt français · app corrigée"),
]
CATEGORY_LABELS = {"instrument": "Instrument", "sfx": "Bruitage", "compositional": "Scène", "mood": "Ambiance"}
SYSTEM_LABELS = {
    "v2_retrieval@app1.1": "v2 recherche (défaut)",
    "v2_hybrid@app1.1": "v2 hybride",
    "v2_neural@app1.1": "v2 prédicteur",
    "oracle_teacher_fullbank@app1.1": "Professeur CLAP, banque complète",
    "v1_synthetic@app1.0": "v1 synthetic · app actuelle",
    "v1_charter@app1.0": "v1 charter · app actuelle",
    "v1_synthetic@app1.1": "v1 synthetic · app corrigée",
    "v1_charter@app1.1": "v1 charter · app corrigée",
    "random@app1.1": "Preset aléatoire",
}


def readout(values) -> str:
    p = dict(zip(PARAM_NAMES, to_physical(values)))
    waves = ["sinus", "triangle", "dent de scie", "carré"]
    return (
        f"{waves[int(p['osc_1_waveform'])]} · bruit {p['noise_level'] * 100:.0f} % · filtre {p['filter_cutoff']:,.0f} Hz · "
        f"attaque {p['amp_attack']:,.0f} ms · decay {p['amp_decay']:,.0f} ms · sustain {p['amp_sustain'] * 100:.0f} % · "
        f"reverb {p['reverb_mix'] * 100:.0f} %"
    ).replace(",", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=BASE_DIR / "data" / "v2" / "bench" / "scores.json")
    parser.add_argument("--template", type=Path, default=BASE_DIR / "scripts" / "v2" / "listening_template.html")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "listening")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    by_key = {(g["system"], g["engine"], g["lang"], g["prompt_en"]): g for g in scores["generations"]}
    (args.out / "audio").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, prompt_en in enumerate(PROMPTS):
        cells = []
        for system, engine, lang, short, long in COLUMNS:
            g = by_key[(system, engine, lang, prompt_en)]
            audio = np.nan_to_num(render_preset(g["values"], note=57, hold_seconds=1.5, total_seconds=4.0, sr=SR, q_mode=engine))
            peak = float(np.abs(audio).max())
            if peak > 0.98:
                audio = audio * (0.98 / peak)
            name = f"audio/{i:02d}_{system}_{lang}.wav"
            sf.write(args.out / name, audio, SR, subtype="PCM_16")
            cells.append(
                f'<div class="take{" take--new" if system.startswith("v2") else ""}">'
                f'<div class="take__head"><span class="take__name">{html.escape(short)}</span>'
                f'<span class="take__score" title="Score du juge indépendant (MS-CLAP)">{g["text_score"]:+.2f}</span></div>'
                f'<p class="take__desc">{html.escape(long)}</p>'
                f'<audio controls preload="none" src="{name}"></audio>'
                f'<p class="take__params">{html.escape(readout(g["values"]))}</p></div>'
            )
        g0 = by_key[(COLUMNS[2][0], COLUMNS[2][1], "fr", prompt_en)]
        rows.append(
            f'<article class="prompt" id="p{i:02d}"><header class="prompt__head">'
            f'<span class="chip chip--{g0["category"]}">{CATEGORY_LABELS[g0["category"]]}</span>'
            f'<h2 class="prompt__fr">« {html.escape(g0["prompt"])} »</h2>'
            f'<p class="prompt__en">{html.escape(prompt_en)}</p></header>'
            f'<div class="takes">{"".join(cells)}</div></article>'
        )

    summary = scores["summary"]
    order = [k for k in SYSTEM_LABELS if k in summary]
    best = max(max(summary[k].get("text_en", 0), summary[k].get("text_fr", 0)) for k in order)
    table = []
    for key in order:
        m = summary[key]
        cells = []
        for metric in ("text_en", "text_fr", "real_en", "real_fr"):
            v = m.get(metric)
            if v is None:
                cells.append('<td class="num">—</td>')
            else:
                width = max(0.0, v) / best * 100 if metric.startswith("text") else max(0.0, v) * 100
                cells.append(f'<td class="num"><span class="bar" style="--w:{width:.1f}%"></span>{round(v, 3) + 0.0:.3f}</td>')
        table.append(f'<tr class="{"row--v2" if key.startswith("v2_retrieval") else ""}"><th scope="row">{html.escape(SYSTEM_LABELS[key])}</th>{"".join(cells)}</tr>')

    page = args.template.read_text(encoding="utf-8")
    page = page.replace("{{ROWS}}", "\n".join(rows)).replace("{{TABLE}}", "\n".join(table))
    page = page.replace("{{N_PROMPTS}}", str(len({g["prompt_en"] for g in scores["generations"]})))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.out / 'index.html'} and {len(PROMPTS) * len(COLUMNS)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
