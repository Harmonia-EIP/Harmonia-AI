"""Harmonia v3 at run time: a prompt becomes a charter-v3 preset, and a preset is refined towards a prompt.

Generate. The prompt goes through the multilingual text encoder and the prior to the sound it asks for (a
point in CLAP's audio space). Each generator (analog engine, DX7 mode) draws candidates for that sound;
Harmonia plays them and CLAP listens. The closest analog one is then polished: a short search around it
(small moves of its settings, each played and heard) keeps what brings it closer to the prompt. The
closest of the polished analog preset and the best DX7 draw is the answer.

Refine. Same search, starting from the preset the user has: a variation is kept only when it is closer to
the prompt and still sounds like the preset it started from, so the sound is kept and moves towards the
description. The engine does not change, and the locked parameters come back with exactly the values they
were sent with.

Refine with a comparative ("plus sombre", "more reverb", "attaque plus lente"...): CLAP does not understand
comparisons, so these are read as directions (src/v3/directives.py). The settings that carry the asked
quality are moved step by step, and a step is kept only when the measure on the sound went the asked way.

Nothing but trained weights is needed: no preset bank, no recording (see scripts/v3/export_model.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src import charter_v3
from src.synth import v3_params as P
from src.v3 import codecs
from src.v3 import directives as D
from src.v3.diffusion import Diffusion
from src.v3.prior import Prior

ENGINES = ("analog", "dx7")
NOTES = (45, 69)  # the notes the generators' training sounds were heard at
SR = 48000
HOLD_SECONDS, TOTAL_SECONDS = 1.5, 4.0
VELOCITY = 100
SILENT = 1e-3  # peak below this: nothing to hear
GUIDANCE = 2.0
MAX_TOKENS = 64  # the text student reads this many tokens of a prompt (src/v2/text_student.py)
# Refine: lowest similarity (CLAP, sound to sound) a refined preset may have to the one it started from,
# at strength 0 and at strength 1. Two renders of one preset an octave apart are around 0.6.
KEEP_AT_ZERO, KEEP_AT_ONE = 0.95, 0.45
STRUCTURE_FREE = 0.75
POLISHED = "analog"  # the engine whose draws the search improves after a generation

# Analog settings a local search may switch (index -> legal values); the others move continuously.
# The coarse tunings only move by octaves: any other interval would transpose or detune the sound.
_ANALOG_DISCRETE = {name: tuple(range(int(P.BY_NAME[name].start), int(P.BY_NAME[name].end) + 1))
                    for name in ("osc_1_waveform", "osc_2_waveform", "filter_type", "filter_slope", "osc_sync",
                                 "lfo_waveform")}
_ANALOG_OCTAVES = ("osc_1_coarse", "osc_2_coarse")
_ANALOG_FIXED = ("unison_voices",)  # changing the voice count changes the level and the cost of a note
# DX7 fields a local search may nudge. Ratios, algorithm, transpose and keyboard scaling change the voice
# itself: only the generator's variations touch them.
_DX7_NUDGE = ("rates", "levels", "output_level", "key_vel_sens", "detune", "feedback", "lfo_speed", "lfo_delay",
              "lfo_pitch_mod_depth", "lfo_amp_mod_depth", "amp_mod_sens", "pitch_mod_sens")


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _dx7_layout() -> Dict[Tuple[Optional[int], str], Tuple[int, int, str]]:
    """(operator or None, field) -> (start, width, kind) in the DX7 codec vector (src/v3/codecs.py)."""
    layout, i = {}, 0
    for op in range(6):
        for name, kind, size in codecs.OP_FIELDS:
            width = 4 if kind == "list" else (size if kind == "cat" else 1)
            layout[(op, name)] = (i, width, kind)
            i += width
    for name, kind, size in codecs.GLOBAL_FIELDS:
        width = 4 if kind == "list" else (size if kind == "cat" else 1)
        layout[(None, name)] = (i, width, kind)
        i += width
    return layout


DX7_LAYOUT = _dx7_layout()


def structure_mask(engine: str) -> np.ndarray:
    """Positions that set the pitch or the architecture of a sound rather than its colour: tunings and voice
    count (analog); algorithm, ratios, fixed-frequency mode and transpose (DX7). A refine holds them unless
    it is asked to go far (strength above STRUCTURE_FREE)."""
    if engine == "analog":
        return np.array([name in _ANALOG_OCTAVES + _ANALOG_FIXED for name in P.NAMES], dtype=bool)
    mask = np.zeros(codecs.DX7_DIM, dtype=bool)
    for (_, name), (start, width, _) in DX7_LAYOUT.items():
        if name in ("algorithm", "freq_coarse", "freq_fine", "osc_mode", "transpose"):
            mask[start:start + width] = True
    return mask


def _dx7_nudge_mask() -> np.ndarray:
    mask = np.zeros(codecs.DX7_DIM, dtype=bool)
    for (_, name), (start, width, kind) in DX7_LAYOUT.items():
        if name in _DX7_NUDGE and kind != "cat":
            mask[start:start + width] = True
    return mask


def lock_mask(engine: str, locked: Sequence[str]) -> np.ndarray:
    """Positions of the engine's codec vector held by the locked parameter ids (others are ignored)."""
    charter = charter_v3.load()
    if engine == "analog":
        return np.array([name in locked for name in P.NAMES], dtype=bool)
    mask = np.zeros(codecs.DX7_DIM, dtype=bool)
    for p in charter.engine_params("dx7"):
        if p.id in locked:
            op, name, stage = p.dx7_path
            start, width, _ = DX7_LAYOUT[(op, name)]
            if stage is None:
                mask[start:start + width] = True
            else:
                mask[start + stage] = True
    return mask


@dataclass
class Settings:
    candidates: int = 8  # drawn per engine for a generation
    polish_rounds: int = 2  # search rounds after a generation (0: keep the best draw as it is)
    refine_rounds: int = 3
    population: int = 8  # sounds played and heard per search round
    strength: float = 0.5  # refine: 0 barely moves the sound, 1 lets it go far towards the prompt


@dataclass
class Result:
    preset: Dict[str, object]
    engine: str
    similarity: float  # of the kept sound to what the prompt asks (CLAP, -1..1)
    info: Dict[str, object] = field(default_factory=dict)


class HarmoniaV3:
    def __init__(self, model_dir: Path, device: Optional[torch.device] = None, settings: Optional[Settings] = None,
                 listener=None, text_encoder: Optional[Callable[[str], np.ndarray]] = None):
        """Loads the trained weights of `model_dir` (scripts/v3/export_model.py). `listener` (an object with
        embed_audio) and `text_encoder` (prompt -> vector in CLAP's text space) replace CLAP's audio tower and
        the text student: the tests use small stand-ins."""
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cpu")
        self.settings = settings or Settings()
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.max_tokens = MAX_TOKENS
        if text_encoder is None:
            from src.v2.text_student import encode_texts, load_trained_student

            student, tokenizer = load_trained_student(self.model_dir / "text_student", self.device)

            def text_encoder(prompt: str) -> np.ndarray:
                return encode_texts(student, tokenizer, [prompt], self.device)[0]

        self.text_encoder = text_encoder
        self.prior = Prior()
        self.prior.load_state_dict(torch.load(self.model_dir / "generator" / "prior.pt", map_location="cpu",
                                              weights_only=True))
        self.prior.eval()
        self.models: Dict[str, Diffusion] = {}
        for engine, dim in (("analog", codecs.ANALOG_DIM), ("dx7", codecs.DX7_DIM)):
            if engine == "dx7" and not self._dx7_available():
                continue  # without the compiled DX7 renderer the server answers with the analog engine only
            weights = torch.load(self.model_dir / "generator" / f"{engine}.pt", map_location="cpu", weights_only=True)
            diffusion = Diffusion.create(dim, width=weights["inp.weight"].shape[0],
                                         depth=len({k.split(".")[1] for k in weights if k.startswith("blocks.")}))
            diffusion.model.load_state_dict(weights)
            diffusion.model.eval()
            self.models[engine] = diffusion
        if listener is None:
            from src.v3.listener import ClapListener

            listener = ClapListener(self.model_dir, self.device)
        self.listener = listener
        self._discrete = {P.INDEX[name]: values for name, values in _ANALOG_DISCRETE.items()}
        self._dx7_nudge = _dx7_nudge_mask()

    # --- pieces -------------------------------------------------------------------------------------------

    @staticmethod
    def _dx7_available() -> bool:
        from src.presets import dx7_render

        return dx7_render.LIBRARY.exists()

    @property
    def engines(self) -> Tuple[str, ...]:
        return tuple(self.models)

    def target(self, prompt: str) -> np.ndarray:
        """The sound the prompt asks for: a unit vector in CLAP's audio space."""
        text = np.asarray(self.text_encoder(prompt), dtype=np.float32)
        with torch.no_grad():
            return self.prior(torch.from_numpy(text[None, :]))[0].numpy()

    def render(self, engine: str, vectors: np.ndarray) -> np.ndarray:
        """(n, notes, samples) mono audio of each codec vector."""
        if engine == "dx7":
            from src.presets import dx7_render

            return np.stack([np.stack([dx7_render.render(codecs.dx7_decode(v), note=n, velocity=VELOCITY,
                                                         hold_seconds=HOLD_SECONDS, total_seconds=TOTAL_SECONDS,
                                                         sample_rate=SR) for n in NOTES]) for v in vectors])
        from src.synth.engine_v3 import render_batch

        physical = np.stack([codecs.analog_decode(v) for v in vectors])
        return render_batch(physical, np.array(NOTES), VELOCITY / 127, HOLD_SECONDS, TOTAL_SECONDS, SR, 7)

    def hear(self, engine: str, vectors: np.ndarray, measures: Optional[List[Dict[str, float]]] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
        """Each vector played and heard: (unit CLAP vectors (n, 512), audible (n,)). With a `measures` list,
        the measures of each sound (src/v3/directives.py) are appended to it."""
        audio = np.nan_to_num(self.render(engine, vectors))
        peak = np.abs(audio).reshape(len(vectors), -1).max(axis=1)
        flat = list(audio.reshape(-1, audio.shape[-1]).astype(np.float32))
        emb = normalize(self.listener.embed_audio(flat)).reshape(len(vectors), len(NOTES), -1)
        if measures is not None:
            measures.extend(D.measure_notes(notes) for notes in audio)
        return normalize(emb.mean(axis=1)), peak >= SILENT

    def _draw(self, engine: str, target: np.ndarray, n: int, seed: int, start: Optional[np.ndarray] = None,
              strength: float = 1.0, keep: Optional[np.ndarray] = None) -> np.ndarray:
        cond = torch.from_numpy(np.repeat(target[None, :].astype(np.float32), n, axis=0))
        kwargs = {}
        if start is not None:
            kwargs["start"] = torch.from_numpy(np.repeat(start[None, :].astype(np.float32), n, axis=0))
            kwargs["strength"] = strength
            if keep is not None and keep.any():
                kwargs["keep"] = torch.from_numpy(keep)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return self.models[engine].sample(cond, guidance=GUIDANCE, generator=generator, **kwargs).numpy()

    def _nudge(self, engine: str, start: np.ndarray, n: int, sigma: float, rng: np.random.Generator,
               keep: np.ndarray) -> np.ndarray:
        """n copies of `start` with small random moves of the settings a search may touch."""
        out = np.repeat(start[None, :], n, axis=0).astype(np.float32)
        if engine == "analog":
            movable = np.array([spec.interval == 0.0 for spec in P.SPECS]) & ~keep
            moves = sigma * 2.0 * rng.standard_normal(out.shape) * (rng.random(out.shape) < 0.5)
            out = np.where(movable[None, :], np.clip(out + moves, -1.0, 1.0), out)
            for index, values in self._discrete.items():
                if keep[index]:
                    continue
                flip = rng.random(n) < 0.08
                spec = P.SPECS[index]
                picks = np.array([2.0 * P.to_normalized(spec, v) - 1.0 for v in rng.choice(values, size=n)])
                out[:, index] = np.where(flip, picks, out[:, index])
            for name in _ANALOG_OCTAVES:
                index = P.INDEX[name]
                if keep[index]:
                    continue
                spec = P.SPECS[index]
                for k in np.nonzero(rng.random(n) < 0.04)[0]:
                    semis = P.from_normalized(spec, (out[k, index] + 1.0) / 2.0) + rng.choice((-12.0, 12.0))
                    if spec.start <= semis <= spec.end:
                        out[k, index] = 2.0 * P.to_normalized(spec, semis) - 1.0
            return out
        movable = self._dx7_nudge & ~keep
        moves = sigma * 2.0 * rng.standard_normal(out.shape) * (rng.random(out.shape) < 0.35)
        return np.where(movable[None, :], np.clip(out + moves, -1.0, 1.0), out).astype(np.float32)

    def _search(self, engine: str, start: np.ndarray, start_emb: np.ndarray, target: np.ndarray, rounds: int,
                seed: int, keep: np.ndarray, strength: float, floor: float = -1.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """Walk from `start` towards `target`: each round plays and hears variations of the best sound so far
        (half drawn by the generator from it, half small moves of its settings) and keeps the closest to the
        target among those that still sound like `start` (similarity to it of at least `floor`)."""
        rng = np.random.default_rng(seed)
        best, best_emb = start.astype(np.float32), start_emb
        best_score = float(best_emb @ target)
        heard = 0
        n_draw = self.settings.population // 2
        n_nudge = self.settings.population - n_draw
        for r in range(rounds):
            shrink = 0.7 ** r
            drawn = self._draw(engine, target, n_draw, seed + 1 + r, start=best, keep=keep,
                               strength=(0.15 + 0.3 * strength) * shrink)
            nudged = self._nudge(engine, best, n_nudge, (0.04 + 0.1 * strength) * shrink, rng, keep)
            pool = np.concatenate([drawn, nudged]).astype(np.float32)
            pool[:, keep] = start[keep]
            emb, audible = self.hear(engine, pool)
            heard += len(pool)
            allowed = audible & (emb @ start_emb >= floor)
            scores = np.where(allowed, emb @ target, -np.inf)
            k = int(np.argmax(scores))
            if scores[k] > best_score:
                best, best_emb, best_score = pool[k], emb[k], float(scores[k])
        return best, best_emb, heard

    def _directed_moves(self, engine: str, start: np.ndarray, wanted: Sequence[D.Directive], keep: np.ndarray,
                        strength: float) -> np.ndarray:
        """Candidates for comparatives: `start` with the settings that carry each asked quality moved the asked
        way, at several sizes. On the DX7, where which operators colour the sound depends on the algorithm,
        each operator is also tried alone and the measure decides."""
        size = D.STEP * (0.4 + 1.2 * strength) * 2.0  # codec vectors span [-1, 1]
        out = []
        if engine == "analog":
            for scale in D.SCALES:
                v = start.copy()
                for directive in wanted:
                    for name, weight in D.ANALOG_MOVES[directive.measure]:
                        i = P.INDEX[name]
                        if not keep[i]:
                            v[i] = np.clip(v[i] + directive.direction * weight * size * scale, -1.0, 1.0)
                out.append(v)
        else:
            groups = [tuple(range(6))] + [(op,) for op in range(6)]
            for scale in D.SCALES[:3]:
                for ops in groups if scale == D.SCALES[1] else groups[:1]:
                    v = start.copy()
                    for directive in wanted:
                        for name, stage, weight in D.DX7_MOVES[directive.measure]:
                            for op in ops:
                                i = DX7_LAYOUT[(op, name)][0] + (stage or 0)
                                if not keep[i]:
                                    v[i] = np.clip(v[i] + directive.direction * weight * size * scale, -1.0, 1.0)
                    out.append(v)
        moved = [v for v in out if np.any(v != start)]
        return np.stack(moved).astype(np.float32) if moved else np.zeros((0, len(start)), dtype=np.float32)

    def _refine_directed(self, engine: str, start: np.ndarray, wanted: Sequence[D.Directive], keep: np.ndarray,
                         strength: float, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Move `start` the way the comparatives ask. A candidate counts by how far each asked measure went the
        right way, up to the goal of this strength (going further earns nothing); the smallest move that
        reaches every goal wins, else the one that went furthest. When the settings that normally carry the
        quality cannot move (locked, or already at their limit), variations of the sound are searched with the
        same measure."""
        before: List[Dict[str, float]] = []
        start_emb, audible = self.hear(engine, start[None, :], before)
        start_emb, before = start_emb[0], before[0]
        goals = {d.measure: D.goal(d.measure, strength) for d in wanted}

        def progress(m: Dict[str, float]) -> Tuple[float, float]:
            gains = [d.direction * (m[d.measure] - before[d.measure]) for d in wanted]
            reached = [min(g, goals[d.measure]) / goals[d.measure] for g, d in zip(gains, wanted)]
            return float(np.mean(reached)), float(min(gains))

        best, best_emb, best_m, best_score, heard = start, start_emb, before, 0.0, 1
        pools = [self._directed_moves(engine, start, wanted, keep, strength)]
        rng = np.random.default_rng(seed)
        for attempt in range(1 + self.settings.refine_rounds):
            if attempt > 0:  # the usual controls could not move the sound: search variations of it instead
                if best_score > 0.0 and (attempt == 1 or best_score >= 1.0):
                    break  # they did, or the search got there: nothing else of the sound has to change
                drawn = self._draw(engine, start_emb, self.settings.population // 2, seed + attempt, start=best,
                                   keep=keep, strength=0.15 + 0.3 * strength)
                nudged = self._nudge(engine, best, self.settings.population - len(drawn), 0.04 + 0.1 * strength,
                                     rng, keep)
                pools.append(np.concatenate([drawn, nudged]).astype(np.float32))
            pool = pools[-1]
            if len(pool) == 0:
                continue
            pool[:, keep] = start[keep]
            measures: List[Dict[str, float]] = []
            emb, ok = self.hear(engine, pool, measures)
            heard += len(pool)
            for k in range(len(pool)):  # in order of size: the first to reach every goal is the smallest move
                score, worst = progress(measures[k])
                if not ok[k] or emb[k] @ start_emb < KEEP_AT_ONE or worst < D.MIN_GAIN:
                    continue
                if score > best_score:
                    best, best_emb, best_m, best_score = pool[k], emb[k], measures[k], score
                    if score >= 1.0 and attempt == 0:
                        break
        info = {"directives": [d.label for d in wanted],
                "measures_before": {d.measure: round(before[d.measure], 3) for d in wanted},
                "measures_after": {d.measure: round(best_m[d.measure], 3) for d in wanted},
                "kept": round(float(best_emb @ start_emb), 4), "heard": heard, "changed": bool(np.any(best != start))}
        return best, best_emb, info

    # --- presets ------------------------------------------------------------------------------------------

    def _preset(self, engine: str, vector: np.ndarray, name: str) -> Dict[str, object]:
        if engine == "dx7":
            values = charter_v3.dx7_values(codecs.dx7_decode(vector))
        else:
            values = dict(zip(P.NAMES, (float(v) for v in codecs.analog_decode(vector))))
        return charter_v3.make_preset(engine, values, name)

    @staticmethod
    def is_v3(parameters: Mapping[str, float]) -> bool:
        """False for a v1/v2 preset: only the 20 v2 parameters, normalized on the v2 ranges."""
        charter = charter_v3.load()
        v3_only = set(charter.by_id) - {p.id for p in charter.v2_params}
        return any(k in v3_only for k in parameters)

    @classmethod
    def read_current(cls, parameters: Mapping[str, float]) -> Tuple[str, np.ndarray]:
        """(engine, codec vector) of the preset a refine starts from: {id: normalized value}, charter v3, or
        a v1/v2 preset, which plays on the analog engine after the charter's upgrade rules."""
        charter = charter_v3.load()
        known = {k: float(v) for k, v in parameters.items() if k in charter.by_id}
        if not cls.is_v3(known):
            values = charter_v3.upgrade_v2_normalized(known)
            return "analog", codecs.analog_encode(P.physical_from_dict(values)).astype(np.float32)
        engine_param = charter.by_id["engine"]
        engine = ENGINES[int(engine_param.from_normalized(known.get("engine", 0.0)))]
        values = {p.id: p.from_normalized(known[p.id]) if p.id in known else p.default
                  for p in charter.engine_params(engine)}
        if engine == "dx7":
            return engine, codecs.dx7_encode(charter_v3.dx7_voice(values)).astype(np.float32)
        return engine, codecs.analog_encode(P.physical_from_dict(values)).astype(np.float32)

    @staticmethod
    def seed_of(*parts: object) -> int:
        digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    # --- the two requests ---------------------------------------------------------------------------------

    def generate(self, prompt: str, variation: int = 0) -> Result:
        """Best preset for the prompt; `variation` > 0 gives other answers to the same prompt.

        Each engine's closest draw is kept. The analog one is then polished by the search, and the engines are
        compared after that. DX7 draws are left as drawn: on the benchmark the search raised their similarity
        for CLAP but not for an independent judge, with or without their structure held (CHANGELOG 0.3.2),
        where it does help analog presets."""
        target = self.target(prompt)
        seed = self.seed_of(prompt, variation)
        best: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
        heard = 0
        for engine in self.engines:
            vectors = self._draw(engine, target, self.settings.candidates, seed)
            emb, audible = self.hear(engine, vectors)
            heard += len(vectors)
            sims = np.where(audible, emb @ target, -np.inf)
            k = int(np.argmax(sims))
            if np.isfinite(sims[k]):
                best[engine] = (float(sims[k]), vectors[k], emb[k])
        if not best:
            raise RuntimeError("no audible candidate")
        drawn = {engine: round(sim, 4) for engine, (sim, _, _) in best.items()}
        if self.settings.polish_rounds > 0 and POLISHED in best:
            _, vector, emb = best[POLISHED]
            vector, emb, more = self._search(POLISHED, vector, emb, target, self.settings.polish_rounds, seed,
                                             np.zeros(len(vector), dtype=bool), strength=0.5)
            heard += more
            best[POLISHED] = (float(emb @ target), vector, emb)
        engine = max(best, key=lambda name: best[name][0])
        similarity, vector, emb = best[engine]
        return Result(self._preset(engine, vector, prompt), engine, similarity,
                      {"engine": engine, "similarity": round(similarity, 4), "drawn_similarity": drawn[engine],
                       "drawn": drawn, "variation": variation, "heard": heard})

    def _refined_preset(self, engine: str, vector: np.ndarray, name: str, parameters: Mapping[str, float],
                        locked: Sequence[str]) -> Dict[str, object]:
        preset = self._preset(engine, vector, name)
        if self.is_v3(parameters):
            for key in locked:  # exactly as sent, whatever rounding the engine's values went through
                if key in preset["parameters"] and key in parameters:
                    preset["parameters"][key] = float(parameters[key])
        # a v1/v2 preset: its locked values were held too, and come back on the v3 ranges like the others
        return preset

    def refine(self, prompt: str, parameters: Mapping[str, float], locked: Sequence[str] = (),
               strength: Optional[float] = None) -> Result:
        """The current preset moved towards the prompt; locked parameters keep their value exactly."""
        strength = self.settings.strength if strength is None else min(1.0, max(0.0, float(strength)))
        engine, start = self.read_current(parameters)
        if engine not in self.models:
            raise RuntimeError(f"the {engine} engine is not available on this server")
        locks = lock_mask(engine, locked)
        keep = locks if strength > STRUCTURE_FREE else locks | structure_mask(engine)
        target = self.target(prompt)
        seed = self.seed_of(prompt, sorted(parameters.items()), sorted(locked))
        asked = D.read(prompt)
        wanted = [d for d in asked if engine == "analog" or D.DX7_MOVES[d.measure]]
        if wanted:  # a comparative: a direction to follow, not a sound to aim at
            push = min(1.0, strength * D.amount(prompt))  # "un peu plus sombre" moves half as far
            vector, emb, info = self._refine_directed(engine, start, wanted, keep, push, seed)
            return Result(self._refined_preset(engine, vector, prompt, parameters, locked), engine, float(emb @ target),
                          {"engine": engine, "strength": strength, **info})
        start_emb, audible = self.hear(engine, start[None, :])
        start_emb = start_emb[0]
        # how much of the current sound must remain (a silent preset has no sound to keep)
        floor = KEEP_AT_ZERO - (KEEP_AT_ZERO - KEEP_AT_ONE) * strength if audible[0] else -1.0
        vector, emb, heard = self._search(engine, start, start_emb, target, self.settings.refine_rounds, seed, keep,
                                          strength, floor)
        preset = self._refined_preset(engine, vector, prompt, parameters, locked)
        before, after = float(start_emb @ target), float(emb @ target)
        return Result(preset, engine, after,
                      {"engine": engine, "similarity": round(after, 4), "similarity_before": round(before, 4),
                       "kept": round(float(emb @ start_emb), 4), "strength": strength, "heard": heard + 1,
                       "changed": bool(np.any(vector != start)),
                       # comparatives this engine has no setting for (reverb on the DX7 mode, which plays dry)
                       **({"unsupported": [d.label for d in asked]} if asked else {})})
