"""Harmonia charter v3: reads charter/harmonia_v3.json, the single source of truth for v3 presets.

The plugin's C++ parameter table and JUCE layout are generated from the same file (scripts/v3/charter_cpp.py),
and src/synth/v3_params.py builds the analog engine's specs from it. Values are normalized like
juce::NormalisableRange (the file's "normalization" section), so a normalized value means the same thing
here and in the plugin. Standard library only, so the backend can use it as is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

CHARTER_PATH = Path(__file__).resolve().parents[1] / "charter" / "harmonia_v3.json"
KINDS = ("float", "int", "choice", "bool")
ENGINES = ("analog", "dx7")


@dataclass(frozen=True)
class V2Rule:
    source: str
    scale: float
    offset: float


@dataclass(frozen=True)
class Param:
    id: str
    module: str
    engine: str  # "global", "analog" or "dx7"
    kind: str
    min: float
    max: float
    default: float
    step: float = 0.0
    centre: Optional[float] = None
    choices: Tuple[str, ...] = ()
    unit: str = ""
    label: str = ""
    label_fr: str = ""
    description: str = ""
    display_offset: int = 0
    v2: Optional[V2Rule] = None
    # DX7 only: where the value lives in a voice dict (operator index 0..5 or None, field, stage 0..3 or None)
    dx7_path: Optional[Tuple[Optional[int], str, Optional[int]]] = None
    fixed_skew: Optional[float] = None  # v2 ranges give their skew directly instead of a centre

    @property
    def interval(self) -> float:
        return self.step if self.kind == "float" else 1.0

    @property
    def skew(self) -> float:
        if self.fixed_skew is not None:
            return self.fixed_skew
        if self.centre is None:
            return 1.0
        return math.log(0.5) / math.log((self.centre - self.min) / (self.max - self.min))

    def snap(self, value: float) -> float:
        """Legal value, rounded exactly as the plugin's JUCE parameters do (see the charter's "snap")."""
        value = min(self.max, max(self.min, float(value)))
        if self.kind == "bool":
            value = self.max if value - self.min >= 0.5 * (self.max - self.min) else self.min
        elif self.kind in ("int", "choice"):
            value = float(round(value))  # half to even, like juce::roundToInt
        elif self.step > 0:
            value = self.min + self.step * math.floor((value - self.min) / self.step + 0.5)
        return min(self.max, max(self.min, value))

    def to_normalized(self, value: float) -> float:
        proportion = (self.snap(value) - self.min) / (self.max - self.min)
        return proportion ** self.skew if self.skew != 1.0 and proportion > 0 else proportion

    def from_normalized(self, normalized: float) -> float:
        proportion = min(1.0, max(0.0, float(normalized)))
        if self.skew != 1.0 and proportion > 0:
            proportion = math.exp(math.log(proportion) / self.skew)
        return self.snap(self.min + (self.max - self.min) * proportion)


@dataclass(frozen=True)
class Charter:
    version: str
    raw: dict
    params: Tuple[Param, ...]  # global, then analog, then DX7, in the file's order
    v2_params: Tuple[Param, ...] = ()  # the 20 v2 ranges, to read v2 presets

    @property
    def by_id(self) -> Dict[str, Param]:
        return {p.id: p for p in self.params}

    def engine_params(self, engine: str) -> Tuple[Param, ...]:
        return tuple(p for p in self.params if p.engine == engine)


def _param(entry: Mapping, engine: str, **overrides) -> Param:
    choices = tuple(entry.get("choices", ()))
    kind = entry["kind"]
    if kind not in KINDS:
        raise ValueError(f"{entry['id']}: unknown kind {kind!r}")
    low = entry.get("min", 0)
    high = entry.get("max", len(choices) - 1 if choices else None)
    rule = entry.get("v2")
    values = dict(
        id=entry["id"], module=entry["module"], engine=engine, kind=kind, min=float(low), max=float(high),
        default=float(entry["default"]), step=float(entry.get("step", 0.0)), centre=entry.get("centre"),
        choices=choices, unit=entry.get("unit", ""), label=entry.get("label", ""), label_fr=entry.get("label_fr", ""),
        description=entry.get("description", ""), display_offset=int(entry.get("display_offset", 0)),
        v2=V2Rule(rule["from"], float(rule["scale"]), float(rule["offset"])) if rule else None,
    )
    values.update(overrides)
    return Param(**values)


def _dx7_params(spec: Mapping) -> List[Param]:
    """Expand the DX7 operator and voice field templates into one parameter per value."""
    out: List[Param] = []

    def expand(fields, prefix: str, module: str, op: Optional[int], name_prefix: str):
        for field in fields:
            count = field.get("count")
            for k in range(count or 1):
                stage = k if count else None
                fmt = {"k": k + 1}
                default = field["default"]
                if isinstance(default, list):
                    default = default[op if len(default) == 6 and op is not None else k]
                entry = dict(field, id=prefix + field["id"].format(**fmt), module=module, default=default,
                             label=name_prefix + field["label"].format(**fmt),
                             label_fr=name_prefix + field["label_fr"].format(**fmt),
                             description=field["description"].format(**fmt))
                out.append(_param(entry, "dx7", dx7_path=(op, field["field"], stage)))

    for op in range(spec["operators"]):
        expand(spec["operator_fields"], f"dx7_op{op + 1}_", "dx7_operator", op, f"OP{op + 1} ")
    expand(spec["voice_fields"], "dx7_", "dx7_voice", None, "")
    return out


@lru_cache(maxsize=4)
def load(path: Path = CHARTER_PATH) -> Charter:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    params = [_param(e, "global") for e in raw["global"]]
    params += [_param(e, "analog") for e in raw["engines"]["analog"]["parameters"]]
    params += _dx7_params(raw["engines"]["dx7"])
    ids = [p.id for p in params]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate parameter ids in the charter")
    v2 = tuple(_param(dict(e, module="v2", default=e["min"]), "v2", fixed_skew=e.get("skew"))
               for e in raw["v2_upgrade"]["parameters"])
    return Charter(raw["version"], raw, tuple(params), v2)


# --- presets ----------------------------------------------------------------------------------------------

def dx7_values(voice: Mapping) -> Dict[str, int]:
    """DX7 voice dict (src/presets/dx7.py) -> {parameter id: value}."""
    out = {}
    for p in load().engine_params("dx7"):
        op, field, stage = p.dx7_path
        value = (voice["operators"][op] if op is not None else voice)[field]
        out[p.id] = int(value[stage] if stage is not None else value)
    return out


def dx7_voice(values: Mapping[str, float], name: str = "HARMONIA") -> Dict[str, object]:
    """{parameter id: value} -> DX7 voice dict; missing values take their default."""
    params = load().engine_params("dx7")
    n_ops = load().raw["engines"]["dx7"]["operators"]
    voice: Dict[str, object] = {"operators": [{} for _ in range(n_ops)]}
    for p in params:
        op, field, stage = p.dx7_path
        target = voice["operators"][op] if op is not None else voice
        value = int(p.snap(values.get(p.id, p.default)))
        if stage is None:
            target[field] = value
        else:
            target.setdefault(field, [0, 0, 0, 0])[stage] = value
    voice["name"] = name[:10]
    return voice


def make_preset(engine: str, values: Mapping[str, float], name: str, brief: Optional[Mapping] = None,
                **metadata) -> Dict[str, object]:
    """Preset document (charter "preset_format") from physical values of one engine."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    charter = load()
    parameters = {"engine": charter.by_id["engine"].to_normalized(ENGINES.index(engine))}
    for p in charter.engine_params(engine):
        parameters[p.id] = p.to_normalized(values.get(p.id, p.default))
    doc: Dict[str, object] = {
        "metadata": {"name": name, "charter_version": charter.version, "engine": engine, **metadata},
        "parameters": parameters,
    }
    if brief is not None:
        doc["brief"] = dict(brief)
    return doc


def read_preset(doc: Mapping) -> Tuple[str, Dict[str, float]]:
    """(engine, {parameter id: physical value}) of a preset document; missing values take their default."""
    errors = validate_preset(doc)
    if errors:
        raise ValueError("; ".join(errors))
    charter = load()
    parameters = doc["parameters"]
    engine_param = charter.by_id["engine"]
    engine = ENGINES[int(engine_param.from_normalized(parameters["engine"]))]
    values = {p.id: p.from_normalized(parameters[p.id]) if p.id in parameters else p.default
              for p in charter.engine_params(engine)}
    return engine, values


def validate_preset(doc: Mapping) -> List[str]:
    """Problems that make a document an invalid v3 preset (empty list when it is valid)."""
    charter = load()
    errors: List[str] = []
    if not isinstance(doc, Mapping) or not isinstance(doc.get("parameters"), Mapping):
        return ["a preset is an object with a 'parameters' object"]
    parameters = doc["parameters"]
    if "engine" not in parameters:
        errors.append("'parameters' must hold 'engine'")
    for key, value in parameters.items():
        if key not in charter.by_id:
            errors.append(f"unknown parameter {key!r}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{key}: expected a normalized number in [0, 1], got {value!r}")
    if "brief" in doc:
        errors += validate_brief(doc["brief"])
    return errors


def validate_brief(brief: Mapping) -> List[str]:
    """Problems in a sound brief (charter "sound_brief"): unknown fields or values outside their list."""
    fields = {f["id"]: f for f in load().raw["sound_brief"]["fields"]}
    if not isinstance(brief, Mapping):
        return ["a brief is an object"]
    errors = []
    for key, value in brief.items():
        if key not in fields:
            errors.append(f"brief: unknown field {key!r}")
        elif isinstance(fields[key]["values"], list) and value not in fields[key]["values"]:
            errors.append(f"brief.{key}: {value!r} is not one of {fields[key]['values']}")
        elif key == "references" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            errors.append("brief.references: expected a list of names")
    return errors


def upgrade_v2(v2_physical: Mapping[str, float]) -> Dict[str, float]:
    """Physical analog values of a v2 preset or state ({v2 name: value}) -> v3 ({id: value}), per the rules."""
    out = {}
    for p in load().engine_params("analog"):
        if p.v2 is not None and p.v2.source in v2_physical:
            out[p.id] = p.snap(p.v2.offset + p.v2.scale * float(v2_physical[p.v2.source]))
        else:
            out[p.id] = p.default
    return out


def upgrade_v2_normalized(v2_normalized: Mapping[str, float]) -> Dict[str, float]:
    """A v2 preset ({v2 name: value normalized on the v2 ranges}) -> physical v3 analog values."""
    v2 = {p.id: p.from_normalized(v2_normalized[p.id]) for p in load().v2_params if p.id in v2_normalized}
    return upgrade_v2(v2)


def ids(engine: str) -> Sequence[str]:
    return tuple(p.id for p in load().engine_params(engine))
