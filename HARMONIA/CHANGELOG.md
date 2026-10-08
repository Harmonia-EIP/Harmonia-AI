# CHANGELOG.md 📜 - 18/01/2026

All notable changes to the **Harmonia** project will be documented in this file.

## [0.2.0] - Harmonia v3 research (prerelease `ai-v3.0.0`, branch `research/ai-v3-presets`, not to be merged)
Goal: generate presets from text with an AI trained only on real, human-made presets and human-written
text (no generated training data), on a richer engine. Progress is checked by ear on a listening page
where Malo rates each sound. Details: [V3.md](V3.md).

### Added
- **Real preset banks** (`scripts/v3/fetch_presets.py`, `presets_to_json.py`): DX7 (27,471 unique voices
  from Dexed's cartridge collection and built-in programs), Surge XT factory + third-party (3,555, GPL-3),
  OB-Xf (492, CC0). Fetched from their sources (pinned commit / sha256), never redistributed.
- **Feature-usage analysis** (`scripts/v3/analyze_presets.py`): which synthesis features real presets use;
  it chose the new engine parameters (e.g. 72 % of OB-Xf presets set an oscillator interval, 63 % use
  unison, 94 % of DX7 voices use 5-6 operators).
- **v3 analog engine** (`src/synth/engine_v3.py`, `v3_params.py`), 46 parameters: the 20 of v2 plus
  coarse tuning per oscillator, pulse width, sync, OB-X style cross-modulation, ring mod, unison (1-16
  voices), stereo width, 12/24 dB filter, keyboard tracking, full filter envelope, pitch envelope,
  velocity to amp, LFO shape/delay/amp/PWM, chorus, delay, reverb size. Band-limited saw/pulse,
  exponential decays, 2x oversampling, analog drift, feedback-delay-network reverb.
- **DX7 mode** (`native/dx7`, `scripts/v3/build_dx7.py`, `src/presets/dx7_render.py`): Dexed's msfa FM
  core (Apache-2.0, embeddable in the app) plays DX7 voices as they are, ~2 ms per note.
- **Original-synth rendering** (`src/presets/originals.py`): OB-Xf and Surge XT presets played by the
  official plugins through pedalboard, to compare every conversion with its original.
- **Converters** (`src/presets/convert_obxf.py`, `convert_surge.py`) using each synth's real scalings, and
  **sound matching** (`src/presets/matching.py`, `perceptual.py`, `scripts/v3/match_presets.py`): CMA-ES
  refines the conversion so it sounds like the original.
- **Listening page with ratings** (`scripts/v3/listening_v3.py`): original / conversion / Harmonia side by
  side, 1-5 ratings and remarks stored in the page's database and read back.
- **v3 bank builder** (`scripts/v3/build_bank_v3.py`) and readable preset text from names/categories
  (`src/presets/labels.py`); `src/presets/filters.py` leaves out arpeggios, step sequences and templates.

### Listening rounds (Malo's ratings, 1-5)
| Round | What changed | OB-Xf | Surge XT | DX7 |
|---|---|---|---|---|
| 1 | first conversions, log-mel matching | 2.5 | 1.9 | 2.8 |
| 2 | perceptual metric, no added modules, stereo, 16-voice unison | 3.3 (17 better, 1 worse) | 3.1 (23 better, 0 worse) | — |
| 3 | fixes below (octave, cross-mod, oversampling, reverb, DX7 velocity) | 3.4 (9 better, 4 worse) | 3.3 (8 better, 6 worse) | 3.2 (8 better, 1 worse vs round 1) |

Round 3 is the frozen v3 sound: 82 ratings, 38 of them 4 or 5, 5 at 1. Remaining remarks are about some
saw-based Surge sounds still "metallic" or lacking depth/clarity ("Church", "Newton was evil", "Fuji"),
and "Tek stab" not resembling its original at all. More engine and matching work goes to a later version.

Round 1 showed the first optimizer "cheated": it lowered its log-mel distance by adding distortion (69 %
of presets), noise (50 %), LFO and effects the originals do not have. The perceptual metric (1/3-octave
timbre, envelope, movement, stereo width) follows the ratings better (Spearman -0.52 vs -0.32) and the
optimizer may no longer add absent modules.

### Fixed after round 2
- Surge scene octave (`a_octave`, 31 % of presets) and OB-Xf global transpose (33 %) were ignored:
  "one octave too high".
- Cross-modulation now works like OB-Xf's (osc1 bends osc2's pitch, harmonic with sync) instead of the
  reverse phase modulation, which turned synced presets inharmonic ("metallic").
- Grit/"metallic" character: 2x oversampling against aliasing of sync, cross-mod and distortion; the
  Freeverb reverb replaced by a modulated feedback delay network (tail spectral ripple 11.8 dB vs 19.0);
  slight analog pitch drift per oscillator.
- DX7 velocity follows the DX7 keyboard range (Dexed's option): E.PIANO 1 was played too hard.
- DX7 voices no longer depend on uninitialized memory (bit-exact repeat renders).
- Pitch envelope is exponential; OB-Xf pitch envelopes on sustained filter envelopes no longer detune.
- Matching weighs attack/impact more and tries the whole preset an octave up/down.

### Step 4 in progress: generating presets from text
- **v3 bank** (`scripts/v3/build_bank_v3.py`): 29,531 real presets heard by CLAP (27,414 DX7 voices,
  425 OB-Xf and 1,692 Surge presets whose Harmonia version scored <= 13, the threshold that separated
  round-2 ratings: 3.6/5 kept vs 2.4/5 dropped).
- **Human-only text corpus** (`scripts/v3/build_text_corpus_v3.py`): FSD50K titles/tags/labels and the
  presets' own names and categories (54,192 sentences, French by `opus-mt` except preset proper names);
  v2's template sentences are gone. The multilingual encoder retrained on it reaches a cosine to the CLAP
  teacher of 0.82 (EN) / 0.79 (FR) on held-out sentences (v2, with templates: 0.84 / 0.80).
- **Generators** (`src/v3/`, `scripts/v3/train_generator_v3.py`): diffusion models that draw DX7 voices or
  analog presets for a CLAP sound embedding, and a text -> sound prior trained on 97,883 human pairs
  (validation: cosine to the true sound 0.62 vs 0.23 for the raw text embedding, right sound in the top 10
  of 2,000 for 63 %). `scripts/v3/generate_v3.py` draws 8 candidates per engine, plays them, keeps the one
  CLAP hears closest; 1-2 s per prompt including rendering. `scripts/v3/listening_prompts.py` builds a blind
  A/B page (generated vs selected from the bank) for Malo's prompts.
- **First run on Malo's prompts** (`benchmarks/v3_prompts_p1.json`, 13 prompts, never used for training;
  "Piano", "Electric Piano", "Soft Piano" and "Soft Pad" are also names of real presets the prior learned
  from). CLAP similarity to the prompt's target sound, before listening:

  | Prompt | Generated | Selected from the bank |
  |---|---|---|
  | Piano | DX7 0.80 | SOFT-PIANO 0.91 |
  | Electric Piano | DX7 0.84 | E.PIANO 58 0.88 |
  | Soft Piano | DX7 0.67 | Steel 11 0.74 |
  | Piano in a big room | analog 0.30 | MIRop4ff 0.39 |
  | Techno Lead | DX7 0.75 | Need'sWork 0.81 |
  | Melencholic House Pad | analog 0.59 | Hardstyle Lead 1 0.59 |
  | Emotional old pluck | analog 0.72 | JP80x0 Pluck 0.71 |
  | Lead from melancholic hill of Gorillaz | analog 0.73 | Like Old Analog 0.74 |
  | Glass breaking | analog 0.33 | Glasscrush 0.49 |
  | Bongo Percussion | analog 0.40 | K SNARE 7 0.48 |
  | Soft Pad | analog 0.50 | SawPW1 0.54 |
  | Minimoog bass | analog 0.73 | Ignitor 0.79 |
  | Hypnotic Futuristic Pluck | analog 0.56 | Utopia 0.58 |

  Generation is within 0.05 of selection on 6 of 13 prompts and weakest on non-instrument sounds (glass,
  percussion) and on a place ("big room"), which the bank barely covers.
- **Blind A/B listening on those prompts** (Malo, 1-5 "matches the description"): **generation 2.0,
  selection 2.8**; generation better on 2 prompts, equal on 2, worse on 9. Remarks: generated glass and
  bongos "wtf", "Soft Piano" "bizarre"; selection's glass (DX7 "Glasscrush") and percussion rated 4, but
  "Soft Pad" got a saw lead and "Melencholic House Pad" a hardstyle lead; nothing sounds like Gorillaz.
- **Why** (1,433 OB-Xf/Surge presets whose category names a type: bass, pad, lead, pluck, keys, drums,
  brass, strings, fx):
  - CLAP's text side barely knows synth words: "a synthesizer pad sound" etc. finds the right type for
    25 % of presets (chance 18 %). Its audio side groups them better (5-NN type vote 56 %).
  - Through the v3 encoder and prior, type precision of the 20 nearest presets is 44 % for a bare type
    word ("lead" 5 %, "strings" 5 %, "fx" 0 %) and drops to 24 % with an adjective ("soft pad": 15 %).
  - The generator is then asked for the wrong sound, and keeping the candidate CLAP scores highest picks
    sounds that fool CLAP (as the first matching optimizer did) rather than good ones.
  - 93 % of the bank is unlabelled DX7 cartridge voices; only 2,117 analog presets.

### v3.1: understanding synth words, fairer choices, variations (after the blind test)
- **Synth vocabulary** (`src/v3/vocabulary.py`): 14 types (bass, pad, lead, pluck, keys, organ, bell, brass,
  strings, winds, voice, guitar, drums, fx) read from the presets' own categories and names and from FSD50K
  class labels; the table only groups their synonyms. 14,050 bank presets get a type (12,303 DX7 voices by
  name, 1,747 analog presets).
- **Synth space** (`src/v3/synth_space.py`, `scripts/v3/train_synth_space.py`): two small adapters on CLAP's
  sound embedding and on the multilingual text encoder, trained so a preset's sound sits next to its own
  name, category and comment (and their opus-mt French versions), with FSD50K for everyday sounds and a
  type head on both sides. A sound matches a prompt when it is close to it and of the type the prompt
  names. Held-out presets (10 %, never trained on):

  | Measure | v3.0 | v3.1 |
  |---|---|---|
  | type of the 20 closest presets, one word ("pad") | 56 % | 78 % |
  | same with an adjective ("soft pad", "dark lead"...) | 47 % | 64 % |
  | French, one word / with an adjective | 53 % / 47 % | 79 % / 69 % |
  | a preset's name finds its own sound in the top 10 of 2,972 | 20 % | 37 % |

  "lead" (30 %), "pluck" (40 %) and "fx" (5 %) stay weak: telling them apart from the sound alone is hard
  even in the training labels (Surge "Keys" are often plucks).
- **Tried and dropped**: measured attack/decay/sustain/brightness next to CLAP (type recognition 64.6 % ->
  65.3 %, CLAP already hears them); an engine chooser (trained on sounds, applied to text it was
  overconfident and contradicted the type).
- **Generators v3.1** (`scripts/v3/train_generator_v3.py --cond synth`): conditioned on the synth space plus
  the type, half the time from the preset's sound and half from its own texts, so a prompt is used directly
  (no text -> sound prior). DX7 voices whose names say what they are come up twice as often.
- **Variations** (`src/v3/diffusion.py`, `start`/`strength`): the generator starts from the real presets
  closest to the prompt, noised to 40 % of the schedule, and walks back towards the prompt.
- **Fairer choice of the candidate** (`scripts/v3/generate_v31.py`): close to the prompt with the right
  type, plus how typical it is of the generator's draws for that prompt (the best match alone favours
  sounds that fool the judge); both engines compete with no engine preference.
- **Blind listening round 2** on the same 13 prompts (Malo: "vraiment nul"): generation 2.0, variation 1.5,
  selection 1.7 (v3.0's selection: 2.8). The synth space learned to put a preset's sound next to its own
  name, so it picks presets whose *names* match the prompt (GlassBreak, MOOG BASS, Tabla 2), mostly amateur
  DX7 cartridge voices that sound poor (v3.0's choices by sound - Glasscrush, Ignitor, K SNARE 7 - had 4/5).
  Renders are identical to the round-3 page (checked on "Fuji"): the gap with that page is which presets
  are played, not the engine.

### v3.2 test: a curated library inside Harmonia, chosen by sound
- 13,002 of the 27,471 DX7 voices appear in a single cartridge of the community collection; the round-3
  page played Yamaha's factory voices and factory OB-Xf / Surge presets.
- `scripts/v3/curated_v32.py`: library of professional presets only (2,941 voices from Yamaha's own DX7
  cartridges - ROM1-4, VRC, TX816, DX5, DX7II - and the 2,110 OB-Xf / Surge presets Harmonia reproduces
  faithfully, minus 17 rated 2 or less in round 3), a few MB with their embeddings, shipped inside the model
  data; chosen by sound (v3.0 target) within the type the prompt names; a light variation (15 % of the
  noise schedule) of the chosen preset.
- Blind round 3 on the same prompts: curated choice 2.5, v3.0's choice in the whole bank 2.6, light
  variation 2.1. Curating the library does not beat choosing by sound in the whole bank, and the best
  choices stay around 3-4/5. Malo: even good presets "feel off" next to professional instruments: the
  next limit is the sound Harmonia makes, not the choice.

### Release `ai-v3.0.0` (GitHub prerelease, models only)
- `scripts/v3/export_model.py` gathers the trained models into `models/harmonia_v3/` with a manifest
  (sha256 per file); packaged by `scripts/v2/package_release.py` and pinned in `models/release.json`;
  `python scripts/v2/fetch_model.py --model harmonia_v3` downloads and verifies it.
- Contents (538 MB): multilingual text encoder (FR/EN), v3.0 prior and generators, v3.1 synth space and
  generators. No preset and no audio: the repository is public and the banks are not ours to
  redistribute; the bank and the curated library are rebuilt locally from their sources.
- Summary of the listening results the release stands on:

  | Listening | Question | Result |
  |---|---|---|
  | Round 3 (presets) | Harmonia vs original | OB-Xf 3.4, Surge 3.3, DX7 mode 3.2 |
  | Prompts 1 | matches the description | v3.0 generation 2.0, selection 2.8 |
  | Prompts 2 | idem | v3.1 generation 2.0, variation 1.5, selection 1.7 |
  | Prompts 3 | idem | curated choice 2.5, v3.0 choice 2.6, light variation 2.1 |

## [0.1.0] - Harmonia v2: Listened Presets and French Prompts
### Added
- **Offline synth renderer (`src/synth/`)**: numba port of the app's `Synth.cpp`, reverb settings and JUCE parameter conversions, validated sample by sample against the real C++ compiled with JUCE (relative RMS error: median 4e-4). `ENGINE_APP_1_0` reproduces the current app, `ENGINE_APP_1_1` the fixed engine (Harmonia-App#40). ~1000 notes/s on 14 cores.
- **v2 training pipeline (`scripts/v2/`, `make v2-train`)**:
  - preset bank: 200,000 presets (concepts, envelope/timbre archetypes, uniform) rendered at A2 and A4 and embedded with CLAP (`laion/larger_clap_general`);
  - FSD50K (51,197 real recordings) embedded with CLAP, used to train and to tune;
  - 66,000-sentence English corpus (FSD50K titles/tags/labels, concepts, compositional templates) translated to French with `opus-mt-en-fr` plus a sound-design glossary;
  - multilingual text encoder (`paraphrase-multilingual-MiniLM-L12-v2`) distilled into the CLAP text space: cosine to teacher 0.84 (EN) / 0.80 (FR);
  - parameter predictor trained on text -> closest-sounding bank preset pairs;
  - retrieval priors: near-inaudible presets excluded, CSLS hub penalty tuned on FSD50K eval;
  - ONNX export with per-channel int8 quantization (160 MB total, encoder cosine to fp32 ≥ 0.999).
- **`harmonia_v2` model served by `scripts/server.py`** (`model_name` `harmonia_v2`, aliases `model-3`, `v2`, `3`) with ONNX Runtime only: 0.4 s load, ~8 ms per request on CPU. Modes `retrieval` (default), `hybrid`, `neural`; `variation` for alternative presets. Same response format as v1 plus `generation`.
- **Release distribution**: `scripts/v2/package_release.py` and `scripts/v2/fetch_model.py` (GitHub release asset pinned by tag + sha256 in `models/release.json`); the Dockerfile fetches it at build time.
- **Benchmark with an independent judge** (`scripts/v2/benchmark_*.py`): 70 prompts in English and hand-written French, held out of all training data, scored by Microsoft CLAP 2023 against the text and against real FSD50K eval recordings. Listening page generator (`scripts/v2/listening_page.py`).

### Results
| System | Text EN | Text FR | Real EN | Real FR |
|---|---|---|---|---|
| v2 retrieval (default) | 0.234 | 0.225 | 0.387 | 0.370 |
| v2 hybrid | 0.214 | 0.216 | 0.376 | 0.374 |
| v1 synthetic, current app | 0.058 | -0.006 | 0.236 | 0.191 |
| random preset | 0.014 | 0.025 | 0.193 | 0.200 |

v2 beats today's v1 by +0.18 (EN) and +0.23 (FR) on the text score (95% CI excludes 0) and wins on 91-93% of the prompts. See [V2.md](V2.md).

### Changed
- Requirements: `onnxruntime==1.30.0` and `tokenizers==0.23.2` for serving; `onnx` and `numba` for CI; new `requirements-train.txt`.

## [0.0.20] - Preset Generation Outage Fix (transformers v5)
### Fixed
- **AI service returned HTTP 500 on every request since 2026-09-25**: the dependency bump to `transformers==5.10.0` broke encoder loading, because transformers v5 no longer infers the model type from the repo name and `prajjwal1/bert-tiny` ships a `config.json` without `model_type`. Preset generation in the app failed as a result.
  - `src/model.py` now resolves the encoder config through `load_encoder_config()` (falls back to `BertConfig` for legacy checkpoints) and exposes `load_tokenizer()`, used by `server.py`, `generate.py`, and `train.py`.
  - Generated presets are unchanged: outputs match the pre-incident stack (`torch 2.2.2` / `transformers 4.38.2`) within 1e-6 on both served models.
- **Dockerfile installed two torch versions**: it installed the CPU wheel of `torch 2.11.0`, then `requirement.txt` (`torch==2.6.0`) replaced it with the CUDA build from PyPI (several GB of `nvidia-*` packages). The CPU wheel now uses the version pinned in `requirement.txt`.

### Changed
- `transformers` pinned to `5.17.0` (`5.10.0` was yanked from PyPI) and `torch` to `2.13.0` (fixes GHSA-rrmf-rvhw-rf47).
- The Docker image now bakes the encoder and tokenizer at build time and runs with `HF_HUB_OFFLINE=1`, so the service no longer depends on the Hugging Face Hub at runtime.
- `deploy-ai-model.yml` fails the deploy when `/health` does not report `model_ready` within 150 seconds, and prints the container logs.
- Removed `deploy-ai.yml`: pushes to `dev` no longer deploy to production. Only merges to `main` deploy, as the release PR workflow states.

### Tests
- Added `tests/test_model_loading.py`: builds the model and tokenizer from a local BERT checkpoint whose `config.json` has no `model_type`, so CI catches this regression without network access.

## [0.0.19] - Dataset Profiles, Auto-Scaffolder, and Live Dashboard Pipeline
### Added
- **Pluggable Dataset Profiles (`src/dataset_profiles.py` + `src/profiles/`)**:
  - JSON-driven mapping from any source synth/dataset to the 20 charter parameters (one profile per source).
  - Strategy primitives: `direct`, `gated`, `max`, `bipolar_amount`, `routed_amount`, `constant`.
  - Canonical `src/profiles/sylenth1.json` (formerly hardcoded in `prepare_dataset.py`).
  - Auto-detection (`autodetect_profile`) based on declarative `detect.required_keys`.
- **Profile Scaffolder (`scripts/inspect_dataset.py`)**:
  - Scans an unknown NPY/JSON dataset, lists every native key, and ranks candidates per charter parameter via token-overlap + substring + SequenceMatcher scoring.
  - Generates an 80-90% pre-filled profile JSON (`--suggest-profile <name>`) ready for manual review.
  - Embedded `_suggestions` block surfaces the top-N alternatives for each charter parameter so the user can refine quickly.
- **Charterized Dataset Builder (`scripts/charterize_dataset.py`)**:
  - Converts a Sylenth1 (or any profile-described) dataset into a charter-shaped NPY/JSON (52,810 records on the current `cleaned_dataset.npy`).
  - `--profile sylenth1 | <name> | auto | legacy` switches the mapping logic at runtime.
  - Anchor presets injected automatically; can be disabled with `--no-anchors`.
- **Live Metrics Dashboard (`metrics_dashboard/`)**:
  - SPA front-end (`index.html`) with 6 views: Overview, Modèles, Entraînement, Presets générés, Activité, Charte 20 paramètres.
  - Muted dark palette (indigo / steel blue / plum) replacing the previous synthwave-rainbow palette.
  - Backend rewrite (`receiver.php`, `api.php`) with multi-kind event store (`events/`), per-model index (`models.json`), and read-only public API.
  - Diagnostic endpoint `GET api.php?action=probe` reports token source visibility without leaking the value.
  - `.env` fallback in `receiver.php` for hosts that strip Apache `SetEnv` (`.htaccess` blocks direct download).
- **Dashboard Event Pipeline (`src/dashboard_events.py`)**:
  - Unified publisher for `training`, `generation`, `command`, `system`, `dataset` events.
  - Local mirror under `metrics_dashboard/events/` always written even when the remote is unreachable.
  - `scripts/flush_events.py` replays the buffered queue once the server is reachable (idempotent: pushed files move to `_uploaded/`).
- **Local Snapshot Command (`scripts/dashboard_stats.py`)**:
  - Aggregates every model, eval report, generated preset, and benchmark history into a single JSON, then pushes it as a `system` event.
- **Terminal Wrapper (`scripts/harmonia.sh`)**:
  - `scripts/harmonia.sh <any command>` captures exit code + duration and publishes a `command` event.
- **Makefile targets**: `charterize-dataset`, `train-charter`, `dashboard-stats`, `dashboard-snapshot`, `dashboard-serve`.

### Changed
- **`prepare_dataset.py`**: emits a `command` dashboard event on completion (best-effort, never breaks the data pipeline).
- **`generate.py` / `server.py`**: every generation now publishes a `generation` event (CLI source vs HTTP source) carrying the prompt + 20 charter values + parameters dict.
- **`train.py`**: replaces the legacy push-only helper with `publish_training` (local mirror + remote POST), and emits a final `command` event with `epochs`, `batch_size`, `dataset_size`, `charter_mode`.
- **`.gitignore`**: tightened scope. Runtime artefacts (`metrics_dashboard/events/`, `*.charter.npy`, snapshots, secrets) stay ignored; dashboard sources, charter anchors (`data/raw/anchor_presets.json`), and registry metadata become tracked.

### Fixed
- **Reverb / Distortion gating**: the legacy `SYLENTH_KEY_MAP` short-circuited the gated synthesisers via `dict.setdefault`, so `Sw ReverbOnOff = 0` records still carried a non-zero `reverb_mix` into the dataset. The new profile path applies the gate correctly: 15,644 records (vs the ~15,495 source records with the switch off) now correctly land at `reverb_mix = 0`.

### Tests
- Added `tests/test_dataset_profiles.py`, `tests/test_inspect_dataset.py`, `tests/test_charterize_dataset.py`, `tests/test_dashboard_events.py`, `tests/test_dashboard_stats.py`.
- Suite grew from 27 → **61 tests** (all passing, bandit + compileall clean).

### Docs
- `metrics_dashboard/README.md` rewritten: architecture table, 6-view tour, push pipeline, deployment.
- `README.md` gains a "Dashboard live (harmonia.mcoet.com)" section with new make targets.

## [0.0.18] - Universal Charter, Perceptual Loss, and Anchor Presets
### Added
- **Centralized Charter (`src/charter.py`)**:
  - Implementation of a single source of truth for the 20 universal parameters.
  - Definition of `name`, `section`, `kind` (continuous/bipolar/discrete), `curve` (linear/log/exponential), physical range, and units.
  - Included discrete step definitions for P1/P2 (Waveforms) and P8 (Filter Types).
  - Utility helpers: `clamp_unit`, `snap_discrete`, `normalise_vector`, and `charter_metadata`.
- **Cold-start Dataset (`data/raw/anchor_presets.json`)**:
  - 12 hand-curated canonical presets with 3-4 prompt variations each (47 training points).
  - Covers: Hard Electro Lead, Soft Piano, Warm Pad, Deep Reese Bass, Acid Bass, Pluck, Snare, Kick, Bright Lead, Strings, Flute, Hard Bass.
- **API Endpoint**: Added `GET /charter` to allow the VST front-end to sync parameter metadata dynamically.

### Changed
- **Enhanced Model Architecture (`src/model.py`)**:
  - Replaced CLS token with **Masked Mean Pooling** for stable signal on short prompts (e.g., "Hard Electro Lead").
  - Upgraded head with `LayerNorm` + `GELU` + `Dropout`.
  - Specialized output heads: Sigmoid for continuous, centered-sigmoid for bipolar (P13), and **Softmax-to-steps** for discrete parameters (P1, P2, P8).
  - Maintained legacy CLS mode for backward compatibility with 214-param Sylenth1 models.
- **Dataset & Preprocessing (`src/dataset.py`, `scripts/prepare_dataset.py`)**:
  - Automatic detection of Charter mode with guaranteed P1..P20 ordering.
  - Improved mapping logic from Sylenth1 to Charter (e.g., Noise estimation from Osc B, Filter Env from xModEnv1).
  - Added prompt augmentation during training (random prefixes/suffixes) for better generalization.
- **Perceptual Training Logic (`scripts/train.py`)**:
  - Implemented **Weighted MSE Loss**: x2 on waveforms/filter types/cutoff, x1.5 on attack/release/distortion, and x0.6 on LFO-to-pitch to prioritize audible features.
- **Refined Inference API (`scripts/server.py`, `scripts/generate.py`)**:
  - `/generate` now returns a rich JSON: `parameters` (dict), `values` (flat list of 20 floats for C++), and `charter_metadata`.
  - All outputs are strictly clamped to [0,1] and snapped to discrete steps, ready for JUCE `NormalisableRange`.

### Fixed
- **Tests**: Rewrote `tests/test_prepare_dataset.py` to validate the new Charter schema.
- **Compatibility**: Verified that the 5 other core test files pass without modification despite the architecture shift.

## [0.0.17] - CI Reliability, Dashboard History, and Secret Hygiene
### Changed
- **CI hardening**:
  - Updated `HARMONIA/.github/workflows/ci.yml` to run real quality gates (no `|| true`, no `--no-deps`).
  - Standardized CI runtime to Python `3.12` and added compile check (`python -m compileall scripts src tests`).
- **Dependency compatibility**:
  - Simplified `numpy` pin in `requirements-ci.txt` and `requirement.txt` to `numpy==2.2.6` for stable installs across environments.
  - Aligned `torch` runtime pin in `requirement.txt` with CI (`torch==2.11.0`).
- **Training/runtime robustness**:
  - `scripts/train.py` now evaluates on the active device (CPU/MPS) and saves best checkpoint by validation MSE when available.
  - Fixed training loop mode switch so the model returns to `train()` after validation, preventing accidental eval-mode epochs.
  - Improved Apple Silicon detection with `torch.backends.mps.is_available()`.
  - Re-enforced strict safe model loading in `scripts/server.py` (`weights_only=True` required).
- **Metrics dashboard pipeline**:
  - `metrics_dashboard/receiver.php` now writes both `latest_metrics.json` and rolling `history_metrics.json`.
  - `metrics_dashboard/index.html` now plots model evolution from `history_metrics.json` and per-parameter errors with a radar chart.

### Security
- Removed hardcoded token fallback from `metrics_dashboard/receiver.php` and require server-side `METRICS_PUSH_TOKEN` configuration.
- `metrics_dashboard/push_metrics.sh` now requires `METRICS_TOKEN` from environment (no plaintext secret in code).

### Docs
- Added/clarified "Commandes de test" in `README.md` and aligned CI notes in `DOC.md`.
- Updated `metrics_dashboard/README.md` examples to match current metrics schema (no accuracy key).

## [0.0.16] - Bandit B310 Fix on Metrics Publisher
### Changed
- **Security hardening** (`src/metrics_publisher.py`):
  - Replaced `urllib.request.urlopen` with `http.client` to remove Bandit `B310` finding.
  - Added strict URL validation before push (scheme + host allowlist).
  - Restricted metrics push hosts to trusted targets (`harmonia.mcoet.com`, `localhost`, `127.0.0.1`).

### Verified
- `python -m bandit -q -r scripts src` -> no `B310` issue
- `python -m pytest -q` -> `25 passed`

## [0.0.15] - Secure Token Handling and Push-Time Dashboard Refresh
### Changed
- **Secret hygiene**:
  - Removed any hardcoded token fallback from `metrics_dashboard/push_metrics.sh`.
  - Hardened `metrics_dashboard/receiver.php` to reject requests when `METRICS_PUSH_TOKEN` is not configured.
- **Dashboard richness**:
  - Upgraded `metrics_dashboard/index.html` to display many more metrics:
    - dynamic KPI cards,
    - complete numeric metrics table,
    - history/per-parameter chart selection,
    - resilient fallback behavior.
- **Push-time refresh**:
  - Updated `.github/workflows/harmonia-ci.yml` to publish a CI metrics payload to `https://harmonia.mcoet.com/receiver.php` on every push (when `METRICS_PUSH_TOKEN` secret is present).

### Security
- Removed local plaintext token file and kept only `.gitignore`-protected secret paths (`metrics_dashboard/.env.local`, generated metrics files).

## [0.0.14] - Automatic Metrics Push and Safer Git Add
### Added
- **Automatic metrics publisher**:
  - Added `src/metrics_publisher.py` to send evaluation reports directly to `https://harmonia.mcoet.com/receiver.php`.
  - Added `scripts/push_latest_metrics.py` to push the latest report from benchmark history.
- **Training hook**:
  - `scripts/train.py` now pushes `eval_*.json` automatically after each successful training run.

### Changed
- **Makefile behavior**:
  - `make test` now runs tests and then pushes latest metrics (non-blocking when no report/token is available).
- **Git safety**:
  - Updated `HARMONIA/.gitignore` to ignore local dashboard secrets and generated metrics files:
    - `metrics_dashboard/.env.local`
    - `metrics_dashboard/latest_metrics.json`
    - `benchmarks/reports/`

### Verified
- `make test` -> `25 passed` + metrics pushed (`HTTP 200`)
- `HARMONIA_EPOCHS=1 make train DATASET=data/processed/presets.json` -> success + auto-push (`HTTP 200`)

## [0.0.13] - Static Web Metrics Dashboard (Apache/PHP)
### Added
- **Static dashboard package** (`HARMONIA/metrics_dashboard/`):
  - `index.html` with TailwindCSS + Chart.js for modern metric visualization.
  - Dynamic cards for key KPIs (`Loss`, `Accuracy`, `MSE`, `MAE`).
  - Automatic chart rendering from `per_param_mse` (bar) or `loss_history` (line).
  - Built-in fallback to simulated JSON when `latest_metrics.json` is missing.
- **Secure receiver endpoint**:
  - Added `receiver.php` to accept `POST` metric payloads and save `latest_metrics.json`.
  - Supports token auth via `Authorization: Bearer ...` or `POST token`.
  - Supports upload modes: multipart file (`metrics_file`), `metrics_json`, or raw JSON body.
- **Push automation scripts**:
  - Added `push_metrics.sh` (curl-based) and `push_metrics.py` (requests-based).
  - Designed for direct integration at the end of PyTorch training pipelines.
- **Usage documentation**:
  - Added `HARMONIA/metrics_dashboard/README.md` with deployment, local test, and production push examples.

### Verified
- `bash -n HARMONIA/metrics_dashboard/push_metrics.sh` -> success
- `python3 -m py_compile HARMONIA/metrics_dashboard/push_metrics.py` -> success
- PHP syntax check not run locally in this environment (`php` executable unavailable).

## [0.0.12] - CI Python Compatibility Fix
### Changed
- **GitHub Actions runtime**:
  - Updated `.github/workflows/harmonia-ci.yml` from Python `3.10` to `3.12`.
  - This fixes CI installation failure with `numpy==2.3.4` (which requires Python `>=3.11`).
- **Dependency compatibility guard**:
  - Added environment markers in `HARMONIA/requirements-ci.txt` and `HARMONIA/requirement.txt`:
    - `numpy==2.3.4` for Python `>=3.11`
    - `numpy==2.2.6` for Python `<3.11`
  - This keeps installs resilient if a runner/local environment uses Python `3.10`.

### Verified
- CI dependency set remains unchanged (`HARMONIA/requirements-ci.txt`, `HARMONIA/requirements-dev.txt`).
- Local quality checks still pass (`pytest`, `bandit`, `compileall`).

## [0.0.11] - Training Profiles and Practical AI Test Commands
### Added
- **Makefile execution profiles**:
  - Added `train-fast` (1 epoch smoke), `train-good` (20 epochs baseline), `generate-cli`, and `pipeline-local`.
  - Added `PROMPT` and `OUTPUT` variables for quick CLI inference tests.
- **Metrics utility script**:
  - Added `scripts/metrics_summary.py` to print latest local metrics and training-time estimates from benchmark history.
- **Operational testing flow**:
  - `pipeline-local` now provides a single local sequence: checks -> fast train -> local metrics -> generation test.

### Changed
- **Documentation UX**:
  - Updated `HARMONIA/README.md` with explicit commands to:
    - train using `cleaned_dataset.npy`,
    - test AI inference in CLI and API mode,
    - read metrics at each step.
  - Added practical training duration estimates based on latest observed benchmark run.
- **Makefile reliability**:
  - `metrics-local` and `estimate-train-time` now call `scripts/metrics_summary.py` (stable shell behavior).

### Verified
- `make check` -> success (`25 passed`, bandit warnings only on existing `# nosec B615`, compileall ok)
- `HARMONIA_EPOCHS=1 HARMONIA_BATCH_SIZE=32 make train-cleaned` -> success on `data/cleaned_dataset.npy`
- `make metrics-local` and `make estimate-train-time` -> success

## [0.0.10] - Makefile Workflow and End-to-End Metrics
### Added
- **Operational Makefile**:
  - Added `HARMONIA/Makefile` with practical targets:
    - `setup`, `check`, `test`, `security`, `compile`
    - `train`, `train-cleaned`, `train-and-report`
    - `serve`, `metrics-local`, `metrics-api`, `estimate-train-time`
- **Metrics Everywhere Workflow**:
  - Added local metrics helpers that print latest benchmark and evaluation report after training.
  - Added training time estimation from benchmark history (`estimate-train-time`).

### Changed
- **Training Configurability** (`scripts/train.py`):
  - `HARMONIA_EPOCHS`, `HARMONIA_BATCH_SIZE`, and `HARMONIA_LR` now override defaults safely.
  - This allows fast debug runs and longer production-quality runs without code edits.
- **Documentation**:
  - Updated `HARMONIA/README.md` and `HARMONIA/DOC.md` with Makefile usage, cleaned dataset training commands, and metrics commands.

### Verified
- `python -m pytest -q` -> `25 passed`
- `python -m bandit -q -r scripts src` -> success (warnings only on existing `# nosec B615` markers)
- `python -m compileall scripts src tests` -> success

## [0.0.9] - Dynamic NPY Training Dataset and Parameter Scaling
### Added
- **Dynamic Dataset Loader**:
  - `src/dataset.py` now supports `.npy` datasets via `numpy.load(..., allow_pickle=True)`.
  - Added flattening for `parameters.continuous`, `parameters.binary`, and `parameters.categorical` into one 1D tensor.
  - Added deterministic key extraction (`param_keys`) to keep output order stable across training and inference.
- **Categorical Normalization**:
  - Added optional per-key categorical normalization (divide by dataset max) to keep targets in `[0, 1]`.
- **Test Coverage**:
  - Added `tests/test_dataset.py` to validate `.npy` loading, key ordering, flattening, and categorical normalization.
  - Extended `tests/test_train.py` for dynamic dataset path resolution and dynamic metric key mapping.

### Changed
- **Dynamic Output Dimension**:
  - `scripts/train.py` now derives `plugin_param_count` from `len(dataset.param_keys)` instead of fixed constants.
  - `src/model.py` now validates output dimension and derives encoder hidden size from the loaded transformer config.
- **Training Dataset Selection**:
  - Added `HARMONIA_DATASET_PATH` environment override.
  - Training now auto-prefers `data/processed/presets.npy` when available, then falls back to `presets.json`.
- **Dependency Compatibility Fixes**:
  - Upgraded `torch` pin from `2.8.0` to `2.11.0` in `requirement.txt` and `requirements-ci.txt`.
  - Upgraded `bandit` pin from `1.8.6` to `1.9.4` in `requirements-dev.txt` to avoid Python 3.14 AST failures.
  - Upgraded `numpy` pin from `2.0.2` to `2.3.4` in `requirements-ci.txt` and added `numpy==2.3.4` to `requirement.txt`.
- **Documentation**:
  - Updated `HARMONIA/README.md` and `HARMONIA/DOC.md` with `.npy` dataset workflow and dynamic parameter behavior.

### Verified
- `python -m pytest -q` -> `25 passed`
- `python -m compileall scripts src tests` -> success
- `python -m bandit -q -r scripts src` -> success (warnings only on existing `# nosec B615` markers)

## [0.0.8] - CI Workflow Restoration and Test Commands Clarity
### Added
- **Missing CI Workflow**:
  - Added `.github/workflows/harmonia-ci.yml` at repository root.
  - Workflow runs on push/pull_request and executes the 3 project checks from `HARMONIA/`:
    - `python -m pytest -q`
    - `python -m bandit -q -r scripts src`
    - `python -m compileall scripts src tests`

### Changed
- **Documentation Clarity**:
  - Added an explicit `Commandes de test (local + CI)` section in `HARMONIA/README.md`.
  - Added the same `Commandes de test (local + CI)` section in `HARMONIA/DOC.md`.
  - Documented a quick targeted run command (`tests/test_server.py`) for fast local validation.

## [0.0.7] - Security Hardening and Dependency Cleanup
### Added
- **Dynamic Inference Configuration**:
  - Training metadata now includes `plugin_param_count`, `param_keys`, and `tokenizer_max_length`.
  - Inference (`scripts/server.py`, `scripts/generate.py`) now reads these metadata fields to avoid hardcoded parameter mapping.
- **Token Context Guard**:
  - `POST /generate` now rejects prompts that exceed model token context instead of silently truncating.

### Changed
- **Dependency Hygiene**:
  - Replaced the non-portable environment dump in `requirement.txt` with a minimal, portable runtime manifest.
- **Safer Model Loading**:
  - Removed unsafe fallback model loading path and enforce `weights_only=True` for PyTorch deserialization.
- **CLI Consistency**:
  - `scripts/generate.py` now mirrors server-side behavior for model/token context validation and metadata-driven output keys.

### Verified
- `python3 -m pytest -q` -> `21 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.6] - Metrics Endpoint, Model Versioning, and CI
### Added
- **API Metrics Endpoint**:
  - Added `GET /metrics/latest` in `scripts/server.py` to expose the latest benchmark entry and latest evaluation report payload.
- **Automatic Model Versioning**:
  - `scripts/train.py` now stores artifacts in `saved_models/<model_version>/`.
  - Added `saved_models/latest_model.json` pointer to resolve the most recent model automatically.
  - Added `src/artifact_registry.py` to centralize model artifact path resolution.
- **CI Automation**:
  - Added GitHub Actions workflow `.github/workflows/harmonia-ci.yml` running:
    - `python -m pytest -q`
    - `python -m bandit -q -r scripts src`
    - `python -m compileall scripts src tests`
  - Added `requirements-ci.txt` for portable CI dependency installation.

### Changed
- **Runtime Model Resolution**:
  - `scripts/server.py` and `scripts/generate.py` now auto-resolve the latest versioned model from `saved_models/`.
- **Training Metadata**:
  - `scripts/train.py` now writes versioned metadata and updates the latest-pointer file each run.
- **Tests and Docs**:
  - Extended `tests/test_server.py` and `tests/test_train.py` for new API and versioning behavior.
  - Updated `README.md` and `DOC.md` with new endpoint, model layout, and CI commands.

### Verified
- `python3 -m pytest -q` -> `19 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.5] - P1 Evaluation Metrics and Model Traceability
### Added
- **Validation Metrics Pipeline**:
  - `scripts/train.py` now creates a deterministic train/validation split (`HARMONIA_VAL_SPLIT`, default `0.2`).
  - Validation metrics now include global MAE/MSE and per-parameter MAE/MSE.
  - Added evaluation report artifacts in `benchmarks/reports/eval_*.json`.
- **Model Metadata Artifact**:
  - Training now writes `saved_models/my_plugin_ai.meta.json` with model version/hash and training context.
- **API Traceability**:
  - `GET /health` now exposes `model_version` and `model_hash`.
  - `POST /generate` now includes `model_version` and `model_hash` in response `metadata`.
- **Test Coverage**:
  - Extended `tests/test_train.py` to cover split sizing and evaluation report creation.
  - Extended API tests for traceability fields.

### Changed
- **Benchmark Enrichment**:
  - `benchmarks/history.json` entries now include train/validation sizes, evaluation summary, report path, and model identifiers.
- **Documentation**:
  - Updated `README.md` and `DOC.md` with P1 outputs and run/test commands (commands unchanged).

### Verified
- `python3 -m pytest -q` -> `15 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.4] - Reliability, Security Baseline, and Tests
### Added
- **API Reliability**:
  - Added `GET /health` endpoint in `scripts/server.py` for runtime status checks.
  - Added strict payload validation for `POST /generate` (`prompt` type, emptiness, max length 512).
- **Reproducible Training**:
  - Added deterministic seed support in `scripts/train.py` via `HARMONIA_SEED` (default: `42`).
  - Benchmark entries now include `seed` and `dataset_size` metadata.
- **Test Suite**:
  - Added `tests/test_server.py` for API behavior and error-path validation.
  - Added `tests/test_prepare_dataset.py` for parser and conversion checks.
  - Added `tests/test_train.py` for deterministic seed and benchmark metadata tests.
  - Added `tests/test_e2e_smoke.py` for a lightweight prepare+generate smoke path.
- **Dev Tooling**:
  - Added `requirements-dev.txt` with `pytest` and `bandit`.

### Changed
- **Model Supply Chain Controls**:
  - Added model source pinning support with `HARMONIA_MODEL_ID` and `HARMONIA_MODEL_REVISION`.
  - Applied pinned revision loading for tokenizer/model in `scripts/train.py`, `scripts/server.py`, `scripts/generate.py`, and `src/model.py`.
- **Safer Weight Loading**:
  - Updated model loading paths to prefer `torch.load(..., weights_only=True)` with compatibility fallback.
- **Auto-Training Execution Safety**:
  - Added a strict script allowlist and resolved-path checks before subprocess execution in `scripts/auto_trainer.py`.
- **Documentation**:
  - Updated `README.md` and `DOC.md` with health endpoint, payload validation, reproducible training, and dev validation commands.

### Verified
- `python3 -m pytest -q` -> `13 passed`
- `python3 -m compileall scripts src tests` -> success
- `python3 -m bandit -q -r scripts src` -> clean run (no reported findings)

## [0.0.3] - Runtime Stability and Technical Audit
### Added
- **Technical Audit**: Added `PROJECT_TECH_AUDIT.md` with current limitations, engineering risks, and a concrete 30-60-90 roadmap.
- **Runtime Guards**:
  - `server.py` now returns `503` when model weights are missing/invalid.
  - `train.py` now stops early on empty datasets.
  - `auto_trainer.py` now ignores non-`.txt` files.

### Changed
- **Path Handling**: Switched critical scripts to absolute paths derived from script location:
  - `prepare_dataset.py`
  - `train.py`
  - `generate.py`
  - `server.py`
  - `benchmark_viewer.py`
  - `auto_trainer.py`
- **Documentation Alignment**:
  - Updated `README.md` and `DOC.md` to match executable commands from repository root.

### Fixed
- **Server Import**: Fixed `ModuleNotFoundError` in `server.py` by ensuring `src/` is available in `sys.path`.
- **Dataset Output Path**: `prepare_dataset.py` default output now correctly targets `data/processed/presets.json`.
- **Benchmark Directory Logic**: `train.py` now creates the real benchmark directory before writing history.
- **Benchmark Viewer Path**: `benchmark_viewer.py` now reads benchmark history correctly regardless of current working directory.
- **Auto-Trainer Subprocesses**: `auto_trainer.py` now executes scripts with deterministic absolute paths and `sys.executable`.

## [0.0.2] - Refactored Architecture
### Changed
- **Project Structure**: Reorganized repository into professional standard structure.
    - `src/`: Core logic (Model definition, Dataset class).
    - `scripts/`: Executable scripts (Train, Generate, Auto-Trainer).
    - `data/`: Storage for raw and processed datasets.
- **Model Definition**: Consolidated `model.py` into `src/` to prevent duplication.
- **Imports**: Updated all scripts to dynamically find the `src` package.

### Fixed
- **Auto Trainer**: Fixed subprocess paths to work within the new `scripts/` directory.
- **Training**: Fixed data paths in `train.py` to point to `../data/`.


## [0.0.1] : Server and Benchmark
### Added
- **Flask API Server (`server.py`)**: A lightweight web server allowing JUCE plugins to request parameters via HTTP POST requests instead of file I/O.
- **Auto-Trainer (`auto_trainer.py`)**: A `watchdog` system that monitors a `drop_zone` folder. Dropping a text file now automatically triggers dataset ingestion, training, and benchmarking.
- **Benchmark System**:
- `train.py` now logs training duration, hyperparameters, and final loss to `benchmarks/history.json`.
- `benchmark_viewer.py` provides a CLI table view to track model improvement (or regression) over time.
- **9-Parameter Mapping**: Updated the model output layer to support specific target knobs: *Frequency, Attack, Cutoff, Decay, Volume, Sustain, Resonance, Release, Waveform*.
- **Robust Data Parser**: `prepare_dataset.py` now handles malformed text dumps (missing parentheses) gracefully.

### Changed
- **Model Architecture**: Switched output layer size from 50 to 9 to match the target VST requirements.
- **Inference Output**: `generate.py` now produces a named JSON dictionary (e.g., `"cutoff": 0.5`) instead of a raw list of floats, making integration with C++ easier.

### Removed
- **Spectrogram Support**: Deprecated the audio-spectrogram approach in favor of direct parameter regression (Text-to-Param).
