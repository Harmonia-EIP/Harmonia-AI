# DX7 renderer (research)

`dx7_render.cpp` plays one DX7 voice with the FM core of Dexed (`Source/msfa`, Copyright Google Inc. and
Pascal Gauthier, Apache License 2.0). It drives one `Dx7Note` the way Dexed's processor does for a single
key (64-sample blocks, LFO, key-up, DC filter) and uses msfa's own `FmCore` ("Modern" engine), so the same
code can run in the Harmonia app.

`shim/` replaces the few Dexed headers msfa includes (tracing, MTS-ESP microtuning, Tunings) with
standard-tuning stand-ins. Build with `python scripts/v3/build_dx7.py`; the msfa sources are fetched by
`scripts/v3/fetch_presets.py --only dexed_src`.
