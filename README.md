# Harmonia-AI

Main project documentation now lives in `HARMONIA/`.

Start here:
- `HARMONIA/README.md` for setup and quickstart
- `HARMONIA/DOC.md` for technical documentation
- `HARMONIA/PROJECT_TECH_AUDIT.md` for limitations, risks, and roadmap

Quick run from repository root:

```bash
python3 HARMONIA/scripts/prepare_dataset.py
python3 HARMONIA/scripts/train.py
python3 HARMONIA/scripts/generate.py "Soft Piano" --output HARMONIA/data/processed/test_preset.json
python3 HARMONIA/scripts/server.py
```

## License

Harmonia is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may use, modify and share it for noncommercial purposes (personal use, study, research, teaching). Commercial use is reserved to the Harmonia team; contact the Harmonia-EIP team for a commercial license.
