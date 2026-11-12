FROM python:3.11

WORKDIR /app

COPY requirement.txt .
# Install the CPU-only wheel of the torch version pinned in requirement.txt, so the
# install below finds it already satisfied instead of pulling the CUDA build from PyPI.
RUN TORCH_VERSION="$(sed -n 's/^torch==//p' requirement.txt)" && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" && \
    pip install --no-cache-dir -r requirement.txt

COPY . .

# Bake the encoder and tokenizer into the image so the service starts without
# reaching the Hugging Face Hub.
RUN python -c "from src.model import TextToParams, load_tokenizer; load_tokenizer(); TextToParams()"
ENV HF_HUB_OFFLINE=1

# v2 model: downloaded from the GitHub release pinned in models/release.json. Optional at build
# time: without it, requests for harmonia_v2 answer 503 and the v1 models keep working.
RUN python scripts/v2/fetch_model.py || echo "harmonia_v2 not installed"

# v3 model (research branch): weights from the GitHub release pinned in models/release.json, and the DX7
# renderer compiled from Dexed's msfa FM core (Apache-2.0, fetched at a pinned commit). Both are optional
# at build time: without the weights harmonia_v3 answers 503, without the renderer it answers with the
# analog engine only. The last step compiles the analog engine once, so the first request does not wait.
RUN python scripts/v2/fetch_model.py --model harmonia_v3 || echo "harmonia_v3 not installed"
RUN (python scripts/v3/fetch_presets.py --only dexed_src --out /opt/harmonia-presets && \
     python scripts/v3/build_dx7.py --msfa /opt/harmonia-presets/dexed_src/Source/msfa) || echo "DX7 renderer not built"
RUN python -c "import numpy as np; from src.synth import v3_params as P; from src.synth.engine_v3 import render_batch; render_batch(P.DEFAULTS[None, :], np.array([60]), 0.8, 0.1, 0.2, 48000, 7)" || echo "v3 engine not compiled"

EXPOSE 5000

CMD ["gunicorn", "--chdir", "scripts", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]
