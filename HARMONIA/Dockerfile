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

EXPOSE 5000

CMD ["gunicorn", "--chdir", "scripts", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]
