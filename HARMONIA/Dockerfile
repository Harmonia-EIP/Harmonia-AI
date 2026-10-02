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

EXPOSE 5000

CMD ["gunicorn", "--chdir", "scripts", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]
