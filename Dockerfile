# CPU-only test image for eeg_steptype: `docker run` executes the full pytest suite.
#
#   docker build -t eeg-steptype .
#   docker run --rm eeg-steptype
#
# TensorFlow is deliberately not installed. The Keras models import it lazily,
# so the suite runs without it: the live Keras-parity tests skip and the
# PyTorch EEGNet port (models/eegnet_torch.py) is exercised instead.
FROM python:3.12-slim

# EEG_STEPTYPE_SKIP_FSAVERAGE_TESTS: the one test that resolves the installed
# fsaverage BEM (a network download via mne.datasets.fetch_fsaverage) skips.
ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    EEG_STEPTYPE_SKIP_FSAVERAGE_TESTS=1

# XGBoost's manylinux wheel links against the system OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, against a stub package, so editing the source does not
# invalidate this (large) layer. torch/torchaudio come from the CPU index up
# front so no later requirement resolves the multi-GB CUDA wheels from PyPI.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
RUN mkdir -p src/eeg_steptype \
    && touch src/eeg_steptype/__init__.py \
    && pip install torch torchaudio --index-url "${TORCH_INDEX}" \
    && pip install -e ".[dev,torch]" --extra-index-url "${TORCH_INDEX}"

COPY . .

CMD ["pytest", "-q"]
