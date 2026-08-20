#!/usr/bin/env bash
# NeuroProfile — local RTX 4060 (Linux) GPU setup.

set -euo pipefail

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

# 0. venv gate 
# Everything must land in an isolated env, or the pins fight system Python.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d neuroprofile-gpu ]]; then
    say "Creating venv ./neuroprofile-gpu"
    python3 -m venv neuroprofile-gpu
  fi
  warn "Not inside a venv. Activate it and re-run:"
  echo "    source neuroprofile-gpu/bin/activate && bash setup_gpu.sh"
  exit 1
fi
say "venv active: $VIRTUAL_ENV"

# 1. python version
PYV=$(python -c 'import sys; print("%d.%d"%sys.version_info[:2])')
say "Python $PYV"
python - <<'PY' || die "Python >= 3.11 required (tribev2 pin)."
import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)
PY

# 2. driver check (informational)
say "GPU / driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  warn "nvidia-smi not found — is the NVIDIA driver installed? torch.cuda will fail without it."
fi

# 3. matched CUDA 12.1 torch triple, then numpy pin
# torch FIRST so it can't drag in a numpy it prefers; numpy pinned SECOND.
say "Installing torch/vision/audio cu121 triple"
python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.2.6

# 4. does torch see the GPU?
say "Verifying torch <-> CUDA"
python - <<'PY' || die "torch.cuda.is_available() is False — driver/torch mismatch. Stop here."
import torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__)
print("cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
raise SystemExit(0 if ok else 1)
PY

# 5. CPU-side + pipeline deps 
say "Installing pipeline deps"
python -m pip install \
  qdrant-client nibabel fastapi python-multipart uvicorn yt-dlp pytest scipy ffmpeg-python


#  7. Hugging Face login (gated Llama-3.2-3B) 
# TRIBE pulls Llama-3.2-3B, which is gated. You also must click "agree" on the
# facebook/tribev2 and Llama-3.2-3B model pages in a browser, or downloads 403.
say "Hugging Face auth"
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "Already logged in as: $(python -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')"
else
  warn "Not logged in. Launching 'hf auth login' — paste a read token from"
  warn "  https://huggingface.co/settings/tokens"
  hf auth login
fi

# 8. NLTK punkt_tab (whisperX needs it, once per env) 
say "NLTK punkt_tab"
python - <<'PY'
import os; os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
import nltk, os.path as p
for r in ["punkt_tab", "punkt"]:
    nltk.download(r)
tgt = p.join(nltk.data.path[0], "tokenizers", "punkt_tab", "english") if nltk.data.path else ""
print("punkt_tab present:", any(p.exists(p.join(d, "tokenizers", "punkt_tab", "english")) for d in nltk.data.path))
PY

say "Setup done."
echo "Next:  python scripts/smoke_test.py <short_clip.mp4 or url>   # OOM gate on the 8 GB card"
echo "Then:  python scripts/batch_encode.py --corpus scripts/corpus.txt"
