#!/usr/bin/env bash
# Download RVC pretrained model files in the background.
# Idempotent: skips files already present.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ASSETS="Retrieval-based-Voice-Conversion-WebUI/assets"
HF="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

mkdir -p "$ASSETS/hubert" "$ASSETS/rmvpe" "$ASSETS/pretrained_v2" "$ASSETS/weights"

dl() {
    local url="$1" dest="$2"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "[skip] $dest ($(du -h "$dest" | cut -f1))"
        return
    fi
    echo "[get ] $url -> $dest"
    curl -fL --retry 3 -o "$dest" "$url"
    echo "[done] $dest ($(du -h "$dest" | cut -f1))"
}

dl "$HF/hubert_base.pt"            "$ASSETS/hubert/hubert_base.pt"
dl "$HF/rmvpe.pt"                  "$ASSETS/rmvpe/rmvpe.pt"
dl "$HF/rmvpe.onnx"                "$ASSETS/rmvpe/rmvpe.onnx"
dl "$HF/pretrained_v2/f0G40k.pth"  "$ASSETS/pretrained_v2/f0G40k.pth"
dl "$HF/pretrained_v2/f0D40k.pth"  "$ASSETS/pretrained_v2/f0D40k.pth"

echo "ALL_DONE"
